from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .documents import IngestedDocument
from .extraction import ExtractionContractError, consensus, validate_extraction_payload
from .models import EvidenceFact, EvidenceGraph, FactStatus, SourceSpan


FACT_TYPES = (
    "procedure", "service", "diagnosis", "condition", "finding", "symptom",
    "anatomy", "medication", "device", "units", "encounter_context", "modifier_evidence",
)
FACT_RELATIONS = (
    "diagnosis_supports_service", "component_of", "integral_to", "separate_site",
    "separate_session", "repeat_service", "laterality_of", "quantity_of",
    "modifier_applies_to",
)


FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts", "relationships"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["fact_type", "text", "status", "start_offset", "end_offset", "certainty", "attributes"],
                "properties": {
                    "fact_type": {"type": "string", "enum": list(FACT_TYPES)},
                    "text": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "enum": [item.value for item in FactStatus]},
                    "start_offset": {"type": "integer", "minimum": 0},
                    "end_offset": {"type": "integer", "minimum": 1},
                    "certainty": {"type": "string", "minLength": 1},
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "value"],
                            "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_index", "target_index", "relation"],
                "properties": {
                    "source_index": {"type": "integer", "minimum": 0},
                    "target_index": {"type": "integer", "minimum": 0},
                    "relation": {"type": "string", "enum": list(FACT_RELATIONS)},
                },
            },
        },
    },
}


SYSTEM_INSTRUCTION = """You extract only source-grounded clinical and administrative facts from a medical note.
Never suggest, copy, infer, or return any billing/classification identifier, medical code, code family, or modifier.
Preserve the shortest verbatim phrase supporting each fact and return its exact zero-based character offsets in DOCUMENT_TEXT.
Classify whether an item was performed, planned, ordered, historical, considered, negated, or unknown.
Use present for an asserted current diagnosis, condition, finding, or symptom; use performed for a completed procedure or service.
Extract anatomy, laterality, encounter timing, technique, findings, diagnosis wording, units, provider/setting clues, and documented relationships as attributes.
Do not add facts that are not stated. Do not convert an ambiguous abbreviation into a single expansion; preserve it verbatim.
Every fact must have a source span. Return only the required JSON structure."""


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    kind: str
    model: str
    endpoint: str
    api_key_env: str
    timeout_seconds: int = 120
    max_attempts: int = 3
    max_document_characters: int = 250_000
    reasoning_effort: str | None = None
    phi_transmission_env: str = "PHI_TRANSMISSION_ALLOWED"


def load_profiles(path: Path) -> tuple[ProviderProfile, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    profiles = tuple(ProviderProfile(**item) for item in value.get("profiles", []))
    if len(profiles) < 2 or len({item.kind for item in profiles}) < 2:
        raise ProviderError("independent extraction requires at least two distinct provider kinds")
    if len({item.name for item in profiles}) != len(profiles):
        raise ProviderError("provider profile names must be unique")
    return profiles


class JsonHttpClient:
    RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

    @classmethod
    def post(cls, profile: ProviderProfile, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            profile.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"content-type": "application/json", **headers},
            method="POST",
        )
        for attempt in range(1, profile.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=profile.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                retryable = error.code in cls.RETRYABLE_STATUS
                error.read(2048)
                if not retryable or attempt == profile.max_attempts:
                    raise ProviderError(f"{profile.name} request failed with HTTP {error.code}") from error
                retry_after = error.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt - 1)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                if attempt == profile.max_attempts:
                    raise ProviderError(f"{profile.name} transport returned no valid response") from error
                delay = 2 ** (attempt - 1)
            time.sleep(delay + random.random() * 0.25)
        raise ProviderError(f"{profile.name} exhausted retries")


class EvidenceProvider:
    def __init__(self, profile: ProviderProfile) -> None:
        self.profile = profile

    def extract(self, document: IngestedDocument) -> EvidenceGraph:
        if len(document.text) > self.profile.max_document_characters:
            raise ProviderError(f"{self.profile.name} document exceeds configured extraction limit")
        api_key = os.environ.get(self.profile.api_key_env, "")
        if not api_key:
            raise ProviderError(f"credential environment variable is missing for {self.profile.name}")
        if os.environ.get(self.profile.phi_transmission_env, "").casefold() not in {"1", "true", "yes"}:
            raise ProviderError(f"PHI transmission is not explicitly authorized for {self.profile.name}")
        prompt = f"DOCUMENT_TEXT_START\n{document.text}\nDOCUMENT_TEXT_END"
        if self.profile.kind == "openai":
            response = JsonHttpClient.post(
                self.profile,
                self._openai_payload(prompt),
                {"authorization": f"Bearer {api_key}"},
            )
            payload = self._parse_openai(response)
        elif self.profile.kind == "anthropic":
            response = JsonHttpClient.post(
                self.profile,
                self._anthropic_payload(prompt),
                {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            payload = self._parse_anthropic(response)
        else:
            raise ProviderError(f"unsupported provider kind: {self.profile.kind}")
        return graph_from_payload(document, payload)

    def _openai_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "instructions": SYSTEM_INSTRUCTION,
            "input": prompt,
            "text": {"format": {"type": "json_schema", "name": "evidence_graph", "strict": True, "schema": FACT_SCHEMA}},
        }
        if self.profile.reasoning_effort:
            payload["reasoning"] = {"effort": self.profile.reasoning_effort}
        return payload

    def _anthropic_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": 20000,
            "system": SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": FACT_SCHEMA}},
        }
        if self.profile.reasoning_effort:
            payload["output_config"]["effort"] = self.profile.reasoning_effort
        return payload

    @staticmethod
    def _parse_openai(response: dict[str, Any]) -> dict[str, Any]:
        if response.get("error"):
            raise ProviderError("OpenAI returned an error response")
        if response.get("status") not in (None, "completed"):
            raise ProviderError(f"OpenAI response is not complete: {response.get('status')}")
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return json.loads(content["text"])
        if isinstance(response.get("output_text"), str):
            return json.loads(response["output_text"])
        raise ProviderError("OpenAI response has no structured output text")

    @staticmethod
    def _parse_anthropic(response: dict[str, Any]) -> dict[str, Any]:
        if response.get("stop_reason") in {"max_tokens", "refusal"}:
            raise ProviderError(f"Anthropic response stopped before a usable result: {response.get('stop_reason')}")
        for content in response.get("content", []):
            if content.get("type") == "text":
                return json.loads(content["text"])
        raise ProviderError("Anthropic response has no structured output text")


def _resolve_span(document: IngestedDocument, text: str, start: int, end: int) -> tuple[int, int]:
    if 0 <= start < end <= len(document.text) and document.text[start:end] == text:
        return start, end
    positions: list[int] = []
    cursor = 0
    while True:
        found = document.text.find(text, cursor)
        if found < 0:
            break
        positions.append(found)
        cursor = found + 1
        if len(positions) > 1:
            break
    if len(positions) != 1:
        raise ExtractionContractError("source phrase offsets are invalid and the phrase is not unique")
    return positions[0], positions[0] + len(text)


def graph_from_payload(document: IngestedDocument, payload: dict[str, Any]) -> EvidenceGraph:
    wrapped = {
        "document_id": document.document_id,
        "document_hash": document.source_sha256,
        "facts": payload.get("facts"),
        "relationships": payload.get("relationships", []),
    }
    validate_extraction_payload(wrapped)
    facts: list[EvidenceFact] = []
    for index, item in enumerate(payload["facts"]):
        text = str(item["text"])
        start, end = _resolve_span(document, text, int(item["start_offset"]), int(item["end_offset"]))
        page = document.page_for_offset(start)
        attributes = {str(value["name"]): str(value["value"]) for value in item.get("attributes", [])}
        fact_identity = hashlib.sha256(
            json.dumps([item["fact_type"], text, item["status"], start, end, attributes], sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        facts.append(
            EvidenceFact(
                fact_id=fact_identity,
                fact_type=str(item["fact_type"]),
                text=text,
                status=FactStatus(item["status"]),
                source_spans=(SourceSpan(document.document_id, start, end, text, page),),
                attributes=attributes,
                certainty=str(item.get("certainty", "asserted")),
            )
        )
    relationships = []
    for item in payload.get("relationships", []):
        source_index, target_index = int(item["source_index"]), int(item["target_index"])
        if not (0 <= source_index < len(facts) and 0 <= target_index < len(facts)):
            raise ExtractionContractError("relationship points outside the fact list")
        relationships.append(
            {"source_fact_id": facts[source_index].fact_id, "target_fact_id": facts[target_index].fact_id, "relation": item["relation"]}
        )
    return EvidenceGraph(document.document_id, document.source_sha256, tuple(facts), tuple(relationships))


class IndependentExtractor:
    def __init__(self, profiles: tuple[ProviderProfile, ...]) -> None:
        if len({profile.kind for profile in profiles}) < 2:
            raise ProviderError("extractor profiles are not provider-independent")
        self.providers = tuple(EvidenceProvider(profile) for profile in profiles)

    def extract_independent(self, document: IngestedDocument) -> tuple[EvidenceGraph, ...]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.providers)) as pool:
            futures = [pool.submit(provider.extract, document) for provider in self.providers]
            return tuple(future.result() for future in futures)

    def extract(self, document: IngestedDocument) -> EvidenceGraph:
        return consensus(list(self.extract_independent(document)))
