"""Deterministic, provenance-bearing clinical abbreviation normalization.

The language model may propose a normalized clinical term, but autonomous
release cannot depend on an untraceable expansion.  This module preserves the
verbatim entity, binds it to exact source offsets, evaluates every registered
abbreviation against section/surrounding/anatomy/laterality/negation context,
and emits raw plus accepted-expanded retrieval forms.

The registry contains terminology only -- never medical codes.  It is a
retrieval/documentation aid, not coding authority.  Ambiguous or unknown terms
remain visible and block autonomous release only when their affirmed context
could alter a billed diagnosis, service, supply, drug, or claim attribute.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.core.config import TERMINOLOGY_PACK_DIR, TERMINOLOGY_REGISTRY_FILE
from app.models.schemas import ClinicalEntity


class TerminologyConfigError(RuntimeError):
    """The governed terminology registry is absent or structurally unsafe."""


_SECTION_KEYS = {
    "CC": "chief_complaint",
    "CHIEF_COMPLAINT": "chief_complaint",
    "HPI": "hpi",
    "PMH": "pmh_medications_allergies",
    "MEDICATIONS": "pmh_medications_allergies",
    "ALLERGIES": "pmh_medications_allergies",
    "PE": "physical_examination",
    "PHYSICAL_EXAMINATION": "physical_examination",
    "IMAGING": "imaging_diagnostics",
    "ASSESSMENT": "assessment_diagnoses",
    "DIAGNOSIS": "assessment_diagnoses",
    "PLAN": "plan",
}

_SCAN_SECTIONS = (
    ("CC", "chief_complaint"),
    ("HPI", "hpi"),
    ("PMH", "pmh_medications_allergies"),
    ("PE", "physical_examination"),
    ("IMAGING", "imaging_diagnostics"),
    ("ASSESSMENT", "assessment_diagnoses"),
    ("PLAN", "plan"),
)

def _canonical_section(value: str) -> str:
    cleaned = re.sub(r"[^A-Z]+", "_", str(value or "").upper()).strip("_")
    if cleaned in _SECTION_KEYS:
        return cleaned
    for alias in _SECTION_KEYS:
        if cleaned.startswith(alias):
            return alias
    return cleaned


def _section_key(value: str) -> str:
    canonical = _canonical_section(value)
    return _SECTION_KEYS.get(canonical, "")


def _dedup_text(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _fingerprint(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def terminology_entity_rows(entities: Iterable[ClinicalEntity | dict]) -> list[dict]:
    """Canonical release/consistency projection of normalized entities."""
    rows = []
    for value in entities:
        entity = value.model_dump() if isinstance(value, ClinicalEntity) else value
        if not isinstance(entity, dict):
            continue
        rows.append({
            "text": entity.get("text") or "",
            "category": entity.get("category") or "",
            "clinical_term": entity.get("clinical_term") or "",
            "normalized_text": entity.get("normalized_text") or "",
            "laterality": entity.get("laterality"),
            "specificity": entity.get("specificity"),
            "source_section": entity.get("source_section") or "",
            "negated": bool(entity.get("negated")),
            "source_span": entity.get("source_span") or {},
            "normalization_status": entity.get("normalization_status") or "",
            "normalization_confidence": entity.get("normalization_confidence") or 0.0,
            "terminology_resolutions": entity.get("terminology_resolutions") or [],
            "retrieval_terms": entity.get("retrieval_terms") or [],
        })
    return sorted(rows, key=lambda row: json.dumps(
        row, sort_keys=True, separators=(",", ":"), default=str))


def terminology_entity_fingerprint(
        entities: Iterable[ClinicalEntity | dict]) -> str:
    return _fingerprint(terminology_entity_rows(entities))


@dataclass(frozen=True)
class _Match:
    entry: dict
    start: int
    end: int
    raw: str


class TerminologyNormalizer:
    """Resolve governed clinical shorthand without changing source text."""

    def __init__(self, registry_path: str | Path | None = None):
        self.path = Path(registry_path or TERMINOLOGY_REGISTRY_FILE)
        paths = [self.path]
        if registry_path is None and TERMINOLOGY_PACK_DIR.is_dir():
            paths.extend(sorted(TERMINOLOGY_PACK_DIR.glob("*.json")))
        try:
            payloads = [(path, path.read_bytes()) for path in paths]
            data = json.loads(payloads[0][1])
            for path, raw in payloads[1:]:
                pack = json.loads(raw)
                if int(pack.get("schema_version") or 0) != 1:
                    raise TerminologyConfigError(
                        f"unsupported terminology pack schema: {path.name}")
                if not isinstance(pack.get("entries"), list):
                    raise TerminologyConfigError(
                        f"terminology pack entries missing: {path.name}")
                pack_sources = pack.get("sources") or {}
                if not isinstance(pack_sources, dict):
                    raise TerminologyConfigError(
                        f"terminology pack sources malformed: {path.name}")
                for source_id, source in pack_sources.items():
                    prior = data.setdefault("sources", {}).get(source_id)
                    if prior is not None and prior != source:
                        raise TerminologyConfigError(
                            f"terminology source collision: {source_id}")
                    data["sources"][source_id] = source
                data.setdefault("entries", []).extend(pack["entries"])
        except Exception as exc:
            raise TerminologyConfigError(
                f"terminology registry unavailable or invalid: {exc}") from exc
        digest = hashlib.sha256()
        for path, raw in payloads:
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(raw)
        self.registry_sha256 = "sha256:" + digest.hexdigest()
        self.registry_files = [str(path) for path, _ in payloads]
        self._load(data)

    def _load(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise TerminologyConfigError("terminology registry root must be an object")
        if int(data.get("schema_version") or 0) != 1:
            raise TerminologyConfigError("unsupported terminology schema_version")
        self.version = str(data.get("version") or "").strip()
        if not self.version:
            raise TerminologyConfigError("terminology registry version is required")
        sources = data.get("sources") or {}
        if not isinstance(sources, dict) or not sources:
            raise TerminologyConfigError("terminology sources are required")
        for source_id, source in sources.items():
            if (not isinstance(source, dict)
                    or source.get("authority_role") != "retrieval_only"
                    or not source.get("provenance_kind")):
                raise TerminologyConfigError(
                    f"{source_id}: terminology sources must be provenance-bearing "
                    "retrieval-only aids")
        acceptance = data.get("acceptance") or {}
        try:
            self.min_confidence = float(acceptance["min_confidence"])
            self.min_margin = float(acceptance["min_margin"])
            self.context_window = int(acceptance["context_window_chars"])
            weights = acceptance["evidence_weights"]
            self.evidence_weights = {
                name: float(weights[name]) for name in (
                    "required_section", "required_context_any",
                    "required_context_all", "anatomy", "laterality")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise TerminologyConfigError("invalid acceptance configuration") from exc
        if not (0 < self.min_confidence <= 1 and 0 <= self.min_margin <= 1
                and self.context_window >= 20
                and all(0 <= value <= 0.25
                        for value in self.evidence_weights.values())):
            raise TerminologyConfigError("unsafe acceptance thresholds")

        unknown = data.get("unknown_detection") or {}
        try:
            self.unknown_pattern = re.compile(str(unknown["pattern"]))
            self.negation_patterns = [
                re.compile(str(pattern), re.IGNORECASE)
                for pattern in unknown["negation_patterns"]
            ]
        except (KeyError, TypeError, re.error) as exc:
            raise TerminologyConfigError("invalid unknown-term pattern") from exc
        if not self.negation_patterns:
            raise TerminologyConfigError("at least one negation pattern is required")
        self.ignored_terms = {
            str(value).casefold() for value in unknown.get("ignored_terms") or []
        }
        self.billing_sections = {
            _canonical_section(value)
            for value in unknown.get("billing_relevant_sections") or []
        }

        entries = data.get("entries") or []
        if not isinstance(entries, list) or not entries:
            raise TerminologyConfigError("terminology entries are required")
        ids: set[str] = set()
        compiled = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise TerminologyConfigError("every terminology entry must be an object")
            entry_id = str(entry.get("id") or "").strip()
            if not entry_id or entry_id in ids:
                raise TerminologyConfigError("terminology entry ids must be unique")
            ids.add(entry_id)
            patterns = entry.get("patterns") or []
            candidates = entry.get("candidates") or []
            if not patterns or not candidates:
                raise TerminologyConfigError(
                    f"{entry_id}: patterns and candidates are required")
            try:
                regexes = [re.compile(str(pattern), re.IGNORECASE)
                           for pattern in patterns]
            except re.error as exc:
                raise TerminologyConfigError(
                    f"{entry_id}: invalid match pattern") from exc
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise TerminologyConfigError(
                        f"{entry_id}: candidate must be an object")
                if not str(candidate.get("expansion") or "").strip():
                    raise TerminologyConfigError(
                        f"{entry_id}: candidate expansion is required")
                source_id = str(candidate.get("source_id") or "")
                if source_id not in sources:
                    raise TerminologyConfigError(
                        f"{entry_id}: unknown source_id {source_id!r}")
                try:
                    confidence = float(candidate["confidence"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise TerminologyConfigError(
                        f"{entry_id}: candidate confidence is invalid") from exc
                if not 0 <= confidence <= 1:
                    raise TerminologyConfigError(
                        f"{entry_id}: confidence must be between zero and one")
            compiled.append((entry, regexes))
        self.sources = sources
        self.entries = compiled

    @staticmethod
    def _sentence(text: str, start: int, end: int) -> str:
        left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start),
                   text.rfind(";", 0, start))
        stops = [pos for pos in (text.find(".", end), text.find("\n", end),
                                 text.find(";", end)) if pos >= 0]
        right = min(stops) if stops else len(text)
        return text[left + 1:right].strip()

    def _context(self, text: str, start: int, end: int) -> str:
        return text[max(0, start - self.context_window):
                    min(len(text), end + self.context_window)]

    def _is_negated(self, text: str, start: int) -> bool:
        prefix = text[max(0, start - 160):start]
        return any(pattern.search(prefix) for pattern in self.negation_patterns)

    def _matches(self, text: str) -> list[_Match]:
        matches: list[_Match] = []
        for entry, regexes in self.entries:
            for regex in regexes:
                for found in regex.finditer(text or ""):
                    start, end = found.span()
                    if start == end:
                        continue
                    matches.append(_Match(entry, start, end, found.group(0)))
        # Prefer the longest governed term when patterns overlap, then keep a
        # deterministic left-to-right order.
        chosen: list[_Match] = []
        for match in sorted(matches, key=lambda m: (m.start, -(m.end - m.start),
                                                    m.entry["id"])):
            if any(match.start < prior.end and prior.start < match.end
                   for prior in chosen):
                continue
            chosen.append(match)
        return sorted(chosen, key=lambda m: (m.start, m.end))

    @staticmethod
    def _has_terms(haystack: str, terms: list[str], *, require_all=False) -> bool:
        low = haystack.casefold()
        hits = [str(term).casefold() in low for term in terms if str(term).strip()]
        if not hits:
            return not require_all
        return all(hits) if require_all else any(hits)

    def _candidate_scores(self, match: _Match, *, section: str, context: str,
                          laterality: str, negated: bool) -> list[dict]:
        scored = []
        for candidate in match.entry.get("candidates") or []:
            rejection_reasons = []
            required_sections = {
                _canonical_section(value)
                for value in candidate.get("required_sections") or []
            }
            if required_sections and section not in required_sections:
                rejection_reasons.append("section context does not apply")
            required_any = candidate.get("required_context_any") or []
            required_all = candidate.get("required_context_all") or []
            excluded = candidate.get("excluded_context_any") or []
            if required_any and not self._has_terms(context, required_any):
                rejection_reasons.append("required surrounding context is absent")
            if required_all and not self._has_terms(
                    context, required_all, require_all=True):
                rejection_reasons.append("required context combination is absent")
            if excluded and self._has_terms(context, excluded):
                rejection_reasons.append("excluded surrounding context is present")
            allowed_laterality = {
                str(value).upper() for value in
                candidate.get("allowed_laterality") or []
            }
            # A structured side is corroborating/contradicting evidence when
            # present.  Its absence does not erase an otherwise unambiguous
            # raw expansion (for note-level scans no entity laterality exists).
            if (allowed_laterality and laterality
                    and laterality not in allowed_laterality):
                rejection_reasons.append("structured laterality contradicts expansion")
            negation = str(candidate.get("negation") or "any").lower()
            if negation == "affirmed" and negated:
                rejection_reasons.append("expansion requires affirmed context")
            if negation == "negated" and not negated:
                rejection_reasons.append("expansion requires negated context")

            score = float(candidate["confidence"])
            if required_sections and section in required_sections:
                score += self.evidence_weights["required_section"]
            if required_any and self._has_terms(context, required_any):
                score += self.evidence_weights["required_context_any"]
            if required_all and self._has_terms(
                    context, required_all, require_all=True):
                score += self.evidence_weights["required_context_all"]
            anatomy = candidate.get("anatomy_any") or []
            if anatomy and self._has_terms(context, anatomy):
                score += self.evidence_weights["anatomy"]
            elif anatomy and candidate.get("require_anatomy"):
                rejection_reasons.append("required anatomy is absent")
            if (allowed_laterality and laterality
                    and laterality in allowed_laterality):
                score += self.evidence_weights["laterality"]
            source_id = str(candidate["source_id"])
            scored.append({
                "expansion": str(candidate["expansion"]).strip(),
                "confidence": round(min(score, 1.0), 4),
                "source_id": source_id,
                "source_version": str(
                    (self.sources.get(source_id) or {}).get("version") or ""),
                "provenance_kind": str(
                    (self.sources.get(source_id) or {}).get(
                        "provenance_kind") or ""),
                "authority_role": "retrieval_only",
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            })
        return sorted(scored, key=lambda row: (not row["eligible"],
                                               -row["confidence"],
                                               row["expansion"].casefold()))

    def _resolve(self, match: _Match, *, section: str, context: str,
                 laterality: str = "", negated: bool = False) -> dict:
        alternatives = self._candidate_scores(
            match, section=section, context=context,
            laterality=laterality, negated=negated)
        eligible = [row for row in alternatives if row["eligible"]]
        top = eligible[0] if eligible else None
        runner_up = eligible[1] if len(eligible) > 1 else None
        margin = (top["confidence"] - runner_up["confidence"]
                  if top and runner_up else 1.0)
        accepted = bool(top and top["confidence"] >= self.min_confidence
                        and margin >= self.min_margin)
        status = "accepted" if accepted else (
            "ambiguous" if eligible else "unresolved")
        return {
            "entry_id": str(match.entry["id"]),
            "raw_text": match.raw,
            "start": match.start,
            "end": match.end,
            "status": status,
            "expansion": top["expansion"] if accepted else "",
            "confidence": top["confidence"] if top else 0.0,
            "margin": round(margin, 4) if top else 0.0,
            "source_id": top["source_id"] if accepted else "",
            "source_version": top["source_version"] if accepted else "",
            "alternatives": alternatives,
            "coding_impact": bool(match.entry.get("coding_impact", True)),
            "negated": bool(negated),
            "section": section,
            "context": context.strip(),
        }

    def _unknowns(self, text: str, covered: list[tuple[int, int]], *,
                  section: str, negation_from_text=True) -> list[dict]:
        unknowns = []
        for found in self.unknown_pattern.finditer(text or ""):
            start, end = found.span()
            raw = found.group(0)
            if raw.casefold() in self.ignored_terms:
                continue
            if any(start < right and left < end for left, right in covered):
                continue
            negated = self._is_negated(text, start) if negation_from_text else False
            unknowns.append({
                "entry_id": "",
                "raw_text": raw,
                "start": start,
                "end": end,
                "status": "unresolved",
                "expansion": "",
                "confidence": 0.0,
                "margin": 0.0,
                "source_id": "",
                "source_version": "",
                "alternatives": [],
                "coding_impact": section in self.billing_sections,
                "negated": negated,
                "section": section,
                "context": self._context(text, start, end).strip(),
                "reason": "unregistered abbreviation-like token",
            })
        return unknowns

    @staticmethod
    def _document_span(full_text: str, raw: str, context: str) -> tuple[int | None, int | None]:
        if not raw or not full_text:
            return None, None
        positions = [m.start() for m in re.finditer(re.escape(raw), full_text)]
        if not positions:
            return None, None
        if len(positions) == 1:
            return positions[0], positions[0] + len(raw)
        context_words = set(re.findall(r"[a-z]+", context.casefold()))
        ranked = []
        for pos in positions:
            window = full_text[max(0, pos - 120):pos + len(raw) + 120]
            overlap = len(context_words & set(re.findall(r"[a-z]+", window.casefold())))
            ranked.append((overlap, pos))
        ranked.sort(reverse=True)
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            return None, None
        return ranked[0][1], ranked[0][1] + len(raw)

    def _entity_span(self, entity: ClinicalEntity, sections: dict) -> dict:
        key = _section_key(entity.source_section)
        section_text = str(sections.get(key) or "")
        raw = str(entity.text or "")
        start = section_text.find(raw) if section_text and raw else -1
        section_end = start + len(raw) if start >= 0 else None
        context = self._context(section_text, start, section_end) \
            if start >= 0 and section_end is not None else ""
        doc_start, doc_end = self._document_span(
            str(sections.get("full_text") or ""), raw, context)
        return {
            "section": _canonical_section(entity.source_section),
            "section_key": key,
            "section_start": start if start >= 0 else None,
            "section_end": section_end,
            "document_start": doc_start,
            "document_end": doc_end,
            "text": raw,
            "verified": start >= 0 and doc_start is not None,
        }

    def _normalize_entity(self, entity: ClinicalEntity, sections: dict) -> ClinicalEntity:
        out = entity.model_copy(deep=True)
        span = self._entity_span(out, sections)
        out.source_span = span
        section = _canonical_section(out.source_section)
        raw = str(out.text or "")
        section_text = str(sections.get(span.get("section_key") or "") or raw)
        section_start = span.get("section_start")
        if isinstance(section_start, int):
            context = self._context(section_text, section_start,
                                    section_start + len(raw))
        else:
            context = raw
        matches = self._matches(raw)
        resolutions = [
            self._resolve(match, section=section, context=context,
                          laterality=str(out.laterality or "").upper(),
                          negated=bool(out.negated))
            for match in matches
        ]
        resolutions.extend(self._unknowns(
            raw, [(m.start, m.end) for m in matches], section=section,
            negation_from_text=False))
        for resolution in resolutions:
            resolution["section_start"] = (
                section_start + resolution["start"]
                if isinstance(section_start, int) else None)
            resolution["section_end"] = (
                section_start + resolution["end"]
                if isinstance(section_start, int) else None)
            doc_start, doc_end = self._document_span(
                str(sections.get("full_text") or ""),
                resolution["raw_text"], resolution["context"])
            resolution["document_start"] = doc_start
            resolution["document_end"] = doc_end
            resolution["source_span_verified"] = doc_start is not None

        normalized = raw
        for resolution in sorted(resolutions, key=lambda row: row["start"],
                                 reverse=True):
            if resolution["status"] == "accepted":
                normalized = (normalized[:resolution["start"]]
                              + resolution["expansion"]
                              + normalized[resolution["end"]:])
        out.normalized_text = normalized
        out.terminology_resolutions = resolutions
        out.retrieval_terms = _dedup_text(
            [out.clinical_term, raw, normalized]
            + [row["expansion"] for row in resolutions
               if row["status"] == "accepted"])
        states = {row["status"] for row in resolutions}
        if "unresolved" in states:
            out.normalization_status = "unresolved"
        elif "ambiguous" in states:
            out.normalization_status = "ambiguous"
        elif "accepted" in states:
            out.normalization_status = "accepted"
        else:
            out.normalization_status = "not_applicable"
        confidences = [row["confidence"] for row in resolutions
                       if row["status"] == "accepted"]
        out.normalization_confidence = min(confidences) if confidences else (
            1.0 if not resolutions else 0.0)
        out.normalization_registry_version = self.version
        out.normalization_registry_sha256 = self.registry_sha256
        return out

    def _note_occurrences(self, sections: dict) -> list[dict]:
        rows = []
        full_text = str(sections.get("full_text") or "")
        for canonical, key in _SCAN_SECTIONS:
            text = str(sections.get(key) or "")
            if not text:
                continue
            matches = self._matches(text)
            for match in matches:
                negated = self._is_negated(text, match.start)
                context = self._context(text, match.start, match.end)
                row = self._resolve(match, section=canonical, context=context,
                                    negated=negated)
                row["section_start"] = match.start
                row["section_end"] = match.end
                doc_start, doc_end = self._document_span(full_text, match.raw, context)
                row["document_start"] = doc_start
                row["document_end"] = doc_end
                row["source_span_verified"] = doc_start is not None
                rows.append(row)
            for row in self._unknowns(
                    text, [(m.start, m.end) for m in matches], section=canonical):
                row["section_start"] = row["start"]
                row["section_end"] = row["end"]
                doc_start, doc_end = self._document_span(
                    full_text, row["raw_text"], row["context"])
                row["document_start"] = doc_start
                row["document_end"] = doc_end
                row["source_span_verified"] = doc_start is not None
                rows.append(row)
        return sorted(rows, key=lambda row: (
            row["section"], row.get("section_start") or -1,
            row["raw_text"].casefold()))

    def normalize_entities(self, entities: list[ClinicalEntity],
                           sections: dict) -> tuple[list[ClinicalEntity], dict]:
        normalized = [self._normalize_entity(entity, sections)
                      for entity in entities]
        note_occurrences = self._note_occurrences(sections)
        canonical_entities = terminology_entity_rows(normalized)
        unresolved = []
        for scope, rows in (("entity", canonical_entities),
                            ("note", note_occurrences)):
            if scope == "entity":
                candidates = [resolution for row in rows
                              for resolution in row["terminology_resolutions"]]
            else:
                candidates = rows
            for row in candidates:
                billing_relevant = bool(row.get("coding_impact")) and not bool(
                    row.get("negated"))
                if row.get("status") in {"ambiguous", "unresolved"} \
                        and billing_relevant:
                    unresolved.append({
                        "scope": scope,
                        "raw_text": row.get("raw_text") or "",
                        "section": row.get("section") or "",
                        "section_start": row.get("section_start"),
                        "document_start": row.get("document_start"),
                        "status": row.get("status") or "",
                        "alternatives": row.get("alternatives") or [],
                        "reason": row.get("reason") or
                                  "no unique high-confidence expansion",
                    })
        # The same term can appear in the note scan and inside an entity. Keep
        # one canonical blocker record per source location/decision.
        unique_unresolved = []
        seen = set()
        for row in unresolved:
            key = (row["raw_text"].casefold(), row["section"],
                   row["section_start"], row["status"])
            if key not in seen:
                seen.add(key)
                unique_unresolved.append(row)
        report_body = {
            "schema_version": 1,
            "registry_version": self.version,
            "registry_sha256": self.registry_sha256,
            "registry_files": self.registry_files,
            "authority_role": "retrieval_only",
            "entities_processed": len(normalized),
            "entities": canonical_entities,
            "note_occurrences": note_occurrences,
            "unresolved_billing_relevant": unique_unresolved,
        }
        report_body["entity_fingerprint"] = _fingerprint(canonical_entities)
        report_body["status"] = (
            "REVIEW_REQUIRED" if unique_unresolved else "PASS")
        report_body["report_fingerprint"] = _fingerprint(report_body)
        return normalized, report_body
