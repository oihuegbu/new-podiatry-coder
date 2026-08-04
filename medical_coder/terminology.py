from __future__ import annotations

import json
import re
import sqlite3
from difflib import SequenceMatcher
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .compiler import sha256_file
from .models import Candidate, EvidenceFact, EvidenceGraph, Normalization


class SnapshotIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    authority: str
    system: str
    version: str
    code: str
    display: str
    concept_id: str | None
    effective_start: str | None
    effective_end: str | None
    status: str
    billable: bool
    properties: dict[str, Any]
    requirements: list[dict[str, Any]] | None
    source_path: str
    source_hash: str
    source_locator: str


@dataclass(frozen=True)
class CoverageEvaluation:
    status: str
    reason: str
    source_refs: tuple[str, ...]
    policy_ids: tuple[str, ...]


class TerminologySnapshot:
    def __init__(self, snapshot_directory: Path) -> None:
        self.directory = snapshot_directory.resolve()
        manifest_path = self.directory / "manifest.json"
        database_path = self.directory / "terminology.sqlite"
        if not manifest_path.is_file() or not database_path.is_file():
            raise SnapshotIntegrityError("snapshot is incomplete")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("database_sha256") != sha256_file(database_path):
            raise SnapshotIntegrityError("snapshot database hash mismatch")
        self.snapshot_id = str(self.manifest["snapshot_id"])
        self._database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self._database_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _fts_query(text: str) -> str:
        tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if not tokens:
            raise ValueError("candidate search requires lexical evidence")
        return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))

    @staticmethod
    def _fts_phrase_query(text: str) -> str:
        tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if not tokens:
            raise ValueError("exact terminology lookup requires lexical evidence")
        escaped = " ".join(token.replace('"', '""') for token in tokens)
        return f'"{escaped}"'

    @lru_cache(maxsize=16)
    def _source_vocabulary(self, systems: tuple[str, ...]) -> tuple[str, ...]:
        if not systems:
            return ()
        placeholders = ",".join("?" for _ in systems)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT DISTINCT token FROM token_statistics WHERE system IN ({placeholders}) ORDER BY token",
                systems,
            ).fetchall()
        return tuple(str(row["token"]) for row in rows)

    @lru_cache(maxsize=16)
    def _source_ngram_index(self, systems: tuple[str, ...], width: int) -> dict[str, tuple[str, ...]]:
        index: dict[str, set[str]] = {}
        for token in self._source_vocabulary(systems):
            for offset in range(len(token) - width + 1):
                index.setdefault(token[offset:offset + width], set()).add(token)
        return {gram: tuple(sorted(tokens)) for gram, tokens in index.items()}

    def source_derived_candidate_recall(
        self,
        text: str,
        system_scope: set[str],
    ) -> tuple[frozenset[str], tuple[str, ...]]:
        """Expand retrieval terms only from the active descriptor vocabulary.

        A hop is accepted only when its best common-substring coverage is uniquely
        separated from the next alternative. These terms may retrieve candidates,
        but they are never evidence of descriptor entailment or code identity.
        """
        original = set(self.normalized_phrase(text).split())
        config = self.manifest.get("semantic_derivations", {}).get("source_derived_candidate_recall", {})
        if not config.get("enabled"):
            return frozenset(original), ()
        minimum_length = int(config["minimum_token_length"])
        minimum_match = int(config["minimum_common_substring_length"])
        minimum_coverage = float(config["minimum_common_substring_coverage"])
        minimum_similarity = float(config["minimum_sequence_similarity"])
        minimum_margin = float(config["minimum_unique_coverage_margin"])
        systems = tuple(sorted(system_scope))
        vocabulary_index = self._source_ngram_index(systems, minimum_match)
        expanded = set(original)
        frontier = {token for token in original if len(token) >= minimum_length}
        factors: list[str] = []
        for hop in range(1, int(config["maximum_hops"]) + 1):
            additions: set[str] = set()
            for source in sorted(frontier):
                possible_targets = {
                    target
                    for offset in range(len(source) - minimum_match + 1)
                    for target in vocabulary_index.get(source[offset:offset + minimum_match], ())
                }
                ranked: list[tuple[float, float, str]] = []
                for target in possible_targets:
                    if target in expanded or len(target) < minimum_length:
                        continue
                    matcher = SequenceMatcher(None, source, target, autojunk=False)
                    match = matcher.find_longest_match()
                    coverage = match.size / min(len(source), len(target))
                    similarity = matcher.ratio()
                    if (
                        match.size < minimum_match
                        or coverage < minimum_coverage
                        or similarity < minimum_similarity
                    ):
                        continue
                    ranked.append((coverage, similarity, target))
                ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
                if not ranked:
                    continue
                best = ranked[0]
                if len(ranked) > 1 and best[0] - ranked[1][0] < minimum_margin:
                    continue
                additions.add(best[2])
                factors.append(
                    f"candidate_recall_only:{source}->{best[2]}:hop={hop}:coverage={best[0]:.3f}"
                )
            additions -= expanded
            if not additions:
                break
            expanded.update(additions)
            frontier = additions
        return frozenset(expanded), tuple(factors)

    def _expanded_fts_query(self, text: str, system_scope: set[str]) -> str:
        tokens, _ = self.source_derived_candidate_recall(text, system_scope)
        if not tokens:
            raise ValueError("candidate search requires lexical evidence")
        return " OR ".join(f'"{token}"' for token in sorted(tokens))

    def search(
        self,
        evidence: EvidenceGraph,
        effective_date: date,
        system_scope: set[str],
        limit_per_fact: int = 20,
    ) -> list[Candidate]:
        if not system_scope:
            raise ValueError("closed-world search requires an explicit system scope")
        combined: dict[str, Candidate] = {}
        placeholders = ",".join("?" for _ in system_scope)
        with self._connect() as connection:
            for fact in evidence.facts:
                query = self._expanded_fts_query(
                    " ".join((fact.text, *map(str, fact.attributes.values()))),
                    system_scope,
                )
                rows = connection.execute(
                    f"""
                    SELECT a.artifact_id, a.system, a.version, a.display, bm25(artifact_search) AS rank
                    FROM artifact_search
                    JOIN artifacts a USING (artifact_id)
                    WHERE artifact_search MATCH ?
                      AND a.system IN ({placeholders})
                      AND (a.effective_start IS NULL OR a.effective_start <= ?)
                      AND (a.effective_end IS NULL OR a.effective_end >= ?)
                      AND a.status = 'active'
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, *sorted(system_scope), effective_date.isoformat(), effective_date.isoformat(), limit_per_fact),
                ).fetchall()
                for row in rows:
                    score = -float(row["rank"])
                    existing = combined.get(row["artifact_id"])
                    fact_ids = set(existing.supporting_fact_ids if existing else ())
                    fact_ids.add(fact.fact_id)
                    combined[row["artifact_id"]] = Candidate(
                        artifact_id=row["artifact_id"],
                        system=row["system"],
                        version=row["version"],
                        display=row["display"],
                        score=max(score, existing.score if existing else score),
                        supporting_fact_ids=tuple(sorted(fact_ids)),
                    )
                lookup_rows = connection.execute(
                    f"""
                    SELECT a.artifact_id, a.system, a.version, a.display, bm25(lookup_search) AS rank
                    FROM lookup_search
                    JOIN lookup_terms l USING (lookup_id)
                    JOIN artifacts a USING (artifact_id)
                    WHERE lookup_search MATCH ?
                      AND a.system IN ({placeholders})
                      AND (a.effective_start IS NULL OR a.effective_start <= ?)
                      AND (a.effective_end IS NULL OR a.effective_end >= ?)
                      AND a.status = 'active'
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, *sorted(system_scope), effective_date.isoformat(), effective_date.isoformat(), limit_per_fact),
                ).fetchall()
                for row in lookup_rows:
                    score = -float(row["rank"])
                    existing = combined.get(row["artifact_id"])
                    fact_ids = set(existing.supporting_fact_ids if existing else ())
                    fact_ids.add(fact.fact_id)
                    combined[row["artifact_id"]] = Candidate(
                        artifact_id=row["artifact_id"],
                        system=row["system"],
                        version=row["version"],
                        display=row["display"],
                        score=max(score, existing.score if existing else score),
                        supporting_fact_ids=tuple(sorted(fact_ids)),
                    )
        return sorted(combined.values(), key=lambda item: (-item.score, item.artifact_id))

    def normalize_exact(
        self,
        fact: EvidenceFact,
        effective_date: date,
        system_scope: set[str],
    ) -> Normalization | None:
        candidates = self.search(
            EvidenceGraph(
                document_id=fact.source_spans[0].document_id,
                document_hash="normalization-only",
                facts=(fact,),
            ),
            effective_date,
            system_scope,
            limit_per_fact=50,
        )
        exact = []
        normalized_text = " ".join(fact.text.casefold().split())
        for candidate in candidates:
            artifact = self.artifact(candidate.artifact_id)
            if artifact and " ".join(artifact.display.casefold().split()) == normalized_text:
                exact.append(artifact)
        if len(exact) != 1:
            return None
        artifact = exact[0]
        return Normalization(
            concept_id=artifact.concept_id or artifact.artifact_id,
            system=artifact.system,
            source=f"snapshot:{self.snapshot_id}",
            confidence=1.0,
            alternatives=(),
            raw_phrase=fact.text,
            candidate_expansions=(artifact.display,),
            resolution_factors=("exact_authoritative_descriptor",),
            target_artifact_id=artifact.artifact_id,
        )

    @staticmethod
    def normalized_phrase(value: str) -> str:
        return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))

    def normalization_candidates(
        self,
        fact: EvidenceFact,
        effective_date: date,
        system_scope: set[str],
    ) -> list[tuple[Artifact, str, tuple[str, ...]]]:
        phrase = self.normalized_phrase(fact.text)
        if not phrase:
            return []
        contextual_tokens = set(self.normalized_phrase(" ".join(map(str, fact.attributes.values()))).split())
        placeholders = ",".join("?" for _ in system_scope)
        if not placeholders:
            raise ValueError("normalization requires an explicit system scope")
        matches: dict[str, tuple[Artifact, str, tuple[str, ...]]] = {}
        # Exact normalization must not use broad OR recall: a frequent token
        # can crowd the intended phrase out of a bounded result set.  Preserve
        # the tokenizer's punctuation handling while requiring token order.
        query = self._fts_phrase_query(fact.text)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT a.*, l.term, l.term_kind
                FROM lookup_search
                JOIN lookup_terms l USING (lookup_id)
                JOIN artifacts a USING (artifact_id)
                WHERE lookup_search MATCH ? AND a.system IN ({placeholders})
                  AND (a.effective_start IS NULL OR a.effective_start <= ?)
                  AND (a.effective_end IS NULL OR a.effective_end >= ?)
                  AND a.status = 'active'
                ORDER BY bm25(lookup_search), l.lookup_id
                LIMIT 500
                """,
                (query, *sorted(system_scope), effective_date.isoformat(), effective_date.isoformat()),
            ).fetchall()
            descriptor_rows = connection.execute(
                f"""
                SELECT a.*, a.display AS term, 'exact_authoritative_descriptor' AS term_kind
                FROM artifacts a
                WHERE lower(trim(a.display)) = ? AND a.system IN ({placeholders})
                  AND (a.effective_start IS NULL OR a.effective_start <= ?)
                  AND (a.effective_end IS NULL OR a.effective_end >= ?)
                  AND a.status = 'active'
                """,
                (phrase, *sorted(system_scope), effective_date.isoformat(), effective_date.isoformat()),
            ).fetchall()
        for row in (*rows, *descriptor_rows):
            if self.normalized_phrase(str(row["term"])) != phrase:
                continue
            artifact = self._artifact_from_row(row)
            display_tokens = set(self.normalized_phrase(artifact.display).split())
            factors = [str(row["term_kind"])]
            overlap = contextual_tokens & display_tokens
            if overlap:
                factors.append("context_overlap:" + ",".join(sorted(overlap)))
            matches[artifact.artifact_id] = (artifact, str(row["term"]), tuple(factors))
        return sorted(matches.values(), key=lambda item: item[0].artifact_id)

    def normalize_fact(
        self,
        fact: EvidenceFact,
        effective_date: date,
        system_scope: set[str],
    ) -> EvidenceFact:
        matches = self.normalization_candidates(fact, effective_date, system_scope)
        if not matches:
            exact = self.normalize_exact(fact, effective_date, system_scope)
            if exact:
                return replace(fact, normalization=exact)
            derived = self.descriptor_resolution(fact, effective_date, system_scope)
            return replace(fact, normalization=derived) if derived else fact
        expansions = tuple(sorted({artifact.display for artifact, _, _ in matches}))
        if len(matches) > 1:
            scored = []
            for match in matches:
                factors = match[2]
                overlap_score = sum(len(factor.removeprefix("context_overlap:").split(",")) for factor in factors if factor.startswith("context_overlap:"))
                scored.append((overlap_score, match))
            scored.sort(key=lambda item: (-item[0], item[1][0].artifact_id))
            if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                artifact, _, factors = scored[0][1]
                return replace(
                    fact,
                    normalization=Normalization(
                        concept_id=artifact.concept_id or artifact.artifact_id,
                        system=artifact.system,
                        source=f"snapshot:{self.snapshot_id}",
                        confidence=0.95,
                        alternatives=(),
                        raw_phrase=fact.text,
                        candidate_expansions=expansions,
                        resolution_factors=(*factors, "unique_context_resolution"),
                        target_artifact_id=artifact.artifact_id,
                    ),
                )
        if len(matches) == 1:
            artifact, _, factors = matches[0]
            return replace(
                fact,
                normalization=Normalization(
                    concept_id=artifact.concept_id or artifact.artifact_id,
                    system=artifact.system,
                    source=f"snapshot:{self.snapshot_id}",
                    confidence=1.0,
                    alternatives=(),
                    raw_phrase=fact.text,
                    candidate_expansions=expansions,
                    resolution_factors=factors,
                    target_artifact_id=artifact.artifact_id,
                ),
            )
        return replace(
            fact,
            normalization=Normalization(
                concept_id="",
                system="AMBIGUOUS",
                source=f"snapshot:{self.snapshot_id}",
                confidence=0.0,
                alternatives=tuple(artifact.artifact_id for artifact, _, _ in matches),
                raw_phrase=fact.text,
                candidate_expansions=expansions,
                resolution_factors=("multiple_authoritative_targets",),
                target_artifact_id="",
            ),
        )

    def descriptor_resolution(
        self,
        fact: EvidenceFact,
        effective_date: date,
        system_scope: set[str],
    ) -> Normalization | None:
        evidence_tokens = set(
            self.normalized_phrase(" ".join((fact.text, *map(str, fact.attributes.values())))).split()
        )
        if not evidence_tokens:
            return None
        candidates = self.search(
            EvidenceGraph(fact.source_spans[0].document_id, "descriptor-resolution", (fact,)),
            effective_date, system_scope, limit_per_fact=100,
        )
        entailed: list[tuple[int, Candidate, Artifact, tuple[str, ...]]] = []
        with self._connect() as connection:
            for candidate in candidates:
                rows = connection.execute(
                    "SELECT token FROM artifact_tokens WHERE artifact_id = ? AND material = 1",
                    (candidate.artifact_id,),
                ).fetchall()
                material = tuple(sorted(row["token"] for row in rows))
                if not material or not set(material).issubset(evidence_tokens):
                    continue
                artifact = self.artifact(candidate.artifact_id)
                if artifact:
                    entailed.append((len(material), candidate, artifact, material))
        if not entailed:
            return None
        entailed.sort(key=lambda item: (-item[0], -item[1].score, item[1].artifact_id))
        best_specificity = entailed[0][0]
        most_specific = [item for item in entailed if item[0] == best_specificity]
        if len(most_specific) != 1:
            return Normalization(
                concept_id="", system="AMBIGUOUS", source=f"snapshot:{self.snapshot_id}", confidence=0.0,
                alternatives=tuple(item[2].artifact_id for item in most_specific), raw_phrase=fact.text,
                candidate_expansions=tuple(item[2].display for item in most_specific),
                resolution_factors=("source_derived_descriptor_tie",), target_artifact_id="",
            )
        _, _, artifact, material = most_specific[0]
        return Normalization(
            concept_id=artifact.concept_id or artifact.artifact_id,
            system=artifact.system,
            source=f"snapshot:{self.snapshot_id}",
            confidence=0.9,
            alternatives=(), raw_phrase=fact.text, candidate_expansions=(artifact.display,),
            resolution_factors=(
                "all_source_derived_material_descriptor_tokens_entailed",
                "material_tokens:" + ",".join(material),
            ),
            target_artifact_id=artifact.artifact_id,
        )

    def normalize_graph(
        self,
        evidence: EvidenceGraph,
        effective_date: date,
        system_scope: set[str],
    ) -> EvidenceGraph:
        return replace(
            evidence,
            facts=tuple(self.normalize_fact(fact, effective_date, system_scope) for fact in evidence.facts),
        )

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> Artifact:
        return Artifact(
            artifact_id=row["artifact_id"], authority=row["authority"], system=row["system"],
            version=row["version"], code=row["code"], display=row["display"], concept_id=row["concept_id"],
            effective_start=row["effective_start"], effective_end=row["effective_end"], status=row["status"],
            billable=bool(row["billable"]), properties=json.loads(row["properties_json"]),
            requirements=json.loads(row["requirements_json"]) if row["requirements_json"] else None,
            source_path=row["source_path"], source_hash=row["source_hash"], source_locator=row["source_locator"],
        )

    def artifact(self, artifact_id: str) -> Artifact | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return self._artifact_from_row(row)

    def artifact_by_code(self, system: str, code: str, service_date: date) -> Artifact | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE system = ? AND code = ? AND status = 'active'
                  AND (effective_start IS NULL OR effective_start <= ?)
                  AND (effective_end IS NULL OR effective_end >= ?)
                """,
                (system, code, service_date.isoformat(), service_date.isoformat()),
            ).fetchall()
        if len(rows) > 1:
            raise SnapshotIntegrityError("more than one active artifact has the same system and code")
        return self._artifact_from_row(rows[0]) if rows else None

    def semantics(self, artifact_id: str, kinds: set[str] | None = None) -> list[dict[str, Any]]:
        parameters: list[Any] = [artifact_id]
        clause = ""
        if kinds:
            clause = " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            parameters.extend(sorted(kinds))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifact_semantics WHERE artifact_id = ?" + clause,
                parameters,
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def material_descriptor_tokens(self, artifact_id: str) -> frozenset[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT token FROM artifact_tokens WHERE artifact_id = ? AND material = 1",
                (artifact_id,),
            ).fetchall()
        return frozenset(str(row["token"]) for row in rows)

    @staticmethod
    def _normalized_code(value: str) -> str:
        return "".join(character for character in value.upper() if character.isalnum())

    def coverage(
        self,
        procedure: Artifact,
        diagnoses: Iterable[Artifact],
        service_date: date,
        jurisdiction: str,
        payer_identifier: str,
        payer_type: str,
        contract_profile: str,
    ) -> CoverageEvaluation:
        procedure_code = self._normalized_code(procedure.code)
        diagnosis_codes = {self._normalized_code(item.code) for item in diagnoses}
        with self._connect() as connection:
            policies = connection.execute(
                """
                SELECT p.* FROM coverage_procedures cp
                JOIN coverage_policies p USING (policy_key)
                WHERE cp.normalized_code = ?
                  AND (p.effective_start IS NULL OR p.effective_start <= ?)
                  AND (p.effective_end IS NULL OR p.effective_end >= ?)
                """,
                (procedure_code, service_date.isoformat(), service_date.isoformat()),
            ).fetchall()
            applicable = []
            normalized_jurisdiction = jurisdiction.casefold().strip()
            normalized_payer_identifier = payer_identifier.casefold().strip()
            normalized_payer_type = payer_type.casefold().strip()
            normalized_contract_profile = contract_profile.casefold().strip()
            alias_rows = connection.execute(
                "SELECT canonical FROM jurisdiction_aliases WHERE alias = ?",
                (normalized_jurisdiction,),
            ).fetchall()
            canonical = {row["canonical"] for row in alias_rows} | {normalized_jurisdiction}
            slots = ",".join("?" for _ in canonical)
            expanded_rows = connection.execute(
                f"SELECT alias FROM jurisdiction_aliases WHERE canonical IN ({slots})",
                tuple(sorted(canonical)),
            ).fetchall()
            accepted_jurisdictions = canonical | {row["alias"] for row in expanded_rows}
            for policy in policies:
                scopes = [str(item).casefold().strip() for item in json.loads(policy["jurisdictions_json"])]
                if scopes and not accepted_jurisdictions.intersection(scopes):
                    continue
                payer_types = {str(item).casefold().strip() for item in json.loads(policy["payer_types_json"])}
                payer_identifiers = {str(item).casefold().strip() for item in json.loads(policy["payer_identifiers_json"])}
                contract_profiles = {str(item).casefold().strip() for item in json.loads(policy["contract_profiles_json"])}
                if payer_types and normalized_payer_type not in payer_types:
                    continue
                if payer_identifiers and normalized_payer_identifier not in payer_identifiers:
                    continue
                if contract_profiles and normalized_contract_profile not in contract_profiles:
                    continue
                applicable.append(policy)
            if not applicable:
                return CoverageEvaluation("not_determinable", "no applicable compiled coverage policy", (), ())
            refs, policy_ids = [], []
            covered = False
            for policy in applicable:
                policy_ids.append(policy["policy_id"])
                refs.append(f"{policy['source_hash']}:{policy['source_locator']}")
                rows = connection.execute(
                    """
                    SELECT normalized_code, disposition, group_id, role FROM coverage_diagnoses
                    WHERE policy_key = ? AND (procedure_scope = '' OR procedure_scope = ?)
                    """,
                    (policy["policy_key"], procedure_code),
                ).fetchall()
                matched = [row for row in rows if row["normalized_code"] in diagnosis_codes]
                dispositions = {row["disposition"] for row in matched}
                if "noncovered" in dispositions:
                    return CoverageEvaluation("fail", "an applicable policy identifies a diagnosis as noncovered", tuple(refs), tuple(policy_ids))
                covered_roles = {row["role"] for row in matched if row["disposition"] == "covered"}
                covered = covered or "unspecified" in covered_roles or (
                    "primary_eligible" in covered_roles and "required_secondary" in covered_roles
                )
            if covered:
                return CoverageEvaluation("pass", "procedure-diagnosis relationship is present in an applicable policy", tuple(refs), tuple(policy_ids))
            return CoverageEvaluation("fail", "no documented diagnosis satisfies the applicable policy", tuple(refs), tuple(policy_ids))

    def active_on(self, artifact: Artifact, service_date: date) -> bool:
        value = service_date.isoformat()
        return (
            artifact.status == "active"
            and (artifact.effective_start is None or artifact.effective_start <= value)
            and (artifact.effective_end is None or artifact.effective_end >= value)
        )

    def constraints(
        self,
        kinds: Iterable[str],
        codes: Iterable[str],
        service_date: date,
    ) -> list[dict[str, Any]]:
        kind_values = tuple(sorted(set(kinds)))
        code_values = tuple(sorted(set(codes)))
        if not kind_values or not code_values:
            return []
        kind_slots = ",".join("?" for _ in kind_values)
        code_slots = ",".join("?" for _ in code_values)
        value = service_date.isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM constraints
                WHERE kind IN ({kind_slots})
                  AND (
                    target_code IN ({code_slots})
                    OR left_code IN ({code_slots})
                    OR right_code IN ({code_slots})
                  )
                  AND (effective_start IS NULL OR effective_start <= ?)
                  AND (effective_end IS NULL OR effective_end >= ?)
                """,
                (*kind_values, *code_values, *code_values, *code_values, value, value),
            ).fetchall()
        return [
            {
                **dict(row),
                "properties": json.loads(row["properties_json"]),
            }
            for row in rows
        ]

