"""Governed, code-system-agnostic retrieval lexicons.

Lexicons may improve candidate recall, but they are never medical-code
authority.  This module validates every configured pack against the exact
authoritative code-source bytes, quarantines uncorroborated generated
mappings, bounds ambiguous term fan-out, and emits a stable report that is
bound into claim readiness.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from app.core import config


_ALLOWED_STATUSES = {"active", "candidate", "disabled"}
_ALLOWED_PROVENANCE = {
    "government_primary", "licensed_primary", "generated",
    "generated_corroborated",
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode()


def _fingerprint(value) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_path(value: str) -> Path:
    path = (config.BASE_DIR / str(value)).resolve()
    if not path.is_relative_to(config.BASE_DIR.resolve()):
        raise ValueError(f"retrieval lexicon path escapes repository: {value}")
    return path


def _normalize_code(value) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _normalize_term(value) -> str:
    return " ".join(unicodedata.normalize(
        "NFKC", str(value or "")).casefold().split())


def _code_keys(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("codes", "data", "items", "results", "entries"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"authoritative code source has no record list: {path}")
    keys = {
        _normalize_code(row.get("code"))
        for row in data if isinstance(row, dict) and row.get("code")
    }
    if not keys:
        raise ValueError(f"authoritative code source has no code identities: {path}")
    return keys


def mapping_key(code: str, term: str) -> str:
    return _fingerprint({"code": _normalize_code(code),
                         "term": _normalize_term(term)})


class RetrievalLexiconRegistry:
    """Load only packs that satisfy the versioned catalog policy."""

    def __init__(self, catalog_path: Path | None = None):
        self.catalog_path = Path(
            catalog_path or config.RETRIEVAL_LEXICON_CATALOG_FILE)
        self._terms: dict[str, dict[str, list[str]]] = defaultdict(dict)
        self._bound_paths: list[Path] = [self.catalog_path]
        self.report = self._load()

    @property
    def bound_paths(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(self._bound_paths))

    def synonyms_for(self, code_system: str) -> dict[str, list[str]]:
        return {
            code: list(values)
            for code, values in self._terms.get(
                str(code_system or "").strip().lower(), {}).items()
        }

    def _load(self) -> dict:
        errors: list[str] = []
        active: list[dict] = []
        quarantined: list[dict] = []
        try:
            catalog = json.loads(self.catalog_path.read_text())
        except Exception as exc:
            return self._report({}, [], [], [f"catalog unavailable: {exc}"])
        if catalog.get("schema_version") != 1:
            return self._report(catalog, [], [],
                                ["unsupported retrieval lexicon catalog schema"])
        policy = catalog.get("policy") or {}
        try:
            min_length = int(policy.get("minimum_term_characters", 2))
            max_length = int(policy.get("maximum_term_characters", 160))
            default_fanout = int(policy.get("maximum_term_code_fanout", 24))
            required_domains = int(
                policy.get("minimum_generated_independence_domains", 2))
        except (TypeError, ValueError):
            return self._report(catalog, [], [],
                                ["retrieval lexicon policy limits are malformed"])
        if not (1 <= min_length <= max_length and default_fanout >= 1
                and required_domains >= 2):
            return self._report(catalog, [], [],
                                ["retrieval lexicon policy limits are unsafe"])

        seen_ids: set[str] = set()
        for row in catalog.get("packs") or []:
            pack_id = str(row.get("id") or "").strip()
            if not pack_id or pack_id in seen_ids:
                errors.append("retrieval lexicon pack ids must be non-empty and unique")
                continue
            seen_ids.add(pack_id)
            try:
                outcome = self._load_pack(
                    row, min_length=min_length, max_length=max_length,
                    default_fanout=default_fanout,
                    required_domains=required_domains)
            except Exception as exc:
                errors.append(f"{pack_id}: {exc}")
                continue
            if outcome["status"] == "active":
                active.append(outcome)
            else:
                quarantined.append(outcome)
        self._activate_terms(active)
        return self._report(catalog, active, quarantined, errors)

    def _activate_terms(self, active: list[dict]) -> None:
        """Apply fan-out limits across all active packs, then publish terms."""
        global_fanout: dict[tuple[str, str], set[str]] = defaultdict(set)
        global_limits: dict[tuple[str, str], list[int]] = defaultdict(list)
        for outcome in active:
            system = outcome["code_system"]
            limit = outcome["maximum_term_code_fanout"]
            for code, values in outcome["_accepted_terms"].items():
                for term in values:
                    key = (system, _normalize_term(term))
                    global_fanout[key].add(code)
                    global_limits[key].append(limit)
        for outcome in active:
            system = outcome["code_system"]
            final: dict[str, list[str]] = defaultdict(list)
            rejected = 0
            for code, values in outcome.pop("_accepted_terms").items():
                for term in values:
                    key = (system, _normalize_term(term))
                    if len(global_fanout[key]) > min(global_limits[key]):
                        rejected += 1
                    else:
                        final[code].append(term)
            outcome["rejected_cross_pack_fanout_count"] = rejected
            outcome["accepted_term_count"] = sum(map(len, final.values()))
            current = self._terms[system]
            for code, values in final.items():
                current.setdefault(code, [])
                current[code].extend(value for value in values
                                     if value not in current[code])

    def _load_pack(self, row: dict, *, min_length: int, max_length: int,
                   default_fanout: int, required_domains: int) -> dict:
        pack_id = str(row["id"])
        status = str(row.get("status") or "").lower()
        provenance = str(row.get("provenance_kind") or "")
        authority_role = str(row.get("authority_role") or "")
        code_system = str(row.get("code_system") or "").strip().lower()
        pack_format = str(row.get("pack_format") or "").strip().lower()
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"unsupported status {status!r}")
        if provenance not in _ALLOWED_PROVENANCE:
            raise ValueError(f"unsupported provenance {provenance!r}")
        if authority_role != "retrieval_only":
            raise ValueError("pack attempted to act as coding authority")
        if not code_system:
            raise ValueError("code system is absent")
        if pack_format not in {"legacy_primary", "legacy_generated",
                               "governed_v1"}:
            raise ValueError("pack format is absent or unsupported")

        pack_path = _safe_path(str(row.get("path") or ""))
        source_path = _safe_path(str(row.get("code_source") or ""))
        self._bound_paths.extend((pack_path, source_path))
        if not pack_path.is_file() or not source_path.is_file():
            raise FileNotFoundError("pack or authoritative code source is absent")
        pack_sha = _sha256(pack_path)
        source_sha = _sha256(source_path)
        if row.get("pack_sha256") != pack_sha:
            raise ValueError("pack bytes do not match the catalog binding")
        if row.get("code_source_sha256") != source_sha:
            raise ValueError("authoritative code-source bytes changed")

        data = json.loads(pack_path.read_text())
        if pack_format == "governed_v1":
            if data.get("schema_version") != 1:
                raise ValueError("governed pack schema is unsupported")
            bindings = {
                "pack_id": str(row.get("id") or ""),
                "code_system": code_system,
                "authority_role": authority_role,
                "provenance_kind": provenance,
                "code_source_sha256": source_sha,
            }
            mismatched = [key for key, expected in bindings.items()
                          if str(data.get(key) or "") != expected]
            if mismatched:
                raise ValueError("governed pack metadata contradicts catalog: "
                                 + ", ".join(mismatched))
        elif status == "active" and provenance.startswith("generated"):
            raise ValueError("generated active packs require governed_v1 format")
        terms = data.get("terms") if isinstance(data, dict) else None
        if not isinstance(terms, dict):
            raise ValueError("pack has no terms mapping")
        if data.get("count") is not None and int(data["count"]) != len(terms):
            raise ValueError("declared term-map count does not match content")

        known_codes = _code_keys(source_path)
        unknown_codes = sorted(
            code for code in (_normalize_code(value) for value in terms)
            if code not in known_codes)
        unknown_policy = str(
            row.get("unknown_code_policy") or "error").strip().lower()
        if unknown_codes and unknown_policy != "reject_terms":
            raise ValueError(
                f"pack references {len(unknown_codes)} unknown code identities")

        generated = provenance.startswith("generated")
        candidate_sha = ""
        mapping_attestations = data.get("mapping_attestations") or {}
        pack_domains = {
            str(value.get("independence_domain") or "").strip().lower()
            for value in row.get("attestations") or []
            if isinstance(value, dict)
            and value.get("pack_sha256") == pack_sha
            and value.get("decision") == "approved"
            and str(value.get("provider") or "").strip().lower()
            == str(value.get("independence_domain") or "").strip().lower()
            and str(value.get("independence_domain") or "").strip()
            and str(value.get("model") or "").strip()
            and str(value.get("profile_id") or "").strip()
        }
        pack_domains.update(
            str(value.get("independence_domain") or "").strip().lower()
            for rows in mapping_attestations.values()
            if isinstance(rows, list)
            for value in rows if isinstance(value, dict)
            and value.get("decision") == "approved"
            and str(value.get("provider") or "").strip().lower()
            == str(value.get("independence_domain") or "").strip().lower()
            and str(value.get("independence_domain") or "").strip()
            and str(value.get("model") or "").strip()
            and str(value.get("profile_id") or "").strip())
        quarantine_reasons = []
        if status == "candidate":
            quarantine_reasons.append("catalog status is candidate")
        elif status == "disabled":
            quarantine_reasons.append("catalog status is disabled")
        if generated and len(pack_domains) < required_domains:
            quarantine_reasons.append(
                "generated pack lacks independent pack attestations")
        if generated and status == "active" and (
                data.get("complete") is not True
                or int(data.get("evaluated_mapping_count", -1))
                != int(data.get("candidate_mapping_count", -2))):
            quarantine_reasons.append(
                "generated pack is not a complete candidate-corpus evaluation")
        if generated and status == "active":
            candidate_path = _safe_path(str(row.get("candidate_source") or ""))
            self._bound_paths.append(candidate_path)
            if not candidate_path.is_file():
                raise FileNotFoundError(
                    "generated active pack candidate source is absent")
            candidate_sha = _sha256(candidate_path)
            if (row.get("candidate_source_sha256") != candidate_sha
                    or data.get("source_candidate_sha256") != candidate_sha):
                raise ValueError(
                    "generated pack is not bound to its candidate-source bytes")
            candidate_data = json.loads(candidate_path.read_text())
            candidate_terms = (candidate_data.get("terms")
                               if isinstance(candidate_data, dict) else None)
            if (candidate_data.get("authority_role") != "retrieval_only"
                    or candidate_data.get("provenance_kind") != "generated"
                    or candidate_data.get("code_system") != code_system
                    or candidate_data.get("code_source_sha256") != source_sha
                    or not isinstance(candidate_terms, dict)):
                raise ValueError("generated candidate-source metadata is invalid")
            candidate_ids_list = [
                mapping_key(code, term)
                for code, values in candidate_terms.items()
                if isinstance(values, list)
                for term in values if isinstance(term, str)
            ]
            candidate_ids = set(candidate_ids_list)
            accepted_count = sum(len(values) for values in terms.values()
                                 if isinstance(values, list))
            declared_accepted = int(
                data.get("accepted_mapping_count", -1))
            declared_rejected = int(
                data.get("rejected_mapping_count", -1))
            candidate_count = int(data.get("candidate_mapping_count", -1))
            rejected_ids = data.get("rejected_mapping_fingerprints") or []
            accepted_ids = {
                mapping_key(code, term)
                for code, values in terms.items() if isinstance(values, list)
                for term in values if isinstance(term, str)
            }
            if (declared_accepted != accepted_count
                    or declared_rejected != len(rejected_ids)
                    or len(set(rejected_ids)) != len(rejected_ids)
                    or accepted_ids.intersection(rejected_ids)
                    or len(candidate_ids_list) != len(candidate_ids)
                    or accepted_ids.union(rejected_ids) != candidate_ids
                    or declared_accepted + declared_rejected != candidate_count
                    or candidate_count != len(candidate_ids)):
                quarantine_reasons.append(
                    "generated pack mapping-accounting proof is incomplete")

        fanout: dict[str, set[str]] = defaultdict(set)
        cleaned: dict[str, list[tuple[str, str]]] = defaultdict(list)
        rejected_shape = 0
        for raw_code, values in terms.items():
            code = _normalize_code(raw_code)
            if code not in known_codes:
                continue
            if not isinstance(values, list):
                rejected_shape += 1
                continue
            seen: set[str] = set()
            for value in values:
                term = str(value).strip() if isinstance(value, str) else ""
                normalized = _normalize_term(term)
                if (not min_length <= len(term) <= max_length
                        or not normalized or normalized in seen):
                    rejected_shape += 1
                    continue
                seen.add(normalized)
                fanout[normalized].add(code)
                cleaned[code].append((term, normalized))
        max_fanout = int(row.get("maximum_term_code_fanout") or default_fanout)
        accepted: dict[str, list[str]] = defaultdict(list)
        rejected_fanout = 0
        rejected_attestation = 0
        for code, values in cleaned.items():
            for term, normalized in values:
                if len(fanout[normalized]) > max_fanout:
                    rejected_fanout += 1
                    continue
                if generated and status == "active":
                    attestations = mapping_attestations.get(
                        mapping_key(code, term)) or []
                    domains = {
                        str(item.get("independence_domain") or "").strip().lower()
                        for item in attestations if isinstance(item, dict)
                        and item.get("decision") == "approved"
                        and str(item.get("provider") or "").strip().lower()
                        == str(item.get("independence_domain") or "").strip().lower()
                        and str(item.get("independence_domain") or "").strip()
                        and str(item.get("model") or "").strip()
                        and str(item.get("profile_id") or "").strip()
                    }
                    if len(domains) < required_domains:
                        rejected_attestation += 1
                        continue
                accepted[code].append(term)

        effective_status = "quarantined" if quarantine_reasons else status
        outcome = {
            "id": pack_id,
            "status": effective_status,
            "configured_status": status,
            "code_system": code_system,
            "provenance_kind": provenance,
            "pack_format": pack_format,
            "authority_role": authority_role,
            "pack_sha256": pack_sha,
            "code_source_sha256": source_sha,
            "candidate_source_sha256": candidate_sha,
            "code_count": len(terms),
            "accepted_term_count": 0,
            "candidate_term_count": sum(map(len, cleaned.values())),
            "rejected_shape_count": rejected_shape,
            "rejected_unknown_code_count": len(unknown_codes),
            "rejected_fanout_count": rejected_fanout,
            "rejected_attestation_count": rejected_attestation,
            "maximum_term_code_fanout": max_fanout,
            "independence_domains": sorted(pack_domains),
            "quarantine_reasons": quarantine_reasons,
        }
        if effective_status == "active":
            outcome["_accepted_terms"] = dict(accepted)
        return outcome

    def _report(self, catalog: dict, active: list[dict],
                quarantined: list[dict], errors: list[str]) -> dict:
        body = {
            "schema_version": 1,
            "catalog_version": str(catalog.get("version") or ""),
            "catalog_sha256": (
                _sha256(self.catalog_path) if self.catalog_path.is_file() else ""),
            "authority_role": "retrieval_only",
            "active_packs": active,
            "quarantined_packs": quarantined,
            "errors": errors,
        }
        body["report_fingerprint"] = _fingerprint(body)
        return body


def current_retrieval_lexicon_registry(
        catalog_path: Path | None = None) -> RetrievalLexiconRegistry:
    """Return a cached registry keyed by every bound file's exact bytes."""
    path = Path(catalog_path or config.RETRIEVAL_LEXICON_CATALOG_FILE)
    identities = []
    candidates = [path]
    try:
        catalog = json.loads(path.read_text())
        for row in catalog.get("packs") or []:
            candidates.extend((_safe_path(str(row.get("path") or "")),
                               _safe_path(str(row.get("code_source") or ""))))
            if row.get("candidate_source"):
                candidates.append(_safe_path(str(row["candidate_source"])))
    except Exception:
        pass
    for candidate in dict.fromkeys(candidates):
        try:
            identities.append((str(candidate), _sha256(candidate)))
        except OSError:
            identities.append((str(candidate), "missing"))
    return _cached_registry(str(path), tuple(identities))


@lru_cache(maxsize=8)
def _cached_registry(path_text: str, _identities: tuple
                     ) -> RetrievalLexiconRegistry:
    return RetrievalLexiconRegistry(Path(path_text))


__all__ = ["RetrievalLexiconRegistry", "mapping_key",
           "current_retrieval_lexicon_registry"]
