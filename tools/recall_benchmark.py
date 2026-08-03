#!/usr/bin/env python3
"""Versioned retrieval evaluation with recall and candidate-burden controls.

Medical-code expectations live in data, never production Python. Every probe
is checked against the live authoritative reference set before measurement.
Candidate reports can be compared with a prior run to detect recall loss,
rank displacement, and additional incorrect candidates ahead of the expected
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BASE_DIR


DEFAULT_CORPUS = BASE_DIR / "data" / "benchmarks" / "retrieval_recall.json"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def _fingerprint(value) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _norm_code(value) -> str:
    return "".join(char for char in str(value or "").upper()
                   if char.isalnum())


def load_corpus(path: Path = DEFAULT_CORPUS) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported retrieval benchmark schema")
    if data.get("purpose") != "retrieval_evaluation_only" or data.get(
            "authority_role") != "non_production_benchmark":
        raise ValueError("retrieval benchmark attempted to become production input")
    probes = data.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("retrieval benchmark has no probes")
    allowed_systems = {"icd10", "cpt", "hcpcs"}
    seen = set()
    for index, row in enumerate(probes, 1):
        if not isinstance(row, dict):
            raise ValueError(f"probe {index} is malformed")
        query = str(row.get("query") or "").strip()
        code = _norm_code(row.get("expected_code"))
        system = str(row.get("code_system") or "").strip().lower()
        key = (query.casefold(), code, system)
        if not query or not code or system not in allowed_systems:
            raise ValueError(f"probe {index} is incomplete")
        if key in seen:
            raise ValueError(f"probe {index} is duplicated")
        seen.add(key)
    body = {key: value for key, value in data.items()
            if key != "corpus_fingerprint"}
    data["corpus_fingerprint"] = _fingerprint(body)
    return data


def load_probes(path: Path = DEFAULT_CORPUS) -> list[tuple[str, str, str]]:
    return [(row["query"], row["expected_code"], row["code_system"])
            for row in load_corpus(path)["probes"]]


# Compatibility for existing offline embedding tools. Values are data-loaded,
# not hardcoded medical-code decisions.
PROBES = load_probes()


def validate_expected_codes(probes: list[tuple[str, str, str]]) -> None:
    from app.rag.code_reference import CodeReferenceDB
    db = CodeReferenceDB()
    db.load_all()
    lookups = {"icd10": db.validate_icd10,
               "cpt": db.validate_cpt,
               "hcpcs": db.validate_hcpcs}
    invalid = []
    for _query, expected, system in probes:
        if not lookups[system](expected):
            invalid.append(f"{system}:{expected}")
    if invalid:
        raise ValueError("benchmark expected identities are absent from the "
                         "authoritative source: " + ", ".join(invalid))


def score_rows(rows: list[dict]) -> dict:
    by_system: dict[str, list[dict]] = {}
    for row in rows:
        by_system.setdefault(row["code_system"], []).append(row)

    def summarize(values: list[dict]) -> dict:
        count = len(values)
        found = sum(row["rank"] is not None for row in values)
        return {
            "probes": count,
            "found": found,
            "recall": found / count if count else 0.0,
            "mrr": sum(1.0 / row["rank"] if row["rank"] else 0.0
                       for row in values) / count if count else 0.0,
            "mean_false_candidates_ahead": sum(
                row["false_candidates_ahead"] for row in values
            ) / count if count else 0.0,
        }
    return {"by_system": {key: summarize(value)
                           for key, value in sorted(by_system.items())},
            "overall": summarize(rows)}


def compare_reports(baseline: dict, candidate: dict) -> dict:
    if baseline.get("corpus_fingerprint") != candidate.get(
            "corpus_fingerprint"):
        raise ValueError("benchmark reports use different corpora")
    base_rows = {row["probe_fingerprint"]: row
                 for row in baseline.get("rows") or []}
    candidate_rows = {row["probe_fingerprint"]: row
                      for row in candidate.get("rows") or []}
    if (len(base_rows) != len(baseline.get("rows") or [])
            or len(candidate_rows) != len(candidate.get("rows") or [])
            or set(base_rows) != set(candidate_rows)):
        raise ValueError("benchmark reports do not contain identical probe sets")
    displacement = []
    for row in candidate_rows.values():
        prior = base_rows[row["probe_fingerprint"]]
        before = prior.get("rank")
        after = row.get("rank")
        displacement.append({
            "probe_fingerprint": row["probe_fingerprint"],
            "baseline_rank": before,
            "candidate_rank": after,
            "rank_displacement": (
                after - before if before is not None and after is not None
                else None),
            "recall_regression": before is not None and after is None,
            "false_candidate_burden_delta": (
                row["false_candidates_ahead"]
                - prior["false_candidates_ahead"]),
        })
    regressions = [row for row in displacement if row["recall_regression"]]
    return {
        "rows": displacement,
        "recall_regressions": len(regressions),
        "rank_regressions": sum(
            (row.get("rank_displacement") or 0) > 0 for row in displacement),
        "additional_false_candidates_ahead": sum(max(
            0, row["false_candidate_burden_delta"]) for row in displacement),
        "passes_no_regression_gate": not regressions and not any(
            (row.get("rank_displacement") or 0) > 0
            or row["false_candidate_burden_delta"] > 0
            for row in displacement),
    }


def run_benchmark(store, corpus: dict, *, top_k: int | None = None) -> dict:
    probes = [(row["query"], row["expected_code"], row["code_system"])
              for row in corpus["probes"]]
    validate_expected_codes(probes)
    rows = []
    for query, expected, system in probes:
        hits = store.search(query, system, top_k=top_k)
        normalized = [_norm_code(hit.get("code")) for hit in hits]
        wanted = _norm_code(expected)
        rank = next((index + 1 for index, code in enumerate(normalized)
                     if code == wanted), None)
        rows.append({
            "probe_fingerprint": _fingerprint({
                "query": query, "expected_code": wanted,
                "code_system": system}),
            "query": query,
            "expected_code": expected,
            "code_system": system,
            "rank": rank,
            "false_candidates_ahead": rank - 1 if rank else len(hits),
            "retrieved_codes": [str(hit.get("code") or "") for hit in hits],
        })
    report = {
        "schema_version": 1,
        "corpus_fingerprint": corpus["corpus_fingerprint"],
        "catalog_version": corpus.get("version") or "",
        "top_k": top_k,
        "retrieval_lexicon": getattr(store, "lexicon_report", {}),
        "rows": rows,
        "metrics": score_rows(rows),
    }
    report["report_fingerprint"] = _fingerprint(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--write-results")
    parser.add_argument("--compare-to")
    parser.add_argument("--require-no-regression", action="store_true")
    args = parser.parse_args()

    from app.rag.vector_store import MedicalCodeVectorStore
    corpus = load_corpus(Path(args.corpus))
    store = MedicalCodeVectorStore()
    store.build_or_load()
    report = run_benchmark(store, corpus, top_k=args.top_k)
    comparison = None
    if args.compare_to:
        comparison = compare_reports(
            json.loads(Path(args.compare_to).read_text()), report)
        report["comparison"] = comparison
        report["report_fingerprint"] = _fingerprint({
            key: value for key, value in report.items()
            if key != "report_fingerprint"})
    if args.write_results:
        output = Path(args.write_results)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"metrics": report["metrics"],
                      "comparison": comparison}, indent=2))
    if args.require_no_regression and (
            not comparison or not comparison["passes_no_regression_gate"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
