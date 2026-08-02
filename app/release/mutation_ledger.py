"""Central accounting for every candidate-to-final claim mutation."""

from __future__ import annotations

import json
from copy import deepcopy


ARRAYS = ("icd_codes", "cpt_codes", "hcpcs_codes")


def _billable_row(array: str, row: dict) -> dict:
    """Fields whose change alters diagnosis selection or a claim line.

    Provenance enrichment, confidence, rationale, and descriptions are
    deliberately excluded: attaching evidence after candidate generation is
    not itself a claim mutation.
    """
    fields = (("code", "type") if array == "icd_codes" else
              ("code", "modifiers", "units", "dx_pointers",
               "diagnosis_pointers", "linked_diagnoses"))
    return {key: deepcopy(row.get(key)) for key in fields
            if row.get(key) not in (None, "", [])}


def normalize_claim(value: dict) -> dict:
    return {
        "icd_codes": deepcopy(value.get("icd_codes")
                              or value.get("icd10_codes") or []),
        "cpt_codes": deepcopy(value.get("cpt_codes") or []),
        "hcpcs_codes": deepcopy(value.get("hcpcs_codes") or []),
    }


def claim_diff(before: dict, after: dict) -> list[dict]:
    """Return stable row-level changes; unchanged claims need no ledger."""
    out = []
    before, after = normalize_claim(before), normalize_claim(after)
    for array in ARRAYS:
        def grouped(rows):
            groups = {}
            for row in rows:
                if not isinstance(row, dict) or not row.get("code"):
                    continue
                code = str(row["code"]).upper()
                groups.setdefault(code, []).append(_billable_row(array, row))
            for values in groups.values():
                values.sort(key=lambda value: json.dumps(
                    value, sort_keys=True, default=str))
            return groups
        old, new = grouped(before[array]), grouped(after[array])
        for code in sorted(set(old) | set(new)):
            old_rows, new_rows = old.get(code, []), new.get(code, [])
            for occurrence in range(max(len(old_rows), len(new_rows))):
                old_row = old_rows[occurrence] if occurrence < len(old_rows) else None
                new_row = new_rows[occurrence] if occurrence < len(new_rows) else None
                if json.dumps(old_row, sort_keys=True, default=str) != \
                        json.dumps(new_row, sort_keys=True, default=str):
                    out.append({"array": array, "code": code,
                                "occurrence": occurrence,
                                "before": old_row, "after": new_row})
    return out


def reconcile_mutation_ledger(candidate: dict, final_claim: dict,
                              reported: list[dict]) -> list[dict]:
    """Bind reported corrections to actual diffs and expose every gap.

    This does not infer authority.  Missing evidence/source/reason fields
    leave an entry unresolved so the release gate fails closed.
    """
    diffs = claim_diff(candidate, final_claim)
    ledger = []
    for diff in diffs:
        matching = [r for r in (reported or []) if isinstance(r, dict)
                    and str(r.get("code") or "").upper() == diff["code"]
                    and r.get("array") == diff["array"]
                    and str(r.get("occurrence", 0)) ==
                    str(diff["occurrence"])]
        record = matching[-1] if matching else {}
        evidence = record.get("evidence_spans") or record.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        sources = record.get("source_record_ids") or []
        complete = bool(record.get("reason") or record.get("rationale")) \
            and bool(evidence) and bool(sources) and bool(record.get("rule_id")) \
            and bool(record.get("effective_on")) \
            and record.get("array") == diff["array"]
        ledger.append({**diff,
                       "component": record.get("component") or
                                    record.get("layer") or "unattributed",
                       "reason": record.get("reason") or
                                 record.get("rationale") or "",
                       "evidence_spans": list(evidence),
                       "source_record_ids": list(sources),
                       "rule_id": record.get("rule_id") or "",
                       "effective_on": record.get("effective_on") or "",
                       "interpretive": bool(record.get("interpretive", True)),
                       "state": "applied" if complete else "unresolved"})
    return ledger
