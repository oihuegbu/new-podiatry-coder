#!/usr/bin/env python3
"""Calibration dataset — rung 8 of the directive's validation ladder.

Directive §9 rung 8 asks for "calibration by decision class and weakest axis — not
one global LLM confidence score". That is what this tool accumulates: one row per
CURRENT ClaimBundle, carrying

  identity  the exact (claim, data snapshot, model profiles, document version) triple
            `app.release.outcome_ledger.claim_identity` derives, so a row can never be
            joined to the wrong run of the same note;
  features  model inputs from the bundle itself — claim shape, resolution-method mix,
            context resolution, decision outcomes, routing breakdown, the WEAKEST
            documented axis;
  class     the canonical `ReleaseDestination` this encounter was routed to. This is
            the calibration bucket: AUTO_READY, AUTO_QUERY, SYSTEM_RETRY,
            NON_BILLABLE, BLOCKED and the residual REVIEW are answerable separately,
            because "how often is the system right" means different things in each;
  labels    what actually happened — a coder's correction from the claims registry
            and payer adjudication from the rung-7 outcome ledger. Both are `None`
            until real outcomes exist. `None` is not zero and is never averaged.

WHAT CHANGED AND WHY: this tool used to read the retired `app.pipeline` result shape
(`cpt_codes`, `icd_codes`, `validation_issues`, `consistency`, `final_disposition`).
The deployed entrypoint has emitted one canonical `ClaimBundle` per note since the
directive's phase 1, so every field it looked for was absent — and because a result
that did not declare `success` was counted as `skipped`, the exporter reported a
clean run over an empty dataset instead of failing. A silently-empty measurement is
worse than a missing one, so a non-bundle artifact is now REFUSED BY NAME.

Artifacts are enumerated through `attempt_ledger.resolve_current`, never a glob: a
superseded, failed or in-flight attempt must not contribute a row (issue #6 F6-R6-B).

Control mode: OBSERVATIONAL. Nothing here changes a claim or a routing decision.

Usage:
  python tools/calibration_dataset.py export [results_dir] [--out FILE]
  python tools/calibration_dataset.py stats  [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "output" / "results"
DEFAULT_OUT = ROOT / "data" / "registry" / "calibration_dataset.jsonl"

#: Bumped when the row shape changes. Rows from an older version are re-exported
#: rather than merged, because a mixed-shape dataset cannot be calibrated.
DATASET_VERSION = 2


class CalibrationInputError(RuntimeError):
    """An artifact cannot contribute a row, and staying silent about it would lie."""


# --------------------------------------------------------------------------
# features / class / labels
# --------------------------------------------------------------------------

def _weakest_axis(bundle) -> tuple[str, float | None]:
    """The lowest-confidence documented axis across every line's evidence.

    The producer records the axis by name in its routing reasons (`autonomy.decide`
    names `weakest axis '<name>'` when it routes on documentation clarity). Read from
    the ROUTING rather than re-derived, so the calibration bucket is the same axis the
    routing decision actually used.
    """
    for item in bundle.audit.routing:
        reason = str(item.get("reason") or "")
        marker = "weakest axis '"
        if marker in reason:
            name = reason.split(marker, 1)[1].split("'", 1)[0]
            return name, None
    return "", None


def extract_features(bundle) -> dict[str, Any]:
    """Model inputs only — the decision class and the labels are recorded separately.

    Nothing here may leak the label: the router's own verdict is the CLASS, and a
    feature that restates it would make any model trained on this dataset relearn the
    router instead of predicting whether it was right.
    """
    from app.contracts.claim_bundle import ContextResolution

    lines = tuple(bundle.service_lines) + tuple(bundle.diagnoses)
    methods = Counter(ln.method.value for ln in lines)
    outcomes = Counter(o.outcome for o in bundle.outcomes)
    routing = Counter(str(i.get("destination") or "") for i in bundle.audit.routing)
    axis, _ = _weakest_axis(bundle)
    return {
        "n_diagnoses": len(bundle.diagnoses),
        "n_service_lines": len(bundle.service_lines),
        "n_modifiers": sum(len(ln.modifiers) for ln in bundle.service_lines),
        "total_units": sum(int(ln.units) for ln in bundle.service_lines),
        "n_excluded_lines": len(bundle.audit.excluded_lines),
        "method_mix": dict(sorted(methods.items())),
        "n_decision_outcomes": len(bundle.outcomes),
        "n_outcomes_blocked": outcomes.get("BLOCKED", 0) + outcomes.get("ERROR", 0),
        "n_outcomes_unknown": outcomes.get("UNKNOWN", 0),
        "n_outcomes_retryable": sum(1 for o in bundle.outcomes if o.retryable),
        "routing_mix": dict(sorted(routing.items())),
        "n_recommendations": len(bundle.audit.recommendations),
        "n_necessity_bindings": len(bundle.audit.necessity_support),
        "context_resolution": bundle.context.resolution.value,
        "context_is_resolved": bundle.context.resolution is ContextResolution.RESOLVED,
        "has_certificate": bundle.certificate is not None,
        "has_source_reconciliation": bool(
            bundle.encounter.source_document.document_version),
        "graph_nodes": len(bundle.graph.clinical_event_ids),
        "graph_relations": len(bundle.graph.relation_ids),
        "weakest_axis": axis,
        "n_integrity_problems": len(bundle.integrity_problems()),
    }


def decision_class(bundle) -> str:
    """The calibration bucket — the canonical routing destination (directive §8)."""
    return bundle.release.destination.value


def extract_labels(bundle, payload: dict, registry_view: dict,
                   outcome_rows: list[dict]) -> dict:
    """What actually happened to this exact claim.

    `None` means UNKNOWN, and every consumer must treat it as unknown rather than as a
    negative: a deployment with no submitted claims has no denials, which is not the
    same as having no denial problem.

    "Did a human change it?" is decided by `claims_registry.extract_claim`, which for a
    canonical artifact is the CONTRACT'S OWN `claim_content()`. This module deliberately
    does not restate what counts as claim-affecting: a second opinion about that is how
    a corrected modifier or a re-ordered diagnosis becomes an invisible correction.
    """
    from tools.claims_registry import extract_claim, _claim_key
    from app.release.outcome_ledger import claim_identity

    identity = claim_identity(bundle)
    doc = str(bundle.encounter.encounter_id or "")
    reg = registry_view.get(doc)

    human_corrected = None
    if reg and reg.get("verification") == "human":
        human_corrected = _claim_key(reg["claim"]) != _claim_key(extract_claim(payload))

    mine = [r for r in outcome_rows if r.get("identity_key") == identity.key]
    denied = [r for r in mine if r.get("body", {}).get("denied")]
    return {
        "human_corrected": human_corrected,
        "payer_outcomes": len(mine),
        "denied": (bool(denied) if mine else None),
        "carcs": sorted({c for r in denied
                         for c in (r.get("body", {}).get("carcs") or [])}),
    }


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _current_artifacts(results_dir: Path) -> list[tuple[str, dict]]:
    """Every CURRENT artifact, through the attempt ledger. Refusals are the caller's
    to report — they are why a note is absent, and an absence with no reason is how a
    superseded result silently becomes a measurement."""
    from app.release.attempt_ledger import resolve_current

    current = resolve_current(results_dir)
    out: list[tuple[str, dict]] = []
    for entry in current.results:
        try:
            out.append((entry.document_id, json.loads(entry.path.read_text())))
        except Exception as exc:
            raise CalibrationInputError(
                f"current artifact for {entry.document_id} at {entry.path} is "
                f"unreadable: {exc}") from exc
    return out


def build_row(document_id: str, payload: dict, registry_view: dict,
              outcome_rows: list[dict]) -> dict:
    from app.contracts.claim_bundle import is_claim_bundle, load_bundle
    from app.release.outcome_ledger import claim_identity

    if not is_claim_bundle(payload):
        raise CalibrationInputError(
            f"{document_id}: artifact is not a ClaimBundle "
            f"(schema_id={payload.get('schema_id')!r}). This exporter calibrates the "
            f"canonical contract only; a legacy artifact is refused rather than "
            f"silently skipped, because an empty dataset that looks healthy is the "
            f"failure this refusal exists to prevent.")
    bundle = load_bundle(payload)
    return {
        "dataset_version": DATASET_VERSION,
        "document_id": document_id,
        "identity": claim_identity(bundle).as_dict(),
        "decision_class": decision_class(bundle),
        "features": extract_features(bundle),
        "labels": extract_labels(bundle, payload, registry_view, outcome_rows),
    }


def export(results_dir: Path, out_path: Path) -> dict:
    from tools.claims_registry import load_events, current_view
    from app.release.outcome_ledger import OutcomeLedger, Rung

    view = current_view(load_events())
    outcome_rows = OutcomeLedger().observations(Rung.OUTCOME_FEEDBACK)

    rows: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("dataset_version") == DATASET_VERSION:
                    rows[r["document_id"]] = r

    stats = {"new": 0, "updated": 0, "unchanged": 0, "refused": 0}
    refusals: list[str] = []
    for document_id, payload in _current_artifacts(results_dir):
        try:
            row = build_row(document_id, payload, view, outcome_rows)
        except CalibrationInputError as exc:
            stats["refused"] += 1
            refusals.append(str(exc))
            continue
        old = rows.get(document_id)
        if old is None:
            stats["new"] += 1
        elif json.dumps(old, sort_keys=True, default=str) == json.dumps(
                row, sort_keys=True, default=str):
            stats["unchanged"] += 1
            continue
        else:
            stats["updated"] += 1
        rows[document_id] = row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for doc in sorted(rows):
            fh.write(json.dumps(rows[doc], sort_keys=True, default=str) + "\n")
    stats["total"] = len(rows)
    stats["refusals"] = refusals
    return stats


# --------------------------------------------------------------------------
# rung 8 — calibration by decision class and weakest axis
# --------------------------------------------------------------------------

def calibrate(rows: list[dict]) -> dict[str, Any]:
    """Per decision class, and per weakest axis, how often the class was right.

    Deliberately NOT one number. A single global accuracy figure over mixed decision
    classes hides the only comparison that matters — an AUTO_READY that a coder had to
    correct is a very different failure from a REVIEW that needed no change — and the
    directive names that conflation explicitly.

    Every bucket reports its label coverage. A bucket with no labelled outcome reports
    `None`, never a rate over zero observations.
    """
    def _bucket(subset: list[dict]) -> dict[str, Any]:
        corrected = [r for r in subset if r["labels"]["human_corrected"] is not None]
        denied = [r for r in subset if r["labels"]["denied"] is not None]
        return {
            "claims": len(subset),
            "human_verdicts": len(corrected),
            "human_correction_rate": (
                sum(1 for r in corrected if r["labels"]["human_corrected"])
                / len(corrected) if corrected else None),
            "payer_outcomes": len(denied),
            "denial_rate": (sum(1 for r in denied if r["labels"]["denied"])
                            / len(denied) if denied else None),
        }

    by_class: dict[str, Any] = {}
    for name in sorted({r["decision_class"] for r in rows}):
        by_class[name] = _bucket([r for r in rows if r["decision_class"] == name])
    by_axis: dict[str, Any] = {}
    for axis in sorted({r["features"].get("weakest_axis") or "" for r in rows}):
        if not axis:
            continue
        by_axis[axis] = _bucket(
            [r for r in rows if r["features"].get("weakest_axis") == axis])
    return {"by_decision_class": by_class, "by_weakest_axis": by_axis,
            "claims": len(rows)}


def print_stats(out_path: Path) -> None:
    if not out_path.exists():
        print("No dataset yet — run 'export' after a batch.")
        return
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("dataset_version") == DATASET_VERSION]
    if not rows:
        print(f"No rows at dataset_version {DATASET_VERSION} — re-run 'export'.")
        return
    cal = calibrate(rows)
    print(f"{cal['claims']} claim(s)\n")
    print("by decision class (directive section 8 routing):")
    for name, b in cal["by_decision_class"].items():
        print(f"  {name:<14} claims={b['claims']:<5} "
              f"human_verdicts={b['human_verdicts']:<4} "
              f"correction_rate={_fmt(b['human_correction_rate'])} "
              f"payer_outcomes={b['payer_outcomes']:<4} "
              f"denial_rate={_fmt(b['denial_rate'])}")
    if cal["by_weakest_axis"]:
        print("\nby weakest documented axis:")
        for name, b in cal["by_weakest_axis"].items():
            print(f"  {name:<20} claims={b['claims']:<5} "
                  f"correction_rate={_fmt(b['human_correction_rate'])}")
    print("\nA rate of 'n/a' means NO labelled outcome exists for that bucket — it is "
          "unknown, not good. Real calibration begins when submitted claims are "
          "adjudicated and recorded through app/release/outcome_ledger.py (ladder "
          "rungs 5-7).")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export", help="(re)build rows from a results dir")
    s.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s = sub.add_parser("stats", help="calibration by decision class and weakest axis")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    if args.cmd == "export":
        stats = export(Path(args.results_dir), Path(args.out))
        print(f"Calibration dataset: {stats['total']} row(s) "
              f"({stats['new']} new, {stats['updated']} updated, "
              f"{stats['unchanged']} unchanged, {stats['refused']} refused) "
              f"-> {args.out}")
        for refusal in stats["refusals"]:
            print(f"  REFUSED {refusal}")
    else:
        print_stats(Path(args.out))


if __name__ == "__main__":
    main()
