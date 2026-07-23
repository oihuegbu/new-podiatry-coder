#!/usr/bin/env python3
"""Flip triage: turn consistency disagreements into a machine-readable
actuation queue.

The consistency gate *detects* run-to-run billing disagreements; this tool
*organizes* them for actuation. It scans batch results, extracts every
billing (non-advisory) disagreement, enriches it with per-run evidence from
the saved consistency traces, clusters recurrences of the same mechanism
into a flip CLASS, and maintains an idempotent queue at
data/registry/flip_queue.jsonl.

A flip class is keyed by (kind, array, code) — e.g. "presence of L3260 in
hcpcs_codes" — because the same code flapping on different notes is one
mechanism to fix, not N. Each class carries per-document evidence: which
runs billed it, that entry's own fields per run, and the note sentences
speaking the code's descriptor vocabulary (the raw material a deterministic
rule needs).

Statuses: open (awaiting actuation) → actuated (a rule was accepted for it)
or escalated (auto-actuation judged no safe deterministic rule exists —
a human's queue). tools/auto_actuate.py consumes open classes and writes
statuses back.

Usage:
  python tools/flip_triage.py scan  [results_dir]   build/refresh the queue
  python tools/flip_triage.py list                  print the queue
  python tools/flip_triage.py reopen <class_key>... re-queue for actuation
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "output" / "results"
QUEUE_PATH = ROOT / "data" / "registry" / "flip_queue.jsonl"

QUEUE_VERSION = 1


def _class_key(d: dict) -> str:
    kind = d.get("kind", "presence")
    if kind == "em_level":
        return "em_level|cpt_codes|EM"
    return f"{kind}|{d.get('array', '?')}|{d.get('code', '?')}"


def _entry_of(run: dict, array: str, code: str) -> dict | None:
    for e in run.get(array) or []:
        if e.get("code") == code:
            return {k: e.get(k) for k in
                    ("code", "description", "modifiers", "units", "type",
                     "dx_pointers")
                    if e.get(k) not in (None, "", [])}
    return None


def _note_text_of(result: dict) -> str:
    return ((result.get("rag_context") or {}).get("note_full_text")
            or "")


def _descriptor_sentences(note_text: str, descriptions: list[str],
                          cap: int = 6) -> list[str]:
    """Note sentences speaking the flip code's own descriptor vocabulary —
    the evidence sentences a note-context rule would have to read."""
    if not note_text:
        return []
    terms = {w for d in descriptions
             for w in re.findall(r"[a-z]+", (d or "").lower()) if len(w) >= 5}
    out = []
    for sent in re.split(r"[.;\n]", note_text):
        s = sent.strip()
        if s and any(t in s.lower() for t in terms):
            out.append(s[:220])
        if len(out) >= cap:
            break
    return out


def _array_of(code: str) -> str:
    """Best-effort claim-array classification of a bare code (dotted or
    long alpha codes are ICD; exactly letter+4-digits is HCPCS; the rest
    is CPT). A miss only mislabels a class key, never loses evidence."""
    c = str(code or "").strip().upper()
    if "." in c or (c[:1].isalpha()
                    and not (len(c) == 5 and c[1:].isdigit())):
        return "icd_codes"
    if c[:1].isalpha():
        return "hcpcs_codes"
    return "cpt_codes"


def _verified_docs() -> set[str]:
    """Documents whose registry claim was verified by a HUMAN coder or the
    expert adjudicator — the only tiers that can serve as the realignment
    target for an audit-dispute actuation (an 'auto' event could BE the
    disputed claim)."""
    try:
        from tools.claims_registry import current_view, load_events
        return {doc for doc, e in current_view(load_events()).items()
                if e.get("verification") in ("human", "adjudicated")}
    except Exception:
        return set()


def _verified_code_targets() -> dict[str, dict[tuple, dict | None]]:
    """{doc: {(array, CODE): row-or-None}} — the PER-CODE verified targets
    a partial adjudication records when the note cannot verify whole (a
    consistency holdout split on other codes, or a claim a replay layer
    keeps overriding). They open exactly the audit-dispute classes whose
    (array, code) they cover: the adjudicated row is the scoped
    realignment goal, so a wrong deterministic rule no longer deadlocks
    behind a full-note verification it is itself blocking."""
    try:
        from tools.claims_registry import verified_code_targets
        return verified_code_targets()
    except Exception:
        return {}


def _verified_advisory_targets() -> dict[str, dict[tuple, bool]]:
    """{doc: {(OBSERVABLE, KEY): emit}} — the verified observable-emission
    targets an audit-dispute adjudication records when the dispute is
    about a measured phenomenon (e.g. a scrubber advisory), not a claim
    line (the claim is correct as billed, so no billing-signature target
    can ever exist). They open exactly the audit-dispute classes whose
    code they cover: the adjudicated emission state is the realignment
    goal the emission-aware replay gate converges on."""
    try:
        from tools.claims_registry import verified_observable_targets
        return verified_observable_targets()
    except Exception:
        return {}


def scan(results_dir: Path, queue_path: Path = QUEUE_PATH) -> dict:
    runs_dir = results_dir / "consistency_runs"
    classes: dict[str, dict] = {}
    if queue_path.exists():
        for line in queue_path.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                classes[c["class_key"]] = c

    stats = {"docs_scanned": 0, "flips_seen": 0, "new_classes": 0,
             "updated_classes": 0, "audit_disputes_seen": 0}
    verified = _verified_docs()
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        result = json.loads(f.read_text())
        doc = str(result.get("document_id") or f.stem.removesuffix("_results"))
        cons = result.get("consistency") or {}
        billing = [d for d in cons.get("disagreements") or []
                   if not d.get("advisory")]
        _scan_audit_disputes(result, doc, classes, stats, verified)
        if not billing:
            continue
        stats["docs_scanned"] += 1
        runs = []
        for i in range(1, (cons.get("runs") or 0) + 1):
            rf = runs_dir / f"{doc}_run{i}.json"
            if rf.exists():
                runs.append(json.loads(rf.read_text()))
        note_text = _note_text_of(result) or next(
            (t for r in runs if (t := _note_text_of(r))), "")

        for d in billing:
            stats["flips_seen"] += 1
            key = _class_key(d)
            array = d.get("array", "")
            codes = d.get("codes") or [d.get("code", "")]
            per_run, descs = [], []
            for r in runs:
                found = [e for c in codes if (e := _entry_of(r, array, c))]
                per_run.append(found[0] if len(found) == 1
                               else (found or None))
                for c in codes:
                    descs += [x.get("description", "")
                              for x in (r.get(array) or [])
                              if x.get("code") == c]
            doc_evidence = {
                "document_id": doc,
                "disagreement": d,
                "per_run_entry": per_run,
                "note_sentences": _descriptor_sentences(
                    note_text, [x for x in descs if x] or codes),
            }
            cls = classes.get(key)
            if cls is None:
                cls = classes[key] = {
                    "queue_version": QUEUE_VERSION,
                    "class_key": key,
                    "kind": d.get("kind", "presence"),
                    "array": array,
                    "code": d.get("code", ""),
                    "status": "open",
                    "documents": [],
                }
                stats["new_classes"] += 1
            docs = {e["document_id"]: e for e in cls["documents"]}
            if doc not in docs:
                stats["updated_classes"] += 1
            # Post-deployment failure detection — the deployed layer's own
            # bug check. A closed class (rule actuated, or baseline declared
            # it resolved) reopens when the same flip surfaces again on
            # evidence the fix should have prevented: a NEW document, or the
            # SAME document reprocessed AFTER the fix went live (result
            # timestamp newer than the status change). Reopening routes it
            # back through actuation instead of letting a dead rule mask a
            # live failure mode.
            if cls["status"] in ("actuated", "resolved_baseline"):
                is_new_doc = doc not in docs
                recurred = is_new_doc or (
                    str(result.get("timestamp") or "")
                    > str(cls.get("status_at") or "9999"))
                if recurred:
                    cls["status"] = "open"
                    cls["reopened"] = {
                        "reason": ("flip recurred on a new document"
                                   if is_new_doc else
                                   "flip persisted on a reprocessed "
                                   "document") + " after actuation — the "
                                  "deployed fix did not hold",
                        "document": doc,
                    }
                    stats["reopened_classes"] = \
                        stats.get("reopened_classes", 0) + 1
            docs[doc] = doc_evidence
            cls["documents"] = [docs[k] for k in sorted(docs)]

    # Audit-dispute classes graduate from awaiting_verification -> open the
    # moment ANY of their documents gains a human/adjudicated registry
    # claim: actuation needs that claim as the realignment target (the
    # runs already AGREE on the disputed content, so 'convergence' for
    # this kind means landing exactly on verified truth). A PER-CODE
    # verified target covering the class's own (array, code) opens it the
    # same way — partial adjudications verify individual codes on notes
    # that cannot verify whole, and the class is exactly that scope.
    code_targets = _verified_code_targets()
    advisory_targets = _verified_advisory_targets()
    for c in classes.values():
        if c.get("kind") != "audit_dispute" \
                or c["status"] != "awaiting_verification":
            continue
        key = (str(c.get("array") or ""), str(c.get("code") or "").upper())
        code = str(c.get("code") or "").upper()
        if any(d["document_id"] in verified
               or key in code_targets.get(d["document_id"], {})
               or any(str(k[1]).rsplit("|", 1)[-1].upper() == code
                      for k in
                      advisory_targets.get(d["document_id"], {}))
               for d in c["documents"]):
            c["status"] = "open"
            stats["opened_after_verification"] = \
                stats.get("opened_after_verification", 0) + 1

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w") as fh:
        for key in sorted(classes):
            fh.write(json.dumps(classes[key], sort_keys=True, default=str)
                     + "\n")
    stats["total_classes"] = len(classes)
    stats["open"] = sum(1 for c in classes.values() if c["status"] == "open")
    return stats


def _scan_audit_disputes(result: dict, doc: str, classes: dict,
                         stats: dict, verified: set[str]) -> None:
    """Clinical-audit disputes are the growth loop's OTHER capture source:
    a consistency flip is a repeatability failure the runs expose, an
    audit dispute is a correctness failure the runs can never expose (a
    wrong deterministic correction is unanimous by construction — measured
    live on routine_00001). Each disputed correction is clustered exactly
    like a flip: same queue, same actuation machinery, same acceptance
    gates. The one difference is the convergence criterion, so the class
    opens only once a human/adjudicated registry claim exists for one of
    its documents — the target every accepted rule's replay must land on
    byte-identically."""
    audit = result.get("clinical_audit") or {}
    findings = [f for f in (audit.get("claim_findings") or [])
                if isinstance(f, dict) and f.get("code")]
    overridden = [c for c in ((result.get("adjudication") or {})
                              .get("overridden_by_replay") or [])
                  if isinstance(c, dict) and c.get("code")]
    if audit.get("verdict") != "disputed" and not findings \
            and not overridden:
        return
    mats = result.get("material_corrections") or []
    note_text = _note_text_of(result)

    def _enqueue(code: str, payload: dict, sentence_seed: str) -> None:
        stats["audit_disputes_seen"] += 1
        array = payload.get("array") or _array_of(code)
        d = {"kind": "audit_dispute", "array": array, "code": code,
             "codes": [code]}
        key = _class_key(d)
        cls = classes.get(key)
        if cls is None:
            cls = classes[key] = {
                "queue_version": QUEUE_VERSION,
                "class_key": key,
                "kind": "audit_dispute",
                "array": array,
                "code": code,
                "status": ("open" if doc in verified
                           else "awaiting_verification"),
                "documents": [],
            }
            stats["new_classes"] += 1
        docs = {e["document_id"]: e for e in cls["documents"]}
        if doc not in docs:
            stats["updated_classes"] += 1
        # Same post-deployment recurrence detection as consistency flips:
        # a closed audit class reopens when the dispute surfaces again.
        if cls["status"] in ("actuated", "resolved_baseline"):
            is_new_doc = doc not in docs
            if is_new_doc or (str(result.get("timestamp") or "")
                              > str(cls.get("status_at") or "9999")):
                cls["status"] = "open"
                cls["reopened"] = {
                    "reason": "audit dispute recurred after actuation — "
                              "the deployed fix did not hold",
                    "document": doc,
                }
                stats["reopened_classes"] = \
                    stats.get("reopened_classes", 0) + 1
        docs[doc] = {
            "document_id": doc,
            "disagreement": dict(payload, kind="audit_dispute",
                                 array=array, codes=[code]),
            "per_run_entry": None,
            "note_sentences": _descriptor_sentences(
                note_text,
                [s for s in (sentence_seed,) if s] or [code]),
        }
        cls["documents"] = [docs[k] for k in sorted(docs)]

    # Capture source 1: disputed correction verdicts (a layer's change
    # the review overturned or could not ground).
    for item in (audit.get("items") or []):
        verdict = str(item.get("verdict") or "").lower()
        if verdict == "uphold":
            continue
        try:
            m = mats[int(item.get("index"))]
        except (TypeError, ValueError, IndexError):
            continue
        code = str(m.get("code") or "")
        if not code:
            continue
        _enqueue(code, {
            "correction": {"category": m.get("category"),
                           "action": m.get("action"),
                           "message": str(m.get("message"))[:400]},
            "audit": {"verdict": verdict,
                      "authority": str(item.get("authority"))[:300],
                      "note_evidence":
                          str(item.get("note_evidence"))[:300]},
        }, str(m.get("message") or ""))

    # Capture source 2: whole-claim review findings — billing-material
    # ones routed the claim to REVIEW; advisory ones (e.g. a system
    # advisory whose recommendation is authoritatively wrong for the fact
    # pattern) change no billing but are confirmed, grounded defects the
    # actuation machinery should still turn into a structural fix. Both
    # enter the same queue and wait for a verified registry claim before
    # any rule is accepted.
    for f in findings:
        _enqueue(str(f["code"]).upper(), {
            "array": f.get("array"),
            "finding": {"kind": f.get("kind"),
                        "materiality": f.get("materiality"),
                        "finding": str(f.get("finding"))[:400]},
            "audit": {"verdict": f.get("materiality"),
                      "authority": str(f.get("authority"))[:300],
                      "note_evidence":
                          str(f.get("note_evidence"))[:300]},
        }, str(f.get("finding") or ""))

    # Capture source 3: layer-vs-adjudicator conflicts — a replay layer
    # overrode an authority-grounded adjudicated decision (measured live
    # on routine_00008: the modifier layer re-added RT after the
    # adjudicator removed it per the CPT descriptor). The layer itself is
    # the defect; the class carries the violated decision so actuation
    # can generalize the layer's missing precondition into a rule.
    for c in overridden:
        _enqueue(str(c["code"]).upper(), {
            "array": c.get("array"),
            "conflict": {"kind": c.get("kind"),
                         "decision": str(c.get("decision"))[:300],
                         "observed": str(c.get("observed"))[:300]},
            "audit": {"verdict": "adjudication_overridden",
                      "authority": str(c.get("authority") or "")[:300],
                      "note_evidence": ""},
        }, str(c.get("decision") or ""))


def set_status(class_key: str, status: str, detail: dict | None = None,
               queue_path: Path = QUEUE_PATH) -> None:
    classes = {}
    for line in queue_path.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            classes[c["class_key"]] = c
    if class_key not in classes:
        raise KeyError(class_key)
    classes[class_key]["status"] = status
    classes[class_key]["status_at"] = datetime.now(timezone.utc).isoformat()
    if detail:
        classes[class_key]["actuation"] = detail
    with open(queue_path, "w") as fh:
        for key in sorted(classes):
            fh.write(json.dumps(classes[key], sort_keys=True, default=str)
                     + "\n")


def load_queue(queue_path: Path = QUEUE_PATH) -> list[dict]:
    if not queue_path.exists():
        return []
    return [json.loads(l) for l in queue_path.read_text().splitlines()
            if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="build/refresh the flip-class queue")
    s.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    sub.add_parser("list", help="print the queue")
    r = sub.add_parser("reopen", help="re-queue class(es) for actuation")
    r.add_argument("class_keys", nargs="+")
    args = p.parse_args()

    if args.cmd == "scan":
        stats = scan(Path(args.results_dir))
        print(f"Flip queue: {stats['total_classes']} class(es) "
              f"({stats['open']} open) from {stats['flips_seen']} flip(s) "
              f"across {stats['docs_scanned']} document(s)")
        return
    if args.cmd == "reopen":
        for key in args.class_keys:
            set_status(key, "open")
            print(f"reopened {key}")
        return

    for c in load_queue():
        docs = ", ".join(e["document_id"][:28] for e in c["documents"])
        print(f"  [{c['status']:9s}] {c['class_key']:40s} docs: {docs}")


if __name__ == "__main__":
    main()
