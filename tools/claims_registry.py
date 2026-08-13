#!/usr/bin/env python3
"""Finalized-claims registry: the durable ledger of verified, billable claims.

The consistency gate decides WHO verifies a claim (unanimous → the pipeline
itself, disagreement → a human coder). This registry is WHERE the verified
claim lives afterward — an append-only event log, one JSON object per line,
human-diffable and bind-mounted on the host so it survives container
rebuilds. Nothing here is ever rewritten or deleted: corrections are new
events, and the "current" view is derived by replaying the log.

Event kinds:
  finalized   a claim reached its final, billable form. verification is
              "auto" (unanimous consistency + CLEAN disposition, recorded by
              the pipeline itself), "adjudicated" (the automated expert-coder
              adjudicator settled a judgment-shaped disagreement — grounded
              in authoritative sources, unanimous across independent passes,
              replay-verified; see tools/coder_adjudicator.py), or "human"
              (a coder's corrected claim). Precedence for the same note:
              human > adjudicated > auto — a lower tier never displaces a
              higher one.
  outcome     payer adjudication for a finalized claim (paid / denied /
              partial, with CARCs). Complements the denial feedback loop:
              denials still flow through tools/denial_feedback.py for
              gap classification; the outcome event ties the adjudication
              to the exact claim version that was submitted.
  adjudicated_codes
              per-code verified targets from a PARTIAL adjudication: the
              expert-coder adjudicator ruled (unanimous across independent
              passes, authority-grounded, mechanically realized) on specific
              codes of a note that cannot be recorded whole — a consistency
              holdout split on OTHER codes, or a claim a replay layer keeps
              overriding. Each target names an (array, code) and the exact
              claim row the verdict mandates there (or null = the code must
              be absent). Scope is the event's whole point: it verifies
              THOSE CODES ONLY, never the note's full claim — actuation uses
              it as the realignment goal for the matching audit-dispute
              class, and nothing else reads it. A full-claim finalized
              record at human/adjudicated tier for the same note supersedes
              its per-code targets entirely.
  adjudicated_advisories
              verified ADVISORY-EMISSION targets from an audit-dispute
              adjudication: the expert coder ruled (unanimous across
              independent passes, authority-grounded) on whether a
              specific compliance-scrubber advisory — a WARN finding,
              identified by (filter_id, code) — should fire on this
              note's claim. emit=false means the advisory is wrong for
              the documented fact pattern and a deterministic fix must
              stop it firing; emit=true means the advisory stands (the
              review's allegation was rejected). Claim lines are NOT
              verified by this event — its whole point is disputes where
              the claim is correct as billed and only the advisory is
              wrong, which billing-signature realignment can never
              measure. Actuation's replay gate uses it as the emission-
              state realignment goal for the matching audit-dispute
              class. Unlike per-code claim targets, a full-claim
              finalized record does NOT supersede it: finalized events
              freeze claim lines, not advisory emissions.
  adjudicated_observables
              the GENERIC form of adjudicated_advisories, one event kind
              for every measurement observable (tools/observables.py) —
              advisory emission was merely the first observable, and the
              measurement vocabulary now grows autonomously (synthesized
              observables), so its verified targets must not require a
              new registry event kind per observable. Each target names
              an {observable, key, emit}: observable is the measurement
              namespace (e.g. "advisory_emission"), key the phenomenon's
              deterministic identity within it (ends with "|<CODE>" by
              contract), emit the adjudicated correct state. Readers use
              verified_observable_targets(), which merges legacy
              adjudicated_advisories events into the same view. Claim
              lines are NOT verified; full-claim records do NOT
              supersede.

Commands:
  ingest  [results_dir]     scan pipeline results and auto-record every
                            eligible claim (success + consistency-unanimous
                            + CLEAN). Idempotent: unchanged claims are
                            skipped; a changed claim appends a new event.
                            Never overrides a human record.
  record  <result_file> --by <coder>
                            record a human-verified claim from a (corrected)
                            result file. Always appends; wins over auto.
  outcome <document_id> --status paid|denied|partial [--carc C ...]
                            attach payer adjudication to the current claim.
  list                      print the current view (latest claim per note,
                            with verification level and outcome).
  export-gold [gold_dir]    materialize the current view as minimal
                            *_results.json files scoreable by
                            tools/benchmark_ab.py — gold re-freezes draw
                            from verified truth instead of ad-hoc snapshots.

Registry file: data/registry/claims_registry.jsonl (append-only JSONL).

TWO PRODUCER SHAPES (issue #6, F6-R4-A1)
----------------------------------------
The deployed entrypoint now writes one canonical `ClaimBundle`
(`app/contracts/claim_bundle.py`). This module reads that contract directly:
`extract_claim`, `eligible_for_auto` and `make_finalized_event` all dispatch on
the artifact's DECLARED schema, never on which keys happen to be present.

  bundle artifact  -> the bundle's own claim content is recorded verbatim, and
                      eligibility is `verify_bundle_readiness()`.
  legacy artifact  -> the pre-existing `app.pipeline` battery, untouched.

The dispatch is on `schema_id`, so a bundle that fails validation raises rather
than falling through to the legacy reader — the failure mode this finding was
about was precisely a new artifact being read under old rules and silently
producing an empty claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY_PATH = ROOT / "data" / "registry" / "claims_registry.jsonl"
DEFAULT_RESULTS = ROOT / "output" / "results"
DEFAULT_GOLD = ROOT / "benchmark" / "gold"

#: 1 = legacy `app.pipeline` claim arrays. 2 = canonical ClaimBundle. Both
#: remain readable; only 2 is written for new artifacts.
REGISTRY_VERSION = 1
BUNDLE_REGISTRY_VERSION = 2

# Verification precedence: a finalized event only displaces the current
# view when its tier is at least as high. Unknown tiers rank lowest.
_VERIFICATION_RANK = {"auto": 0, "adjudicated": 1, "human": 2}


def _rank(verification: str) -> int:
    return _VERIFICATION_RANK.get(str(verification or "").lower(), -1)

# The fields that make a claim billable — exactly what benchmark_ab.py
# scores and what a CMS-1500 carries. Everything else in a result
# (rationale, evidence spans, audit trails) stays in output/results.
_ICD_FIELDS = ("code", "type", "description", "laterality")
_LINE_FIELDS = ("code", "modifiers", "units", "dx_pointers",
                "diagnosis_pointers", "linked_diagnoses", "description",
                "laterality", "place_of_service", "ndc")


# --------------------------------------------------------------------------
# storage — append-only JSONL
# --------------------------------------------------------------------------

def load_events(path: Path = REGISTRY_PATH) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # RuntimeError, not sys.exit: SystemExit is a BaseException and
            # would sail through run.py's `except Exception` guard, killing
            # a whole batch at the very end over one bad line.
            raise RuntimeError(
                f"Corrupt registry line {i} in {path}: {exc} — the registry "
                f"is append-only; restore the line or remove it after "
                f"investigating, never rewrite history") from None
    return events


def append_events(new: list[dict], path: Path = REGISTRY_PATH) -> None:
    if not new:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for e in new:
            f.write(json.dumps(e, sort_keys=True, default=str) + "\n")


# --------------------------------------------------------------------------
# claim extraction / normalization
# --------------------------------------------------------------------------

def _slim(entry: dict, fields: tuple) -> dict:
    out = {}
    for k in fields:
        v = entry.get(k)
        if v not in (None, "", []):
            out[k] = v
    return out


def is_bundle_artifact(result: dict) -> bool:
    """Does this artifact declare the canonical claim contract?"""
    from app.contracts.claim_bundle import is_claim_bundle
    return is_claim_bundle(result)


def as_bundle(result: dict):
    """Parse a canonical artifact, or raise. Never returns a legacy artifact."""
    from app.contracts.claim_bundle import load_bundle
    return load_bundle(result)


def document_id_of(result: dict) -> str:
    """The note identity, from whichever contract the artifact declares.

    A bundle keeps it under `encounter.document_id`; reading the legacy
    top-level `document_id` off a bundle would return "" and every bundle would
    be filed under its filename instead of its encounter.
    """
    if is_bundle_artifact(result):
        encounter = result.get("encounter") or {}
        return str(encounter.get("document_id")
                   or encounter.get("encounter_id") or "")
    return str(result.get("document_id") or "")


def extract_claim(result: dict) -> dict:
    """The billable payload of a result.

    For a canonical `ClaimBundle` this is the contract's own `claim_content()`
    — ordered diagnoses, service lines with units/modifiers/diagnosis pointers,
    the source-document identity and the encounter-context fingerprint. Defined
    by the contract, not restated here, so the registry cannot develop its own
    opinion about what is claim-affecting (which is how the old slimming lists
    silently dropped everything a bundle carries).

    For a legacy `app.pipeline` result it is the historical projection: the
    three code arrays reduced to claim-relevant fields plus disposition.
    (Suppressed lines never appear there — the validator removes them from the
    arrays before saving.)
    """
    if is_bundle_artifact(result):
        return as_bundle(result).claim_content()

    def _lines(key: str, fields: tuple) -> list[dict]:
        out = []
        for e in result.get(key) or []:
            if not isinstance(e, dict) or not e.get("code"):
                continue
            out.append(_slim(e, fields))
        return out

    return {
        "icd_codes": _lines("icd_codes", _ICD_FIELDS),
        "cpt_codes": _lines("cpt_codes", _LINE_FIELDS),
        "hcpcs_codes": _lines("hcpcs_codes", _LINE_FIELDS),
        "final_disposition": result.get("final_disposition") or "",
    }


def _claim_key(claim: dict) -> str:
    """Canonical fingerprint for change detection (idempotent ingest)."""
    return json.dumps(claim, sort_keys=True, default=str)


def _verification_key(event: dict) -> str:
    """What must be identical for a re-ingest to be a no-op.

    For a legacy event this is the claim arrays, as it always was.

    For a canonical event it is the claim AND the certificate identity AND the
    context fingerprint. The certificate is part of the key deliberately: a
    re-coded note produces a byte-identical claim but a NEW certificate (it
    binds that run's durable audit records), and the submitter requires the
    registry and the live artifact to carry the SAME certificate. Keying on the
    claim alone would call the re-ingest "unchanged", leave the old certificate
    recorded, and deadlock the note — permanently unsubmittable, with no event
    a re-ingest could ever add. Re-attesting a claim is a new verification, and
    the append-only log should say so.
    """
    if event.get("registry_version") == BUNDLE_REGISTRY_VERSION:
        return _claim_key({
            "claim": event.get("claim") or {},
            "certificate_sha256": event.get("certificate_sha256") or "",
            "encounter_context_fingerprint": event.get(
                "encounter_context_fingerprint") or "",
        })
    return _claim_key(event.get("claim") or {})


def _payer_of(result: dict) -> str:
    if is_bundle_artifact(result):
        payer = as_bundle(result).context.payer
        return payer.name or payer.payer_id or ""
    meta = result.get("patient_metadata") or {}
    for k in ("insurance", "payer", "insurance_provider", "insurance_payer"):
        v = meta.get(k)
        if v:
            return str(v)
    return ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_fingerprint(result: dict) -> dict:
    """Compact clinical fingerprint of the encounter behind a claim — what
    exemplar retrieval (app/coding/exemplars.py) matches new notes against.
    Only note-level facts, never patient identity."""
    if is_bundle_artifact(result):
        bundle = as_bundle(result)
        # A bundle carries no note SECTIONS (the canonical contract binds
        # evidence spans, not the retired pipeline's section split), so the
        # section-derived fields stay empty rather than being approximated from
        # a rationale string. The procedure descriptors are real authoritative
        # descriptors and are the part exemplar retrieval actually matches on.
        return {
            "note_category": "",
            "chief_complaint": "",
            "assessment": "",
            "procedures": [line.descriptor for line in bundle.service_lines
                           if line.descriptor],
        }
    sections = result.get("note_sections") or {}
    vc = (result.get("rag_context") or {}).get("vision_context") or {}
    return {
        "note_category": vc.get("note_category") or "",
        "chief_complaint": (sections.get("chief_complaint") or "")[:200],
        "assessment": (sections.get("assessment_diagnoses") or "")[:300],
        "procedures": list(vc.get("procedures_performed_today") or []),
    }


def make_finalized_event(document_id: str, result: dict, verification: str,
                         verified_by: str, source: str) -> dict:
    if is_bundle_artifact(result):
        return _make_bundle_event(document_id, as_bundle(result), verification,
                                  verified_by, source)
    from app.release.claim_readiness import encounter_context_fingerprint
    return {
        "registry_version": REGISTRY_VERSION,
        "event": "finalized",
        "document_id": document_id,
        "recorded_at": _now(),
        "verification": verification,       # "auto" | "human"
        "verified_by": verified_by,
        "source": source,                   # result filename / batch id
        "payer": _payer_of(result),
        "consistency": {
            "runs": (result.get("consistency") or {}).get("runs"),
            "unanimous": (result.get("consistency") or {}).get("unanimous"),
        },
        "fingerprint": make_fingerprint(result),
        "claim": extract_claim(result),
        "encounter_context_fingerprint": encounter_context_fingerprint(result),
        "claim_readiness_certificate": (
            result.get("claim_readiness_certificate") or {}),
    }


def _make_bundle_event(document_id: str, bundle, verification: str,
                       verified_by: str, source: str) -> dict:
    """A finalized event recording one canonical `ClaimBundle`.

    The WHOLE bundle is recorded, not a projection of it. The registry is the
    submitter's sole source of billable content, so anything the bundle carries
    that the event drops is content the 837P can never see — the mechanism by
    which diagnosis and service lines disappeared in the first place.
    `claim` is the contract's own `claim_content()`, retained as the
    change-detection key (`_claim_key`) that ingest idempotence and the
    submitter's ledger both hash.
    """
    from app.release.claim_readiness import bundle_encounter_context_fingerprint
    certificate = bundle.certificate
    return {
        "registry_version": BUNDLE_REGISTRY_VERSION,
        "event": "finalized",
        "document_id": document_id,
        "recorded_at": _now(),
        "verification": verification,
        "verified_by": verified_by,
        "source": source,
        "payer": bundle.context.payer.name or bundle.context.payer.payer_id or "",
        "fingerprint": make_fingerprint(bundle.to_payload()),
        "claim": bundle.claim_content(),
        "claim_bundle": bundle.to_payload(),
        "encounter_context_fingerprint": bundle_encounter_context_fingerprint(bundle),
        "certificate_sha256": (certificate.certificate_sha256
                               if certificate else ""),
    }


def bundle_of_event(event: dict):
    """The `ClaimBundle` a registry event recorded, or None for a legacy event.

    Returns None ONLY when the event carries no bundle at all. A bundle that is
    present but unreadable raises: a corrupt canonical event must not read as
    "this is a legacy event", which would silently authorize it under the wrong
    battery.
    """
    payload = event.get("claim_bundle")
    if payload is None:
        return None
    return as_bundle(payload)


# --------------------------------------------------------------------------
# current view — replay the log
# --------------------------------------------------------------------------

def current_view(events: list[dict]) -> dict[str, dict]:
    """Latest finalized claim per document_id, honoring verification
    precedence (human > adjudicated > auto) regardless of order — an
    automated re-ingest must never displace a coder's (or adjudicator's)
    verdict with a lower-tier record. Latest outcome attached."""
    view: dict[str, dict] = {}
    for e in events:
        doc = str(e.get("document_id", ""))
        if not doc or e.get("event") != "finalized":
            continue
        cur = view.get(doc)
        if cur and _rank(e.get("verification")) < _rank(cur["verification"]):
            continue
        view[doc] = dict(e)
    for e in events:
        doc = str(e.get("document_id", ""))
        if doc in view and e.get("event") == "outcome":
            view[doc]["outcome"] = {k: e[k] for k in
                                    ("status", "carcs", "recorded_at", "note")
                                    if k in e}
    return view


# --------------------------------------------------------------------------
# ingest — auto-record eligible pipeline results
# --------------------------------------------------------------------------

def eligible_for_auto(result: dict) -> tuple[bool, str]:
    """The auto-submission bar for whichever contract this artifact declares.

    CANONICAL `ClaimBundle`: the bar is `verify_bundle_readiness()`, which
    requires an AUTO_READY destination with a certificate that self-addresses
    and attests THIS encounter's codes, a reproducing claim fingerprint, an
    encounter context that an authoritative provider RESOLVED (complete and
    conflict-free), every service line linked to a diagnosis that exists, and a
    bound authoritative-data/source-manifest identity.

    It deliberately does NOT require the legacy battery's N-run consistency,
    compliance scrub, mutation ledger or clinical-audit fields. Those are
    artifacts of the retired `app.pipeline` and the coder that produces bundles
    has none of them: it replaces repeatability-by-re-running with
    eligibility-before-retrieval, propose-then-verify against authoritative
    descriptors, and independent cross-model corroboration, all of which must
    already have cleared inside the pipeline for an AUTO_READY verdict to exist
    at all, and all of which the certificate binds. Demanding the legacy fields
    of a bundle would not make it safer — it would make every bundle
    permanently ineligible for a reason unrelated to its correctness, which is
    the bug being fixed, not a stricter version of it.

    LEGACY `app.pipeline` result: unchanged below — the pipeline
    succeeded, every billing array was unanimous across the consistency
    runs (>= 2 — a single run proves nothing about repeatability), the
    scrub disposition is CLEAN, and the clinical-correctness review — a
    whole-claim expert review plus a verdict on every interpretive layer
    correction — upheld the claim, judged against its CURRENT corrections
    and claim shape. Unanimity measures repeatability, never correctness:
    a wrong deterministic correction is unanimous by construction, so the
    review is the gate that can catch it, and it is required for EVERY
    claim (the absence of recorded corrections is exactly what an
    unreported mutation looks like — measured live, routine_00003)."""
    if is_bundle_artifact(result):
        from app.release.claim_readiness import verify_bundle_artifact
        return verify_bundle_artifact(result)
    if not result.get("success"):
        return False, "pipeline did not succeed"
    cons = result.get("consistency") or {}
    if not cons:
        return False, "no consistency report (single run)"
    if (cons.get("runs") or 0) < 2:
        return False, "fewer than 2 consistency runs"
    if not cons.get("unanimous"):
        return False, "billing disagreement across runs — awaiting human review"
    if (result.get("final_disposition") or "").upper() != "CLEAN":
        return False, f"disposition {result.get('final_disposition')} != CLEAN"
    if (result.get("adjudication") or {}).get("overridden_by_replay"):
        return False, ("a deterministic layer overrode the adjudicator's "
                       "verdict on replay — layer-vs-adjudicator conflict, "
                       "human decision required")
    audit = result.get("clinical_audit") or {}
    if not audit:
        return False, ("claim not yet clinically reviewed "
                       "(tools/clinical_auditor.py)")
    if audit.get("verdict") != "upheld":
        return False, ("clinical review disputed the claim — "
                       "awaiting human review")
    try:
        from tools.clinical_auditor import corrections_fingerprint
        if audit.get("fingerprint") != corrections_fingerprint(result):
            return False, ("clinical review verdict is stale — the claim "
                           "changed since it was reviewed")
    except Exception:
        return False, "clinical review fingerprint could not be verified"
    # Last, the whole-record contradiction gate: the checks above trust
    # individual fields; this one requires the fields to agree with EACH
    # OTHER (adjudication realized on the claim, correction ledger vs
    # claim state, linkage integrity, one first-listed diagnosis, ...).
    try:
        from tools.record_coherence import coherence_violations
        violations = coherence_violations(result)
    except Exception as exc:  # fail closed: unverifiable ≠ coherent
        violations = [f"coherence gate could not run ({exc})"]
    if violations:
        return False, ("record contradicts itself: " + "; ".join(violations))
    try:
        from app.release.claim_readiness import verify_readiness_certificate
        ready, reason = verify_readiness_certificate(result)
    except Exception as exc:
        return False, f"claim readiness authorization could not run ({exc})"
    if not ready:
        return False, f"claim readiness authorization failed: {reason}"
    return True, ""


def ingest(results_dir: Path, registry_path: Path = REGISTRY_PATH) -> dict:
    events = load_events(registry_path)
    view = current_view(events)
    new_events: list[dict] = []
    stats = {"recorded": 0, "unchanged": 0, "skipped": 0,
             "human_protected": 0, "skip_reasons": {}}

    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        result = json.loads(f.read_text())
        doc = str(document_id_of(result) or f.stem.removesuffix("_results"))

        try:
            ok, why = eligible_for_auto(result)
        except Exception as exc:
            # A malformed artifact is ONE note's problem, recorded with its
            # reason — never a crash that silently ends the scan and leaves the
            # remaining notes un-ingested.
            ok, why = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            stats["skipped"] += 1
            stats["skip_reasons"][doc] = why
            continue

        cur = view.get(doc)
        if cur and _rank(cur["verification"]) > _rank("auto"):
            # A coder's (or the adjudicator's) verdict is final for this
            # note; the pipeline's auto ingest never displaces it.
            # Re-verification is an explicit `record` at the higher tier.
            stats["human_protected"] += 1
            continue

        ev = make_finalized_event(doc, result, verification="auto",
                                  verified_by="pipeline/consistency-gate",
                                  source=f.name)
        if cur and _verification_key(cur) == _verification_key(ev):
            stats["unchanged"] += 1
            continue
        new_events.append(ev)
        view[doc] = ev
        stats["recorded"] += 1

    append_events(new_events, registry_path)
    return stats


# --------------------------------------------------------------------------
# record_adjudicated — the automated expert coder's verdict
# --------------------------------------------------------------------------

def record_adjudicated(document_id: str, result: dict, source: str,
                       by: str, registry_path: Path = REGISTRY_PATH) -> dict | None:
    """Record a claim settled by the automated expert-coder adjudicator
    (tools/coder_adjudicator.py). Outranks auto, never displaces a human
    record (precedence enforced here AND in current_view). Idempotent on
    unchanged claims. Returns the appended event, or None if skipped."""
    view = current_view(load_events(registry_path))
    cur = view.get(document_id)
    if cur and _rank(cur["verification"]) > _rank("adjudicated"):
        return None
    ev = make_finalized_event(document_id, result,
                              verification="adjudicated",
                              verified_by=by, source=source)
    adj = result.get("adjudication") or {}
    ev["adjudication"] = {k: adj.get(k) for k in
                          ("at", "model", "passes", "protocol") if k in adj}
    if cur and cur["verification"] == "adjudicated" \
            and _verification_key(cur) == _verification_key(ev):
        return None
    append_events([ev], registry_path)
    return ev


# --------------------------------------------------------------------------
# adjudicated_codes — per-code verified targets from partial adjudications
# --------------------------------------------------------------------------

def record_adjudicated_codes(document_id: str, targets: list[dict],
                             source: str, by: str,
                             adjudication: dict | None = None,
                             registry_path: Path = REGISTRY_PATH
                             ) -> dict | None:
    """Record per-code verified targets from a partial adjudication.

    `targets` is a list of {"array", "code", "row"} where row is the exact
    claim row the adjudicated verdict mandates ({"code", "modifiers",
    "units"[, "type"]}) or None when the verdict is that the code must be
    ABSENT. Never displaces a full-claim human/adjudicated record (that
    record already covers every code); idempotent when the latest
    per-code event for this note carries identical targets."""
    if not targets:
        return None
    events = load_events(registry_path)
    cur = current_view(events).get(document_id)
    if cur and _rank(cur["verification"]) >= _rank("adjudicated"):
        return None
    last = None
    for e in events:
        if e.get("event") == "adjudicated_codes" \
                and str(e.get("document_id")) == document_id:
            last = e
    key = json.dumps(targets, sort_keys=True, default=str)
    if last and json.dumps(last.get("targets"), sort_keys=True,
                           default=str) == key:
        return None
    ev = {
        "registry_version": REGISTRY_VERSION,
        "event": "adjudicated_codes",
        "document_id": document_id,
        "recorded_at": _now(),
        "verification": "adjudicated",
        "verified_by": by,
        "source": source,
        "targets": targets,
    }
    if adjudication:
        ev["adjudication"] = {k: adjudication.get(k) for k in
                              ("at", "model", "passes") if k in adjudication}
    append_events([ev], registry_path)
    return ev


def record_adjudicated_advisories(document_id: str, targets: list[dict],
                                  source: str, by: str,
                                  adjudication: dict | None = None,
                                  registry_path: Path = REGISTRY_PATH
                                  ) -> dict | None:
    """Record verified advisory-emission targets from an audit-dispute
    adjudication.

    `targets` is a list of {"filter_id", "code", "emit", "authority"}:
    filter_id names the scrubber filter whose WARN finding is disputed,
    code the claim line it fires on, emit the adjudicated correct state
    (False = must not fire for this documented fact pattern, True = the
    advisory stands). Idempotent when the latest advisory event for this
    note carries identical targets. Never superseded by full-claim
    records — those verify claim lines, not advisory emissions."""
    targets = [t for t in (targets or [])
               if isinstance(t, dict) and t.get("filter_id")
               and t.get("code") and isinstance(t.get("emit"), bool)]
    if not targets:
        return None
    events = load_events(registry_path)
    last = None
    for e in events:
        if e.get("event") == "adjudicated_advisories" \
                and str(e.get("document_id")) == document_id:
            last = e
    key = json.dumps(targets, sort_keys=True, default=str)
    if last and json.dumps(last.get("targets"), sort_keys=True,
                           default=str) == key:
        return None
    ev = {
        "registry_version": REGISTRY_VERSION,
        "event": "adjudicated_advisories",
        "document_id": document_id,
        "recorded_at": _now(),
        "verification": "adjudicated",
        "verified_by": by,
        "source": source,
        "targets": targets,
    }
    if adjudication:
        ev["adjudication"] = {k: adjudication.get(k) for k in
                              ("at", "model", "passes") if k in adjudication}
    append_events([ev], registry_path)
    return ev


def verified_advisory_targets(events: list[dict] | None = None,
                              registry_path: Path = REGISTRY_PATH
                              ) -> dict[str, dict[tuple, bool]]:
    """{document_id: {(FILTER_ID, CODE): emit}} — the advisory_emission
    slice of verified_observable_targets, kept for callers that speak
    the advisory-specific vocabulary."""
    out: dict[str, dict[tuple, bool]] = {}
    for doc, m in verified_observable_targets(events, registry_path).items():
        for (obs, key), emit in m.items():
            if obs != "advisory_emission" or "|" not in key:
                continue
            fid, code = key.rsplit("|", 1)
            out.setdefault(doc, {})[(fid, code)] = emit
    return out


def record_adjudicated_observables(document_id: str, targets: list[dict],
                                   source: str, by: str,
                                   adjudication: dict | None = None,
                                   registry_path: Path = REGISTRY_PATH
                                   ) -> dict | None:
    """Record verified observable-emission targets from an audit-dispute
    adjudication — the generic successor of record_adjudicated_advisories.

    `targets` is a list of {"observable", "key", "emit", "authority"}:
    observable names the measurement namespace (tools/observables.py),
    key the phenomenon's deterministic identity within it, emit the
    adjudicated correct state (False = must not fire for this documented
    fact pattern, True = it stands). Idempotent when the latest
    observable event for this note carries identical targets. Never
    superseded by full-claim records — those verify claim lines, not
    measured emissions."""
    targets = [t for t in (targets or [])
               if isinstance(t, dict) and t.get("observable")
               and t.get("key") and isinstance(t.get("emit"), bool)]
    if not targets:
        return None
    events = load_events(registry_path)
    last = None
    for e in events:
        if e.get("event") == "adjudicated_observables" \
                and str(e.get("document_id")) == document_id:
            last = e
    key = json.dumps(targets, sort_keys=True, default=str)
    if last and json.dumps(last.get("targets"), sort_keys=True,
                           default=str) == key:
        return None
    ev = {
        "registry_version": REGISTRY_VERSION,
        "event": "adjudicated_observables",
        "document_id": document_id,
        "recorded_at": _now(),
        "verification": "adjudicated",
        "verified_by": by,
        "source": source,
        "targets": targets,
    }
    if adjudication:
        ev["adjudication"] = {k: adjudication.get(k) for k in
                              ("at", "model", "passes") if k in adjudication}
    append_events([ev], registry_path)
    return ev


def _anchorable(t: dict) -> bool:
    """Whether a recorded target may anchor actuation (rule minting,
    class verification, replay realignment goals).

    Attestation tiers are stamped at RECORDING time by
    tools/policy_corpus.attest():
      document_quoted  verdict's policy quote verified against a stored
                       source -> anchorable
      data_backed      verdict declared derivable from the case file's
                       own reference data (no prose quote applies; the
                       deterministic gates re-verify against the data)
                       -> anchorable
      unverified       no policy corpus existed when recorded (nothing
                       could be looked up) -> anchorable, grandfathered
      attested_only    the corpus WAS available and the prose-policy
                       verdict still carried no verifiable quote -> the
                       weakest grounding; recorded for audit visibility
                       but NEVER anchorable
    Targets recorded before the field existed carry no tier and stay
    anchorable (same grandfathering as 'unverified')."""
    return str(t.get("attestation") or "") != "attested_only"


def verified_observable_targets(events: list[dict] | None = None,
                                registry_path: Path = REGISTRY_PATH
                                ) -> dict[str, dict[tuple, bool]]:
    """{document_id: {(OBSERVABLE, KEY): emit}} — the emission-state
    realignment goals for observable-shaped audit-dispute actuation, from
    adjudicated_observables events PLUS legacy adjudicated_advisories
    events (read as observable "advisory_emission", key "FILTER|CODE").
    Latest event wins per (observable, key). attested_only targets are
    excluded (see _anchorable)."""
    events = load_events(registry_path) if events is None else events
    out: dict[str, dict[tuple, bool]] = {}
    for e in events:
        doc = str(e.get("document_id") or "")
        if not doc:
            continue
        if e.get("event") == "adjudicated_observables":
            for t in (e.get("targets") or []):
                if not isinstance(t, dict) or not t.get("observable") \
                        or not t.get("key") \
                        or not isinstance(t.get("emit"), bool):
                    continue
                key = (str(t["observable"]).strip(),
                       str(t["key"]).strip())
                if not _anchorable(t):
                    # latest event wins: an attested_only re-adjudication
                    # RETIRES any earlier anchorable value for this key
                    # rather than letting the stale one keep anchoring
                    out.setdefault(doc, {}).pop(key, None)
                    continue
                out.setdefault(doc, {})[key] = bool(t["emit"])
        elif e.get("event") == "adjudicated_advisories":
            for t in (e.get("targets") or []):
                if not isinstance(t, dict) or not t.get("filter_id") \
                        or not t.get("code") \
                        or not isinstance(t.get("emit"), bool):
                    continue
                key = ("advisory_emission",
                       f"{str(t['filter_id']).strip()}"
                       f"|{str(t['code']).strip().upper()}")
                out.setdefault(doc, {})[key] = bool(t["emit"])
    return {doc: m for doc, m in out.items() if m}


def verified_code_targets(events: list[dict] | None = None,
                          registry_path: Path = REGISTRY_PATH
                          ) -> dict[str, dict[tuple, dict | None]]:
    """{document_id: {(array, CODE): row-or-None}} from adjudicated_codes
    events — the per-code realignment goals for audit-dispute actuation.
    Latest event wins per (array, code). Documents with a full-claim
    record at human/adjudicated tier are excluded entirely: the full
    claim is the (stronger) target there, and two sources of truth for
    one note would be one too many. attested_only targets are excluded
    (see _anchorable)."""
    events = load_events(registry_path) if events is None else events
    full = {doc for doc, e in current_view(events).items()
            if _rank(e.get("verification")) >= _rank("adjudicated")}
    out: dict[str, dict[tuple, dict | None]] = {}
    for e in events:
        if e.get("event") != "adjudicated_codes":
            continue
        doc = str(e.get("document_id") or "")
        if not doc or doc in full:
            continue
        for t in (e.get("targets") or []):
            if not isinstance(t, dict) or not t.get("code"):
                continue
            key = (str(t.get("array") or ""), str(t["code"]).upper())
            if not _anchorable(t):
                out.setdefault(doc, {}).pop(key, None)
                continue
            out.setdefault(doc, {})[key] = t.get("row")
    return {doc: m for doc, m in out.items() if m}


# --------------------------------------------------------------------------
# export-gold — registry is the source of truth for benchmark re-freezes
# --------------------------------------------------------------------------

def export_gold(gold_dir: Path, registry_path: Path = REGISTRY_PATH) -> int:
    view = current_view(load_events(registry_path))
    gold_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for doc, e in sorted(view.items()):
        review = e.get("gold_review") or {}
        # A pipeline or adjudicator answer must never promote itself into
        # benchmark truth.  Export requires two distinct human attestations.
        if e.get("verification") != "human" or not review.get("approved") \
                or not review.get("reviewed_by") \
                or review.get("reviewed_by") == e.get("verified_by"):
            continue
        if e.get("registry_version") == BUNDLE_REGISTRY_VERSION:
            # `tools/benchmark_ab.py` scores the legacy code-array shape. A
            # bundle claim is NOT that shape, and emitting it under the same
            # filename would give the scorer a file it reads as a claim with
            # zero codes — a silent zero, which is worse than an absent file.
            # Skipped loudly until the benchmark speaks the contract.
            print(f"  [skip] {doc}: canonical ClaimBundle claims are not yet "
                  f"scoreable by the benchmark gold format")
            continue
        payload = {
            "document_id": doc,
            **e["claim"],
            "gold_provenance": {
                "verification": e["verification"],
                "verified_by": e["verified_by"],
                "recorded_at": e["recorded_at"],
                "independent_review": review,
            },
        }
        (gold_dir / f"{doc}_results.json").write_text(
            json.dumps(payload, indent=2, default=str))
        n += 1
    return n


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="auto-record eligible pipeline results")
    s.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))

    s = sub.add_parser("record", help="record a human-verified claim")
    s.add_argument("result_file")
    s.add_argument("--by", required=True,
                   help="coder identity (name/initials) — the audit trail")
    s.add_argument("--note", default="", help="optional review note")
    s.add_argument("--gold-approved", action="store_true",
                   help="approve this claim for benchmark export")
    s.add_argument("--independent-reviewer", default="",
                   help="second reviewer; must differ from --by")

    s = sub.add_parser("outcome", help="attach payer adjudication")
    s.add_argument("document_id")
    s.add_argument("--status", required=True,
                   choices=("paid", "denied", "partial"))
    s.add_argument("--carc", action="append", default=[],
                   help="CARC code(s) for denied/partial (repeatable)")
    s.add_argument("--note", default="")

    s = sub.add_parser("list", help="print the current view")

    s = sub.add_parser("export-gold",
                       help="materialize the current view for benchmark_ab")
    s.add_argument("gold_dir", nargs="?", default=str(DEFAULT_GOLD))

    args = p.parse_args()

    if args.cmd == "ingest":
        stats = ingest(Path(args.results_dir))
        print(f"Ingest complete: {stats['recorded']} recorded, "
              f"{stats['unchanged']} unchanged, "
              f"{stats['human_protected']} human-protected, "
              f"{stats['skipped']} not eligible")
        for doc, why in sorted(stats["skip_reasons"].items()):
            print(f"  [skip] {doc}: {why}")
        return

    if args.cmd == "record":
        result = json.loads(Path(args.result_file).read_text())
        doc = str(document_id_of(result)
                  or Path(args.result_file).stem.removesuffix("_results"))
        ev = make_finalized_event(doc, result, verification="human",
                                  verified_by=args.by,
                                  source=Path(args.result_file).name)
        if args.note:
            ev["note"] = args.note
        if args.gold_approved:
            if not args.independent_reviewer or \
                    args.independent_reviewer == args.by:
                sys.exit("--gold-approved requires a distinct "
                         "--independent-reviewer")
            ev["gold_review"] = {
                "approved": True,
                "reviewed_by": args.independent_reviewer,
                "reviewed_at": _now(),
            }
        append_events([ev])
        print(f"Recorded human-verified claim for {doc} (by {args.by})")
        return

    if args.cmd == "outcome":
        view = current_view(load_events())
        if args.document_id not in view:
            sys.exit(f"No finalized claim for '{args.document_id}' — "
                     f"outcomes attach to finalized claims only")
        ev = {
            "registry_version": REGISTRY_VERSION,
            "event": "outcome",
            "document_id": args.document_id,
            "recorded_at": _now(),
            "status": args.status,
            "carcs": args.carc,
        }
        if args.note:
            ev["note"] = args.note
        append_events([ev])
        print(f"Recorded {args.status} outcome for {args.document_id}"
              + (f" (CARCs: {', '.join(args.carc)})" if args.carc else ""))
        return

    if args.cmd == "list":
        view = current_view(load_events())
        if not view:
            print("Registry is empty — run 'ingest' after a batch, or "
                  "'record' a human-verified claim.")
            return
        for doc, e in sorted(view.items()):
            c = e["claim"]
            outcome = e.get("outcome", {}).get("status", "-")
            if e.get("registry_version") == BUNDLE_REGISTRY_VERSION:
                shape = (f"dx={len(c.get('diagnoses') or [])} "
                         f"svc={len(c.get('service_lines') or [])} "
                         f"contract=claim_bundle")
            else:
                shape = (f"icd={len(c['icd_codes'])} cpt={len(c['cpt_codes'])} "
                         f"hcpcs={len(c['hcpcs_codes'])} "
                         f"disp={c['final_disposition']}")
            print(f"  {doc:45s} {e['verification']:11s} "
                  f"by {e['verified_by'][:28]:28s} "
                  f"{shape} outcome={outcome}")
        print(f"\n{len(view)} finalized claim(s)")
        return

    if args.cmd == "export-gold":
        n = export_gold(Path(args.gold_dir))
        print(f"Exported {n} verified claim(s) -> {args.gold_dir}")


if __name__ == "__main__":
    main()
