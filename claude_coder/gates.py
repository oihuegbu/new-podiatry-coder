"""Stage 4 — Positive release gates (fail-CLOSED).

Every gate is a POSITIVE assertion about the claim. Release is earned only when
each gate returns PASS or a proven NOT_APPLICABLE. UNKNOWN, ERROR and BLOCKED
all stop autonomy — a check that could not run is never treated as success.
This is the opposite of "clean = no failure findings": here, silence is not
consent.

Every gate answers from the authoritative source (activity window, NCCI, MUE) or
from the note itself (evidence), never from a code list baked into this file.
"""
from __future__ import annotations

import json
import re

from .data_access import AUTHORITY_UNAVAILABLE, CodeSource
from .models import (CodingResult, FactKind, GateResult, Outcome, RelationPredicate,
                     RelationState, ResolvedLine)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _worst(outcomes: list[Outcome]) -> Outcome:
    """Fail-closed precedence: any BLOCKED/ERROR dominates, then UNKNOWN, then
    NOT_APPLICABLE if that's all there was, else PASS."""
    for bad in (Outcome.ERROR, Outcome.BLOCKED, Outcome.UNKNOWN):
        if bad in outcomes:
            return bad
    if outcomes and all(o is Outcome.NOT_APPLICABLE for o in outcomes):
        return Outcome.NOT_APPLICABLE
    return Outcome.PASS


def dos_gate(result: CodingResult) -> GateResult:
    if not result.date_of_service:
        return GateResult("date_of_service", Outcome.BLOCKED,
                          "no date of service — every date-sensitive check is unverifiable",
                          "input contract")
    return GateResult("date_of_service", Outcome.PASS,
                      f"DOS = {result.date_of_service}", "input contract")


def evidence_gate(result: CodingResult, note_text: str) -> GateResult:
    note = _norm(note_text)
    outcomes: list[Outcome] = []
    misses: list[str] = []
    for ln in result.billable_lines:
        spans = ln.fact.evidence
        present = any(_norm(s.text) and _norm(s.text) in note for s in spans)
        outcomes.append(Outcome.PASS if present else Outcome.BLOCKED)
        if not present:
            misses.append(ln.chosen.code if ln.chosen else ln.fact.description)
    if not outcomes:
        return GateResult("verbatim_evidence", Outcome.NOT_APPLICABLE,
                          "no billable lines", "evidence")
    return GateResult("verbatim_evidence", _worst(outcomes),
                      "all lines have verbatim note support" if not misses
                      else f"unsupported: {misses}", "note text")


def code_active_gate(result: CodingResult, source: CodeSource) -> GateResult:
    outcomes, detail = [], []
    for ln in result.billable_lines:
        o = source.active_on(ln.chosen.code, ln.chosen.system, result.date_of_service)
        outcomes.append(o)
        if o is not Outcome.PASS:
            detail.append(f"{ln.chosen.code}:{o.value}")
    if not outcomes:
        return GateResult("code_active_on_dos", Outcome.NOT_APPLICABLE,
                          "no billable lines", "authoritative source")
    return GateResult("code_active_on_dos", _worst(outcomes),
                      "all codes active for DOS" if not detail else ", ".join(detail),
                      "authoritative source")


def ncci_gate(result: CodingResult, source: CodeSource) -> GateResult:
    lines = [ln for ln in result.billable_lines
             if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")]
    outcomes, detail, retryable = [], [], False
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            ind = source.ncci_indicator(lines[i].chosen.code, lines[j].chosen.code,
                                        result.date_of_service)
            if ind == AUTHORITY_UNAVAILABLE:   # the check could not run -> not clean
                outcomes.append(Outcome.UNKNOWN)
                retryable = True               # operational failure -> SYSTEM_HOLD, not review
                detail.append(f"{lines[i].chosen.code}/{lines[j].chosen.code} "
                              f"NCCI check unavailable")
                continue
            if ind is None:
                continue                       # ran; this pair not in the edit table
            if ind == "0":                     # not separately reportable, no bypass
                outcomes.append(Outcome.BLOCKED)
                detail.append(f"{lines[i].chosen.code}/{lines[j].chosen.code} bundled (0)")
            elif ind == "1":                   # bypassable with a justified modifier
                pair = frozenset((lines[i].chosen.code, lines[j].chosen.code))
                if pair in result.bypassed_ncci:   # a distinct-service modifier was applied
                    outcomes.append(Outcome.PASS)
                else:
                    outcomes.append(Outcome.UNKNOWN)
                    detail.append(f"{lines[i].chosen.code}/{lines[j].chosen.code} needs modifier (1)")
    if len(lines) < 2:
        supp = getattr(result, "ncci_suppressed", [])
        if supp:
            pairs = "; ".join(f"{c} bundled into {p}" for c, p in supp)
            return GateResult("ncci_ptp", Outcome.PASS,
                              f"NCCI PTP applied in reconciliation ({pairs}); no unresolved "
                              f"conflicts among released line(s)", "NCCI PTP (data)")
        return GateResult("ncci_ptp", Outcome.NOT_APPLICABLE, "fewer than two procedures",
                          "NCCI PTP (data)")
    return GateResult("ncci_ptp", _worst(outcomes) if outcomes else Outcome.PASS,
                      "no unresolved PTP conflicts" if not detail else "; ".join(detail),
                      "NCCI PTP (data)", retryable=retryable)


def mue_gate(result: CodingResult, source: CodeSource) -> GateResult:
    outcomes, detail = [], []
    for ln in result.billable_lines:
        limit = source.mue_limit(ln.chosen.code, result.date_of_service)
        units = ln.units                        # descriptor-driven billing units
        if limit is None:
            continue                            # no MUE published for this code
        if units > limit:
            outcomes.append(Outcome.BLOCKED)
            detail.append(f"{ln.chosen.code}: {units} > MUE {limit}")
        else:
            outcomes.append(Outcome.PASS)
    if result.billable_lines and not source.mue_available():
        return GateResult("mue", Outcome.UNKNOWN,
                          "MUE table unavailable — unit limits cannot be asserted",
                          "MUE (data)", retryable=True)
    if not outcomes:
        return GateResult("mue", Outcome.NOT_APPLICABLE, "no MUE-constrained lines",
                          "MUE (data)")
    return GateResult("mue", _worst(outcomes),
                      "units within MUE" if not detail else "; ".join(detail), "MUE (data)")


def drug_units_gate(result: CodingResult, source: CodeSource) -> GateResult:
    """A documented drug DOSE may only become billing units from the authoritative
    per-unit dose table.

    Without this gate, an absent dose table made `drug_unit` return None — the same
    answer as "this code is not a dosed drug" — and the units silently fell back to the
    documented COUNT (30 mg of a 'per 15 mg' code billed as 1 unit instead of 2). That is
    an absent optional source changing a released claim, which is precisely what
    "optional" may never mean. Unavailable authority now HOLDS (retryable) instead.
    (Codex F6-R5, round 5.)
    """
    from . import ontology
    dosed = []
    for ln in result.billable_lines:
        fact = getattr(ln, "fact", None)
        if fact is None or getattr(fact, "kind", None) is not FactKind.DRUG:
            continue
        if ontology.parse_dose(ontology.documented_dose_text(fact)) is None:
            continue                            # no dose documented -> count-based units
        dosed.append(ln)
    if not dosed:
        return GateResult("drug_units", Outcome.NOT_APPLICABLE,
                          "no drug line with a documented dose", "drug dosing (data)")
    # Authority availability is asked ONCE: "the table did not load" is a different
    # answer from "this code carries no per-unit dose", and only the first one holds.
    available = getattr(source, "drug_dose_table_available", None)
    if not (callable(available) and available()):
        return GateResult(
            "drug_units", Outcome.UNKNOWN,
            "authoritative drug per-unit dose table unavailable — a documented dose "
            "cannot be converted into billing units",
            "drug dosing (data)", retryable=True)
    unresolved = [ln.chosen.code for ln in dosed
                  if ln.chosen is not None and source.drug_unit(ln.chosen.code) is None]
    if unresolved:
        return GateResult(
            "drug_units", Outcome.PASS,
            "dose documented but no per-unit dose is published for: "
            + ", ".join(sorted(unresolved)) + "; units stay count-based",
            "drug dosing (data)")
    return GateResult("drug_units", Outcome.PASS,
                      "documented doses converted from the authoritative per-unit dose",
                      "drug dosing (data)")


# The necessity relation control lives in REVIEWED VERSIONED CONFIGURATION, not as an inline
# Python constant: confidence floor, required relation properties, and which reconciliation
# statuses count as independent support are all control decisions that must be diffable and
# citable to an authority. Loading is fail-closed -- there is no built-in default to fall back
# to, because a silently-defaulted control is exactly the thing being fixed. (Codex F6-R3.)
# The path is resolved from the RELEASE-SOURCE DECLARATION, not from __file__: the control's
# bytes are part of what the release certificate attests, so the file the gate loads and the
# file the manifest content-addresses must be the same object by construction, and an
# undeclared control raises here instead of releasing claims unattested. (Codex F6-R5.)
_NECESSITY_CONTROL_ID = "necessity_relation_control"
_NECESSITY_CONTROL_CACHE: dict | None = None

_REQUIRED_CONTROL_KEYS = ("version", "control_mode", "authority", "min_relation_confidence",
                          "require_anchored_relation_evidence", "require_asserted_state",
                          "conflicting_edge_disqualifies_support",
                          "accepted_reconciliation_statuses")


class NecessityControlError(RuntimeError):
    """The necessity relation control configuration is missing or malformed. Raised (never
    defaulted) so the gate reports ERROR and autonomy stops."""


def load_necessity_control() -> dict:
    """The reviewed necessity relation control, fully validated. Cached per process."""
    global _NECESSITY_CONTROL_CACHE
    if _NECESSITY_CONTROL_CACHE is not None:
        return _NECESSITY_CONTROL_CACHE
    try:
        from app.release.source_manifest import declared_source_path
        path = declared_source_path(_NECESSITY_CONTROL_ID)
        cfg = json.loads(path.read_text())
    except Exception as exc:
        raise NecessityControlError(
            f"necessity relation control unreadable ({_NECESSITY_CONTROL_ID}): {exc}") from exc
    if not isinstance(cfg, dict):
        raise NecessityControlError("necessity relation control must be a JSON object")
    missing = [k for k in _REQUIRED_CONTROL_KEYS if k not in cfg]
    if missing:
        raise NecessityControlError(
            f"necessity relation control is missing required key(s): {missing}")
    floor = cfg["min_relation_confidence"]
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) \
            or not 0.0 <= float(floor) <= 1.0:
        raise NecessityControlError("min_relation_confidence must be a number in [0.0, 1.0]")
    accepted = cfg["accepted_reconciliation_statuses"]
    if not isinstance(accepted, list) or not accepted or \
            not all(isinstance(s, str) and s.strip() for s in accepted):
        raise NecessityControlError(
            "accepted_reconciliation_statuses must be a non-empty array of strings")
    # A claim-affecting control may accept ONLY a status that means the SOURCE RECORD grounds
    # the edge. This is validated against the provenance layer's own declaration rather than
    # trusted to the config's contents, so no config edit -- including restoring an older
    # revision of this file -- can re-admit an agreement-only status as justification. It
    # fails closed: the gate reports ERROR and autonomy stops. (Codex F6-R3, round 5.)
    try:
        from . import provenance as _prov
    except Exception as exc:                          # pragma: no cover - defensive
        # Typed and fail-closed: without the grounding declaration the control cannot be
        # validated, so the gate must ERROR (SYSTEM_HOLD) rather than run unvalidated.
        raise NecessityControlError(
            f"cannot validate accepted_reconciliation_statuses: the provenance grounding "
            f"declaration is unavailable ({exc})") from exc
    for status in accepted:
        norm = str(status).strip().lower()
        if norm in _prov.GROUNDED_RECONCILIATION_STATUSES:
            continue
        if norm in _prov.RETIRED_RECONCILIATION_STATUSES:
            raise NecessityControlError(
                f"accepted_reconciliation_statuses lists {norm!r}, which is no longer a "
                f"reconciliation status: agreement between assertion origins is recorded on "
                f"the separate corroboration axis and can never ground a claim-affecting "
                f"relation. Accepted values are grounded statuses only: "
                f"{sorted(_prov.GROUNDED_RECONCILIATION_STATUSES)}")
        raise NecessityControlError(
            f"accepted_reconciliation_statuses lists {norm!r}, which does not establish that "
            f"the source record grounds the relation. Accepted values are grounded statuses "
            f"only: {sorted(_prov.GROUNDED_RECONCILIATION_STATUSES)}")
    if not cfg.get("authority"):
        raise NecessityControlError("a control must cite its authority")
    _NECESSITY_CONTROL_CACHE = cfg
    return cfg


def _necessity_support_relations(result: CodingResult, control: dict) -> tuple[list, set]:
    """(usable REASON_FOR edges, disqualified (subject, object) pairs).

    An edge may support necessity only when it is, per the reviewed control:
      - predicate REASON_FOR and state ASSERTED (a merged conflict collapses to UNCERTAIN);
      - EVIDENCE-ANCHORED -- it carries verified span references, so the asserted relationship
        is tied to the source document rather than existing only in the model's output;
      - at or above the configured confidence floor;
      - GROUNDED IN THE RECORD -- stamped by the deterministic provenance layer with a
        reconciliation status the control accepts, which the loader has already constrained to
        `provenance.GROUNDED_RECONCILIATION_STATUSES` (a clause of the source document that
        states the relationship directionally). Never from repetition, co-occurrence, or
        agreement between assertion origins: how many model runs asserted an edge is recorded
        on the separate `corroboration_status` axis and is read here for the audit record
        only, never as justification. (Codex F6-R3, round 5.)
      - and NAMING the verified spans that grounded it: a grounded status with an empty
        `reconciliation_evidence` list would certify necessity while citing no source text, so
        it is refused here as well as being unreachable in the provenance layer.
    Any conflicting (non-asserted) edge between the same endpoints disqualifies that pair
    outright, so an uncertain/negated duplicate can never be out-voted by a confident one.
    """
    accepted = {str(s).strip().lower() for s in control["accepted_reconciliation_statuses"]}
    floor = float(control["min_relation_confidence"])
    require_anchor = bool(control["require_anchored_relation_evidence"])
    require_asserted = bool(control["require_asserted_state"])
    disqualified: set = set()
    usable: list = []
    for r in (result.relations or []):
        if r.predicate is not RelationPredicate.REASON_FOR:
            continue
        pair = (r.subject_event_id, r.object_event_id)
        if require_asserted and r.state is not RelationState.ASSERTED:
            if control.get("conflicting_edge_disqualifies_support"):
                disqualified.add(pair)
            continue
        if require_anchor and not list(getattr(r, "evidence_span_ids", None) or []):
            continue
        if float(getattr(r, "confidence", 0.0) or 0.0) < floor:
            continue
        if str(getattr(r, "reconciliation_status", "") or "").strip().lower() not in accepted:
            continue
        if not list(getattr(r, "reconciliation_evidence", None) or []):
            continue
        usable.append(r)
    return usable, disqualified


def _support_record(dx_line, relation, *, policy_qualifying: bool | None) -> dict:
    """The auditable justification for ONE released service: which claim-line diagnosis, and
    the full provenance of the relation that linked it. The certificate binds this."""
    return {
        "diagnosis_event_id": relation.subject_event_id,
        "diagnosis_code": dx_line.chosen.code if dx_line.chosen else None,
        "diagnosis_system": dx_line.chosen.system if dx_line.chosen else None,
        "relation_id": relation.relation_id,
        # GROUNDING (what released this) and AGREEMENT (audit/confidence only) are recorded as
        # the two separate facts they are, so a reader of the certificate can tell which one
        # justified the service without having to know the vocabulary. (Codex F6-R3, round 5.)
        "reconciliation_status": relation.reconciliation_status,
        "reconciliation_evidence": list(getattr(relation, "reconciliation_evidence", []) or []),
        "corroboration_status": str(getattr(relation, "corroboration_status", "") or ""),
        "assertion_origins": sorted(str(o) for o in (relation.assertion_origins or [])),
        "independent_support": int(getattr(relation, "independent_support", 0) or 0),
        "support": int(getattr(relation, "support", 0) or 0),
        "confidence": float(getattr(relation, "confidence", 0.0) or 0.0),
        "evidence_span_ids": list(relation.evidence_span_ids or []),
        "policy_qualifying": policy_qualifying,
    }


def medical_necessity_gate(result: CodingResult,
                           source: "CodeSource | None" = None) -> GateResult:
    """Every released procedure needs a diagnosis that JUSTIFIES IT IN THIS ENCOUNTER, and
    that justification must come from something other than the extraction model's own
    confidence in itself.

    EVERY released procedure requires an ENCOUNTER-SPECIFIC RESOLVED LINKAGE: a diagnosis
    --REASON_FOR--> service edge that is asserted, evidence-anchored, at/above the configured
    confidence floor, and GROUNDED IN THE RECORD by `provenance.reconcile_relations` -- a
    clause of the source document that states the relationship directionally, naming the
    verified spans that state it.

    Agreement between extraction runs is NOT such a grounding and can never substitute for it.
    Distinct assertion origins are still counted and recorded (`corroboration_status`,
    `assertion_origins`, `independent_support`) for the audit trail and confidence display,
    but N runs of a model agreeing -- same provider or different providers -- is that model's
    self-confidence sampled N times, not documentation. The control config cannot re-admit it:
    `load_necessity_control` refuses any accepted status outside
    `provenance.GROUNDED_RECONCILIATION_STATUSES`. (Codex F6-R3, round 5.)

    Where an authoritative CMS coverage policy GOVERNS the service (`qualifying_dx_for`
    returns a set), that is an ADDITIONAL requirement, not a substitute: the diagnosis linked
    in this encounter must itself qualify under the policy. Coverage membership proves a code
    pair CAN qualify; it never proves that this diagnosis justified this service in this note,
    so an unrelated covered diagnosis on the encounter can no longer release a governed
    service with no relation at all. (Codex F6-R3.)

    That coverage check stays an ADDITIONAL requirement in BOTH directions: it is also never
    accepted IN PLACE OF record grounding. A coverage policy says a code pair can qualify in
    general; it says nothing about what this note documents, so admitting it as a substitute
    would weaken the ungrounded-edge hold this gate exists to enforce rather than strengthen
    it. Grounding comes from the record; policy compatibility is checked on top of it.

    Everything else HOLDs: an unanchored edge, an edge the record does not ground (including
    one that only agreeing extraction runs assert), an edge below the floor, a
    conflicting/uncertain edge, a governed procedure whose linked diagnosis does not qualify,
    a governed procedure whose qualifying diagnosis is not the linked one, and an unavailable
    coverage evaluation.
    """
    result.necessity_support = []
    procs = result.procedure_lines
    if not procs:
        return GateResult("medical_necessity", Outcome.NOT_APPLICABLE,
                          "no procedures to justify", "necessity")
    dx_lines = result.diagnosis_lines
    if not dx_lines:
        return GateResult("medical_necessity", Outcome.BLOCKED,
                          "performed procedure(s) with no documented diagnosis",
                          "necessity (structural)")
    try:
        control = load_necessity_control()
    except NecessityControlError as exc:
        return GateResult("medical_necessity", Outcome.ERROR, str(exc),
                          "necessity relation control (config)", retryable=True)
    authority = f"necessity (structural + coverage) [{control['version']}]"
    released_dx = {ln.fact.fact_id: ln for ln in dx_lines
                   if ln.fact is not None and ln.fact.fact_id}
    reason_for, disqualified = _necessity_support_relations(result, control)
    holds: list[str] = []
    bindings: list[dict] = []
    for ln in procs:
        pid = ln.fact.fact_id if ln.fact is not None else None
        label = (ln.chosen.code if ln.chosen else None) or (
            ln.fact.description if ln.fact else "procedure")
        # Encounter-specific linkage: which RELEASED diagnoses this note actually documents
        # as the reason for THIS service, with the edge that says so (best edge per pair,
        # chosen deterministically so the binding is reproducible).
        linked: dict = {}
        for r in reason_for:
            if r.object_event_id != pid or r.subject_event_id not in released_dx:
                continue
            if (r.subject_event_id, pid) in disqualified:
                continue
            prev = linked.get(r.subject_event_id)
            if prev is None or (r.confidence, r.relation_id) > (prev.confidence, prev.relation_id):
                linked[r.subject_event_id] = r
        # Authoritative coverage policy: does one govern this service, and which released
        # diagnoses qualify under it?
        policy_linked: set = set()
        qualifying = None
        if source is not None and ln.chosen is not None:
            try:
                qualifying = source.qualifying_dx_for(ln.chosen.code, ln.chosen.system)
            except Exception:
                holds.append(f"{label}: coverage-policy evaluation unavailable")
                continue
            if qualifying is not None:
                want = {str(q).replace(".", "").upper() for q in qualifying}
                policy_linked = {e for e, dln in released_dx.items() if dln.chosen
                                 and dln.chosen.code.replace(".", "").upper() in want}
        no_link = (f"{label}: no record-grounded, evidence-anchored REASON_FOR link to a "
                   f"released diagnosis (neither the model's own confidence nor agreement "
                   f"between extraction runs is clinical support)")
        if qualifying is not None:
            # GOVERNED: encounter linkage AND policy compatibility, on the SAME diagnosis.
            accepted = sorted(set(linked) & policy_linked)
            if not accepted:
                if not linked:
                    holds.append(
                        f"{no_link}; coverage-list membership alone does not establish that a "
                        f"diagnosis justified this service in this encounter")
                elif not policy_linked:
                    holds.append(
                        f"{label}: linked diagnosis does not qualify under CMS coverage policy")
                else:
                    holds.append(
                        f"{label}: the diagnosis linked in this encounter is not the one that "
                        f"qualifies under CMS coverage policy")
                continue
        else:
            # UNGOVERNED: no policy exists for this service, so the encounter linkage is the
            # whole justification and is strictly required.
            accepted = sorted(linked)
            if not accepted:
                holds.append(no_link)
                continue
        bindings.append({
            "procedure_event_id": pid,
            "procedure_code": ln.chosen.code if ln.chosen else None,
            "procedure_system": ln.chosen.system if ln.chosen else None,
            "policy_governed": qualifying is not None,
            "control_version": control["version"],
            "supports": [_support_record(released_dx[e], linked[e],
                                         policy_qualifying=(e in policy_linked)
                                         if qualifying is not None else None)
                         for e in accepted],
        })
    result.necessity_support = bindings
    if holds:
        return GateResult("medical_necessity", Outcome.UNKNOWN, "; ".join(holds), authority)
    return GateResult("medical_necessity", Outcome.PASS,
                      f"{len(procs)} procedure(s) each justified by a record-grounded, "
                      f"anchored diagnosis link in this encounter, and by authoritative "
                      f"coverage policy where one governs the service",
                      authority)


def icd_excludes_gate(result: CodingResult, source: CodeSource) -> GateResult:
    """ICD-10-CM Tabular Excludes1: two diagnoses in an Excludes1 relationship are
    'NOT CODED HERE' — mutually exclusive, not reportable together, UNLESS the two
    conditions are genuinely unrelated (an FY-guideline exception that is a human
    judgement). This is the diagnosis-axis analogue of the NCCI procedure gate, read
    from the authoritative instructional notes — never a baked-in pair list.

    Fail-closed: a detected Excludes1 co-occurrence returns UNKNOWN (stops autonomy,
    routes to review) rather than PASS or BLOCKED — the coder cannot deterministically
    assert the two conditions are unrelated, so it cannot certify the set clean, and
    it also must not silently drop a possibly-valid code."""
    dx = [ln.chosen for ln in result.billable_lines
          if ln.chosen and ln.chosen.system == "icd10"]
    if len(dx) < 2:
        return GateResult("icd_excludes1", Outcome.NOT_APPLICABLE,
                          "fewer than two diagnoses", "ICD-10-CM Tabular (data)")
    conflicts: list[str] = []
    seen: set[frozenset[str]] = set()
    for a in dx:
        refs = source.excludes1_refs(a.code, "icd10")
        if not refs:
            continue
        for b in dx:
            if a.code == b.code:
                continue
            bu = str(b.code).replace(".", "").upper()
            if any(bu.startswith(r) for r in refs):
                pair = frozenset((a.code, b.code))
                if pair not in seen:
                    seen.add(pair)
                    conflicts.append(f"{a.code}/{b.code}")
    if conflicts:
        return GateResult("icd_excludes1", Outcome.UNKNOWN,
                          "Excludes1 pair(s) — confirm the conditions are unrelated "
                          f"before reporting together: {'; '.join(conflicts)}",
                          "ICD-10-CM Tabular (data)")
    return GateResult("icd_excludes1", Outcome.PASS,
                      "no Excludes1 conflicts among diagnoses", "ICD-10-CM Tabular (data)")


def claim_ownership_gate(result: CodingResult) -> GateResult:
    """Tri-state claim ownership over structured actor and organization identities."""
    from .ownership import fact_ownership, classify_ownership
    blocked: list[str] = []
    unstated = 0
    for ln in result.billable_lines:
        if ln.fact.kind is FactKind.DIAGNOSIS:
            continue
        o = fact_ownership(ln.fact)
        st = classify_ownership(o.performer_id, o.billing_entity_id,
                                o.organization_id, o.performer_function)
        if st is Outcome.BLOCKED:
            blocked.append(ln.chosen.code if ln.chosen else ln.fact.description)
        elif st is Outcome.UNKNOWN:
            unstated += 1
    owned_lines = [ln for ln in result.billable_lines if ln.fact.kind is not FactKind.DIAGNOSIS]
    if not owned_lines:
        return GateResult("claim_ownership", Outcome.NOT_APPLICABLE, "no billable lines",
                          "ownership")
    if blocked:
        return GateResult("claim_ownership", Outcome.BLOCKED,
                          f"billed by an entity that did not perform the service: {blocked}",
                          "ownership")
    if unstated:
        return GateResult("claim_ownership", Outcome.UNKNOWN,
                          f"ownership unresolved for {unstated} billed line(s)",
                          "ownership", retryable=True)
    return GateResult("claim_ownership", Outcome.PASS,
                      "all billed services are owned by the billing entity",
                      "ownership")


def source_manifest_gate(result: CodingResult) -> GateResult:
    """Fail closed on a MISSING REQUIRED authoritative source. Degradation is loud:
    an absent required source (a code table / edit-policy file) BLOCKS release; absent
    OPTIONAL recall aids are recorded (NOT_APPLICABLE detail) but do not block."""
    try:
        from .capability import build_manifest
        man = build_manifest()
    except Exception as exc:
        return GateResult("source_manifest", Outcome.ERROR, f"manifest unavailable: {exc}",
                          "capability manifest", retryable=True)
    if man.get("missing_required"):
        return GateResult("source_manifest", Outcome.BLOCKED,
                          "required source(s) missing: " + ", ".join(man["missing_required"]),
                          "capability manifest")
    # A source that IS present but whose bytes cannot be identified -- undigestible, or
    # drifted from its reviewed lock -- is just as unsafe as an absent one: the claim would
    # be certified against data nobody can name. (Codex F6-R5.)
    if man.get("integrity_errors"):
        return GateResult("source_manifest", Outcome.BLOCKED,
                          "authoritative source integrity: "
                          + "; ".join(man["integrity_errors"]), "capability manifest")
    degraded = man.get("degraded_optional") or []
    detail = ("all required sources loaded"
              + (f"; optional absent: {', '.join(degraded)}" if degraded else ""))
    return GateResult("source_manifest", Outcome.PASS, detail, "capability manifest")


def run_gates(result: CodingResult, note_text: str, source: CodeSource) -> list[GateResult]:
    """All mandatory gates. Add a gate here (never a code list) as coverage grows."""
    try:
        return [
            source_manifest_gate(result),
            claim_ownership_gate(result),
            dos_gate(result),
            evidence_gate(result, note_text),
            code_active_gate(result, source),
            medical_necessity_gate(result, source),
            ncci_gate(result, source),
            mue_gate(result, source),
            drug_units_gate(result, source),
            icd_excludes_gate(result, source),
        ]
    except Exception as exc:  # a gate that crashes is ERROR, never a silent pass
        return [GateResult("gate_execution", Outcome.ERROR, str(exc), "runtime")]
