import re

TEMPLATE_NAME = "coverage_evidence_advisory_suppression"

SCHEMA_DOC = """coverage_evidence_advisory_suppression
=======================================
WARN-only mechanic: when the encounter note documents every coverage
finding a policy's documentation requirements demand, suppress the
compliance scrubber's ADVISORY (WARN) for the selected billed lines via
v.suppress_scrub_advisory(filter_id, code, rule_id, authority, note).
The claim itself is NEVER touched: no line, modifier, unit, or diagnosis
is written. Use this template ONLY for adjudicated advisory disputes
whose verified_state is "must_not_fire" -- the claim is verified correct
as billed and every claim line must replay byte-identical. The scrubber
records each suppression as its own PASS finding carrying rule_id and
authority; a FAIL can never be suppressed (WARN-only by contract).

RULE FIELDS
-----------
filter_id (string, required)
    The scrubber filter class whose WARN advisory is suppressed, exactly
    as it appears in the advisory key "FILTER_ID|CODE" (e.g. a
    medical-necessity filter class).

applies_to (object, required)
    array (required): "cpt_codes" | "hcpcs_codes" | "icd_codes".
    code_regex (required): BROAD structural regex over the code string
        (section-shaped, like a leading-digit pattern). It narrows; it
        never selects alone. Never a literal code.
    descriptor_requires_all (list of lowercase substrings, optional)
    descriptor_requires_any (list of lowercase substrings, optional)
        Matched against the code's OFFICIAL descriptor from the engine's
        reference tables (e.g. "nail" + "debrid"). At least one of the
        two lists must be non-empty or the rule is a no-op, so selection
        is always descriptor grammar, never code identity. A line whose
        reference lookup or descriptor is missing is never selected.

exclusion_contexts (list, optional)
    [{"label": str, "regex": str}, ...]. Any negation-scrubbed sentence
    matching any context regex is EXCLUDED from evidence evaluation.
    Use for history-of / prior-care, patient-education, return-
    precautions, and planned/future/deferred care -- text that does not
    describe today's covered condition. An unparseable regex makes the
    whole rule a no-op.

evidence_classes (list, required; labels unique)
    [{"label": str, "note_regex": str,
      "contradiction_regex": str (optional)}, ...]
    One class per documentation requirement of the governing coverage
    policy. Each note_regex is evaluated over the negation-scrubbed,
    lowercased sentences of the FULL note that survive
    exclusion_contexts. Keep multi-term proximity inside ONE clause by
    writing [^;.]* between terms. A class is "documented" if any
    surviving sentence matches its note_regex. If ANY class's
    contradiction_regex matches ANY surviving sentence, the whole rule
    is a no-op -- never suppress against conflicting documentation.
    Classes are evaluated in sorted-label order; an unparseable regex,
    missing/duplicate label, or malformed class makes the whole rule a
    no-op.

require ("all" or positive integer, default "all")
    "all": every evidence class must be documented.
    N (integer >= 1): at least N of the M classes documented.
    Anything else is a no-op.

action (object, optional)
    note (string): template for the suppression audit note the scrubber
        records in its PASS finding. Placeholders: {code} {desc}
        {evidence_labels} (comma-joined sorted labels of the documented
        classes). A conservative default is used if omitted.

authority (string, required by the pack)
    Passed verbatim to v.suppress_scrub_advisory; cite the coverage
    policy whose documentation requirements the evidence classes mirror.

GUARANTEES
----------
Deterministic: lines are processed in claim order (first occurrence of
each code); evidence classes in sorted-label order; evidence labels in
the note are sorted. Conservative: missing note text, missing reference
lookup, unparseable regex, unmet threshold, or any contradiction =>
no suppression at all. Single-surface: the ONLY effect is
v.suppress_scrub_advisory calls -- billed arrays are read-only.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    def safe_compile(pattern):
        if not isinstance(pattern, str) or not pattern.strip():
            return None
        try:
            return re.compile(pattern)
        except re.error:
            return None

    # ---- advisory selection -------------------------------------------
    filter_id = rule.get("filter_id")
    if not isinstance(filter_id, str) or not filter_id.strip():
        return
    filter_id = filter_id.strip()

    applies = rule.get("applies_to") or {}
    if not isinstance(applies, dict):
        return
    array_name = applies.get("array")
    if array_name == "cpt_codes":
        lines = cpt
    elif array_name == "hcpcs_codes":
        lines = hcpcs
    elif array_name == "icd_codes":
        lines = icd
    else:
        return
    if not isinstance(lines, list) or not lines:
        return

    code_rx = safe_compile(applies.get("code_regex"))
    if code_rx is None:
        return

    raw_all = applies.get("descriptor_requires_all") or []
    raw_any = applies.get("descriptor_requires_any") or []
    if not isinstance(raw_all, list) or not isinstance(raw_any, list):
        return
    req_all = [t.strip().lower() for t in raw_all
               if isinstance(t, str) and t.strip()]
    req_any = [t.strip().lower() for t in raw_any
               if isinstance(t, str) and t.strip()]
    if not req_all and not req_any:
        return  # descriptor grammar is mandatory; a regex never selects alone

    def lookup(code):
        if array_name == "cpt_codes":
            return v.db.validate_cpt(code)
        if array_name == "hcpcs_codes":
            return v.db.validate_hcpcs(code)
        return v.db.validate_icd10(code)

    selected = []
    seen = set()
    for entry in lines:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "").strip()
        if not code or code in seen:
            continue
        if not code_rx.search(code):
            continue
        info = lookup(code)
        if not isinstance(info, dict):
            continue  # missing reference lookup: never selected on faith
        desc = str(info.get("description")
                   or info.get("long_description") or "").strip()
        if not desc:
            continue
        low = desc.lower()
        if any(term not in low for term in req_all):
            continue
        if req_any and not any(term in low for term in req_any):
            continue
        seen.add(code)
        selected.append((code, desc))
    if not selected:
        return

    # ---- evidence -------------------------------------------------------
    if not isinstance(note_full_text, str) or not note_full_text.strip():
        return
    try:
        scrubbed = v._note_evidence(note_full_text)[1]
    except (TypeError, IndexError, KeyError):
        return
    if not isinstance(scrubbed, str) or not scrubbed.strip():
        return
    scrubbed = scrubbed.lower()

    sentences = [s.strip() for s in re.split(r"[.!?\n]+", scrubbed)
                 if s.strip()]
    if not sentences:
        return

    excl_cfg = rule.get("exclusion_contexts") or []
    if not isinstance(excl_cfg, list):
        return
    excl_rx = []
    for ctx in excl_cfg:
        if not isinstance(ctx, dict):
            return
        rx = safe_compile(ctx.get("regex"))
        if rx is None:
            return  # unparseable exclusion regex => whole rule is a no-op
        excl_rx.append(rx)
    kept = [s for s in sentences
            if not any(rx.search(s) for rx in excl_rx)]
    if not kept:
        return

    ev_cfg = rule.get("evidence_classes")
    if not isinstance(ev_cfg, list) or not ev_cfg:
        return
    ev_classes = []
    labels_seen = set()
    for cls in ev_cfg:
        if not isinstance(cls, dict):
            return
        label = cls.get("label")
        if not isinstance(label, str) or not label.strip():
            return
        label = label.strip()
        if label in labels_seen:
            return  # duplicate labels are ambiguous => no-op
        labels_seen.add(label)
        rx = safe_compile(cls.get("note_regex"))
        if rx is None:
            return
        crx = None
        contradiction = cls.get("contradiction_regex")
        if contradiction is not None:
            crx = safe_compile(contradiction)
            if crx is None:
                return
        ev_classes.append((label, rx, crx))
    ev_classes = sorted(ev_classes, key=lambda c: c[0])

    documented = []
    for label, rx, crx in ev_classes:
        if crx is not None and any(crx.search(s) for s in kept):
            return  # conflicting documentation => never suppress
        if any(rx.search(s) for s in kept):
            documented.append(label)

    need = rule.get("require", "all")
    if need == "all":
        if len(documented) != len(ev_classes):
            return
    elif isinstance(need, int) and not isinstance(need, bool):
        if need < 1 or len(documented) < need:
            return
    else:
        return

    # ---- action: advisory suppression only; claim arrays untouched ------
    action = rule.get("action") or {}
    if not isinstance(action, dict):
        action = {}
    note_tpl = action.get("note")
    if not isinstance(note_tpl, str) or not note_tpl.strip():
        note_tpl = ("Coverage documentation verified in the encounter note "
                    "for {code} ('{desc}'): {evidence_labels}. Advisory "
                    "suppressed; claim is correct as billed.")
    labels_txt = ", ".join(sorted(documented))
    rule_id = str(rule.get("id") or "")
    authority = str(rule.get("authority") or "")
    for code, desc in selected:
        note = (note_tpl.replace("{code}", code)
                        .replace("{desc}", desc)
                        .replace("{evidence_labels}", labels_txt))
        v.suppress_scrub_advisory(filter_id, code, rule_id=rule_id,
                                  authority=authority, note=note)
