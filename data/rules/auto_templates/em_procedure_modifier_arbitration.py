import re

TEMPLATE_NAME = "em_procedure_modifier_arbitration"

SCHEMA_DOC = """
Rule JSON schema for template "em_procedure_modifier_arbitration".

Purpose: single-axis presence arbitration of ONE anchored modifier class on
exactly ONE E/M line, driven entirely by whether the claim also bills a
same-day procedural service. The template mutates ONLY the matching
modifier(s) on ONLY the selected line; it never touches units, other
modifiers, diagnosis lines, or any other claim line.

Fields:
  target (object, required)
    array: "cpt_codes" | "hcpcs_codes" -- which claim array holds the E/M
      family line.
    descriptor_prefix: lowercase prefix the code's OFFICIAL reference
      descriptor must start with (e.g. "office or other outpatient visit").
      Selection is by descriptor grammar only -- NEVER put literal medical
      codes in any selecting field of this rule.
  modifier (object, required)
    regex: anchored pattern naming the arbitrated modifier class. Anchor it
      (^...$). An unparseable pattern disables the rule entirely (no action).
    value: the exact modifier string to append on the add path; it must
      itself match regex. Only needed when action.add_modifier is true.
  procedural (object, required -- configure at least one axis)
    global_indicator_field: name of the reference-record field that carries
      the global-surgery indicator (read from the other line's own db
      record; the template never interprets code numbers).
    global_indicator_values: list of field values that mark a billed line as
      a same-day procedural service (e.g. ["000", "010", "090"]).
    descriptor_requires_any: fallback grammar -- lowercase substrings; a line
      whose reference descriptor contains any of them counts as procedural.
      A billed line that NO configured axis can classify makes the claim
      ambiguous and the template does nothing at all.
  evidence (object, optional)
    raw_text: bool, default true -- search the raw note text; set false to
      search the negation-scrubbed text instead. NOTE: 'no procedure
      performed' affirmations are themselves negated sentences, so the
      negation scrub would delete exactly that evidence; raw_text true is
      almost always correct for the removal affirmation.
    no_procedure_affirmation_regex: case-insensitive pattern that must match
      the note before a present modifier is removed (when required is true).
    required: bool, default true -- whether the affirmation match is
      mandatory for the removal path when the regex is configured.
    separately_identifiable_regex: case-insensitive pattern of documented
      separately identifiable E/M work; MANDATORY for the add path.
  action (object, required)
    severity / category / denial_risk / message / recommendation: issue
      fields for the removal path. Placeholders: {code} {desc} {modifier}.
    add_modifier: bool, default false -- gate for the (rare) add path, taken
      only when procedural lines exist, the modifier is absent, and
      separately_identifiable_regex matches the note. message_add /
      recommendation_add are the issue texts for that path (same
      placeholders).

Deterministic behavior:
  1. Select E/M lines: entries in target.array whose reference descriptor
     starts with descriptor_prefix. Exactly one must exist, else no action.
  2. Classify every OTHER billed CPT/HCPCS line from reference data only.
     Any missing lookup, unclassifiable line, blank code, or second family
     E/M line -> no action. Count the procedural lines.
  3. Zero procedural lines: a present target modifier has nothing to be
     'separately identifiable' FROM -> remove ONLY the matching modifier(s)
     from ONLY the selected line (an emptied modifier list drops the key so
     replays converge byte-for-byte), and emit exactly one issue. If the
     modifier is already absent, the template is a no-op (no issue).
  4. One or more procedural lines: a present modifier is KEPT untouched;
     a missing one is added only via the gated, evidence-backed add path.
  All ambiguity (ties, unparseable regex, missing lookups) -> do nothing.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    target = rule.get("target") or {}
    array_name = str(target.get("array", "cpt_codes"))
    if array_name == "cpt_codes":
        target_lines = cpt
    elif array_name == "hcpcs_codes":
        target_lines = hcpcs
    else:
        return

    prefix = str(target.get("descriptor_prefix", "") or "").strip().lower()
    if not prefix:
        return

    mod_cfg = rule.get("modifier") or {}
    mod_pattern = mod_cfg.get("regex")
    if not isinstance(mod_pattern, str) or not mod_pattern:
        return
    try:
        mod_rx = re.compile(mod_pattern)
    except re.error:
        return

    def lookup(code_str):
        info = v.db.validate_cpt(code_str)
        if info:
            return info
        return v.db.validate_hcpcs(code_str)

    def descriptor_of(info):
        raw = info.get("description") or info.get("long_description") or ""
        return str(raw).lower()

    # ---- select the family E/M line by descriptor grammar, never code identity
    selected = []
    for entry in target_lines:
        if not isinstance(entry, dict):
            continue
        code_str = str(entry.get("code", "") or "").strip()
        if not code_str:
            continue
        info = lookup(code_str)
        if not info:
            continue
        if descriptor_of(info).startswith(prefix):
            selected.append((code_str, entry, info))
    if len(selected) != 1:
        return  # zero family lines: out of scope; several: ambiguous -> nothing
    em_code = selected[0][0]
    em_entry = selected[0][1]
    em_desc = descriptor_of(selected[0][2])

    # ---- classify every OTHER billed line from reference data only
    proc_cfg = rule.get("procedural") or {}
    gfield = str(proc_cfg.get("global_indicator_field", "") or "").strip()
    gvalues = set()
    for gval in (proc_cfg.get("global_indicator_values") or []):
        gtxt = str(gval).strip()
        if gtxt:
            gvalues.add(gtxt)
    has_global_axis = bool(gfield) and bool(gvalues)
    grammar_terms = []
    for term in (proc_cfg.get("descriptor_requires_any") or []):
        ttxt = str(term).strip().lower()
        if ttxt:
            grammar_terms.append(ttxt)
    if not has_global_axis and not grammar_terms:
        return  # no classification axis configured -> do nothing

    procedural_count = 0
    for other in list(cpt) + list(hcpcs):
        if other is em_entry or not isinstance(other, dict):
            continue
        code_str = str(other.get("code", "") or "").strip()
        if not code_str:
            return  # unidentifiable billed line -> ambiguous -> do nothing
        info = lookup(code_str)
        if not info:
            return  # missing reference lookup -> ambiguous -> do nothing
        desc = descriptor_of(info)
        if desc.startswith(prefix):
            return  # a second family E/M line -> ambiguous -> do nothing
        classified = False
        is_procedural = False
        if has_global_axis:
            raw = info.get(gfield)
            if raw is not None:
                classified = True
                if str(raw).strip() in gvalues:
                    is_procedural = True
        if not is_procedural and grammar_terms:
            classified = True
            for term in grammar_terms:
                if term in desc:
                    is_procedural = True
                    break
        if not classified:
            return  # no axis could classify this line -> ambiguous -> nothing
        if is_procedural:
            procedural_count = procedural_count + 1

    # ---- optional note evidence text
    evidence_cfg = rule.get("evidence") or {}
    text_for_evidence = str(note_full_text or "")
    if not evidence_cfg.get("raw_text", True):
        scrubbed = v._note_evidence(text_for_evidence)
        text_for_evidence = str(scrubbed[1])

    action = rule.get("action") or {}
    current_mods = em_entry.get("modifiers") or []
    if not isinstance(current_mods, list):
        return
    matching = []
    for m in current_mods:
        mtxt = str(m).strip()
        if mod_rx.match(mtxt):
            matching.append(mtxt)

    if procedural_count == 0:
        # no procedural service for the E/M to be separately identifiable FROM
        if not matching:
            return  # already aligned -> no-op
        aff_pattern = evidence_cfg.get("no_procedure_affirmation_regex")
        if isinstance(aff_pattern, str) and aff_pattern:
            try:
                aff_rx = re.compile(aff_pattern, re.IGNORECASE)
            except re.error:
                return  # unparseable regex -> do nothing
            if evidence_cfg.get("required", True) and not aff_rx.search(text_for_evidence):
                return  # note does not affirm absence of a procedure -> nothing
        kept = []
        for m in current_mods:
            if not mod_rx.match(str(m).strip()):
                kept.append(m)
        if kept:
            em_entry["modifiers"] = kept
        else:
            em_entry.pop("modifiers", None)
        removed = ", ".join(sorted(set(matching)))
        message = str(action.get("message", "") or "")
        message = message.replace("{code}", em_code)
        message = message.replace("{desc}", em_desc)
        message = message.replace("{modifier}", removed)
        recommendation = str(action.get("recommendation", "") or "")
        recommendation = recommendation.replace("{code}", em_code)
        recommendation = recommendation.replace("{desc}", em_desc)
        recommendation = recommendation.replace("{modifier}", removed)
        v._add(
            str(action.get("severity", "WARNING")),
            em_code,
            str(action.get("category", "em_modifier_arbitration")),
            message,
            recommendation,
            denial_risk=str(action.get("denial_risk", "MEDIUM")),
        )
        return

    # ---- same-day procedural service(s) present
    if matching:
        return  # a present modifier is supported by the claim's own lines
    if not action.get("add_modifier"):
        return  # add path disabled -> conservative no-op
    add_value = str(mod_cfg.get("value", "") or "").strip()
    if not add_value or not mod_rx.match(add_value):
        return
    sep_pattern = evidence_cfg.get("separately_identifiable_regex")
    if not isinstance(sep_pattern, str) or not sep_pattern:
        return  # adding requires documented evidence -> do nothing
    try:
        sep_rx = re.compile(sep_pattern, re.IGNORECASE)
    except re.error:
        return
    if not sep_rx.search(text_for_evidence):
        return
    new_mods = list(current_mods)
    new_mods.append(add_value)
    em_entry["modifiers"] = new_mods
    message = str(action.get("message_add", "") or "")
    message = message.replace("{code}", em_code)
    message = message.replace("{desc}", em_desc)
    message = message.replace("{modifier}", add_value)
    recommendation = str(action.get("recommendation_add", "") or "")
    recommendation = recommendation.replace("{code}", em_code)
    recommendation = recommendation.replace("{desc}", em_desc)
    recommendation = recommendation.replace("{modifier}", add_value)
    v._add(
        str(action.get("severity", "WARNING")),
        em_code,
        str(action.get("category", "em_modifier_arbitration")),
        message,
        recommendation,
        denial_risk=str(action.get("denial_risk", "MEDIUM")),
    )
