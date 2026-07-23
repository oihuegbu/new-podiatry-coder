import re

TEMPLATE_NAME = "descriptor_scope_modifier_arbitration"

SCHEMA_DOC = """
TEMPLATE: descriptor_scope_modifier_arbitration

PURPOSE
  Deterministically removes ONE payment-modifier class from billed CPT/HCPCS
  lines whose OFFICIAL reference descriptor (from the validator's code
  tables, NOT the claim or note text) grammatically encodes that the
  modifier's payment axis is already priced into the code. Classic case:
  bilateral modifier on a timed 'each 15 minutes' descriptor that already
  spans 'extremity(ies) ... and/or trunk' -- bilateral work is billed as
  additional timed units, so the modifier double-pays an axis the code
  already covers. The template mutates ONLY modifiers matching
  modifier_regex on ONLY the lines its selectors match. It never touches
  units, other modifiers, other lines, or other arrays, and it never adds
  anything.

RULE FIELDS
  template            (string, required) must equal
                      "descriptor_scope_modifier_arbitration".
  applies_to          (object, required)
    .array            "cpt_codes" or "hcpcs_codes" -- which claim array's
                      lines are candidates.
    .code_regex       (string, required) broad STRUCTURAL regex applied with
                      re.search against each line's code. Must NOT be a
                      literal code list -- use structural shape only (e.g.
                      a leading-digit family pattern). Lines whose code does
                      not match are never touched.
  modifier_regex      (string, required) anchored regex (e.g. "^50$")
                      defining the SINGLE modifier class this rule
                      arbitrates. Matched with re.search against each
                      whitespace-stripped modifier string. Only matching
                      modifiers are ever removed; every non-matching
                      modifier is preserved verbatim and in original order.
  descriptor_inherency_markers (array, required, non-empty) list of
                      {"label": str, "regex": str}. Each regex is matched
                      (re.search) against the LOWERCASED official descriptor
                      of the line's code from the reference store. Labels
                      appear in the emitted message via {markers}. Choose
                      markers that quote the descriptor's own grammar (a
                      timed-unit phrase, an inherent-plural-anatomy phrase),
                      never code identity.
  markers_require     (string, optional, default "all") "all" = every marker
                      must match the descriptor; "any" = at least one must.
                      Prefer "all" for conservatism.
  action              (object, optional) issue reporting:
    .severity         default "WARNING"
    .category         default "descriptor_scope_modifier"
    .denial_risk      "LOW"|"MEDIUM"|"HIGH", default "MEDIUM"
    .message          template string; placeholders {code} {desc}
                      {modifier} {markers}
    .recommendation   template string; same placeholders

SEMANTICS / GUARANTEES
  - For each candidate line (code matches code_regex) carrying at least one
    modifier matching modifier_regex, the template looks up the code's
    official descriptor via the validator's reference tables. If the lookup
    fails or yields no descriptor text, the line is SKIPPED (never guess).
  - If the required inherency markers match the descriptor, the matching
    modifiers -- and ONLY those -- are removed from that line's modifiers
    list, and one issue is emitted per corrected line.
  - Conservatism: any unparseable regex (code, modifier, or any marker),
    malformed marker entry, unknown markers_require value, or missing
    reference descriptor causes the template to do NOTHING for the rule or
    line in question. No marker match => no action.
  - Determinism: lines are processed in claim order; markers are evaluated
    in the order the rule lists them; removed-modifier text in messages is
    sorted and de-duplicated.
  - Single-axis: nothing but the matched modifier class on matched lines is
    ever written. Units, descriptions, other modifiers, ICD lines, and
    coding_result are never modified. Nothing is ever added.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    applies = rule.get("applies_to")
    if not isinstance(applies, dict):
        return

    array = applies.get("array")
    if array == "cpt_codes":
        lines = cpt
        lookup = v.db.validate_cpt
    elif array == "hcpcs_codes":
        lines = hcpcs
        lookup = v.db.validate_hcpcs
    else:
        return
    if not isinstance(lines, list):
        return

    code_src = applies.get("code_regex")
    mod_src = rule.get("modifier_regex")
    if not isinstance(code_src, str) or not code_src:
        return
    if not isinstance(mod_src, str) or not mod_src:
        return
    try:
        code_rx = re.compile(code_src)
        mod_rx = re.compile(mod_src)
    except Exception:
        return

    markers_cfg = rule.get("descriptor_inherency_markers")
    if not isinstance(markers_cfg, list) or not markers_cfg:
        return
    compiled_markers = []
    for marker in markers_cfg:
        if not isinstance(marker, dict):
            return
        label = marker.get("label")
        pattern = marker.get("regex")
        if not isinstance(label, str) or not label:
            return
        if not isinstance(pattern, str) or not pattern:
            return
        try:
            compiled_markers.append((label, re.compile(pattern)))
        except Exception:
            return

    require = rule.get("markers_require", "all")
    if require not in ("any", "all"):
        return

    action = rule.get("action")
    if not isinstance(action, dict):
        action = {}
    severity = action.get("severity", "WARNING")
    category = action.get("category", "descriptor_scope_modifier")
    denial_risk = action.get("denial_risk", "MEDIUM")
    message_tpl = action.get(
        "message",
        "AUTO-CORRECTED: modifier {modifier} removed from {code} "
        "('{desc}') - the official descriptor already prices the payment "
        "axis the modifier claims ({markers}).",
    )
    recommendation_tpl = action.get(
        "recommendation",
        "Do not append modifier {modifier} to {code}; the descriptor "
        "itself defines the unit of service across sides/sites.",
    )

    for entry in lines:
        if not isinstance(entry, dict):
            continue
        code_raw = entry.get("code", "")
        code = str(code_raw).strip() if code_raw is not None else ""
        if not code or not code_rx.search(code):
            continue

        mods = entry.get("modifiers")
        if not isinstance(mods, list) or not mods:
            continue

        removed = []
        kept = []
        for m in mods:
            m_txt = m.strip() if isinstance(m, str) else ""
            if m_txt and mod_rx.search(m_txt):
                removed.append(m_txt)
            else:
                kept.append(m)
        if not removed:
            continue

        info = lookup(code)
        if not isinstance(info, dict):
            continue
        desc = info.get("description") or info.get("long_description") or ""
        if not isinstance(desc, str) or not desc.strip():
            continue
        desc_l = desc.lower()

        matched_labels = []
        for label, marker_rx in compiled_markers:
            if marker_rx.search(desc_l):
                matched_labels.append(label)
        if require == "all":
            if len(matched_labels) != len(compiled_markers):
                continue
        else:
            if not matched_labels:
                continue

        entry["modifiers"] = kept

        fields = {
            "code": code,
            "desc": desc,
            "modifier": ", ".join(sorted(set(removed))),
            "markers": "; ".join(matched_labels),
        }
        try:
            message = message_tpl.format(**fields)
        except Exception:
            message = message_tpl
        try:
            recommendation = recommendation_tpl.format(**fields)
        except Exception:
            recommendation = recommendation_tpl

        v._add(severity, code, category, message, recommendation,
               denial_risk=denial_risk)
