import re

TEMPLATE_NAME = "coverage_findings_modifier_arbitration"

SCHEMA_DOC = """
coverage_findings_modifier_arbitration
======================================
Single-axis PRESENCE arbitration of ONE coverage-attestation modifier
family (e.g. the routine-foot-care class-findings modifier family) on
procedure lines. The rule declares the governing coverage policy's
finding classes as regex lexicons drawn from the policy's own wording;
each arbitrated modifier value carries a requirement over class-finding
counts. If a billed line carries a family modifier whose requirement is
NOT met by the documented (non-contradicted) findings, that ONE modifier
is removed and one issue is emitted. A supported modifier is kept
silently. Adding a missing modifier happens ONLY when action.add_modifier
is true, the requirement is fully met, the line carries no family
modifier, and exactly one candidate qualifies. Units, other modifiers,
other lines, and the diagnosis arrays are NEVER touched.

Rule JSON fields
----------------
id          kebab-case string (required)
template    "coverage_findings_modifier_arbitration" (required)
enabled     true (required)
authority   prose citation of the governing coverage policy (required)

applies_to  (required)
  array        "cpt_codes" | "hcpcs_codes"
  code_regex   broad STRUCTURAL regex the billed code must match
               (re.search, case-insensitive). Never a code list; the
               real selection is descriptor grammar (line_select).

line_select (optional, strongly recommended)
  descriptor_requires_all  [lowercase substrings] each must appear in
                           the code's official reference descriptor
  descriptor_requires_any  [lowercase substrings] at least one must
                           appear
  Descriptors come from the engine's reference tables; a line whose
  code has no reference entry is SKIPPED, never guessed at.

modifier_class_regex (required)
  Anchored, case-insensitive regex for the ONE modifier family this
  rule arbitrates. Matching uses fullmatch. Only modifiers fullmatching
  it are ever candidates for removal or addition; everything else on
  the line is invisible to the template.

classes (required) list of finding classes:
  [{ "class_label": "B", "findings": [ FINDING, ... ] }]
  FINDING is either atomic:
    { "label": "...",
      "note_regex": "...",              (required)
      "contradiction_regex": "..." }    (optional)
  or composite (N-of-M sub-findings, e.g. advanced trophic changes):
    { "label": "...", "min_count": 3,
      "sub_findings": [ atomic FINDING, ... ] }
  Semantics:
    DOCUMENTED    note_regex matches a negation-scrubbed sentence that
                  lies outside every exclusion context
    CONTRADICTED  contradiction_regex matches any negation-scrubbed
                  sentence (affirmative exam statements survive the
                  scrub, e.g. 'pulses palpable', 'sensation intact')
    AMBIGUOUS     a finding or sub-finding both documented and
                  contradicted => the ENTIRE rule takes NO action
    composite     documented iff at least min_count sub-findings are
                  documented and non-contradicted
  A class's count = number of documented, non-contradicted findings.
  Write the regexes in the coverage policy's own vocabulary. Sentences
  are split on . ; ! ? and newlines, so use [^;.]* rather than .* to
  keep proximity constraints inside one clause.

exclusion_contexts (optional) [regex]
  A sentence matching any of these is ignored for DOCUMENTATION only
  (history-of, patient-education, return-precautions boilerplate...).
  Contradictions still count from every sentence.

modifiers (required) one entry per arbitrated modifier value:
  [{ "modifier_regex": "...",   anchored; fullmatched against each
                                billed family modifier
     "requires": [ {"class_label": "B", "min": 2}, ... ],
                                ALL entries must hold (logical AND);
                                every class_label must exist in classes;
                                min is an integer >= 1
     "add_value": "..." }]      optional; the literal modifier the
                                gated add path may append (must
                                fullmatch this entry's regex AND
                                modifier_class_regex)
  A billed family modifier matching zero or several entries is left
  untouched (conservative: the template never guesses).

action (required)
  severity / category / denial_risk   for the emitted issue
  message / recommendation            templates with placeholders
                                      {code} {modifier} {desc} {unmet}
  add_modifier    bool, default false; enables the add path described
                  above
  add_message / add_recommendation    templates for the add path,
                  placeholders {code} {modifier} {desc}

Guarantees the mechanic enforces (rule authors rely on these):
- unparseable regex, malformed config, missing descriptor lookup, or an
  empty note => the rule does nothing at all
- any ambiguity (documented AND contradicted) => nothing
- deterministic: classes evaluated in sorted label order; lines,
  billed modifiers, and config entries in listed order
- single-axis: only family modifiers on selected lines ever change;
  nothing else is rewritten or normalized
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    def make_rx(pattern):
        if not isinstance(pattern, str) or not pattern:
            return None
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error:
            return None

    applies = rule.get("applies_to") or {}
    array_name = applies.get("array")
    if array_name == "cpt_codes":
        lines = cpt
        lookup = v.db.validate_cpt
    elif array_name == "hcpcs_codes":
        lines = hcpcs
        lookup = v.db.validate_hcpcs
    else:
        return
    if not isinstance(lines, list):
        return

    code_rx = make_rx(applies.get("code_regex"))
    class_rx = make_rx(rule.get("modifier_class_regex"))
    if code_rx is None or class_rx is None:
        return

    excl = []
    for pat in (rule.get("exclusion_contexts") or []):
        er = make_rx(pat)
        if er is None:
            return
        excl.append(er)

    def make_atomic(cfg):
        if not isinstance(cfg, dict):
            return None
        nr = make_rx(cfg.get("note_regex"))
        if nr is None:
            return None
        cr = None
        if cfg.get("contradiction_regex") is not None:
            cr = make_rx(cfg.get("contradiction_regex"))
            if cr is None:
                return None
        return (nr, cr)

    classes_cfg = rule.get("classes")
    if not isinstance(classes_cfg, list) or not classes_cfg:
        return
    compiled_classes = {}
    for cls in classes_cfg:
        if not isinstance(cls, dict):
            return
        label = cls.get("class_label")
        finds = cls.get("findings")
        if not isinstance(label, str) or not label:
            return
        if label in compiled_classes:
            return
        if not isinstance(finds, list) or not finds:
            return
        compiled = []
        for f in finds:
            if not isinstance(f, dict):
                return
            subs = f.get("sub_findings")
            if isinstance(subs, list) and subs:
                mc = f.get("min_count")
                if isinstance(mc, bool) or not isinstance(mc, int) or mc < 1:
                    return
                csubs = []
                for sf in subs:
                    pair = make_atomic(sf)
                    if pair is None:
                        return
                    csubs.append(pair)
                compiled.append(("composite", mc, csubs))
            else:
                pair = make_atomic(f)
                if pair is None:
                    return
                compiled.append(("atomic", pair, None))
        compiled_classes[label] = compiled

    mods_cfg = rule.get("modifiers")
    if not isinstance(mods_cfg, list) or not mods_cfg:
        return
    compiled_mods = []
    for m in mods_cfg:
        if not isinstance(m, dict):
            return
        mr = make_rx(m.get("modifier_regex"))
        if mr is None:
            return
        reqs = m.get("requires")
        if not isinstance(reqs, list) or not reqs:
            return
        creq = []
        for r in reqs:
            if not isinstance(r, dict):
                return
            cl = r.get("class_label")
            mn = r.get("min")
            if cl not in compiled_classes:
                return
            if isinstance(mn, bool) or not isinstance(mn, int) or mn < 1:
                return
            creq.append((cl, mn))
        add_value = m.get("add_value")
        if add_value is not None and not isinstance(add_value, str):
            return
        compiled_mods.append((mr, creq, add_value))

    text = note_full_text if isinstance(note_full_text, str) else ""
    if not text.strip():
        return
    ev = v._note_evidence(text)
    scrubbed = ""
    if isinstance(ev, tuple) and len(ev) > 1 and isinstance(ev[1], str):
        scrubbed = ev[1]
    if not scrubbed.strip():
        return
    sentences = [s.strip() for s in re.split(r"[.;!?\n]+", scrubbed)
                 if s.strip()]
    if not sentences:
        return
    doc_sentences = [s for s in sentences
                     if not any(er.search(s) for er in excl)]

    def eval_atomic(pair):
        nr, cr = pair
        doc = any(nr.search(s) for s in doc_sentences)
        con = (cr is not None) and any(cr.search(s) for s in sentences)
        return doc, con

    counts = {}
    for label in sorted(compiled_classes):
        n = 0
        for f in compiled_classes[label]:
            if f[0] == "atomic":
                doc, con = eval_atomic(f[1])
                if doc and con:
                    return
                if doc:
                    n += 1
            else:
                sub_n = 0
                for pair in f[2]:
                    doc, con = eval_atomic(pair)
                    if doc and con:
                        return
                    if doc:
                        sub_n += 1
                if sub_n >= f[1]:
                    n += 1
        counts[label] = n

    sel = rule.get("line_select") or {}
    req_all = [t.lower() for t in (sel.get("descriptor_requires_all") or [])
               if isinstance(t, str) and t]
    req_any = [t.lower() for t in (sel.get("descriptor_requires_any") or [])
               if isinstance(t, str) and t]

    action = rule.get("action") or {}
    severity = action.get("severity", "WARNING")
    category = action.get("category", "coverage_findings_modifier")
    denial = action.get("denial_risk", "MEDIUM")
    msg_tpl = action.get("message") or ""
    rec_tpl = action.get("recommendation") or ""
    add_msg_tpl = action.get("add_message") or ""
    add_rec_tpl = action.get("add_recommendation") or ""

    def render(tpl, ctx, fallback):
        if not tpl:
            return fallback
        try:
            return tpl.format(**ctx)
        except (KeyError, IndexError, ValueError):
            return fallback

    for entry in lines:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "")
        if not code or not code_rx.search(code):
            continue
        info = lookup(code)
        if not isinstance(info, dict):
            continue
        desc = str(info.get("description")
                   or info.get("long_description") or "").lower()
        if not desc:
            continue
        if req_all and not all(t in desc for t in req_all):
            continue
        if req_any and not any(t in desc for t in req_any):
            continue

        mods = entry.get("modifiers")
        if isinstance(mods, list) and mods:
            kept = []
            changed = False
            for mod in mods:
                mstr = str(mod) if mod is not None else ""
                if not class_rx.fullmatch(mstr):
                    kept.append(mod)
                    continue
                hits = [cm for cm in compiled_mods
                        if cm[0].fullmatch(mstr)]
                if len(hits) != 1:
                    kept.append(mod)
                    continue
                unmet = [(cl, mn, counts.get(cl, 0))
                         for (cl, mn) in hits[0][1]
                         if counts.get(cl, 0) < mn]
                if not unmet:
                    kept.append(mod)
                    continue
                changed = True
                unmet_txt = "; ".join(
                    "Class " + cl + " requires " + str(mn)
                    + " documented finding(s), the note supports "
                    + str(have)
                    for (cl, mn, have) in unmet)
                ctx = {"code": code, "modifier": mstr,
                       "desc": desc, "unmet": unmet_txt}
                fallback = ("AUTO-CORRECTED: modifier " + mstr
                            + " removed from " + code + " -- "
                            + unmet_txt)
                v._add(severity, code, category,
                       render(msg_tpl, ctx, fallback),
                       render(rec_tpl, ctx,
                              "Document the coverage policy's class "
                              "findings before appending this modifier"),
                       denial_risk=denial)
            if changed:
                entry["modifiers"] = kept

        if action.get("add_modifier") is True:
            current = entry.get("modifiers")
            cur = current if isinstance(current, list) else []
            has_family = any(
                class_rx.fullmatch(str(m)) for m in cur if m is not None)
            if not has_family:
                cands = []
                for (mr, creq, add_value) in compiled_mods:
                    if not add_value:
                        continue
                    if not mr.fullmatch(add_value):
                        continue
                    if not class_rx.fullmatch(add_value):
                        continue
                    if all(counts.get(cl, 0) >= mn for (cl, mn) in creq):
                        cands.append(add_value)
                if len(cands) == 1:
                    entry["modifiers"] = cur + [cands[0]]
                    ctx = {"code": code, "modifier": cands[0],
                           "desc": desc, "unmet": ""}
                    fallback = ("AUTO-CORRECTED: modifier " + cands[0]
                                + " added to " + code
                                + " -- documented findings satisfy the "
                                + "coverage requirement")
                    v._add(severity, code, category,
                           render(add_msg_tpl, ctx, fallback),
                           render(add_rec_tpl, ctx,
                                  "Confirm the appended coverage "
                                  "modifier against the exam findings"),
                           denial_risk=denial)
