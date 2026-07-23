import re

TEMPLATE_NAME = "em_same_day_presence_arbitration"

SCHEMA_DOC = """
em_same_day_presence_arbitration -- arbitrates the PRESENCE of exactly one
E/M family line on claims that also bill same-day minor procedures. The
template's ONLY mutation is suppression of that one selected E/M line via
v._non_billable_codes_to_suppress; it never adds lines and never edits
modifiers, units, diagnoses, or any procedural line.

Rule fields:
  applies_to.array      : "cpt_codes" or "hcpcs_codes" -- the claim array
                          the E/M line lives in.
  applies_to.code_regex : optional broad STRUCTURAL regex a candidate code
                          must match. Never a literal code list.
  family.descriptor_prefix : lowercase official-descriptor prefix defining
                          the E/M family (e.g. "office or other outpatient
                          visit"). Matched against the reference-database
                          descriptor, never the line's own text.
  family.descriptor_requires_any : optional lowercase phrases; the official
                          descriptor must contain at least one (patient-
                          status grammar such as "established patient").
  procedural.global_fields : reference-record field names that may carry a
                          global-surgery indicator.
  procedural.global_values : indicator values (strings, compared after
                          strip+lowercase) marking a global-package minor/
                          major procedure (e.g. "000", "010", "090").
                          These are payment indicators, not medical codes.
  procedural.action_terms : lowercase surgical-action grammar phrases used
                          as fallback classifier when no global field is
                          present (e.g. "debridement", "destruction",
                          "excision", "application of").
  time_proof.threshold_regex : regex with ONE numeric capture group, run
                          against the selected E/M code's OWN official
                          descriptor to extract its minute threshold
                          ("N minutes must be met or exceeded").
  time_proof.capture_regexes : regexes, each with ONE numeric capture
                          group, run against the negation-scrubbed
                          lowercase note to collect documented total-time
                          minute values.
  separate_problem.lexicon_regexes : sentence-level regexes signalling a
                          separately identifiable problem ("separate
                          problem", "also evaluated", ...).
  separate_problem.min_token_len : minimum alphabetic token length when
                          harvesting anatomy/condition tokens from the
                          procedural lines' official descriptors
                          (default 3; descriptor stopwords always drop).
  verify_modifier_regex : optional broad structural regex for the
                          distinct-service modifier class; when the E/M
                          line is KEPT and no billed modifier matches, an
                          INFO issue is reported. Report only -- never a
                          mutation.
  action.severity / action.category / action.denial_risk /
  action.message / action.recommendation : issue text for the suppression
                          path. Placeholders: {code} {desc} {procedures}.
  action.verify_category / action.verify_message /
  action.verify_recommendation : issue text for the report-only modifier
                          check on the keep path.

Deterministic behavior:
  1. Select E/M family members by official-descriptor grammar. Anything
     other than EXACTLY ONE billed family line => do nothing. A missing
     E/M line is never added.
  2. Classify every OTHER billed CPT/HCPCS line from reference data only:
     global-surgery indicator in procedural.global_values, else official-
     descriptor action grammar. ANY reference-lookup failure => do
     nothing.
  3. Zero procedural lines => do nothing (the E/M stands on its own; out
     of scope).
  4. With one or more same-day procedures the E/M line is kept ONLY on
     affirmative proof of separately identifiable work:
       (a) exactly one distinct documented total-time minute value meeting
           or exceeding the threshold parsed from the E/M code's own
           descriptor; two or more distinct values => ambiguous => do
           nothing; a documented value with an unparseable descriptor
           threshold => do nothing; or
       (b) a negation-scrubbed sentence matching the separate-problem
           lexicon in which NONE of the anatomy/condition token stems
           harvested from the procedural lines' official descriptors
           appear (evaluative work disjoint from every procedure target).
     Proof => keep the line untouched (optional report-only modifier
     check). No proof => suppress the E/M line and report, citing NCCI
     Ch.1: the evaluation leading to the decision to perform a minor
     procedure is included in the procedure and not separately reportable.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    applies = rule.get("applies_to") or {}
    array_name = str(applies.get("array") or "")
    if array_name == "cpt_codes":
        em_src = cpt
    elif array_name == "hcpcs_codes":
        em_src = hcpcs
    else:
        return
    if not isinstance(em_src, list):
        return

    code_pat = None
    code_regex = applies.get("code_regex")
    try:
        if code_regex:
            code_pat = re.compile(str(code_regex))
    except re.error:
        return

    fam = rule.get("family") or {}
    prefix = str(fam.get("descriptor_prefix") or "").strip().lower()
    if not prefix:
        return
    req_any = [str(t).strip().lower()
               for t in (fam.get("descriptor_requires_any") or [])
               if str(t).strip()]

    def _official(code, prefer_cpt):
        if prefer_cpt:
            info = v.db.validate_cpt(code)
            if not isinstance(info, dict):
                info = v.db.validate_hcpcs(code)
        else:
            info = v.db.validate_hcpcs(code)
            if not isinstance(info, dict):
                info = v.db.validate_cpt(code)
        if isinstance(info, dict):
            return info
        return None

    def _desc_of(info):
        if not isinstance(info, dict):
            return ""
        return str(info.get("description")
                   or info.get("long_description") or "").strip().lower()

    # 1. select the single E/M family line by official-descriptor grammar
    family_hits = []
    for entry in em_src:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "").strip()
        if not code:
            continue
        if code_pat is not None and code_pat.search(code) is None:
            continue
        desc = _desc_of(_official(code, array_name == "cpt_codes"))
        if not desc.startswith(prefix):
            continue
        if req_any and not any(t in desc for t in req_any):
            continue
        family_hits.append((code, entry, desc))
    if len(family_hits) != 1:
        return
    em_code = family_hits[0][0]
    em_entry = family_hits[0][1]
    em_desc = family_hits[0][2]

    # 2. classify every other billed line from reference data only
    proc_cfg = rule.get("procedural") or {}
    g_fields = [str(f) for f in (proc_cfg.get("global_fields") or [])]
    g_values = set(str(x).strip().lower()
                   for x in (proc_cfg.get("global_values") or []))
    action_terms = [str(t).strip().lower()
                    for t in (proc_cfg.get("action_terms") or [])
                    if str(t).strip()]

    others = []
    for pair in (("cpt_codes", cpt), ("hcpcs_codes", hcpcs)):
        arr = pair[1]
        if not isinstance(arr, list):
            continue
        for entry in arr:
            if entry is em_entry or not isinstance(entry, dict):
                continue
            code = str(entry.get("code") or "").strip()
            if not code:
                continue
            others.append((pair[0], code))

    proc_codes = []
    proc_descs = []
    for arr_name, code in sorted(set(others)):
        info = _official(code, arr_name == "cpt_codes")
        if info is None:
            return  # unclassifiable line: reference lookup failed -> no action
        desc = _desc_of(info)
        procedural = False
        for f in g_fields:
            val = info.get(f)
            if val is not None and str(val).strip().lower() in g_values:
                procedural = True
        if not procedural and desc and action_terms:
            procedural = any(t in desc for t in action_terms)
        if procedural:
            proc_codes.append(code)
            proc_descs.append(desc)

    # 3. no same-day procedures: the E/M stands on its own; out of scope
    if not proc_codes:
        return

    # negation-scrubbed lowercase note text
    try:
        evidence = v._note_evidence(str(note_full_text or ""))
        scrubbed = str(evidence[1])
    except (TypeError, IndexError, KeyError, ValueError):
        return
    if not scrubbed.strip():
        return

    # 4a. documented total-time proof against the descriptor's own threshold
    tp = rule.get("time_proof") or {}
    threshold = None
    minutes_seen = set()
    try:
        thr_regex = tp.get("threshold_regex")
        if thr_regex:
            m = re.search(str(thr_regex), em_desc)
            if m is not None:
                threshold = int(m.group(1))
        for pat in (tp.get("capture_regexes") or []):
            for m in re.finditer(str(pat), scrubbed):
                minutes_seen.add(int(m.group(1)))
    except (re.error, ValueError, IndexError, TypeError):
        return
    time_proof = False
    if len(minutes_seen) > 1:
        return  # ambiguous: more than one distinct documented time value
    if len(minutes_seen) == 1:
        if threshold is None:
            return  # documented time but no parseable descriptor threshold
        if sorted(minutes_seen)[0] >= threshold:
            time_proof = True

    # 4b. separate-problem sentence disjoint from every procedure's target
    sep_proof = False
    if not time_proof:
        sp = rule.get("separate_problem") or {}
        min_len = 3
        try:
            min_len = int(sp.get("min_token_len") or 3)
        except (TypeError, ValueError):
            min_len = 3
        lex_pats = []
        try:
            lex_pats = [re.compile(str(x))
                        for x in (sp.get("lexicon_regexes") or []) if str(x)]
        except re.error:
            return
        proc_tokens = set()
        for d in proc_descs:
            for tok in v._tokens(d):
                if (len(tok) >= min_len and tok.isalpha()
                        and tok not in v._DESC_STOPWORDS):
                    proc_tokens.add(v._stem(tok))
        for sent in re.split(r"[.!?\n]+", scrubbed):
            if sep_proof:
                continue
            if not any(p.search(sent) is not None for p in lex_pats):
                continue
            sent_stems = set(v._stem(t) for t in v._tokens(sent))
            if not (sent_stems & proc_tokens):
                sep_proof = True

    act = rule.get("action") or {}
    proc_list = ", ".join(sorted(set(proc_codes)))

    if time_proof or sep_proof:
        # keep the line untouched; optional REPORT-ONLY modifier check
        mod_regex = rule.get("verify_modifier_regex")
        if mod_regex:
            try:
                mod_pat = re.compile(str(mod_regex))
            except re.error:
                return
            mods = em_entry.get("modifiers") or []
            if not isinstance(mods, list):
                mods = []
            if not any(mod_pat.search(str(x)) is not None for x in mods):
                vmsg = str(act.get("verify_message")
                           or "E/M line {code} is separately identifiable "
                              "but carries no distinct-service modifier.")
                vmsg = vmsg.replace("{code}", em_code)
                vmsg = vmsg.replace("{procedures}", proc_list)
                vrec = str(act.get("verify_recommendation")
                           or "Confirm the distinct-service modifier on "
                              "{code}.")
                vrec = vrec.replace("{code}", em_code)
                v._add("INFO", em_code,
                       str(act.get("verify_category")
                           or "em_distinct_service_modifier_check"),
                       vmsg, vrec, denial_risk="LOW")
        return

    # no proof: suppress the E/M line (the template's only mutation)
    msg = str(act.get("message") or "")
    msg = msg.replace("{code}", em_code)
    msg = msg.replace("{desc}", em_desc)
    msg = msg.replace("{procedures}", proc_list)
    rec = str(act.get("recommendation") or "")
    rec = rec.replace("{code}", em_code)
    rec = rec.replace("{procedures}", proc_list)
    v._non_billable_codes_to_suppress.add(em_code)
    v._add(str(act.get("severity") or "WARNING"), em_code,
           str(act.get("category") or "em_bundled_into_minor_procedure"),
           msg, rec,
           denial_risk=str(act.get("denial_risk") or "HIGH"))
