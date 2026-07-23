import re

TEMPLATE_NAME = "primary_flag_encounter_reason_arbitration"

SCHEMA_DOC = """
TEMPLATE primary_flag_encounter_reason_arbitration

PURPOSE
  Single-axis arbitration of the type='primary' flag among billed ICD lines
  when runs disagree which documented diagnosis is first-listed. Implements
  ICD-10-CM guidelines IV.A.1 / IV.G deterministically: the first-listed
  diagnosis is the condition the note's own assessment (or operative-
  procedure) statement names FIRST. The template writes ONLY the 'type'
  field of the candidate lines its rule selects. It never adds, removes, or
  reorders lines and never touches codes, modifiers, units, or any other
  attribute of any line.

RULE FIELDS
  applies_to (required): {"array": "icd_codes", "code_regex": "..."}
      array MUST be "icd_codes". code_regex is a broad STRUCTURAL regex
      (e.g. a chapter-letter prefix) -- never a literal code list.
  families (required, non-empty list): each family is
      {"label": str,
       "descriptor_terms": [lowercase words; a billed line joins the family
           when any term is a word-boundary PREFIX match inside the line's
           reference-data descriptor (v.db lookup, falling back to the
           line's own description)],
       "condition_stems": [lowercase note-vocabulary stems/phrases;
           word-boundary prefix-matched against negation-scrubbed note text
           to locate the family's condition in prose]}
      A line matching MORE THAN ONE family aborts the entire rule (no-op).
  assessment (required object): how to locate the encounter-reason text.
      label_regex (optional): matches an assessment/impression heading
          inside a paragraph; the text after the match is the assessment.
      plan_regex (optional): matches the plan/follow-up paragraph; when no
          labeled assessment is found, the paragraph IMMEDIATELY BEFORE the
          first plan match (provided it does not itself match plan_regex)
          is taken as the assessment. Engine-extracted assessment text,
          when the pipeline supplies it, is preferred over both heuristics.
      procedure_regex (optional): identifies the operative-procedure
          sentence (searched within the plan paragraph when plan_regex is
          given); used only as the FALLBACK focus text when no candidate
          appears in the assessment text.
  exclusion_contexts (optional list of {"label", "regex"}): note segments
      matching any regex (history sections, risk counseling, patient
      education) are removed before verifying that each candidate's
      condition is actually documented at this encounter.
  action (optional): severity, category, denial_risk, message,
      recommendation. Placeholders: {code} {desc} {role} {reason}.

MECHANIC
  1. candidates = billed icd lines matching code_regex AND exactly one
     family by descriptor grammar.
  2. Hard no-ops (never guess): any configured regex unparseable; fewer
     than 2 candidates; any candidate's stems absent from the
     exclusion-filtered, negation-scrubbed note; any line currently flagged
     primary that is NOT a candidate; no assessment or procedure focus text
     found; no candidate appearing in any focus text; a tie for the
     earliest mention.
  3. Otherwise, in the first focus text (assessment, else procedure
     sentence) where at least one candidate's stems appear, the candidate
     with the UNIQUE earliest character position is pinned type='primary';
     every other candidate is pinned type='secondary'. Lines already
     carrying the pinned value are left byte-identical and unreported.
  Deterministic: claim-list order and character positions only; no dict
  iteration affects the outcome.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v
    if not isinstance(rule, dict) or not isinstance(icd, list):
        return

    def rx(pattern):
        if not isinstance(pattern, str) or not pattern.strip():
            return None
        try:
            return re.compile(pattern, re.IGNORECASE)
        except Exception:
            return None

    applies = rule.get("applies_to")
    if not isinstance(applies, dict) or applies.get("array") != "icd_codes":
        return
    code_rx = rx(applies.get("code_regex"))
    if code_rx is None:
        return

    fam_cfg = rule.get("families")
    if not isinstance(fam_cfg, list) or not fam_cfg:
        return
    families = []
    for fam in fam_cfg:
        if not isinstance(fam, dict):
            return
        terms = [t.strip().lower() for t in (fam.get("descriptor_terms") or [])
                 if isinstance(t, str) and t.strip()]
        stems = [s.strip().lower() for s in (fam.get("condition_stems") or [])
                 if isinstance(s, str) and s.strip()]
        if not terms or not stems:
            return
        families.append((str(fam.get("label") or ""), terms, stems))

    excl = []
    for ctx in (rule.get("exclusion_contexts") or []):
        if not isinstance(ctx, dict):
            return
        crx = rx(ctx.get("regex"))
        if crx is None:
            return
        excl.append(crx)

    a_cfg = rule.get("assessment")
    if not isinstance(a_cfg, dict):
        return
    label_rx = None
    if a_cfg.get("label_regex") is not None:
        label_rx = rx(a_cfg.get("label_regex"))
        if label_rx is None:
            return
    plan_rx = None
    if a_cfg.get("plan_regex") is not None:
        plan_rx = rx(a_cfg.get("plan_regex"))
        if plan_rx is None:
            return
    proc_rx = None
    if a_cfg.get("procedure_regex") is not None:
        proc_rx = rx(a_cfg.get("procedure_regex"))
        if proc_rx is None:
            return

    full_text = note_full_text if isinstance(note_full_text, str) else ""

    def scrub(text):
        pair = None
        try:
            pair = v._note_evidence(text)
        except Exception:
            pair = None
        if isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], str):
            return pair[1].lower()
        return text.lower()

    def earliest(stems, text):
        best = None
        for s in stems:
            m = re.search("\\b" + re.escape(s), text)
            if m is not None and (best is None or m.start() < best):
                best = m.start()
        return best

    def descriptor_of(entry):
        desc = ""
        code = entry.get("code")
        if isinstance(code, str) and code.strip():
            info = None
            try:
                info = v.db.validate_icd10(code)
            except Exception:
                info = None
            if isinstance(info, dict):
                desc = info.get("long_description") or info.get("description") or ""
        if not desc:
            fallback = entry.get("description")
            desc = fallback if isinstance(fallback, str) else ""
        return desc.lower() if isinstance(desc, str) else ""

    # candidate selection: structural regex + exactly one descriptor family
    candidates = []
    for idx, entry in enumerate(icd):
        if not isinstance(entry, dict):
            continue
        code = entry.get("code")
        if not isinstance(code, str) or not code_rx.search(code):
            continue
        desc = descriptor_of(entry)
        if not desc:
            continue
        matched = []
        for label, terms, stems in families:
            hit = False
            for t in terms:
                if re.search("\\b" + re.escape(t), desc):
                    hit = True
                    break
            if hit:
                matched.append((label, stems))
        if len(matched) > 1:
            return  # ambiguous family mapping -> never guess
        if len(matched) == 1:
            candidates.append((idx, entry, matched[0][0], matched[0][1]))

    if len(candidates) < 2:
        return
    cand_idx = set(c[0] for c in candidates)
    for idx, entry in enumerate(icd):
        if isinstance(entry, dict) and entry.get("type") == "primary" and idx not in cand_idx:
            return  # a non-candidate holds the flag; arbitrating would conflict

    # every candidate's condition must be documented at this encounter
    segments = [s for s in re.split("(?<=[.!?])\\s+|\\n+", full_text) if s.strip()]
    kept = []
    for seg in segments:
        drop = False
        for crx in excl:
            if crx.search(seg):
                drop = True
                break
        if not drop:
            kept.append(seg)
    note_scrubbed = scrub(" ".join(kept))
    for idx, entry, label, stems in candidates:
        if earliest(stems, note_scrubbed) is None:
            return

    # locate the encounter-reason focus text
    paragraphs = [p.strip() for p in re.split("\\n\\s*\\n", full_text) if p.strip()]
    assessment_text = ""
    if isinstance(note_assessment_text, str) and note_assessment_text.strip():
        assessment_text = note_assessment_text.strip()
    if not assessment_text and label_rx is not None:
        for p in paragraphs:
            m = label_rx.search(p)
            if m is not None:
                tail = p[m.end():].strip()
                assessment_text = tail if tail else p
                break
    if not assessment_text and plan_rx is not None:
        for i, p in enumerate(paragraphs):
            if plan_rx.search(p):
                if i > 0 and not plan_rx.search(paragraphs[i - 1]):
                    assessment_text = paragraphs[i - 1]
                break
    procedure_text = ""
    if proc_rx is not None:
        for p in paragraphs:
            if plan_rx is not None and not plan_rx.search(p):
                continue
            sentences = [s for s in re.split("(?<=[.!?])\\s+", p) if s.strip()]
            for s in sentences:
                if proc_rx.search(s):
                    procedure_text = s
                    break
            if procedure_text:
                break

    focus_texts = []
    if assessment_text:
        focus_texts.append(scrub(assessment_text))
    if procedure_text:
        focus_texts.append(scrub(procedure_text))
    if not focus_texts:
        return

    winner_idx = None
    for ftext in focus_texts:
        scored = []
        for idx, entry, label, stems in candidates:
            pos = earliest(stems, ftext)
            if pos is not None:
                scored.append((pos, idx))
        if not scored:
            continue
        best = min(sc[0] for sc in scored)
        top = [sc for sc in scored if sc[0] == best]
        if len(top) != 1:
            return  # tie for earliest mention -> never guess
        winner_idx = top[0][1]
        break
    if winner_idx is None:
        return

    # single-axis mutation: ONLY the type field, ONLY matched candidates
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    severity = action.get("severity") or "WARNING"
    category = action.get("category") or "first_listed_diagnosis_arbitration"
    risk = action.get("denial_risk") or "MEDIUM"
    msg_t = action.get("message") or (
        "AUTO-CORRECTED: {code} ('{desc}') set to {role} -- {reason}.")
    rec_t = action.get("recommendation") or (
        "Confirm the first-listed diagnosis is the condition chiefly "
        "responsible for the encounter (ICD-10-CM guideline IV.A.1/IV.G).")

    for idx, entry, label, stems in candidates:
        target = "primary" if idx == winner_idx else "secondary"
        if entry.get("type") == target:
            continue
        entry["type"] = target
        code = str(entry.get("code") or "")
        desc = str(entry.get("description") or "")
        if target == "primary":
            reason = ("the note's assessment/procedure statement names this "
                      "condition first, making it the condition chiefly "
                      "responsible for the encounter")
        else:
            reason = ("another billed condition is named earlier in the "
                      "note's assessment/procedure statement and is "
                      "therefore first-listed")
        message = (msg_t.replace("{code}", code).replace("{desc}", desc)
                   .replace("{role}", target).replace("{reason}", reason))
        rec = (rec_t.replace("{code}", code).replace("{desc}", desc)
               .replace("{role}", target))
        v._add(severity, code, category, message, rec, denial_risk=risk)
