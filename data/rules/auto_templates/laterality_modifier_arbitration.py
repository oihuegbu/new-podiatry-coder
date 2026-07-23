import re

TEMPLATE_NAME = "laterality_modifier_arbitration"

SCHEMA_DOC = """
TEMPLATE laterality_modifier_arbitration — single-axis arbitration of the
anatomic laterality modifier class (generic side modifiers such as
right/left/bilateral) on procedure lines. It mutates ONLY the laterality
modifier class on ONLY the lines the rule selects; it never touches units,
other modifiers, other lines, or any other array.

ELIGIBILITY (all must hold, else the line is skipped):
 1. line's code matches applies_to.code_regex (broad structural regex —
    never a literal code list);
 2. the line's OFFICIAL reference descriptor (v.db lookup, not the claim's
    own description text) contains at least one word from
    paired_anatomy_terms (stem-matched);
 3. v.store.bilat_surg(code) — the CMS Physician Fee Schedule bilateral-
    surgery indicator, THE authority on laterality-modifier applicability —
    returns a value listed in bilat_surg_indicators. Never assume
    applicability from a code's section or descriptor alone.

EVIDENCE: the note is split into sentences; a sentence QUALIFIES when its
stemmed tokens intersect the descriptor's matched anatomy words OR the
operative_terms vocabulary, and it does not match any exclusion_contexts
regex (non-operative contexts: vascular/neuro exam findings, tourniquet
placement, history sections). Word-bounded side lexemes from sides.*.words
are collected across all qualifying sentences. If EXACTLY ONE side key is
found, the line's modifier set is arbitrated to include exactly that
side's modifier (added when absent; a conflicting member of the configured
class is removed). Zero sides, or two-plus distinct side keys, means the
evidence is ambiguous and the template does NOTHING.

ALREADY-SIDED SAFETY (reference-data lookup, never a modifier list): for
every existing modifier OUTSIDE the configured class, the template asks
v.store.modifier_laterality(mod). If a site-specific modifier already
denotes the documented side, the generic modifier is NOT stacked on top
(only a conflicting class member would be dropped). If any such modifier
denotes a DIFFERENT side than documented, the line is left untouched
entirely — conflicts are never resolved by guessing.

RULE FIELDS
  id (kebab-case), template = "laterality_modifier_arbitration",
  enabled (bool), authority (prose citation of the governing source).
  applies_to (required): { "array": "cpt_codes" | "hcpcs_codes",
      "code_regex": broad structural regex — NO literal code lists }.
  bilat_surg_indicators (required): list of CMS PFS bilateral-surgery
      indicator strings that make a laterality modifier applicable
      (typically ["1","3"]). Empty/missing disables the rule.
  paired_anatomy_terms (required): fixed human-anatomy lexicon words
      (e.g. heel, calcaneus, ankle, foot, toe, knee). This is anatomy
      vocabulary — a language fact — never a curated code list. The
      descriptor must name one of these for the line to be eligible;
      the words that matched the descriptor also anchor sentence
      qualification.
  operative_terms (optional): operative-action vocabulary (incision,
      excised, resected, repair, prepped...) that also qualifies a
      sentence as procedure-site evidence.
  descriptor_excludes (optional): lowercase substrings; a descriptor
      containing any of them makes the line ineligible.
  exclusion_contexts (optional): [{"label": str, "regex": str}] —
      sentences matching any regex are ignored as evidence (exam
      findings, tourniquet, history).
  sides (required): { side_key: {"words": [lexemes], "modifier": "XX"} }.
      words are word-bounded lowercase lexemes ('right','rt',...).
      modifier is the ARBITRATION TARGET VALUE for that side — an
      attribute value the rule writes, not a code selector; the template
      cross-checks it against v.store.modifier_laterality at runtime.
  raw_text (optional bool, default false): false = evidence is collected
      from the negation-scrubbed note text; true = raw note text.
  action (required): { severity, category, denial_risk, message,
      recommendation } — message/recommendation may use placeholders
      {code}, {modifier}, {side}. An issue is reported ONLY when the
      modifier list actually changed.

GUARANTEES: deterministic (sides and words iterated sorted; sentence
order is note order); conservative (any ambiguity => no mutation);
single-axis (only the configured laterality class on selected lines is
ever written; the relative order of untouched modifiers is preserved and
the target is appended last when added).
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    applies = rule.get("applies_to") or {}
    array = applies.get("array") or ""
    if array == "cpt_codes":
        lines = cpt
        lookup = v.db.validate_cpt
    elif array == "hcpcs_codes":
        lines = hcpcs
        lookup = v.db.validate_hcpcs
    else:
        return

    rx = applies.get("code_regex")
    if not rx:
        return
    try:
        code_re = re.compile(str(rx))
    except Exception:
        return

    # --- side lexeme patterns and arbitration targets (sorted: determinism)
    sides_cfg = rule.get("sides") or {}
    side_pats = []
    for key in sorted(sides_cfg):
        cfg = sides_cfg.get(key) or {}
        words = sorted({str(w).lower().strip() for w in (cfg.get("words") or []) if w})
        mod = str(cfg.get("modifier") or "").strip()
        if not words or not mod:
            continue
        try:
            pat = re.compile("\\b(?:" + "|".join(re.escape(w) for w in words) + ")\\b")
        except Exception:
            continue
        side_pats.append((str(key), pat, mod))
    if not side_pats:
        return
    class_mods = {m for (_k, _p, m) in side_pats}

    # --- anatomy lexicon and operative vocabulary (stemmed)
    lex_stems = set()
    for w in rule.get("paired_anatomy_terms") or []:
        for t in v._tokens(str(w)):
            lex_stems.add(v._stem(t))
    if not lex_stems:
        return
    op_stems = set()
    for w in rule.get("operative_terms") or []:
        for t in v._tokens(str(w)):
            op_stems.add(v._stem(t))

    # --- exclusion context patterns
    excl = []
    for ctx in rule.get("exclusion_contexts") or []:
        if not isinstance(ctx, dict):
            continue
        crx = ctx.get("regex")
        if not crx:
            continue
        try:
            excl.append(re.compile(str(crx)))
        except Exception:
            continue

    # --- note text -> qualifying-sentence corpus
    text = str(note_full_text or "")
    if not text:
        return
    if not rule.get("raw_text"):
        try:
            ev = v._note_evidence(text)
            text = str(ev[1])
        except Exception:
            pass
    low = text.lower()
    sent_data = []
    for s in re.split("[\\.\\n;!?]+", low):
        s2 = s.strip()
        if not s2:
            continue
        if any(p.search(s2) for p in excl):
            continue
        stems = set()
        for t in v._tokens(s2):
            stems.add(v._stem(t))
        sent_data.append((s2, stems))
    if not sent_data:
        return

    allowed_ind = {str(a).strip() for a in (rule.get("bilat_surg_indicators") or []) if str(a).strip()}
    if not allowed_ind:
        return
    desc_excl = [str(x).lower() for x in (rule.get("descriptor_excludes") or []) if x]

    act = rule.get("action") or {}

    for entry in lines:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "").strip()
        if not code or not code_re.search(code):
            continue

        # official descriptor — reference data, never the claim's own text
        info = lookup(code)
        if not info or not isinstance(info, dict):
            continue
        desc = str(info.get("description") or info.get("long_description") or "")
        if not desc:
            continue
        dlow = desc.lower()
        if any(x in dlow for x in desc_excl):
            continue
        d_stems = set()
        for t in v._tokens(desc):
            d_stems.add(v._stem(t))
        matched = d_stems & lex_stems
        if not matched:
            continue

        # CMS bilateral-surgery indicator: THE authority on applicability
        ind = v.store.bilat_surg(code)
        if ind is None or str(ind).strip() not in allowed_ind:
            continue

        # collect documented sides across qualifying sentences
        found = set()
        for s2, stems in sent_data:
            if not (stems & matched) and not (stems & op_stems):
                continue
            for key, pat, _m in side_pats:
                if pat.search(s2):
                    found.add(key)
        if len(found) != 1:
            continue  # zero or conflicting sides: never guess
        side_key = sorted(found)[0]
        target = None
        for k, _p, m in side_pats:
            if k == side_key:
                target = m
        if not target:
            continue
        target_lat = v.store.modifier_laterality(target)

        mods = list(entry.get("modifiers") or [])

        # already-sided / conflict check on NON-class modifiers (data lookup)
        already_sided = False
        conflict = False
        for m in mods:
            if m in class_mods:
                continue
            lat = v.store.modifier_laterality(str(m))
            if lat is None:
                continue
            if target_lat is not None and lat == target_lat:
                already_sided = True
            else:
                conflict = True
        if conflict:
            continue

        # single-axis mutation: only the configured laterality class
        new_mods = [m for m in mods if m not in class_mods or m == target]
        if target not in new_mods and not already_sided:
            new_mods.append(target)
        if new_mods == mods:
            continue
        entry["modifiers"] = new_mods

        msg = str(act.get("message") or "")
        msg = msg.replace("{code}", code).replace("{modifier}", target).replace("{side}", side_key)
        rec = str(act.get("recommendation") or "")
        rec = rec.replace("{code}", code).replace("{modifier}", target).replace("{side}", side_key)
        v._add(
            act.get("severity") or "WARNING",
            code,
            act.get("category") or "laterality_modifier_arbitration",
            msg or ("Laterality modifier arbitrated to " + target + " for " + code),
            rec or "Confirm the anatomic modifier matches the documented operative side",
            denial_risk=act.get("denial_risk") or "MEDIUM",
        )
