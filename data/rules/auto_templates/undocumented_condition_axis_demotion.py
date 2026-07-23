import re

TEMPLATE_NAME = "undocumented_condition_axis_demotion"

SCHEMA_DOC = """
Template: undocumented_condition_axis_demotion

Purpose: presence/primacy arbitration for billed ICD lines whose OFFICIAL
descriptor names condition-axis terms the note never affirmatively
documents. If the note contains even ONE affirmative (negation-scrubbed)
sentence naming the axis, the template does NOTHING. Otherwise every
selected line is DEMOTED (never deleted): moved from the icd_codes array
to coding_result['supporting_conditions'] with needs_review=True and a
review_reason. If a demoted line was flagged type='primary' and no billed
primary remains, the template promotes the lowest-(code, description)-
sorting remaining billed ICD line whose descriptor's non-stopword tokens
(minus the declared laterality/qualifier drop_words) are all covered by
the note's evidence word set, setting type='primary' with
needs_review=True. If no billed line qualifies, it emits a WARNING that
the documented first-listed diagnosis is missing from the claim.

Rule JSON fields:

  applies_to (required, object):
    array      : must be the literal string "icd_codes".
    code_regex : broad STRUCTURAL regex (no literal medical codes). It is
                 tried against both the dotted and undotted forms of each
                 billed code via re.search. Unparseable regex => no action.

  descriptor (required, object) — grammar that selects the family within
  the regex matches, applied to the code's OFFICIAL descriptor (v.db
  lookup; falls back to the claim line's own description; if neither
  exists the line is skipped). All matching is lowercase substring.
    requires_all : list of phrases that must ALL appear in the descriptor.
    requires_any : list of phrases of which at least ONE must appear.
    excludes     : list of phrases (sibling axes) none of which may appear.
  At least one of requires_all / requires_any must be nonempty, otherwise
  the template refuses to act (regex alone must never select).

  condition_stems (required, nonempty list of lowercase phrases): the note
  vocabulary for the descriptor's OWN condition axis (e.g. "flexor
  tendon", "spontaneous ruptur"). Each phrase is matched as a substring
  of each negation-scrubbed, lowercased note sentence. If ANY sentence
  outside the exclusion contexts contains ANY stem, the condition is
  considered documented and the template does nothing. More stems = more
  ways for documentation to be found = MORE conservative. Empty list or
  missing note text => no action.

  exclusion_contexts (optional, list of {label, regex}): sentences
  matching any of these case-insensitive regexes do not count as
  affirmative documentation (e.g. rule-out / differential language). An
  unparseable regex => no action (conservative).

  promotion (optional, object):
    drop_words : laterality/qualifier grammar words (e.g. left, right,
                 bilateral, unspecified) dropped — along with the engine's
                 descriptor stopwords — before the fallback-promotion
                 coverage check of a candidate descriptor against the
                 note's evidence word set (tokens and their stems).

  action (optional, object) — reporting text. Placeholders {code}, {desc},
  {axis} (axis = the joined condition_stems) are substituted literally.
    severity, category, denial_risk        : for the demotion issue.
    message, recommendation                : demotion issue text.
    demote_review_reason                   : review_reason on demoted lines.
    promote_message, promote_review_reason : fallback-promotion text
                                             (promotion is reported INFO/LOW).
    missing_primary_message,
    missing_primary_recommendation         : WARNING/HIGH emitted when the
                                             primary was demoted and no
                                             billed line qualifies.

Guarantees:
  - Deterministic: line selection by index, sentence scan in document
    order, promotion candidates sorted by (code, description).
  - Single-axis: only the selected family's presence and (when required)
    exactly one promoted line's type flag are written. No other line,
    modifier, unit, or array is touched.
  - Conservative: unparseable regexes, empty condition_stems, missing note
    text, missing descriptor grammar, or ANY affirmative documentation of
    the axis => no action. ICD lines are never deleted, only demoted to
    supporting_conditions with needs_review=True.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    applies = rule.get("applies_to") or {}
    if str(applies.get("array") or "") != "icd_codes":
        return
    pattern = str(applies.get("code_regex") or "")
    if not pattern:
        return
    try:
        rx = re.compile(pattern)
    except re.error:
        return

    stems = []
    for s in (rule.get("condition_stems") or []):
        t = str(s).strip().lower()
        if t:
            stems.append(t)
    if not stems:
        return

    text = str(note_full_text or "")
    if not text.strip():
        return

    desc_spec = rule.get("descriptor") or {}
    req_all = [str(p).lower() for p in (desc_spec.get("requires_all") or []) if str(p).strip()]
    req_any = [str(p).lower() for p in (desc_spec.get("requires_any") or []) if str(p).strip()]
    excludes = [str(p).lower() for p in (desc_spec.get("excludes") or []) if str(p).strip()]
    if not req_all and not req_any:
        return

    def official_description(code):
        info = None
        try:
            info = v.db.validate_icd10(code)
        except Exception:
            info = None
        if not isinstance(info, dict):
            table = v.db.icd10 or {}
            candidate = table.get(str(code).replace(".", ""))
            if isinstance(candidate, dict):
                info = candidate
        if isinstance(info, dict):
            return str(info.get("long_description") or info.get("description") or "")
        return ""

    # --- select billed lines by structural regex + descriptor grammar ---
    sel_idx = set()
    for i, entry in enumerate(icd):
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "")
        if not code:
            continue
        undotted = code.replace(".", "")
        if not (rx.search(code) or rx.search(undotted)):
            continue
        desc = official_description(code)
        if not desc.strip():
            desc = str(entry.get("description") or "")
        if not desc.strip():
            continue  # cannot verify the axis -> conservative skip
        d = desc.lower()
        ok = True
        for p in req_all:
            if p not in d:
                ok = False
                break
        if ok and req_any:
            if not any(p in d for p in req_any):
                ok = False
        if ok:
            for p in excludes:
                if p in d:
                    ok = False
                    break
        if ok:
            sel_idx.add(i)
    if not sel_idx:
        return

    # --- evidence: any affirmative sentence naming the axis => do nothing ---
    evidence_words, scrubbed = v._note_evidence(text)
    if evidence_words is None:
        evidence_words = set()
    scrubbed = str(scrubbed or "")

    ctx_rx = []
    for ctx in (rule.get("exclusion_contexts") or []):
        if not isinstance(ctx, dict):
            continue
        pat = str(ctx.get("regex") or "")
        if not pat:
            continue
        try:
            ctx_rx.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            return  # unparseable exclusion context -> no action

    for sent in re.split(r"[.!?\n]+", scrubbed):
        s = sent.strip().lower()
        if not s:
            continue
        if any(x.search(s) for x in ctx_rx):
            continue
        for stem in stems:
            if stem in s:
                return  # the axis IS documented -> conservative no-op

    # --- demote every selected line (never delete) ---
    action = rule.get("action") or {}
    severity = str(action.get("severity") or "WARNING")
    category = str(action.get("category") or "undocumented_condition_axis")
    denial = str(action.get("denial_risk") or "HIGH")
    axis = ", ".join(stems)

    def fill(tmpl, code, desc):
        out = str(tmpl)
        out = out.replace("{code}", code)
        out = out.replace("{desc}", desc)
        out = out.replace("{axis}", axis)
        return out

    msg_tmpl = action.get("message") or (
        "AUTO-CORRECTED: {code} ('{desc}') demoted to supporting_conditions "
        "— the descriptor's condition axis ({axis}) is never affirmatively "
        "documented in the note, so the code is unsupported as a billed "
        "diagnosis.")
    rec_tmpl = action.get("recommendation") or (
        "Bill the diagnosis the note actually documents; restore {code} only "
        "if the provider documents the condition its descriptor names.")
    demote_reason_tmpl = action.get("demote_review_reason") or (
        "Confirm removal of {code}: its descriptor's condition axis ({axis}) "
        "is not documented anywhere in the note.")

    supporting = coding_result.get("supporting_conditions")
    if not isinstance(supporting, list):
        supporting = []
        coding_result["supporting_conditions"] = supporting

    had_primary = False
    first_demoted_code = ""
    kept = []
    for i, entry in enumerate(icd):
        if i in sel_idx and isinstance(entry, dict):
            code = str(entry.get("code") or "")
            desc = official_description(code)
            if not desc.strip():
                desc = str(entry.get("description") or "")
            if str(entry.get("type") or "").lower() == "primary":
                had_primary = True
            if not first_demoted_code:
                first_demoted_code = code
            moved = dict(entry)
            moved["needs_review"] = True
            moved["review_reason"] = fill(demote_reason_tmpl, code, desc)
            supporting.append(moved)
            v._add(severity, code, category,
                   fill(msg_tmpl, code, desc),
                   fill(rec_tmpl, code, desc),
                   denial_risk=denial)
        else:
            kept.append(entry)
    icd[:] = kept

    # --- primary-flag repair, only if the demoted line was the primary ---
    if not had_primary:
        return
    for entry in icd:
        if isinstance(entry, dict) and str(entry.get("type") or "").lower() == "primary":
            return  # another billed primary remains -> nothing to repair

    promo = rule.get("promotion") or {}
    drop = set()
    for w in (promo.get("drop_words") or []):
        t = str(w).strip().lower()
        if t:
            drop.add(t)
            drop.add(v._stem(t))
    stop = v._DESC_STOPWORDS

    candidates = []
    for entry in icd:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "")
        if not code:
            continue
        desc = official_description(code)
        if not desc.strip():
            desc = str(entry.get("description") or "")
        if not desc.strip():
            continue
        covered = True
        for tok in sorted(v._tokens(desc)):
            if tok in stop or tok in drop or v._stem(tok) in drop:
                continue
            if tok in evidence_words or v._stem(tok) in evidence_words:
                continue
            covered = False
            break
        if covered:
            candidates.append((code, desc, entry))

    if candidates:
        candidates.sort(key=lambda c: (c[0], c[1]))
        code, desc, best = candidates[0]
        best["type"] = "primary"
        best["needs_review"] = True
        promote_reason_tmpl = action.get("promote_review_reason") or (
            "Confirm {code} as the first-listed diagnosis; it was promoted "
            "deterministically after an undocumented primary was demoted.")
        best["review_reason"] = fill(promote_reason_tmpl, code, desc)
        promote_msg_tmpl = action.get("promote_message") or (
            "AUTO-CORRECTED: {code} ('{desc}') promoted to primary — the "
            "flagged primary was demoted as undocumented, and this is the "
            "lowest-sorting billed diagnosis whose descriptor is fully "
            "covered by the note's documentation.")
        v._add("INFO", code, category,
               fill(promote_msg_tmpl, code, desc),
               fill(rec_tmpl, code, desc),
               denial_risk="LOW")
    else:
        miss_msg = str(action.get("missing_primary_message") or (
            "The flagged primary diagnosis was demoted as undocumented and no "
            "remaining billed diagnosis is fully supported by the note — the "
            "documented first-listed diagnosis is missing from the claim."))
        miss_rec = str(action.get("missing_primary_recommendation") or (
            "Add the diagnosis the note documents as the reason for the "
            "encounter and flag it primary."))
        v._add("WARNING", first_demoted_code, category, miss_msg, miss_rec,
               denial_risk="HIGH")
