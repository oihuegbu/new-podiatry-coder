import re

TEMPLATE_NAME = "nec_residual_etiology_arbitration"

SCHEMA_DOC = """
nec_residual_etiology_arbitration -- presence arbitration for ICD-10-CM
codes whose Tabular CATEGORY title carries the classification's own
residual marker ('not elsewhere classified' or descriptor-grammar
equivalent).

Mechanic (deterministic):
 1. For every billed ICD entry matching applies_to.code_regex, look up
    the 3-character category's Tabular title. If no residual_markers
    phrase appears in that lowercased title, the entry is out of scope.
 2. Derive condition stems from the category title: tokenize, drop
    descriptor stopwords, drop the residual-marker words, drop the stems
    of condition_qualifier_words (chronicity / pressure-axis / site /
    laterality qualifiers the rule declares). If nothing remains,
    DO NOTHING (conservative no-op).
 3. If any surviving condition stem appears in the note's
    negation-scrubbed evidence word set, the condition is affirmatively
    named by its own term -> the code stands (DO NOTHING).
 4. Otherwise scan the negation-scrubbed sentences against
    etiology_classes. A class fires when its trigger_regex matches a
    sentence (and, when problem_regex is set, the SAME sentence also
    matches problem_regex, anchoring the mechanism word to the
    presenting problem rather than an incidental mention). If zero
    classes fire, or more than one distinct class fires (ambiguous),
    DO NOTHING.
 5. Exactly one class fired and the condition is never named: the NEC
    line is demoted. ICD lines are never deleted -- the entry is moved
    to coding_result['supporting_conditions'] with needs_review=True and
    a review_reason. Then:
      (a) if another billed ICD entry's UNdotted code matches the firing
          class's target_chapter_regex, that entry carries the claim; if
          the demoted NEC line was primary, the lowest-sorting such
          entry is promoted to primary (needs_review=True, review_reason
          attached); or
      (b) if none exists, an issue with severity WARNING and the
          configured category is emitted stating that the mandated
          etiology-chapter replacement code is MISSING -- the claim is
          never silently left without surfacing the required
          replacement (including its 7th-character requirement, which
          the rule's recommendation text should spell out).

Rule JSON fields:
  id            (string, kebab-case)  rule identifier
  template      (string)              must equal
                                      "nec_residual_etiology_arbitration"
  enabled       (bool)
  authority     (string)  governing citation (prose; MAY name codes)
  applies_to    (object, required)
      array       must be "icd_codes" (only ICD entries have Tabular
                  categories; anything else makes the rule a no-op)
      code_regex  broad chapter-level STRUCTURAL regex selecting entries
                  to test (letter + digits shape). NEVER a literal full
                  code.
  residual_markers (array[string], required, lowercase)
      phrases searched verbatim in the lowercased category title, e.g.
      the classification's own residual wording. Their words are also
      excluded from the condition-stem derivation.
  condition_qualifier_words (array[string], optional)
      title words that are qualifier axes (chronicity, pressure axis,
      site, laterality), NOT the condition noun itself. Their stems are
      dropped when deriving condition stems. Hyphenated title words are
      split into tokens, so list each token separately. Accidentally
      listing the condition noun empties the stem set and neutralizes
      the rule (safe no-op, never a wrong action).
  etiology_classes (array[object], required, at least one)
      label                (string, required) human label for messages
      trigger_regex        (string, required) lowercase regex over
                           negation-scrubbed sentences naming the
                           etiology mechanism vocabulary
      target_chapter_regex (string, required) STRUCTURAL regex the
                           UNdotted code of the mandated etiology
                           chapter must match. NEVER a literal full
                           code.
      Classes are evaluated in sorted-label order; if more than one
      class fires the template does nothing.
  problem_regex (string, optional)
      when present, a trigger only fires inside a sentence that also
      matches this regex (e.g. wound/lesion vocabulary), so an
      immunization-history or unrelated mention cannot trigger.
  action (object, required)
      severity, category, denial_risk   issue metadata; the
                                        missing-replacement branch is
                                        always escalated to WARNING
      message_promoted / message_missing (strings) issue text.
          Placeholders: {code} {desc} {etiology} {target_chapter}
          {promoted}
      recommendation_promoted / recommendation_missing (strings)
      review_reason_suppressed / review_reason_promoted (strings)
          attached to the mutated entries (same placeholders)

Conservatism guarantees: missing/empty Tabular title, empty derived
condition-stem set, condition named anywhere in the scrubbed note, zero
or multiple firing etiology classes, or any unparseable regex all result
in NO action.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    def compile_pat(pattern):
        if not isinstance(pattern, str) or not pattern:
            return None
        try:
            return re.compile(pattern)
        except re.error:
            return None

    def fmt(text, fields):
        if not isinstance(text, str) or not text:
            return ""
        try:
            return text.format(**fields)
        except (KeyError, IndexError, ValueError):
            return text

    if not isinstance(rule, dict):
        return
    applies = rule.get("applies_to") or {}
    if not isinstance(applies, dict) or applies.get("array") != "icd_codes":
        return
    code_pat = compile_pat(applies.get("code_regex"))
    if code_pat is None:
        return

    markers = []
    for m in rule.get("residual_markers") or []:
        if isinstance(m, str) and m.strip():
            markers.append(m.strip().lower())
    if not markers:
        return
    marker_stems = set()
    for m in markers:
        for t in v._tokens(m):
            marker_stems.add(v._stem(t))

    qual_stems = set()
    for w in rule.get("condition_qualifier_words") or []:
        if isinstance(w, str):
            for t in v._tokens(w):
                qual_stems.add(v._stem(t))

    classes = []
    for ec in rule.get("etiology_classes") or []:
        if not isinstance(ec, dict):
            continue
        label = ec.get("label")
        trig = compile_pat(ec.get("trigger_regex"))
        targ = compile_pat(ec.get("target_chapter_regex"))
        if not isinstance(label, str) or not label or trig is None or targ is None:
            continue
        classes.append((label, trig, targ))
    if not classes:
        return
    classes.sort(key=lambda c: c[0])

    problem_pat = None
    if rule.get("problem_regex") is not None:
        problem_pat = compile_pat(rule.get("problem_regex"))
        if problem_pat is None:
            return

    note_words, scrubbed = v._note_evidence(note_full_text or "")
    sentences = []
    for s in re.split(r"[.;!?\n]+", scrubbed or ""):
        s = s.strip()
        if s:
            sentences.append(s)
    if not sentences:
        return

    fired = []
    for label, trig, targ in classes:
        for s in sentences:
            if trig.search(s) and (problem_pat is None or problem_pat.search(s)):
                fired.append((label, targ))
                break
    if len(fired) != 1:
        return
    etiology_label, target_pat = fired[0]

    action = rule.get("action") or {}
    severity = action.get("severity") or "WARNING"
    category = action.get("category") or "nec_etiology_conflict"
    denial_risk = action.get("denial_risk") or "MEDIUM"

    entries = []
    for e in list(icd):
        if not isinstance(e, dict):
            continue
        code = str(e.get("code") or "")
        if code and code_pat.search(code):
            entries.append((code, e))
    entries.sort(key=lambda p: p[0])

    for code, entry in entries:
        undotted = code.replace(".", "")
        cat3 = undotted[:3]
        title = v.store.icd10_tabular_description(cat3) or ""
        title = str(title).lower()
        if not title:
            continue
        has_marker = False
        for m in markers:
            if m in title:
                has_marker = True
                break
        if not has_marker:
            continue

        cond_stems = set()
        for t in v._tokens(title):
            if t in v._DESC_STOPWORDS:
                continue
            st = v._stem(t)
            if st in marker_stems or st in qual_stems:
                continue
            cond_stems.add(st)
        if not cond_stems:
            continue
        if cond_stems & note_words:
            continue

        idx = -1
        for i, e in enumerate(icd):
            if e is entry:
                idx = i
                break
        if idx < 0:
            continue

        candidates = []
        for other in icd:
            if other is entry or not isinstance(other, dict):
                continue
            ocode = str(other.get("code") or "")
            if ocode and target_pat.search(ocode.replace(".", "")):
                candidates.append((ocode, other))
        candidates.sort(key=lambda p: p[0])

        was_primary = str(entry.get("type") or "").lower() == "primary"
        fields = {
            "code": code,
            "desc": str(entry.get("description") or ""),
            "etiology": etiology_label,
            "target_chapter": target_pat.pattern,
            "promoted": candidates[0][0] if candidates else "",
        }

        icd.pop(idx)
        entry["type"] = "supporting"
        entry["needs_review"] = True
        entry["review_reason"] = fmt(
            action.get("review_reason_suppressed")
            or "Demoted per the residual (NEC) convention: the note documents a {etiology} and never names the residual condition by its own term; verify against the record before final submission.",
            fields)
        supp = coding_result.get("supporting_conditions")
        if not isinstance(supp, list):
            supp = []
            coding_result["supporting_conditions"] = supp
        supp.append(entry)

        if candidates:
            promoted = candidates[0][1]
            if was_primary:
                promoted["type"] = "primary"
            promoted["needs_review"] = True
            promoted["review_reason"] = fmt(
                action.get("review_reason_promoted")
                or "Carries the claim in place of the demoted residual (NEC) line; confirm the code and any required 7th character against the documentation.",
                fields)
            v._add(severity, code, category,
                   fmt(action.get("message_promoted")
                       or "AUTO-CORRECTED: {code} ('{desc}') demoted -- its Tabular category is a residual (not-elsewhere-classified) bucket, the note never names the condition by its own term, and the documented {etiology} places the condition in the mandated etiology chapter; {promoted}, already on the claim, carries it.",
                       fields),
                   fmt(action.get("recommendation_promoted")
                       or "Confirm the promoted etiology-chapter code and any required 7th character against the documentation.",
                       fields),
                   denial_risk=denial_risk)
        else:
            v._add("WARNING", code, category,
                   fmt(action.get("message_missing")
                       or "AUTO-CORRECTED: {code} ('{desc}') demoted -- its Tabular category is a residual (not-elsewhere-classified) bucket, the note never names the condition by its own term, and the documented {etiology} means the classification codes it elsewhere; NO replacement etiology-chapter code is on the claim.",
                       fields),
                   fmt(action.get("recommendation_missing")
                       or "Add the mandated etiology-chapter code for the documented {etiology}, with any required 7th character, and re-bill.",
                       fields),
                   denial_risk=denial_risk)
