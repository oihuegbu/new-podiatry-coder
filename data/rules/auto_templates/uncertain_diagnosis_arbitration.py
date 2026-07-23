import re

TEMPLATE_NAME = 'uncertain_diagnosis_arbitration'

SCHEMA_DOC = '''
Template: uncertain_diagnosis_arbitration
=========================================
Presence/primacy arbitration for outpatient ICD lines documented ONLY as
uncertain, per ICD-10-CM guideline IV.H (uncertain diagnoses are not coded
in the outpatient setting; code to the highest degree of certainty, such
as the presenting symptom).

Deterministic mechanic:
 1. Every negation-scrubbed note sentence containing a condition_stems
    entry (lowercase substring match) is collected.
 2. If there are ZERO such sentences, or ANY such sentence matches no
    uncertainty_lexicon regex (i.e. at least one certain assertion of the
    condition exists), the template does NOTHING.
 3. Otherwise every billed ICD line selected by applies_to.code_regex AND
    the family_descriptor grammar is DEMOTED: moved from icd_codes into
    coding_result['supporting_conditions'] with needs_review=True and a
    review_reason (ICD lines are never deleted), and an issue is reported
    per demoted line.
 4. If a demoted line carried type='primary' and no billed primary
    remains: the lowest-(code, description)-sorting billed ICD line
    matching symptom_fallback whose required descriptor tokens the
    scrubbed note documents is promoted (type='primary',
    needs_review=True). If no such line is billed, a WARNING issue is
    emitted stating the guideline-mandated symptom / first-listed code is
    missing.
 5. Any unparseable regex, empty condition_stems, missing/empty
    symptom_fallback selector, or missing note text => no action at all
    (all config is validated BEFORE any mutation).
Single-axis guarantee: only the selected ICD family's presence and the
promoted symptom line's type flag are ever written. Other lines, other
attributes, CPT/HCPCS lines, units and modifiers are never touched.

Rule JSON fields
----------------
id            : kebab-case rule id.
template      : must equal 'uncertain_diagnosis_arbitration'.
enabled       : true.
authority     : prose citation of the governing guideline. Prose fields
                may mention codes; selecting fields may NOT.
applies_to    : {'array': 'icd_codes', 'code_regex': '<BROAD structural
                regex>'}. The regex is tried against the billed code both
                dotted and undotted. Use a broad shape (letter + digits),
                never a literal code.
family_descriptor : descriptor grammar narrowing the family among lines
                passing code_regex. Object with 'requires_all' (lowercase
                substrings that must ALL appear in the line descriptor)
                and/or 'requires_any' (at least one must appear). At
                least one list must be non-empty. The descriptor comes
                from the line itself, falling back to the reference DB.
condition_stems : non-empty list of lowercase NOTE-vocabulary stems or
                phrases naming the condition (clinical note terms may
                differ from the official descriptor, e.g. 'fasciitis'
                vs 'fibromatosis'). Substring match against scrubbed
                lowercase sentences.
uncertainty_lexicon : non-empty list of {'label': str, 'regex': str}
                drawn from the guideline's own words plus standard
                clinical equivalents: probable, suspected, questionable,
                rule out, r/o, versus, vs, differential, working
                diagnosis, compatible with, consistent with, likely.
                Every regex must compile or the rule does nothing.
symptom_fallback : selector for the mandated symptom line to promote.
                At least one of:
                  'code_regex' - broad structural regex (dotted and
                      undotted forms tried);
                  'descriptor_requires_any' - list of stems; a candidate
                      matches when any of its descriptor TOKENS equals or
                      starts with a listed stem (token-prefix, so 'pain'
                      does not match 'sprain').
                Optional 'note_evidence': {'min_token_len': int (default
                4; shorter descriptor tokens are never required),
                'ignore_tokens': [laterality/qualifier grammar words
                never required in the note]}. A candidate qualifies only
                if EVERY remaining non-stopword descriptor token (or its
                stem) appears in the note evidence word set. Ties resolve
                by (code, description) sort - deterministic.
action        : {'severity' (default WARNING), 'category',
                'denial_risk' (default MEDIUM), 'message',
                'recommendation', 'missing_primary_message',
                'promotion_message' (optional; INFO issue when set),
                'review_reason_demote', 'review_reason_promote'}.
                Placeholders in all message/review strings: {code} {desc}
                {labels} (sorted uncertainty labels found); promotion
                strings also receive {target}.
'''


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v
    if not isinstance(rule, dict) or not isinstance(icd, list):
        return
    if not isinstance(coding_result, dict):
        return

    def as_str(x):
        if isinstance(x, str):
            return x
        return ''

    def compile_rx(src):
        if not isinstance(src, str) or not src:
            return None
        try:
            return re.compile(src)
        except re.error:
            return None

    applies = rule.get('applies_to')
    if not isinstance(applies, dict) or applies.get('array') != 'icd_codes':
        return
    code_rx = compile_rx(applies.get('code_regex'))
    if code_rx is None:
        return

    fam_cfg = rule.get('family_descriptor')
    if not isinstance(fam_cfg, dict):
        return
    req_all = [s.strip().lower() for s in (fam_cfg.get('requires_all') or [])
               if isinstance(s, str) and s.strip()]
    req_any = [s.strip().lower() for s in (fam_cfg.get('requires_any') or [])
               if isinstance(s, str) and s.strip()]
    if not req_all and not req_any:
        return

    stems = [s.strip().lower() for s in (rule.get('condition_stems') or [])
             if isinstance(s, str) and s.strip()]
    if not stems:
        return

    lex_cfg = rule.get('uncertainty_lexicon')
    if not isinstance(lex_cfg, list) or not lex_cfg:
        return
    lexicon = []
    for item in lex_cfg:
        if not isinstance(item, dict):
            return
        rx = compile_rx(item.get('regex'))
        if rx is None:
            return
        lexicon.append((as_str(item.get('label')) or 'uncertainty term', rx))

    fb_cfg = rule.get('symptom_fallback')
    if not isinstance(fb_cfg, dict):
        return
    fb_rx = None
    if fb_cfg.get('code_regex') is not None:
        fb_rx = compile_rx(fb_cfg.get('code_regex'))
        if fb_rx is None:
            return
    fb_any = [s.strip().lower()
              for s in (fb_cfg.get('descriptor_requires_any') or [])
              if isinstance(s, str) and s.strip()]
    if fb_rx is None and not fb_any:
        return
    fb_ev = fb_cfg.get('note_evidence') or {}
    if not isinstance(fb_ev, dict):
        return
    min_len = fb_ev.get('min_token_len')
    if isinstance(min_len, bool) or not isinstance(min_len, int) or min_len < 1:
        min_len = 4
    ignore_toks = set(s.strip().lower()
                      for s in (fb_ev.get('ignore_tokens') or [])
                      if isinstance(s, str) and s.strip())

    text = note_full_text if isinstance(note_full_text, str) else ''
    if not text.strip():
        return
    evidence = v._note_evidence(text)
    note_words = evidence[0]
    scrubbed = evidence[1]
    if not isinstance(scrubbed, str) or not scrubbed.strip():
        return

    def line_desc(entry):
        d = entry.get('description')
        if isinstance(d, str) and d.strip():
            return d.strip().lower()
        code = as_str(entry.get('code'))
        info = v.db.validate_icd10(code) if code else None
        if isinstance(info, dict):
            d2 = info.get('description') or info.get('long_description')
            if isinstance(d2, str) and d2.strip():
                return d2.strip().lower()
        return ''

    def code_matches(rx, code):
        if rx.match(code):
            return True
        return bool(rx.match(code.replace('.', '')))

    family = []
    for entry in icd:
        if not isinstance(entry, dict):
            continue
        code = as_str(entry.get('code'))
        if not code or not code_matches(code_rx, code):
            continue
        desc = line_desc(entry)
        if not desc:
            continue
        if req_all and not all(s in desc for s in req_all):
            continue
        if req_any and not any(s in desc for s in req_any):
            continue
        family.append(entry)
    if not family:
        return

    sentences = [s.strip() for s in re.split(r'[.!?\n]+', scrubbed)
                 if s.strip()]
    cond_sents = [s for s in sentences if any(st in s for st in stems)]
    if not cond_sents:
        return
    seen = set()
    for sent in cond_sents:
        hits = [lab for (lab, rx) in lexicon if rx.search(sent)]
        if not hits:
            return
        for lab in hits:
            seen.add(lab)
    labels_txt = ', '.join(sorted(seen))

    action = rule.get('action')
    if not isinstance(action, dict):
        action = {}
    severity = as_str(action.get('severity')) or 'WARNING'
    category = as_str(action.get('category')) or 'uncertain_diagnosis'
    denial = as_str(action.get('denial_risk')) or 'MEDIUM'

    def fmt(tmpl, **kw):
        if not isinstance(tmpl, str) or not tmpl:
            return ''
        try:
            return tmpl.format(**kw)
        except (KeyError, IndexError, ValueError):
            return tmpl

    support = coding_result.get('supporting_conditions')
    if not isinstance(support, list):
        support = []
        coding_result['supporting_conditions'] = support

    def sort_key(e):
        return (as_str(e.get('code')), as_str(e.get('description')))

    family_sorted = sorted(family, key=sort_key)
    had_primary = False
    for entry in family_sorted:
        code = as_str(entry.get('code'))
        desc = line_desc(entry)
        if as_str(entry.get('type')).strip().lower() == 'primary':
            had_primary = True
        for i in range(len(icd)):
            if icd[i] is entry:
                icd.pop(i)
                break
        entry['needs_review'] = True
        entry['review_reason'] = fmt(
            action.get('review_reason_demote'),
            code=code, desc=desc, labels=labels_txt) or (
            'Demoted per ICD-10-CM guideline IV.H: every note mention of '
            'this condition is uncertain (' + labels_txt + '); uncertain '
            'diagnoses are not coded for outpatient encounters.')
        support.append(entry)
        v._add(severity, code, category,
               fmt(action.get('message'),
                   code=code, desc=desc, labels=labels_txt) or (
                   'AUTO-CORRECTED: ' + code + ' moved to supporting '
                   'conditions - every note mention of the condition is '
                   'uncertain (' + labels_txt + '); guideline IV.H '
                   'prohibits coding uncertain outpatient diagnoses.'),
               fmt(action.get('recommendation'),
                   code=code, desc=desc, labels=labels_txt) or (
                   'Code the documented symptom instead, or obtain '
                   'definitive provider documentation.'),
               denial_risk=denial)

    if not had_primary:
        return
    for entry in icd:
        if isinstance(entry, dict) and \
                as_str(entry.get('type')).strip().lower() == 'primary':
            return

    def fb_desc_hit(desc):
        if not fb_any:
            return True
        toks = v._tokens(desc)
        for s in fb_any:
            for t in toks:
                if t == s or t.startswith(s):
                    return True
        return False

    candidates = []
    for entry in icd:
        if not isinstance(entry, dict):
            continue
        code = as_str(entry.get('code'))
        if not code:
            continue
        if fb_rx is not None and not code_matches(fb_rx, code):
            continue
        desc = line_desc(entry)
        if not desc:
            continue
        if not fb_desc_hit(desc):
            continue
        needed = sorted(t for t in v._tokens(desc)
                        if t not in v._DESC_STOPWORDS
                        and t not in ignore_toks
                        and len(t) >= min_len)
        if not needed:
            continue
        documented = all((t in note_words) or (v._stem(t) in note_words)
                         for t in needed)
        if documented:
            candidates.append(entry)

    anchor_code = as_str(family_sorted[0].get('code'))
    if candidates:
        candidates.sort(key=sort_key)
        target = candidates[0]
        tcode = as_str(target.get('code'))
        tdesc = line_desc(target)
        target['type'] = 'primary'
        target['needs_review'] = True
        target['review_reason'] = fmt(
            action.get('review_reason_promote'),
            code=anchor_code, desc=tdesc, labels=labels_txt,
            target=tcode) or (
            'Promoted to primary per guideline IV.H: the uncertain '
            'diagnosis was demoted and this documented symptom code is '
            'the mandated first-listed code.')
        promo = fmt(action.get('promotion_message'),
                    code=anchor_code, desc=tdesc, labels=labels_txt,
                    target=tcode)
        if promo:
            v._add('INFO', tcode, category, promo,
                   fmt(action.get('recommendation'),
                       code=tcode, desc=tdesc, labels=labels_txt) or '',
                   denial_risk='LOW')
    else:
        v._add('WARNING', anchor_code, category,
               fmt(action.get('missing_primary_message'),
                   code=anchor_code, desc='', labels=labels_txt) or (
                   'The uncertain diagnosis was demoted per guideline '
                   'IV.H (' + labels_txt + ') but no billed symptom line '
                   'matches the documented presentation - the claim is '
                   'left without a first-listed diagnosis; add the '
                   'documented symptom/sign code.'),
               fmt(action.get('recommendation'),
                   code=anchor_code, desc='', labels=labels_txt) or (
                   'Add the documented symptom/sign code as the '
                   'first-listed diagnosis.'),
               denial_risk='HIGH')
