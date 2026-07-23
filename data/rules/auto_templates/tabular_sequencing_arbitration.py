import re

TEMPLATE_NAME = "tabular_sequencing_arbitration"

SCHEMA_DOC = """
TEMPLATE tabular_sequencing_arbitration
=======================================
Enforces the ICD-10-CM Tabular 'Code first' / 'Use additional code'
sequencing convention between billed diagnosis lines. The template selects
an UNDERLYING (etiology, code-first) candidate and COMPANION
(manifestation) candidates purely by structural regex + descriptor
grammar, then PROVES the pairing from the reference store's own Tabular
instructional notes: a category named in the underlying line's 'Use
additional code' targets must prefix the companion's code (and/or a
category named in the companion's 'Code first' refs must prefix the
underlying's code). When exactly one underlying line is linked, the
template pins type='secondary' on every linked companion line, pins
type='primary' on the underlying line (only when no NON-member line
already holds primary), and resequences the underlying line ahead of the
first companion line. It never adds or deletes lines. On any ambiguity
(no linked pair, more than one distinct linked underlying line, a line
matching both selectors) it does NOTHING.

RULE FIELDS
-----------
id           kebab-case rule id.
template     must equal "tabular_sequencing_arbitration".
enabled      true|false.
authority    prose citation of the governing Tabular note / guideline
             (prose MAY mention codes; selecting fields may NOT).
applies_to   {"array": "icd_codes", "code_regex": <regex>} - cheap gate;
             the rule no-ops unless some billed ICD line matches.
             array must be "icd_codes": this mechanic sequences
             diagnosis lines only.
underlying   selector for the etiology (code-first) line:
               code_regex        structural regex on the dotted code
                                 (chapter/shape only, e.g. "^I[0-9]{2}";
                                 NEVER a literal full code)
               desc_contains_any [terms] descriptor must contain >= 1
               desc_contains_all [terms] descriptor must contain all
               desc_excludes     [terms] descriptor must contain none
companion    selector for the manifestation line; same shape as
             underlying. Any line matching BOTH selectors is discarded
             from both (ambiguous) - keep the two code_regex fields
             structurally disjoint (different chapters).
link         {"require": "use_additional"|"code_first"|"both"|"any"}
             (default "use_additional"). Which Tabular lookup(s) must
             prove the pairing:
               use_additional - a category token found in the underlying
                 line's 'Use additional code' target text prefixes the
                 companion's undotted code
               code_first     - a category token found in the companion
                 line's 'Code first' etiology refs prefixes the
                 underlying's undotted code
action       {"severity", "category", "denial_risk", "message",
              "recommendation"}. message/recommendation may use the
             placeholders {underlying} {underlying_desc} {companion}
             {instruction} {changes}: {instruction} is the verbatim
             Tabular target text that proved the link; {changes} lists
             the exact mutations performed.

CONSTRAINTS FOR RULE AUTHORS
----------------------------
- No literal medical codes in any selecting field; broad structural
  regexes and descriptor grammar only. The pairing is proven at runtime
  by the Tabular notes in the reference store, never by code identity.
- Deterministic: candidates and mutations are processed in sorted order.
- Conservative: zero linked pairs, competing underlying lines, or
  selector overlap => no action at all. The underlying line is only
  promoted to primary when no non-member line already holds primary;
  linked companion lines are always pinned secondary (a manifestation
  code can never be first-listed when its etiology is billed).
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v
    if not isinstance(icd, list) or len(icd) < 2:
        return

    applies = rule.get("applies_to") or {}
    if str(applies.get("array") or "icd_codes") != "icd_codes":
        return

    def low(x):
        return str(x or "").strip().lower()

    def undot(c):
        return str(c or "").replace(".", "").strip().upper()

    def desc_of(entry):
        d = entry.get("description")
        if d:
            return low(d)
        info = None
        try:
            info = v.db.validate_icd10(str(entry.get("code") or ""))
        except Exception:
            info = None
        if isinstance(info, dict):
            return low(info.get("description") or info.get("long_description") or "")
        return ""

    cat_pat = re.compile(r"\b([A-Z][0-9]{2})\b")

    def cats_from_text(t):
        return set(cat_pat.findall(str(t or "").upper()))

    def sel_match(entry, sel):
        if not isinstance(sel, dict) or not sel:
            return False
        code = str(entry.get("code") or "")
        cr = sel.get("code_regex")
        if cr:
            try:
                if not re.search(cr, code):
                    return False
            except re.error:
                return False
        d = desc_of(entry)
        any_terms = sel.get("desc_contains_any") or []
        if any_terms and not any(low(t) in d for t in any_terms):
            return False
        all_terms = sel.get("desc_contains_all") or []
        if all_terms and not all(low(t) in d for t in all_terms):
            return False
        excl = sel.get("desc_excludes") or []
        if excl and any(low(t) in d for t in excl):
            return False
        return True

    def ua_target_texts(code_und):
        texts = []
        for key in [code_und, code_und[:3]]:
            groups = None
            try:
                groups = v.store.use_additional_code_groups(key)
            except Exception:
                groups = None
            if not groups:
                continue
            if isinstance(groups, dict):
                groups = [groups]
            for g in groups:
                if isinstance(g, dict):
                    tl = g.get("targets")
                    if tl is None:
                        tl = []
                    if isinstance(tl, str):
                        tl = [tl]
                    for t in tl:
                        texts.append(str(t))
                elif isinstance(g, (list, tuple, set, frozenset)):
                    for t in sorted([str(x) for x in g]):
                        texts.append(t)
                else:
                    texts.append(str(g))
            if texts:
                break
        return texts

    def cf_categories(code_und):
        cats = set()
        for key in [code_und, code_und[:3]]:
            refs = None
            try:
                refs = v.store.code_first_etiology_refs(key)
            except Exception:
                refs = None
            if not refs:
                continue
            if isinstance(refs, (str, dict)):
                refs = [refs]
            for r in refs:
                if isinstance(r, dict):
                    for k in sorted(r):
                        cats |= cats_from_text(r[k])
                else:
                    cats |= cats_from_text(r)
            if cats:
                break
        return cats

    live = [(i, e) for i, e in enumerate(icd)
            if isinstance(e, dict) and e.get("code")]
    if len(live) < 2:
        return

    gate_re = applies.get("code_regex")
    if gate_re:
        hit = False
        try:
            hit = any(re.search(gate_re, str(e.get("code") or ""))
                      for _, e in live)
        except re.error:
            return
        if not hit:
            return

    u_sel = rule.get("underlying") or {}
    c_sel = rule.get("companion") or {}
    u_cands = [(i, e) for i, e in live if sel_match(e, u_sel)]
    c_cands = [(i, e) for i, e in live if sel_match(e, c_sel)]
    overlap = set(i for i, _ in u_cands) & set(i for i, _ in c_cands)
    if overlap:
        # a line matching both selectors is ambiguous: drop it from both
        u_cands = [(i, e) for i, e in u_cands if i not in overlap]
        c_cands = [(i, e) for i, e in c_cands if i not in overlap]
    if not u_cands or not c_cands:
        return

    require = low((rule.get("link") or {}).get("require") or "use_additional")

    pairs = []  # (underlying_index, companion_index, verbatim_instruction)
    for ui, ue in u_cands:
        u_und = undot(ue.get("code"))
        if not u_und:
            continue
        cat_to_text = {}
        for t in ua_target_texts(u_und):
            for c in sorted(cats_from_text(t)):
                if c not in cat_to_text:
                    cat_to_text[c] = t
        for ci, ce in c_cands:
            if ci == ui:
                continue
            c_und = undot(ce.get("code"))
            if not c_und or c_und == u_und:
                continue
            ua_cat = ""
            for cat in sorted(cat_to_text):
                if c_und.startswith(cat):
                    ua_cat = cat
                    break
            ua_ok = bool(ua_cat)
            cf_ok = False
            if require in ("code_first", "both", "any"):
                cf_ok = any(u_und.startswith(cat)
                            for cat in sorted(cf_categories(c_und)))
            if require == "use_additional":
                ok = ua_ok
            elif require == "code_first":
                ok = cf_ok
            elif require == "both":
                ok = ua_ok and cf_ok
            else:
                ok = ua_ok or cf_ok
            if ok:
                pairs.append((ui, ci, str(cat_to_text.get(ua_cat, "")).strip()))

    if not pairs:
        return
    u_indices = sorted(set(p[0] for p in pairs))
    if len(u_indices) != 1:
        # competing code-first chains: never guess
        return
    ui = u_indices[0]
    u_entry = icd[ui]
    c_indices = sorted(set(p[1] for p in pairs))
    members = set([ui]) | set(c_indices)
    companion_codes = sorted(set(str(icd[ci].get("code") or "")
                                 for ci in c_indices))

    non_member_primary = any(low(e.get("type")) == "primary"
                             for i, e in live if i not in members)

    changes = []
    for ci in c_indices:
        ce = icd[ci]
        if low(ce.get("type")) != "secondary":
            old = str(ce.get("type") or "unset")
            ce["type"] = "secondary"
            changes.append(str(ce.get("code") or "") + " type " + old
                           + " -> secondary")
    if not non_member_primary and low(u_entry.get("type")) != "primary":
        old = str(u_entry.get("type") or "unset")
        u_entry["type"] = "primary"
        changes.append(str(u_entry.get("code") or "") + " type " + old
                       + " -> primary")

    first_c = min(c_indices)
    if ui > first_c:
        moved = icd.pop(ui)
        icd.insert(first_c, moved)
        changes.append(str(moved.get("code") or "")
                       + " resequenced ahead of "
                       + str(icd[first_c + 1].get("code") or ""))

    if not changes:
        return

    instruction = ""
    for p in sorted(pairs):
        if p[2]:
            instruction = p[2]
            break

    mapping = {
        "underlying": str(u_entry.get("code") or ""),
        "underlying_desc": desc_of(u_entry),
        "companion": ", ".join(companion_codes),
        "instruction": instruction,
        "changes": "; ".join(changes),
    }

    def fill(t, m):
        s = str(t or "")
        for k in sorted(m):
            s = s.replace("{" + k + "}", str(m[k]))
        return s

    act = rule.get("action") or {}
    default_msg = ("AUTO-CORRECTED: {changes} - the Tabular entry for "
                   "{underlying} carries a 'Use additional code' "
                   "instruction ({instruction}) naming the billed "
                   "companion {companion}; the etiology must be "
                   "sequenced ahead of the manifestation.")
    default_rec = ("Keep {underlying} first-listed with {companion} as "
                   "secondary per the ICD-10-CM code-first/use-additional "
                   "convention")
    msg = fill(act.get("message") or default_msg, mapping)
    rec = fill(act.get("recommendation") or default_rec, mapping)
    v._add(str(act.get("severity") or "WARNING"),
           mapping["underlying"],
           str(act.get("category") or "tabular_sequencing_arbitration"),
           msg,
           rec,
           denial_risk=str(act.get("denial_risk") or "MEDIUM"))
