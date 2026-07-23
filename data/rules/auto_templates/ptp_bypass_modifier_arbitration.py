import re

TEMPLATE_NAME = "ptp_bypass_modifier_arbitration"

SCHEMA_DOC = """
ptp_bypass_modifier_arbitration -- deterministic arbitration of the NCCI-PTP
distinct-service bypass modifier class (e.g. 59 / XE / XP / XS / XU -- payment
modifiers, not procedure or diagnosis codes) on billed procedure lines.
SINGLE-AXIS: the template mutates ONLY the modifiers matching
bypass_modifier_regex, ONLY on the lines selected by applies_to. It never
rewrites a line's other modifiers, never touches units or other lines, and
never deletes a line. Whether two billed lines bundle is a reference-store
fact (v.store.ncci_pair) -- never code identity. Adjudication honors the
bypass modifier on either line of a PTP pair, so arbitration is symmetric
over the pair regardless of which column the selected line occupies.

Per selected line, against every OTHER billed CPT/HCPCS line on the claim:
 1. NO PTP edit exists with any other billed line -> a present bypass
    modifier has nothing to bypass; it is removed
    (action.message_removed_superfluous).
 2. Any PTP edit carries modifier indicator "0" -> the edit can never be
    bypassed; a present bypass modifier is removed and the column-2 service
    flagged as not separately payable (action.message_removed_bundled).
    The line itself is NOT deleted.
 3. All edits carry modifier indicator "1" -> the modifier is supportable
    only with documented distinctness, proven either by
      a. a distinct-session/lesion sentence: any
         evidence.distinct_session_regexes match in a (negation-scrubbed)
         note sentence that no evidence.exclusion_regexes veto; or
      b. a different-anatomic-site test: the two lines' OFFICIAL reference
         descriptors (v.db) resolve via evidence.site_lexicon to DISJOINT
         anatomy groups AND the note documents words from both lines'
         groups.
    Every pair distinct -> a present modifier is kept silently; a missing
    one is appended only when action.add_modifier is configured, itself
    matches bypass_modifier_regex, and the line is the column-2 code of an
    edit (action.message_kept_distinct).
    Any pair whose descriptors share an anatomy group (same region) with no
    session/lesion proof -> a present modifier is removed
    (action.message_conflict).
    No proof either way -> evidence.unproven_action decides: "remove"
    strips the undocumented modifier (NCCI requires the record to support
    it); "none" (the default) leaves the line untouched.
CONSERVATISM: unparseable regexes, missing reference lookups, indicator
values other than "0"/"1", or conflicting evidence (distinct for one pair,
same-region for another) -> no action at all.

Rule fields:
  applies_to.array         "cpt_codes" | "hcpcs_codes" (required)
  applies_to.code_regex    broad STRUCTURAL regex over line codes (required;
                           never a literal code list)
  bypass_modifier_regex    required anchored regex matched against
                           upper-cased, stripped modifier strings; defines
                           the single modifier class this rule arbitrates
  evidence.scrub_negation  bool, default true: evaluate note evidence on the
                           negation-scrubbed text from v._note_evidence
  evidence.site_lexicon    {group_name: [anatomy words...]} -- a fixed
                           human-anatomy vocabulary (like the tier
                           lexicons), NOT a curated code list; matched
                           against official descriptors and the note
  evidence.distinct_session_regexes  [{label, regex}] sentence-level proofs
                           of a separate session / lesion / site
  evidence.exclusion_regexes         [{label, regex}] a sentence matching
                           any of these cannot supply distinctness proof
  evidence.unproven_action "remove" | "none" (default "none")
  action.severity / action.category / action.denial_risk (strings)
  action.recommendation    string; supports the same placeholders
  action.add_modifier      optional modifier appended on proven
                           distinctness; must itself match
                           bypass_modifier_regex or no add ever occurs
  action.message_removed_superfluous / action.message_removed_bundled /
  action.message_conflict / action.message_kept_distinct
    Placeholders: {code} {desc} {pair_code} {indicator} {modifier} {sites}.
    An empty/missing message suppresses only the issue text; the
    deterministic modifier mutation still occurs.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result, note_full_text, note_assessment_text):
    v = engine.v
    applies = rule.get("applies_to") or {}
    array = str(applies.get("array") or "")
    if array == "cpt_codes":
        selected_list = cpt if isinstance(cpt, list) else []
    elif array == "hcpcs_codes":
        selected_list = hcpcs if isinstance(hcpcs, list) else []
    else:
        return
    code_regex = str(applies.get("code_regex") or "")
    mod_regex = str(rule.get("bypass_modifier_regex") or "")
    if not code_regex or not mod_regex:
        return
    try:
        code_pat = re.compile(code_regex)
        bypass_pat = re.compile(mod_regex)
    except Exception:
        return

    evidence = rule.get("evidence") or {}
    act = rule.get("action") or {}

    distinct_pats = []
    excl_pats = []
    try:
        for spec in (evidence.get("distinct_session_regexes") or []):
            if isinstance(spec, dict):
                distinct_pats.append((str(spec.get("label") or "distinct service"),
                                      re.compile(str(spec.get("regex") or ""), re.IGNORECASE)))
        for spec in (evidence.get("exclusion_regexes") or []):
            if isinstance(spec, dict):
                excl_pats.append(re.compile(str(spec.get("regex") or ""), re.IGNORECASE))
    except Exception:
        return

    raw_note = str(note_full_text or "")
    if evidence.get("scrub_negation", True):
        note_words, note_text = v._note_evidence(raw_note)
        note_words = set(note_words)
        note_text = str(note_text)
    else:
        note_text = raw_note.lower()
        note_words = set(v._tokens(note_text))
    note_words = note_words | set([v._stem(t) for t in note_words])
    note_low = " " + note_text + " "

    raw_lex = evidence.get("site_lexicon") or {}
    lexicon = {}
    for g in sorted(raw_lex):
        terms = raw_lex.get(g) or []
        lexicon[str(g)] = sorted(set([str(t).lower().strip() for t in terms if str(t).strip()]))

    def groups_in(text):
        toks = set(v._tokens(str(text)))
        toks = toks | set([v._stem(t) for t in toks])
        low = " " + str(text).lower() + " "
        found = set()
        for g in sorted(lexicon):
            for term in lexicon[g]:
                if (" " in term and term in low) or (term in toks) or (v._stem(term) in toks):
                    found.add(g)
                    break
        return found

    def group_documented(g):
        for term in lexicon.get(g, []):
            if (" " in term and term in note_low) or (term in note_words) or (v._stem(term) in note_words):
                return True
        return False

    sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", note_text) if s.strip()]
    session_label = ""
    for item in distinct_pats:
        if session_label:
            break
        for s in sentences:
            if item[1].search(s):
                vetoed = False
                for xp in excl_pats:
                    if xp.search(s):
                        vetoed = True
                        break
                if not vetoed:
                    session_label = item[0] + ": '" + s[:120] + "'"
                    break

    arr_of = {}
    for entry in (cpt if isinstance(cpt, list) else []):
        if isinstance(entry, dict) and entry.get("code"):
            c = str(entry.get("code"))
            if c not in arr_of:
                arr_of[c] = "cpt_codes"
    for entry in (hcpcs if isinstance(hcpcs, list) else []):
        if isinstance(entry, dict) and entry.get("code"):
            c = str(entry.get("code"))
            if c not in arr_of:
                arr_of[c] = "hcpcs_codes"

    def desc_of(c, arr):
        info = v.db.validate_cpt(c) if arr == "cpt_codes" else v.db.validate_hcpcs(c)
        if not isinstance(info, dict):
            return ""
        return str(info.get("description") or info.get("long_description") or "")

    def emit(msg_key, line_code, mapping):
        msg = str(act.get(msg_key) or "")
        if not msg:
            return
        rec = str(act.get("recommendation") or "")
        for k in sorted(mapping):
            token = "{" + k + "}"
            msg = msg.replace(token, str(mapping[k]))
            rec = rec.replace(token, str(mapping[k]))
        v._add(str(act.get("severity") or "WARNING"), line_code,
               str(act.get("category") or "ncci_ptp_bypass_modifier"),
               msg, rec, denial_risk=str(act.get("denial_risk") or "MEDIUM"))

    for entry in selected_list:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "")
        if not code or not code_pat.match(code):
            continue
        mods = entry.get("modifiers")
        mods = mods if isinstance(mods, list) else []
        bypass = [m for m in mods if bypass_pat.match(str(m).strip().upper())]
        kept = [m for m in mods if not bypass_pat.match(str(m).strip().upper())]
        desc = desc_of(code, array) or str(entry.get("description") or "")
        mod_text = ", ".join(sorted(set([str(m) for m in bypass])))

        edits = []
        for pc in sorted([c for c in arr_of if c != code]):
            e = v.store.ncci_pair(code, pc)
            if not isinstance(e, dict):
                e = v.store.ncci_pair(pc, code)
            if isinstance(e, dict):
                edits.append((pc, e))

        if not edits:
            if bypass:
                entry["modifiers"] = kept
                emit("message_removed_superfluous", code,
                     {"code": code, "desc": desc, "modifier": mod_text,
                      "pair_code": "", "indicator": "", "sites": ""})
            continue

        zeros = [pe for pe in edits if str(pe[1].get("modifier_indicator") or "") == "0"]
        ones = [pe for pe in edits if str(pe[1].get("modifier_indicator") or "") == "1"]

        if zeros:
            if bypass:
                entry["modifiers"] = kept
                emit("message_removed_bundled", code,
                     {"code": code, "desc": desc, "modifier": mod_text,
                      "pair_code": zeros[0][0], "indicator": "0", "sites": ""})
            continue

        if not ones or len(ones) != len(edits):
            continue

        my_groups = groups_in(desc)
        decisions = set()
        same_pair = ("", [])
        distinct_notes = set()
        for pc, e in ones:
            if session_label:
                decisions.add("distinct")
                distinct_notes.add(session_label)
                continue
            pg = groups_in(desc_of(pc, arr_of.get(pc) or ""))
            if my_groups and pg:
                overlap = sorted(my_groups & pg)
                if overlap:
                    decisions.add("same")
                    if not same_pair[0]:
                        same_pair = (pc, overlap)
                elif any([group_documented(g) for g in sorted(my_groups)]) and any([group_documented(g) for g in sorted(pg)]):
                    decisions.add("distinct")
                    distinct_notes.add("different documented anatomic regions: " + ", ".join(sorted(my_groups)) + " vs " + ", ".join(sorted(pg)))
                else:
                    decisions.add("inconclusive")
            else:
                decisions.add("inconclusive")

        if "distinct" in decisions and "same" in decisions:
            continue

        if decisions == set(["distinct"]):
            if not bypass:
                addm = str(act.get("add_modifier") or "").strip().upper()
                is_col2 = any([str(pe[1].get("col2") or "") == code for pe in ones])
                if addm and is_col2 and bypass_pat.match(addm):
                    entry["modifiers"] = list(mods) + [addm]
                    emit("message_kept_distinct", code,
                         {"code": code, "desc": desc, "modifier": addm,
                          "pair_code": ones[0][0], "indicator": "1",
                          "sites": "; ".join(sorted(distinct_notes))})
            continue

        if "same" in decisions:
            if bypass:
                entry["modifiers"] = kept
                emit("message_conflict", code,
                     {"code": code, "desc": desc, "modifier": mod_text,
                      "pair_code": same_pair[0], "indicator": "1",
                      "sites": "both descriptors resolve to the same anatomic region (" + ", ".join(same_pair[1]) + ") and the note documents no distinct site, lesion, or session"})
            continue

        if str(evidence.get("unproven_action") or "none") == "remove" and bypass:
            entry["modifiers"] = kept
            emit("message_conflict", code,
                 {"code": code, "desc": desc, "modifier": mod_text,
                  "pair_code": ones[0][0], "indicator": "1",
                  "sites": "no distinct site, lesion, or session is documented anywhere in the note"})
