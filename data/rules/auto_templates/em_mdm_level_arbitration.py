import re

TEMPLATE_NAME = "em_mdm_level_arbitration"

SCHEMA_DOC = """
TEMPLATE em_mdm_level_arbitration
=================================
Presence-and-level arbitration for an evaluation-and-management (E/M) code
family selected PURELY by descriptor grammar (never code identity). The
template finds the single billed line whose OFFICIAL descriptor starts with
family.descriptor_prefix and contains exactly one family.status_phrases
entry (the patient-status axis, e.g. 'new patient' / 'established patient').
It then computes an MDM level from the reference store's mdm_requirements
table (per family member) and the note's negation-scrubbed sentences, and
swaps the billed line to the same-status family member whose reference MDM
level equals the computed level. SINGLE AXIS: only entry['code'] and
entry['description'] on that one line are rewritten; modifiers, units, and
every other line are preserved byte-for-byte.

RULE FIELDS
-----------
id, template (= "em_mdm_level_arbitration"), enabled, authority: as usual.

applies_to.array : "cpt_codes" or "hcpcs_codes" — which claim array holds
  the E/M line. (No code regex: selection is by descriptor grammar only.)

family.descriptor_prefix : lowercase prefix the official descriptor must
  start with (e.g. an office-visit stem). REQUIRED.
family.status_phrases : list of lowercase patient-status phrases; the billed
  descriptor must contain EXACTLY ONE of them, and only members carrying the
  SAME phrase are arbitration candidates. REQUIRED.

level_order : the MDM level names, lowest to highest, exactly as the
  reference table's mdm_requirements 'level' field spells them (e.g.
  ["straightforward","low","moderate","high"]). REQUIRED, >= 2 names.

problems : list of evidence classes for the 'problems addressed' element.
  Each class: {label, note_regex, exclude_regex (optional), table_regex}.
  A class FIRES when note_regex matches a negation-scrubbed sentence in
  which exclude_regex does NOT match. A firing class maps to the LOWEST
  family level whose mdm_requirements.requirements.problems_addressed row
  texts match table_regex (case-insensitive search over the rows joined
  with ' || '). The element level is the MAX over firing classes
  (met-or-exceeded semantics). Zero firing (or zero mappable) classes =>
  the whole rule does nothing. table_regex is matched against the
  reference table's OWN row wording, never against codes.

risk : same shape and semantics as problems, but table_regex is matched
  against the risk_of_management label plus its example phrases (joined
  with ' || ', lowercased). Zero firing classes => no action. Use
  exclude_regex to suppress hypothetical/future statements ('possible',
  'if symptoms persist', 'consider', 'deferred', ...).

data_events : list of countable event classes for the 'data reviewed and
  analyzed' element. Each class: {label, note_regex, exclude_regex
  (optional), item_regex (optional), category_regex (optional), max_count
  (optional positive int)}. The count of a class is the number of DISTINCT
  lowercased match texts of note_regex across non-excluded sentences,
  capped at max_count. For a candidate level, each data category from the
  reference table is tested: a category WITH items is satisfied when the
  summed counts of classes whose item_regex matches the category's own
  item texts reach the integer parsed from the category's 'requirement'
  text ('Any combination of N ...'); a category WITHOUT items is satisfied
  when any class with count > 0 has a category_regex matching the
  category's name+requirement text. The element is met when the number of
  satisfied categories reaches the integer parsed from the level's data
  label ('at least N of ...', default 1). A level whose reference data has
  NO categories (e.g. 'Minimal or none') is always met. A category whose
  requirement text carries no parseable integer is conservatively treated
  as unsatisfiable.

mutually_exclusive : optional list of [labelA, labelB] pairs; if both
  labels fired (across problems and risk), the evidence is conflicting and
  the rule does nothing.

total_time_regex : optional; a regex with ONE capture group of integer
  minutes matched against the raw lowercased note. If EXACTLY ONE distinct
  minute value is captured, any member whose own descriptor's stated
  threshold (parsed via descriptor_minutes_regex, default
  '(\\d+)\\s+minutes\\s+must\\s+be\\s+met') is <= that value may raise the
  target level. Multiple distinct captured values = ambiguous => time is
  ignored. No explicit total-time sentence => time is never used.

action : {severity, category, denial_risk, message, recommendation}.
  Placeholders: {code} billed code, {target} target code, {desc} target
  official descriptor, {level} computed level name, {status} patient-status
  phrase.

MECHANIC
--------
1. Exactly one billed family line, with exactly one status phrase, and the
   billed code itself a family member with usable mdm_requirements —
   otherwise no action.
2. Family members = all reference-table codes with the same descriptor
   prefix + status phrase and a usable mdm_requirements level found in
   level_order. Need >= 2 members, each at a DISTINCT level; duplicates =>
   no action.
3. Element levels computed as above from the scrubbed note. The final MDM
   level is the HIGHEST member level whose own selection_rule ('N of the M
   elements met or exceeded'; N parsed, default 2) is satisfied by
   (problems_level >= member, risk_level >= member, data_met(member)).
4. Documented total time (see total_time_regex) may only RAISE the target.
5. If the target member differs from the billed code, swap code +
   description on that line only, and report via action. If equal, silence.

CONSERVATISM (all => no action): unparseable configured regex (whole rule
disabled), missing/duplicate reference levels, zero firing problems or risk
classes, mutually-exclusive labels both firing, no member at the computed
level, ambiguous total-time values (time ignored), more or fewer than one
billed family line. With affirmative evidence supporting only the lower
elements, arbitration lands on the lowest supportable member (audit-safe
downcode). All lexicons live in the rule; matching is order-deterministic
(classes sorted by label, members sorted by code then level).
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result, note_full_text, note_assessment_text):
    v = engine.v

    def rx(pat):
        if not isinstance(pat, str) or not pat:
            return None
        try:
            return re.compile(pat, re.I)
        except re.error:
            return None

    fam = rule.get("family") or {}
    prefix = str(fam.get("descriptor_prefix") or "").strip().lower()
    statuses = sorted([str(s).lower() for s in (fam.get("status_phrases") or []) if isinstance(s, str) and s])
    order = [str(x).lower() for x in (rule.get("level_order") or []) if isinstance(x, str) and x]
    if not prefix or not statuses or len(order) < 2 or len(set(order)) != len(order):
        return
    level_value = {}
    for i, name in enumerate(order):
        level_value[name] = i + 1

    def load(key, need_table, extras):
        raw = rule.get(key)
        if raw is None:
            return []
        if not isinstance(raw, list):
            return None
        out = []
        for c in raw:
            if not isinstance(c, dict):
                return None
            nr = rx(c.get("note_regex"))
            if nr is None:
                return None
            ex = None
            if c.get("exclude_regex") is not None:
                ex = rx(c.get("exclude_regex"))
                if ex is None:
                    return None
            tr = None
            if need_table:
                tr = rx(c.get("table_regex"))
                if tr is None:
                    return None
            item = {"label": str(c.get("label") or ""), "note": nr, "exclude": ex, "table": tr}
            for k in extras:
                r2 = None
                if c.get(k) is not None:
                    r2 = rx(c.get(k))
                    if r2 is None:
                        return None
                item[k] = r2
            mc = c.get("max_count")
            item["max_count"] = mc if isinstance(mc, int) and mc > 0 else None
            out.append(item)
        return sorted(out, key=lambda d: d["label"])

    prob = load("problems", True, [])
    risk = load("risk", True, [])
    data = load("data_events", False, ["item_regex", "category_regex"])
    if prob is None or risk is None or data is None or not prob or not risk:
        return

    array_name = str((rule.get("applies_to") or {}).get("array") or "cpt_codes")
    if array_name == "cpt_codes":
        lines = cpt
    elif array_name == "hcpcs_codes":
        lines = hcpcs
    else:
        return

    def lookup(code):
        try:
            if array_name == "hcpcs_codes":
                return v.db.validate_hcpcs(code)
            return v.db.validate_cpt(code)
        except (AttributeError, TypeError, KeyError, ValueError):
            return None

    def desc_of(info):
        if not isinstance(info, dict):
            return ""
        return str(info.get("description") or info.get("long_description") or "")

    def status_of(desc):
        low = desc.lower()
        if not low.startswith(prefix):
            return None
        found = [s for s in statuses if s in low]
        if len(found) != 1:
            return None
        return found[0]

    billed = []
    for entry in lines:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "")
        if not code:
            continue
        st = status_of(desc_of(lookup(code)))
        if st is not None:
            billed.append((code, st, entry))
    if len(billed) != 1:
        return
    billed_code, status, entry = billed[0]

    def mdm_of(code, info):
        m = info.get("mdm_requirements") if isinstance(info, dict) else None
        if not isinstance(m, dict):
            try:
                m = v.store.mdm_requirements(code)
            except (AttributeError, TypeError, KeyError, ValueError):
                m = None
        return m if isinstance(m, dict) else None

    try:
        table = v.db.cpt if array_name == "cpt_codes" else v.db.hcpcs
        pairs = sorted(table.items())
    except (AttributeError, TypeError, ValueError):
        return

    members = []
    for code, info in pairs:
        d = desc_of(info)
        if not d or status_of(d) != status:
            continue
        m = mdm_of(str(code), info)
        if m is None:
            continue
        name = str(m.get("level") or "").lower()
        if name not in level_value:
            continue
        members.append({"code": str(code), "desc": d, "mdm": m, "value": level_value[name], "level": name})
    if len(members) < 2:
        return
    vals = [m["value"] for m in members]
    if len(set(vals)) != len(vals):
        return
    if billed_code not in set(m["code"] for m in members):
        return
    ordered = sorted(members, key=lambda m: m["value"])

    try:
        pair = v._note_evidence(str(note_full_text or ""))
        scrub = pair[1]
    except (AttributeError, TypeError, ValueError, IndexError):
        return
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(scrub)) if s and s.strip()]

    def fires(cls):
        for s in sentences:
            if cls["exclude"] is not None and cls["exclude"].search(s):
                continue
            if cls["note"].search(s):
                return True
        return False

    def prob_text(m):
        req = m["mdm"].get("requirements") or {}
        rows = req.get("problems_addressed") or []
        if not isinstance(rows, list):
            return ""
        return " || ".join(str(r).lower() for r in rows)

    def risk_text(m):
        req = m["mdm"].get("requirements") or {}
        rk = req.get("risk_of_management") or {}
        if not isinstance(rk, dict):
            return ""
        parts = [str(rk.get("label") or "")]
        ex = rk.get("examples") or []
        if isinstance(ex, list):
            for e in ex:
                parts.append(str(e))
        return " || ".join(parts).lower()

    fired_labels = set()

    def element_level(classes, text_fn):
        best = 0
        for cls in classes:
            if not fires(cls):
                continue
            fired_labels.add(cls["label"])
            mapped = 0
            for m in ordered:
                if cls["table"].search(text_fn(m)):
                    mapped = m["value"]
                    break
            if mapped > best:
                best = mapped
        return best

    problems_level = element_level(prob, prob_text)
    risk_level = element_level(risk, risk_text)
    if problems_level == 0 or risk_level == 0:
        return

    mx = rule.get("mutually_exclusive") or []
    if isinstance(mx, list):
        for pairx in mx:
            if isinstance(pairx, list) and len(pairx) == 2:
                if str(pairx[0]) in fired_labels and str(pairx[1]) in fired_labels:
                    return

    counts = []
    for cls in data:
        found = set()
        for s in sentences:
            if cls["exclude"] is not None and cls["exclude"].search(s):
                continue
            for mt in cls["note"].finditer(s):
                found.add(mt.group(0).lower())
        n = len(found)
        if cls["max_count"] is not None and n > cls["max_count"]:
            n = cls["max_count"]
        counts.append((cls, n))

    def data_met(m):
        req = m["mdm"].get("requirements") or {}
        dr = req.get("data_reviewed_analyzed") or {}
        if not isinstance(dr, dict):
            return False
        cats = dr.get("categories") or []
        if not isinstance(cats, list) or not cats:
            return True
        nm = re.search(r"at least\s+(\d+)", str(dr.get("label") or "").lower())
        need_cats = int(nm.group(1)) if nm else 1
        satisfied = 0
        for cat in cats:
            if not isinstance(cat, dict):
                continue
            items = cat.get("items") or []
            if isinstance(items, list) and items:
                rn = re.search(r"(\d+)", str(cat.get("requirement") or ""))
                if not rn:
                    continue
                need_n = int(rn.group(1))
                itxt = " || ".join(str(i).lower() for i in items)
                total = 0
                for cls, n in counts:
                    if n > 0 and cls["item_regex"] is not None and cls["item_regex"].search(itxt):
                        total += n
                if total >= need_n:
                    satisfied += 1
            else:
                ctxt = (str(cat.get("name") or "") + " " + str(cat.get("requirement") or "")).lower()
                for cls, n in counts:
                    if n > 0 and cls["category_regex"] is not None and cls["category_regex"].search(ctxt):
                        satisfied += 1
                        break
        return satisfied >= need_cats

    def need_of(m):
        nm = re.search(r"(\d+)\s+of\s+the\s+\d+", str(m["mdm"].get("selection_rule") or "").lower())
        return int(nm.group(1)) if nm else 2

    target_value = 0
    for m in ordered:
        met = 0
        if problems_level >= m["value"]:
            met += 1
        if risk_level >= m["value"]:
            met += 1
        if data_met(m):
            met += 1
        if met >= need_of(m) and m["value"] > target_value:
            target_value = m["value"]
    if target_value == 0:
        return

    ttr = rule.get("total_time_regex")
    if ttr is not None:
        trx = rx(ttr)
        if trx is None:
            return
        mins = set()
        ok = True
        for g in trx.finditer(str(note_full_text or "").lower()):
            try:
                val = g.group(1)
            except IndexError:
                ok = False
                break
            if isinstance(val, str) and val.isdigit():
                mins.add(int(val))
        if not ok:
            return
        if len(mins) == 1:
            t = sorted(mins)[0]
            dmr = rx(str(rule.get("descriptor_minutes_regex") or r"(\d+)\s+minutes\s+must\s+be\s+met"))
            if dmr is None:
                return
            for m in ordered:
                dm = dmr.search(m["desc"].lower())
                if dm is None:
                    continue
                try:
                    dval = dm.group(1)
                except IndexError:
                    return
                if isinstance(dval, str) and dval.isdigit() and int(dval) <= t and m["value"] > target_value:
                    target_value = m["value"]

    targets = [m for m in members if m["value"] == target_value]
    if len(targets) != 1:
        return
    target = targets[0]
    if target["code"] == billed_code:
        return

    entry["code"] = target["code"]
    entry["description"] = target["desc"]

    action = rule.get("action") or {}

    def fill(t):
        out = str(t)
        out = out.replace("{code}", billed_code)
        out = out.replace("{target}", target["code"])
        out = out.replace("{desc}", target["desc"])
        out = out.replace("{level}", target["level"])
        out = out.replace("{status}", status)
        return out

    v._add(
        str(action.get("severity") or "WARNING"),
        target["code"],
        str(action.get("category") or "em_mdm_level_mismatch"),
        fill(action.get("message") or "AUTO-CORRECTED: {code} swapped to {target} ('{desc}') - the documented MDM supports the '{level}' level for this {status} visit."),
        fill(action.get("recommendation") or "Verify the documented problems, data, and risk support the billed visit level."),
        denial_risk=str(action.get("denial_risk") or "MEDIUM"),
    )
