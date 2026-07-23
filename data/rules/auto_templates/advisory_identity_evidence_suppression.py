import re

TEMPLATE_NAME = "advisory_identity_evidence_suppression"

SCHEMA_DOC = """
Template: advisory_identity_evidence_suppression

Purpose
-------
WARN-only suppression of a compliance-scrubber ADVISORY for codes whose
leaf descriptor is missing or empty in the reference store (common for
status/administrative chapters), where descriptor-grammar selection is
impossible. Selection is by the advisory's own identity (filter_id) plus
code STRUCTURE, never by literal code. Billed arrays are strictly
read-only: the ONLY effect of this template is one or more calls to
v.suppress_scrub_advisory(filter_id, code, rule_id, authority, note).
FAIL findings are never suppressible (WARN-only by API contract). Use
this template ONLY for adjudicated_advisory_targets disputes with
verified_state must_not_fire, where the claim is verified correct as
billed and must replay byte-identical.

Rule fields
-----------
filter_id (string, required)
    The scrubber filter whose WARN advisory (keyed "FILTER_ID|CODE")
    this rule targets. Exact match.

applies_to (object, required)
    array: one of "icd_codes" | "cpt_codes" | "hcpcs_codes".
    code_regex: broad STRUCTURAL regex (anchor it yourself) matched
        against each candidate code -- a chapter-letter-plus-digits
        shape, never a literal code. Compiled case-insensitive.

category_title_requires_any (array of strings, optional; icd_codes only)
    Lowercase substrings; at least one must appear in the Tabular title
    of the candidate's 3-character category (the store carries category
    titles even when leaf descriptors are absent), keeping selection
    grammar-driven. If configured and the title lookup is missing or
    empty, the candidate is skipped (conservative). If configured on a
    non-ICD array, nothing selects.

require_leaf_descriptor_missing (bool, optional, default false)
    If true, a candidate qualifies only when its reference-store leaf
    descriptor is missing or empty -- the exact situation this template
    exists for; codes with a real descriptor belong to grammar-selected
    templates instead.

select_from_claim_lines (bool, optional, default true)
    Candidate (filter_id, code) pairs are gathered first from any WARN
    findings already present in coding_result["claim_scrub"]["findings"]
    (findings order; key "FILTER_ID|CODE" or explicit filter_id/code
    fields; severity WARN only), then -- if this flag is true -- from
    the applies_to array's billed lines in claim order (the scrubber
    may run after validator rules, so findings may not exist yet).
    First occurrence per code wins; duplicates are ignored.

evidence_classes (array of objects, required, non-empty)
    Each: {"label": unique non-blank string,
           "note_regex": regex (required),
           "contradiction_regex": regex (optional)}.
    Evaluated over negation-scrubbed sentences of the FULL note that
    survive exclusion_contexts. A class is DOCUMENTED when its
    note_regex matches any surviving sentence. If ANY class's
    contradiction_regex matches ANY surviving sentence, the WHOLE rule
    is a no-op. Classes are evaluated in sorted-label order; the sorted
    labels of documented classes fill the {evidence} placeholder.

exclusion_contexts (array of regexes, optional)
    A sentence matching any of these is removed before evidence
    evaluation (e.g. allergy lists, family-history sections). Any
    unparseable pattern makes the whole rule a no-op.

evidence_threshold ("all" | positive integer, optional, default "all")
    "all": every evidence class must be documented. Integer N: at least
    N classes documented. An N larger than the class count can never be
    met (permanent no-op -- do not do that).

note (string, optional)
    Audit-note template for the suppression record; placeholders
    {code} and {evidence} (comma-joined sorted documented labels).

authority (string, required by the pack)
    Governing-source citation, passed through to the suppression
    record verbatim.

No-op guarantees (conservative by design)
-----------------------------------------
The entire rule does nothing if: any regex fails to compile; the note
text is missing/empty; evidence_classes is missing, empty, or malformed
(duplicate or blank labels); any contradiction matches; the threshold
is not met; or no candidate code qualifies. The template never mutates
icd/cpt/hcpcs, never adds entries, never suppresses billed lines, and
emits no validator findings of its own -- the scrubber records each
suppression as its own PASS finding carrying rule_id and authority.

Determinism
-----------
Candidates: findings order first, then claim-line order; first
occurrence per code. Evidence labels: sorted. No dependence on dict
iteration order anywhere the outcome could change.
"""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v

    def compile_rx(pattern):
        if not isinstance(pattern, str) or not pattern.strip():
            return None
        try:
            return re.compile(pattern, re.IGNORECASE)
        except Exception:
            return None

    # --- rule identity / selection config (any defect => no-op) ---
    filter_id = rule.get("filter_id", "")
    if not isinstance(filter_id, str) or not filter_id.strip():
        return
    applies = rule.get("applies_to", {})
    if not isinstance(applies, dict):
        return
    array_name = applies.get("array", "")
    arrays = {"icd_codes": icd, "cpt_codes": cpt, "hcpcs_codes": hcpcs}
    lines = arrays.get(array_name)
    if lines is None:
        return
    code_rx = compile_rx(applies.get("code_regex", ""))
    if code_rx is None:
        return

    # --- evidence-class config ---
    classes = rule.get("evidence_classes", [])
    if not isinstance(classes, list) or not classes:
        return
    compiled_classes = []
    seen_labels = set()
    for cls in classes:
        if not isinstance(cls, dict):
            return
        label = cls.get("label", "")
        if not isinstance(label, str) or not label.strip():
            return
        if label in seen_labels:
            return
        seen_labels.add(label)
        note_rx = compile_rx(cls.get("note_regex", ""))
        if note_rx is None:
            return
        contra_rx = None
        contra_pat = cls.get("contradiction_regex", "")
        if isinstance(contra_pat, str) and contra_pat.strip():
            contra_rx = compile_rx(contra_pat)
            if contra_rx is None:
                return
        compiled_classes.append((label, note_rx, contra_rx))
    compiled_classes = sorted(compiled_classes, key=lambda item: item[0])

    exclusions = rule.get("exclusion_contexts", [])
    if not isinstance(exclusions, list):
        return
    exclusion_rxs = []
    for pat in exclusions:
        rx = compile_rx(pat)
        if rx is None:
            return
        exclusion_rxs.append(rx)

    threshold = rule.get("evidence_threshold", "all")
    if threshold == "all":
        needed = len(compiled_classes)
    elif isinstance(threshold, int) and not isinstance(threshold, bool) and threshold > 0:
        needed = threshold
    else:
        return

    # --- note text -> negation-scrubbed surviving sentences ---
    text = note_full_text if isinstance(note_full_text, str) else ""
    if not text.strip():
        return
    scrubbed = ""
    ev = v._note_evidence(text)
    if isinstance(ev, (list, tuple)) and len(ev) >= 2 and isinstance(ev[1], str):
        scrubbed = ev[1]
    if not scrubbed.strip():
        return
    sentences = []
    for raw in re.split(r"[.!?;\n]+", scrubbed):
        sent = raw.strip()
        if not sent:
            continue
        excluded = False
        for rx in exclusion_rxs:
            if rx.search(sent):
                excluded = True
                break
        if not excluded:
            sentences.append(sent)
    if not sentences:
        return

    # --- evaluate evidence classes (any contradiction => no-op) ---
    documented = []
    for label, note_rx, contra_rx in compiled_classes:
        hit = False
        for sent in sentences:
            if contra_rx is not None and contra_rx.search(sent):
                return
            if note_rx.search(sent):
                hit = True
        if hit:
            documented.append(label)
    if len(documented) < needed:
        return

    # --- candidate qualification (structure + Tabular category grammar) ---
    reqs = rule.get("category_title_requires_any", [])
    if not isinstance(reqs, list):
        return
    title_reqs = []
    for req in reqs:
        if isinstance(req, str) and req.strip():
            title_reqs.append(req.strip().lower())

    require_missing = bool(rule.get("require_leaf_descriptor_missing", False))

    def leaf_info(code):
        if array_name == "icd_codes":
            return v.db.validate_icd10(code)
        if array_name == "cpt_codes":
            return v.db.validate_cpt(code)
        return v.db.validate_hcpcs(code)

    def qualifies(code):
        if not isinstance(code, str) or not code.strip():
            return False
        if not code_rx.search(code):
            return False
        if require_missing:
            info = leaf_info(code)
            desc = ""
            if isinstance(info, dict):
                desc = info.get("description", "") or info.get("long_description", "") or ""
            if isinstance(desc, str) and desc.strip():
                return False
        if title_reqs:
            if array_name != "icd_codes":
                return False
            category = code.split(".")[0][:3]
            title = v.store.icd10_tabular_description(category)
            if not isinstance(title, str) or not title.strip():
                return False
            title_l = title.lower()
            matched = False
            for req in title_reqs:
                if req in title_l:
                    matched = True
                    break
            if not matched:
                return False
        return True

    ordered_codes = []
    seen_codes = set()
    scrub_block = coding_result.get("claim_scrub", {}) if isinstance(coding_result, dict) else {}
    findings = scrub_block.get("findings", []) if isinstance(scrub_block, dict) else []
    if isinstance(findings, list):
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            severity = finding.get("severity", "")
            if not isinstance(severity, str) or severity.upper() != "WARN":
                continue
            key = finding.get("key", "")
            fid = ""
            fcode = ""
            if isinstance(key, str) and "|" in key:
                fid = key.split("|")[0]
                fcode = key[len(fid) + 1:]
            else:
                fid = finding.get("filter_id", "")
                fcode = finding.get("code", "")
            if fid != filter_id:
                continue
            if fcode in seen_codes or not qualifies(fcode):
                continue
            seen_codes.add(fcode)
            ordered_codes.append(fcode)
    if bool(rule.get("select_from_claim_lines", True)):
        for entry in lines:
            if not isinstance(entry, dict):
                continue
            code = entry.get("code", "")
            if code in seen_codes or not qualifies(code):
                continue
            seen_codes.add(code)
            ordered_codes.append(code)
    if not ordered_codes:
        return

    # --- suppress exactly the selected advisories; claim is read-only ---
    evidence_txt = ", ".join(sorted(documented))
    note_tmpl = rule.get("note", "")
    for code in ordered_codes:
        if isinstance(note_tmpl, str) and note_tmpl.strip():
            audit_note = note_tmpl.replace("{code}", code).replace(
                "{evidence}", evidence_txt)
        else:
            audit_note = "documented evidence: " + evidence_txt
        v.suppress_scrub_advisory(
            filter_id,
            code,
            rule_id=rule.get("id", ""),
            authority=rule.get("authority", ""),
            note=audit_note,
        )
