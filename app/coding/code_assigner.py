import json
import re

from app.core.llm_client import chat_completion
from app.core.config import (
    CODING_TEMPERATURE,
    CODING_MAX_TOKENS,
    LLM_PROVIDER,
    CLAUDE_VERIFY_MODEL,
    CLAUDE_VERIFY_EFFORT,
    STRUCTURED_OUTPUTS,
)
from app.coding.schemas import (
    ICD_PASS_SCHEMA,
    CPT_PASS_SCHEMA,
    HCPCS_PASS_SCHEMA,
    VERIFY_PASS_SCHEMA,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


def _pass_schema(schema: dict) -> dict | None:
    """The structured-output schema for a pass, honoring the kill switch."""
    return schema if STRUCTURED_OUTPUTS else None


# ---------------------------------------------------------------------------
# Database Description Reconciliation
# ---------------------------------------------------------------------------

def _enrich_with_db_descriptions(entries: list[dict], code_system: str, db) -> list[dict]:
    """Lookup each assigned code in the authoritative database and attach its
    real description as ``db_description``.  The verification pass uses this
    to catch LLM hallucinations (e.g. M21.611 assigned for flat foot when the
    database says 'Bunion of right foot')."""
    if db is None:
        return entries
    for entry in entries:
        code = entry.get("code", "").strip()
        if not code:
            continue
        if code_system == "icd10":
            rec = db.validate_icd10(code)
            desc = (rec or {}).get("description", "")
        elif code_system == "cpt":
            rec = db.validate_cpt(code)
            desc = (rec or {}).get("long_description", "") or (rec or {}).get("short_description", "")
        elif code_system == "hcpcs":
            rec = db.validate_hcpcs(code)
            desc = (rec or {}).get("description", "")
        else:
            desc = ""
        entry["db_description"] = desc if desc else "NOT FOUND IN DATABASE"
    return entries


def _enforce_real_descriptions(final_result: dict, db) -> None:
    """Overwrite every code's own "description" field with the real
    database text — deterministic, not a prompt instruction. Applied last,
    to the actual returned code arrays (icd10_codes/cpt_codes/hcpcs_codes),
    after Pass 4 has already written whatever it wrote. Only overwrites
    when a real record is found; a code that doesn't validate is left
    alone here (existence is a separate concern, handled by _hard_db_gate
    upstream and validator.py's code-existence check downstream) rather
    than silently blanking a description for a code that turns out to be
    invalid.
    """
    if db is None:
        return
    for entry in final_result.get("icd10_codes", []) + final_result.get("supporting_conditions", []):
        rec = db.validate_icd10(entry.get("code", "").strip())
        if rec and rec.get("description"):
            entry["description"] = rec["description"]
    for entry in final_result.get("cpt_codes", []):
        rec = db.validate_cpt(entry.get("code", "").strip())
        if rec:
            desc = rec.get("long_description") or rec.get("short_description")
            if desc:
                entry["description"] = desc
    for entry in final_result.get("hcpcs_codes", []):
        rec = db.validate_hcpcs(entry.get("code", "").strip())
        if rec and rec.get("description"):
            entry["description"] = rec["description"]


def _build_db_description_block(combined: dict) -> str:
    """Build the authoritative code description block for the verification prompt.

    When the LLM's assigned description or rationale conflicts with the database
    description, the verification pass can catch and correct the wrong code."""
    lines = ["## AUTHORITATIVE DATABASE DESCRIPTIONS (ground truth — use to catch wrong codes)"]
    lines.append("If a code's DB_DESCRIPTION contradicts the clinical context or rationale → that code is WRONG. Fix it.\n")

    icd_lines = []
    for e in combined.get("icd10_codes", []):
        code = e.get("code", "")
        db_desc = e.get("db_description", "NOT FOUND")
        llm_desc = e.get("description", "")
        rationale = e.get("rationale", "")[:60]
        flag = " ⚠ NOT IN DATABASE" if db_desc == "NOT FOUND IN DATABASE" else ""
        icd_lines.append(f"  ICD {code}: DB='{db_desc}'{flag} | LLM='{llm_desc}' | rationale='{rationale}'")

    cpt_lines = []
    for e in combined.get("cpt_codes", []):
        code = e.get("code", "")
        db_desc = e.get("db_description", "NOT FOUND")
        llm_desc = e.get("description", "")[:60]
        flag = " ⚠ NOT IN DATABASE" if db_desc == "NOT FOUND IN DATABASE" else ""
        cpt_lines.append(f"  CPT {code}: DB='{db_desc}'{flag} | LLM='{llm_desc}'")

    hcpcs_lines = []
    for e in combined.get("hcpcs_codes", []):
        code = e.get("code", "")
        db_desc = e.get("db_description", "NOT FOUND")
        llm_desc = e.get("description", "")[:60]
        flag = " ⚠ NOT IN DATABASE" if db_desc == "NOT FOUND IN DATABASE" else ""
        hcpcs_lines.append(f"  HCPCS {code}: DB='{db_desc}'{flag} | LLM='{llm_desc}'")

    if icd_lines:
        lines.append("### ICD-10-CM")
        lines.extend(icd_lines)
    if cpt_lines:
        lines.append("### CPT")
        lines.extend(cpt_lines)
    if hcpcs_lines:
        lines.append("### HCPCS")
        lines.extend(hcpcs_lines)

    lines.append("\nCRITICAL INSTRUCTION: If any DB_DESCRIPTION does NOT match the clinical reason this")
    lines.append("code was assigned, CHANGE that code to the correct one from the RAG candidates above.")
    lines.append("The DB_DESCRIPTION is authoritative ground truth. Trust it over the LLM-assigned description.\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PASS 1 — ICD-10-CM Diagnosis Coding
# ---------------------------------------------------------------------------

ICD_SYSTEM_PROMPT = """You are a medical coding assistant selecting ICD-10-CM entries from an authoritative, effective-dated candidate list.

SOURCE OF TRUTH
- Select only from the supplied authoritative candidates and category-family descriptors.
- Treat the official descriptor, billable-leaf status, effective dates, and supplied Tabular/Index instructional notes as the definition. Never use a memorized code mapping, prefix heuristic, or example.
- Resolve laterality, encounter character, anatomy, etiology/manifestation, temporal qualifier, severity, and sequencing only when explicit note evidence and the supplied authority support them.
- If no candidate fully matches, omit the code and flag review; never invent specificity.

CLAIM SCOPE
- Put current, documented encounter diagnoses in icd10_codes.
- Keep history-only or informational conditions in supporting_conditions unless the Assessment/Plan addresses them as a current problem.
- Do not infer diagnoses from medications, measurements, procedures, family history, or clinical knowledge.
- Cite one contiguous verbatim note span for every entry.

OUTPUT
Return JSON matching the supplied schema with complete icd10_codes and supporting_conditions arrays, sequencing rationale, evidence, confidence, and review fields. JSON only."""


# ---------------------------------------------------------------------------
# PASS 2 — CPT Procedure/E&M Coding
# ---------------------------------------------------------------------------

# Runtime CPT policy is deliberately compact and source-driven. Changing code
# sets and policy examples are supplied from the effective-dated datastore,
# never duplicated in prompt prose.
CPT_SYSTEM_PROMPT = """You are a medical coding assistant selecting CPT entries from an authoritative, effective-dated candidate list.

SOURCE OF TRUTH
- Select codes only from the authoritative candidates supplied in the user message.
- Treat each candidate's official descriptor and attached CMS/AMA attributes as its definition. Do not use memorized code mappings, ranges, examples, or specialty heuristics.
- Use the effective-dated MDM grid supplied in the user message for E/M leveling. If the grid or patient status is unresolved, omit the status-specific E/M line and flag review.
- Use only the supplied NCCI, global-period, billability, and other authoritative data blocks for modifier or bundling decisions. Absence of a data block is not permission to guess.

DOCUMENTATION
- Code only services explicitly documented as performed by this billing provider on the DOS. Planned, ordered, historical, outside-party, and integral services are not separately reported.
- Match every material descriptor attribute to the note, including anatomy, extent, technique, laterality, view/count, and units. If no candidate matches, omit and flag review.
- Every line must cite one or more contiguous verbatim evidence spans from the note and link only diagnoses present in the supplied diagnosis array.

OUTPUT
Return JSON matching the supplied schema, including the complete CPT list, modifiers with structured rationale, units, diagnosis linkage, MDM details where applicable, and review flags. Return an empty list when no defensible candidate exists. JSON only."""


# ---------------------------------------------------------------------------
# PASS 3 — HCPCS Level II + SNOMED
# ---------------------------------------------------------------------------

# HCPCS identity and descriptor semantics come from the effective-dated code
# candidates supplied at runtime. Do not duplicate a changing medical code set
# in prompt prose: the former prompt did so and contradicted the repository's
# own authoritative CMS data on construction, fitting, size, and billing-unit
# axes. That duplicated mapping has been removed rather than retained as a
# second, stale source of truth.
HCPCS_SNOMED_SYSTEM_PROMPT = """You are a medical coding assistant selecting HCPCS Level II entries from an authoritative, effective-dated candidate list and optionally mapping documented clinical concepts to SNOMED CT.

HCPCS SOURCE-OF-TRUTH RULES
- Select a HCPCS code only from the authoritative candidates in the user message.
- Treat each candidate's official descriptor as the definition. Never substitute memorized mappings, infer a code outside the list, or alter its descriptor.
- A candidate list is evidence of code identity, not evidence that the service is covered, medically necessary, or separately payable.
- If no candidate descriptor matches every material documented attribute, omit the HCPCS line. Do not guess.

DOCUMENTATION GATE
- Code only an item, supply, drug, or service explicitly documented as performed, administered, applied, fitted, or physically dispensed at this encounter.
- An order, prescription, recommendation, future plan, historical use, or instruction to continue an existing item is not current dispensing or administration.
- Cite a contiguous verbatim note span for every line. Never manufacture or splice evidence.

DESCRIPTOR MATCHING
- Compare the note with every defining descriptor attribute, including item or drug identity, formulation, construction, prefabricated versus custom manufacture, fitting or customization, dimensions, quantity, laterality, and the descriptor's billing unit.
- Do not invent undocumented attributes to reach a more specific candidate.
- Compute units only from the documented quantity and the selected descriptor's billing unit. Do not encode drug quantity with laterality or bilateral modifiers.
- Add a modifier only when the documentation and an applicable authoritative rule support it. Do not infer a blanket modifier rule from a code prefix.
- If multiple candidates remain plausible, choose none unless one is clearly best supported; otherwise mark the selected line for review and state the unresolved descriptor axis.

SNOMED CT
- Do not invent or recall a concept identifier from memory. Output a SNOMED concept only when an exact concept identifier has been supplied by a controlled terminology lookup in the input.
- Map only concepts explicitly documented in the current note and use the most specific supplied concept.

OUTPUT
Return JSON matching the supplied schema. Each HCPCS entry must include code, description, confidence, modifiers, modifier_reasoning, units, linked_diagnoses, rationale, supporting_text, needs_review, and review_reason. Each SNOMED entry must include concept_id, description, entity_text, category, and confidence. Return empty arrays when no defensible entry is available.

Return JSON only. No markdown or code fences."""


# ---------------------------------------------------------------------------
# PASS 4 — Self-Verification & Correction (Anchor-and-Audit)
# ---------------------------------------------------------------------------


VERIFICATION_SYSTEM_PROMPT = """You are a medical-coding verification agent auditing a complete candidate claim.

- Every retained or introduced code must be present in the authoritative candidate/family blocks and must match explicit note evidence on every material descriptor axis.
- Use only the supplied effective-dated reference descriptions, MDM grid, NCCI status, billability status, instructional notes, and policy blocks. Never apply a memorized code mapping, code range, modifier rule, or bundling convention.
- Preserve every documented diagnosis, performed service, and dispensed item unless a supplied authority proves it is nonbillable, integral, bundled, inactive, or unsupported. Never silently drop documented work.
- Keep code arrays, modifiers, units, diagnosis linkage, correction records, and rationales mutually consistent. Record every code-value change with the old and new code.
- Evidence must be a contiguous verbatim note span. Ambiguity, missing authority, or unresolved patient status requires review rather than a guess.

Return the complete corrected claim and correction ledger using the supplied JSON schema. JSON only."""


_EVIDENCE_MIN_LEN = 14


def _resolve_patient_status(patient_metadata: dict,
                            vision_context: dict | None) -> str | None:
    """Resolve new/established only from explicit extracted encounter facts.

    The old boolean default classified every blank or unfamiliar note type as
    established.  Conflicting or absent evidence is now an unresolved state,
    which prevents the coder from guessing an E/M family.
    """
    values = [str((patient_metadata or {}).get("note_type") or ""),
              str((vision_context or {}).get("note_category") or "")]
    found: set[str] = set()
    for value in values:
        normalized = value.lower().replace("_", " ").replace("-", " ")
        if re.search(r"\bnew\s+patient\b", normalized):
            found.add("new patient")
        if re.search(r"\bestablished\s+patient\b", normalized):
            found.add("established patient")
    return next(iter(found)) if len(found) == 1 else None


def _build_mdm_reference_block(store, dos) -> str:
    """Render the effective-dated MDM grid already loaded from source data."""
    if store is None:
        return ("## AUTHORITATIVE E/M MDM GRID\n"
                "Unavailable — do not assign an MDM-leveled E/M code.")
    try:
        from app.compliance.engine import _parse_dos
        parsed_dos = _parse_dos({"date_of_service": dos})
        grid = store.mdm_grid(parsed_dos) if parsed_dos is not None else None
    except Exception:
        grid = None
    if not grid:
        return ("## AUTHORITATIVE E/M MDM GRID\n"
                "No effective grid covers the DOS — do not assign an "
                "MDM-leveled E/M code.")
    payload = {
        "source": grid.get("source", ""),
        "effective_from": grid.get("effective_from", ""),
        "selection_rule": grid.get("selection_rule", ""),
        "time_rule": grid.get("time_rule", ""),
        "levels": grid.get("levels") or {},
    }
    return ("## AUTHORITATIVE E/M MDM GRID (effective for this DOS)\n"
            + json.dumps(payload, indent=2, sort_keys=True))


def _candidate_code(system: str, code) -> str:
    value = str(code or "").strip().upper()
    return value.replace(".", "") if system == "icd10" else value


def _expand_allowed_icd_family_candidates(codes: set[str], db) -> set[str]:
    """Add authoritative siblings for small retrieved ICD categories.

    Verification is shown the same family for descriptor disambiguation, so
    its evidence-backed sibling correction must be able to survive the final
    candidate gate. Large categories remain retrieval-bound to keep the model
    prompt and selection surface bounded.
    """
    expanded = {_candidate_code("icd10", code) for code in codes if code}
    if db is None:
        return expanded
    for offered in list(expanded):
        try:
            siblings = db.icd10_siblings(offered[:3])
        except Exception as exc:
            logger.warning("ICD family expansion unavailable for %s: %s",
                           offered, exc)
            continue
        if 1 < len(siblings) <= _ICD_FAMILY_SIZE_CAP:
            expanded.update(_candidate_code("icd10", code)
                            for code, _description in siblings)
    return expanded


def _evidence_norm(text: str) -> str:
    """Whitespace/case/punctuation-normalized text for verbatim matching —
    the same normalization the validator's downstream check uses, so the two
    agree on what 'verbatim' means."""
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower())).strip()


def _strip_nonverbatim_spans(result: dict, note_text: str) -> None:
    """Strip cited quotes that do not actually appear in the note, in place.
    Each evidence_spans entry / supporting_text must be a CONTIGUOUS substring
    of the note (a space-stripped fallback tolerates a line break inside a
    real quote but not a splice, whose fragments are far apart). A short
    fragment is left alone (too small to judge). Fail-open on missing/short
    note text — never strip when the source cannot be verified."""
    if not note_text or len(note_text) < 40:
        return
    note_n = _evidence_norm(note_text)
    note_ns = note_n.replace(" ", "")

    def _verbatim(sp) -> bool:
        sn = _evidence_norm(sp)
        return (len(sn) >= _EVIDENCE_MIN_LEN and
                (sn in note_n or sn.replace(" ", "") in note_ns))

    stripped = 0
    for arr_key, field, is_list in (("cpt_codes", "evidence_spans", True),
                                    ("hcpcs_codes", "evidence_spans", True),
                                    ("hcpcs_codes", "supporting_text", False),
                                    ("icd10_codes", "supporting_text", False)):
        for e in result.get(arr_key, []) or []:
            if not isinstance(e, dict):
                continue
            raw = e.get(field)
            if is_list:
                spans = raw if isinstance(raw, list) else []
                kept = [s for s in spans if _verbatim(s)]
                if len(kept) != len(spans):
                    stripped += len(spans) - len(kept)
                    e[field] = kept
            elif raw and not _verbatim(raw):
                stripped += 1
                e[field] = ""
    if stripped:
        logger.info(f"  Evidence gate: stripped {stripped} non-verbatim "
                    f"citation(s) from coder output (not found in the note)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def assign_codes(
    note_text: str,
    note_sections: dict,
    patient_metadata: dict,
    entities: list[dict],
    rag_candidates: dict,
    vision_context: dict | None = None,
    prior_surgery_info: dict | None = None,
    db=None,
    physician_documented_codes: list[dict] | None = None,
    store=None,
    exemplar_block: str = "",
) -> tuple[dict, dict]:
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                   "cache_read_tokens": 0, "cache_write_tokens": 0}

    note_context = _build_note_context(note_sections, patient_metadata)
    # Verified-claim exemplars (live mode only — empty otherwise). Appended
    # to the shared note context so the ICD and CPT passes both see the
    # worked examples in a stable, cache-friendly position.
    if exemplar_block:
        note_context = f"{note_context}\n\n{exemplar_block}"
    # Coding memorandum: the rule pack's PROVEN corrections compiled into
    # upstream guidance, so the generative passes stop re-making error
    # classes the deterministic stack already corrects. Appended to the
    # shared context (all four passes see it); recompiles automatically
    # when the pack changes; CODING_MEMORANDUM=0 disables.
    from app.coding.memorandum import memorandum_block
    memo = memorandum_block()
    if memo:
        note_context = f"{note_context}\n\n{memo}"
    entity_summary = _format_entities(entities)
    vision_block = _format_vision_context(vision_context) if vision_context else ""
    global_block = _format_global_period_context(
        prior_surgery_info, store=store) if prior_surgery_info else ""

    # A generative pass may select only from the immutable retrieval set or
    # an exact code visibly documented by the physician.  Reference-database
    # existence alone is not a candidate source: otherwise a model can emit
    # any currently valid code from memory and still pass the old DB gate.
    allowed_codes = {
        system: {
            _candidate_code(system, row.get("code"))
            for row in rag_candidates.get(system, [])[:25]
            if isinstance(row, dict)
        }
        for system in ("icd10", "cpt", "hcpcs")
    }
    for row in physician_documented_codes or []:
        if not isinstance(row, dict):
            continue
        system = str(row.get("system") or "").strip().lower()
        code = _candidate_code(system, row.get("code"))
        if system in allowed_codes and code:
            allowed_codes[system].add(code)

    # Small ICD category families are an authoritative candidate-expansion
    # mechanism, not model memory.  The verification pass is intentionally
    # shown every sibling when retrieval may have anchored on the wrong one;
    # those same siblings must therefore be eligible to survive the final
    # candidate gate.  Expansion remains bounded by the family-size cap and
    # every member comes from the loaded code reference.
    allowed_codes["icd10"] = _expand_allowed_icd_family_candidates(
        allowed_codes["icd10"], db)

    # --- PASS 1: ICD-10-CM ---
    logger.info("  Pass 1/4: ICD-10-CM diagnosis coding...")
    icd_prompt = f"""{note_context}
{vision_block}
{global_block}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## ICD-10-CM CANDIDATE CODES (effective-dated authoritative retrieval)
{_format_candidates_for_system(rag_candidates, 'icd10')}

Assign ICD-10-CM codes following the billable/advisory split:
- icd10_codes: current encounter diagnoses documented in the Assessment/Plan
- supporting_conditions: history-only or informational conditions not managed today
Use only candidate descriptors and supplied instructional data to decide whether an
additional status, manifestation, etiology, or other companion code is supported."""

    icd_raw, usage = chat_completion(ICD_SYSTEM_PROMPT, icd_prompt, temperature=CODING_TEMPERATURE,
                                     max_tokens=2500, json_schema=_pass_schema(ICD_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    icd_result = _safe_parse(icd_raw, "icd10_codes")
    # Hard DB gate — remove any hallucinated/invalid ICD codes immediately
    icd_result["icd10_codes"] = _hard_db_gate(
        icd_result.get("icd10_codes", []), "icd10", db,
        allowed_codes["icd10"])
    icd_result["supporting_conditions"] = _hard_db_gate(
        icd_result.get("supporting_conditions", []), "icd10", db,
        allowed_codes["icd10"])
    logger.info(f"    → {len(icd_result.get('icd10_codes', []))} ICD-10-CM codes, "
                f"{len(icd_result.get('supporting_conditions', []))} supporting conditions")

    # --- PASS 2: CPT ---
    logger.info("  Pass 2/4: CPT procedure/E&M/imaging coding...")
    icd_summary = _summarize_icd(icd_result.get("icd10_codes", []))
    patient_status = _resolve_patient_status(patient_metadata, vision_context)
    is_post_op = (prior_surgery_info or {}).get("is_post_op_visit", False)
    days_post_op = (prior_surgery_info or {}).get("days_post_op")
    prior_cpt = (prior_surgery_info or {}).get("prior_surgery_cpt", "")

    mdm_block = _build_mdm_reference_block(
        store, patient_metadata.get("date_of_service"))

    cpt_prompt = f"""{note_context}
{vision_block}
{global_block}

## PATIENT STATUS: {patient_status.upper() if patient_status else "UNRESOLVED — DO NOT GUESS AN E/M FAMILY"}
{f"## POST-OPERATIVE CONTEXT: Prior surgery {prior_cpt or 'unknown'}, day {days_post_op or '?'}; use only the authoritative global-period block below." if is_post_op else ""}

## ASSIGNED ICD-10-CM CODES (from Pass 1)
{icd_summary}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## CPT CANDIDATE CODES (from official database via semantic search)
## Candidate annotations are authoritative attributes, not standalone billing
## decisions. Do not infer a modifier solely from a global-period value.
{_format_candidates_for_system(rag_candidates, 'cpt', store=store)}

{mdm_block}

Assign CPT codes and link each line only to diagnoses in the supplied claim.
Treat modifier selection and separate-reportability as unresolved unless the
note plus a supplied authoritative rule block establishes them. The downstream
deterministic compliance engine owns final NCCI, global-period, and modifier
adjudication; do not replace an absent rule with memorized policy.
For imaging and every other family, select solely by comparing the documented
attributes with the official candidate descriptors.
Do not make a pairwise bundling claim until the authoritative NCCI block is
available in verification."""

    cpt_raw, usage = chat_completion(CPT_SYSTEM_PROMPT, cpt_prompt, temperature=CODING_TEMPERATURE,
                                     max_tokens=2500, json_schema=_pass_schema(CPT_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    cpt_result = _safe_parse(cpt_raw, "cpt_codes")
    cpt_result["cpt_codes"] = _hard_db_gate(
        cpt_result.get("cpt_codes", []), "cpt", db,
        allowed_codes["cpt"])
    logger.info(f"    → {len(cpt_result.get('cpt_codes', []))} CPT codes")

    # --- PASS 3: HCPCS + SNOMED ---
    logger.info("  Pass 3/4: HCPCS + SNOMED coding...")
    hcpcs_prompt = f"""{note_context}
{vision_block}

## ASSIGNED ICD-10-CM CODES
{icd_summary}

## CPT CODES (for laterality reference)
{_summarize_cpt(cpt_result.get('cpt_codes', []), store=store)}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## HCPCS CANDIDATE CODES (from official database via semantic search)
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

Assign HCPCS codes only from the authoritative candidates for items or drugs documented as
physically dispensed, applied, fitted, or administered today. Match every defining descriptor
attribute and billing unit. Apply modifiers only when documentation and an applicable rule support
them. For SNOMED, omit any concept whose identifier was not supplied by controlled terminology
data. Do not code ordered, prescribed, recommended, historical, or continued items."""

    hcpcs_raw, usage = chat_completion(HCPCS_SNOMED_SYSTEM_PROMPT, hcpcs_prompt, temperature=CODING_TEMPERATURE,
                                       max_tokens=2500, json_schema=_pass_schema(HCPCS_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    hcpcs_result = _safe_parse(hcpcs_raw, "hcpcs_codes")
    hcpcs_result["hcpcs_codes"] = _hard_db_gate(
        hcpcs_result.get("hcpcs_codes", []), "hcpcs", db,
        allowed_codes["hcpcs"])
    logger.info(f"    → {len(hcpcs_result.get('hcpcs_codes', []))} HCPCS, {len(hcpcs_result.get('snomed_codes', []))} SNOMED")

    # --- PASS 4: Constrained Self-Verification (Anchor-and-Audit) ---
    logger.info("  Pass 4/4: Constrained verification (anchor-and-audit)...")
    combined = {
        "icd10_codes": icd_result.get("icd10_codes", []),
        "supporting_conditions": icd_result.get("supporting_conditions", []),
        "cpt_codes": cpt_result.get("cpt_codes", []),
        "hcpcs_codes": hcpcs_result.get("hcpcs_codes", []),
        "snomed_codes": hcpcs_result.get("snomed_codes", []),
        "em_level_reasoning": cpt_result.get("em_level_reasoning", ""),
    }

    # Enrich every assigned code with its authoritative database description.
    # This gives the verification LLM ground truth to detect mismatches
    # (e.g. LLM assigned M21.611 for "flat foot" → DB says "Bunion of right foot" → wrong).
    _enrich_with_db_descriptions(combined["icd10_codes"],  "icd10",  db)
    _enrich_with_db_descriptions(combined["cpt_codes"],    "cpt",    db)
    _enrich_with_db_descriptions(combined["hcpcs_codes"],  "hcpcs",  db)
    db_description_block = _build_db_description_block(combined)
    ncci_pair_block = _build_ncci_pair_block(
        combined["cpt_codes"], rag_candidates.get("cpt", []), store,
        patient_metadata.get("date_of_service"),
    )
    billability_block = _build_billability_block(
        combined["cpt_codes"], combined["hcpcs_codes"], store
    )
    icd_excludes1_block = _build_icd_excludes1_block(combined["icd10_codes"], store)
    code_family_block = _build_code_family_block(combined["cpt_codes"], rag_candidates.get("cpt", []), db)
    icd_family_block = _build_icd_family_block(
        combined["icd10_codes"] + combined["supporting_conditions"], db
    )

    assessment_text = note_sections.get("assessment_diagnoses", "")
    pmh_text = note_sections.get("pmh_medications_allergies", "")
    anchor_block = _build_anchor_block(assessment_text, pmh_text, vision_context, prior_surgery_info)

    physician_block = _format_physician_codes(physician_documented_codes or [])

    verify_prompt = f"""{note_context}
{vision_block}
{global_block}

## PATIENT STATUS: {patient_status.upper() if patient_status else "UNRESOLVED"}

{anchor_block}

{physician_block}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

{db_description_block}

{ncci_pair_block}

{billability_block}

{icd_excludes1_block}

{code_family_block}

{icd_family_block}

## CURRENTLY ASSIGNED CODES (to audit)
{json.dumps(combined, indent=2)}

## RAG CANDIDATE CODES (verified in official databases)
### ICD-10-CM Candidates
{_format_candidates_for_system(rag_candidates, 'icd10')}

### CPT Candidates (annotated with actual CMS global periods)
{_format_candidates_for_system(rag_candidates, 'cpt', store=store)}

### HCPCS Candidates
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

## AUDIT INSTRUCTIONS
1. Reconcile every protected diagnosis, performed-service, and dispensed-item anchor with the complete claim. Add or retain a line only when an effective-dated candidate descriptor matches explicit note evidence.
2. Use the supplied authoritative database descriptions and family-disambiguation blocks to correct code identity. Record every code-field change in corrections_made; changing only prose is not a correction.
3. Apply the NCCI pair-status, billability, ICD instructional-note, global-period, MDM, and coverage blocks exactly as supplied. Do not invent an edit, modifier, code mapping, or coverage rule when a block is absent or inconclusive; flag review instead.
4. Preserve diagnosis-pointer, modifier_reasoning, units, evidence_spans, and correction-ledger consistency after every change.
5. Remove unsupported lines and nonverbatim evidence. Keep documented work when no supplied authority establishes that it is integral or nonreportable.
6. Return the complete corrected arrays, including supporting_conditions in their original role except for evidence-backed identity corrections."""

    # Elastic max_tokens: scale with note complexity so simple notes are fast
    # and complex notes get enough room without hitting API limits.
    # Signals: entity count, total codes assigned, tokens consumed so far.
    _n_entities = len(entities or [])
    _n_codes = (
        len(combined.get("icd10_codes", []))
        + len(combined.get("supporting_conditions", []))
        + len(combined.get("cpt_codes", []))
        + len(combined.get("hcpcs_codes", []))
        + len(combined.get("snomed_codes", []))
    )
    _token_pressure = min(total_usage.get("total_tokens", 0) / 5000, 10)  # 0-10 scale
    _complexity = _n_entities + (_n_codes * 2) + _token_pressure
    # Scale 4000 (simple) → 10000 (very complex), capped at safe API max
    _verify_max_tokens = max(4000, min(10000, int(4000 + (_complexity / 90) * 6000)))
    logger.info(f"    verify max_tokens={_verify_max_tokens} (entities={_n_entities}, codes={_n_codes}, token_pressure={_token_pressure:.1f})")
    # Escalation tiering: the verify pass optionally runs on a stronger
    # model/effort (CLAUDE_VERIFY_MODEL / CLAUDE_VERIFY_EFFORT) — it is the
    # one judgment-concentrated call per note. Claude-provider only.
    _verify_model = (CLAUDE_VERIFY_MODEL or None) if LLM_PROVIDER == "claude" else None
    _verify_effort = (CLAUDE_VERIFY_EFFORT or None) if LLM_PROVIDER == "claude" else None
    if _verify_model or _verify_effort:
        logger.info(f"    verify tier: model={_verify_model or 'default'}, "
                    f"effort={_verify_effort or 'default'}")
    verify_raw, usage = chat_completion(VERIFICATION_SYSTEM_PROMPT, verify_prompt,
                                        temperature=CODING_TEMPERATURE, max_tokens=_verify_max_tokens,
                                        model=_verify_model, effort=_verify_effort,
                                        json_schema=_pass_schema(VERIFY_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    verified = _safe_parse(verify_raw, "icd10_codes")

    # Fix 7 — added/changed-correction enforcement + modifier hygiene
    verified = _normalize_corrections(verified)
    verified = _enforce_added_corrections(verified, db)
    verified = _enforce_changed_corrections(verified, db)
    verified = _strip_invalid_cpt_modifiers(verified, store)

    corrections = verified.get("corrections_made", [])
    if corrections:
        logger.info(f"    → {len(corrections)} corrections made:")
        for c in corrections:
            logger.info(f"      [{c.get('type', '?')}] {c.get('code', '?')}: {c.get('reason', '')[:70]}")
    else:
        logger.info("    → No corrections needed")

    final_result = {
        # .get(key, fallback), not `verified.get(key) or fallback` — the
        # verification pass legitimately returning an empty list (e.g. "zero
        # billable diagnoses after audit") is a real correction, not a
        # missing field; `or` can't distinguish that from a parse failure
        # that omitted the key entirely, and silently discarded the real
        # correction by reverting to the pre-verification list.
        "icd10_codes": verified.get("icd10_codes", combined["icd10_codes"]),
        "supporting_conditions": verified.get("supporting_conditions", combined["supporting_conditions"]),
        "cpt_codes": verified.get("cpt_codes", combined["cpt_codes"]),
        "hcpcs_codes": verified.get("hcpcs_codes", combined["hcpcs_codes"]),
        "snomed_codes": verified.get("snomed_codes", combined["snomed_codes"]),
        "em_level_reasoning": verified.get("em_level_reasoning", combined["em_level_reasoning"]),
        "corrections_made": corrections,
        "audit_notes": verified.get("audit_notes", ""),
        "auto_coding_review_reasons": verified.get("auto_coding_review_reasons", []),
        "auto_coding_summary": verified.get("auto_coding_summary", ""),
    }

    # Pass 4 is the only pass whose output was never re-gated against the
    # reference DB — codes IT introduces (a re-added anchor the Pass-1 gate
    # already removed, or a wholly new line) reached the claim unvalidated.
    # Observed live twice in one batch: D48.1 (non-billable category header,
    # re-added as a bare string) and CPT 20926 (deleted code, added as a new
    # line with modifiers). Same gate policy as Passes 1-3: a code must
    # either have survived its own pass's gate (present pre-verification) or
    # validate in its claimed system now.
    final_result = _gate_verify_additions(
        final_result, combined, db, store, allowed_codes)

    # Pass 4 rewrites whole entries and routinely omits fields it wasn't
    # asked to change — observed live: every ICD entry came back without
    # "type", so the schema default ("secondary") silently erased the
    # claim's primary-diagnosis designation on three notes in one batch.
    # Deterministic merge: any contracted field missing from a verified
    # entry is inherited from the same code's pre-verification entry.
    _inherit_dropped_fields(final_result, combined)

    # Overwrite every code's description with the real database text —
    # deterministic, not prompt-following-dependent. Root cause: Pass 4 can
    # write anything into a code's own "description" field, and it isn't
    # just cosmetic — the LLM was observed using its own fabricated
    # descriptor text as justification for a coding decision within the
    # same completion (28730's real long_description is "Arthrodesis,
    # midtarsal or tarsometatarsal, multiple or transverse;" — no bone-graft
    # language at all; Pass 4 appended fabricated "with or without primary
    # bone graft (includes obtaining graft)" text and cited it to justify
    # dropping 20900, despite _build_db_description_block already showing
    # the real descriptor as ground truth, and despite the real NCCI pair
    # block correctly showing no edit exists — a second, different
    # fabrication after the NCCI-based one was already closed). Enrichment
    # (_enrich_with_db_descriptions) already computes the real description
    # for prompt context; this applies the same lookup to the FINAL output
    # so a fabricated descriptor can never survive into the returned data
    # regardless of what Pass 4 wrote.
    _enforce_real_descriptions(final_result, db)

    # SOURCE-side verbatim gate: the coder is INSTRUCTED to quote the note
    # verbatim in evidence_spans/supporting_text, but the model demonstrably
    # splices/paraphrases (measured: the 27654 evidence splice). Strip any
    # citation that is not a contiguous substring of the note BEFORE the
    # result is persisted, so a fabricated citation never enters the record.
    # The validator's identical downstream check remains as the net for
    # codes ADDED after coding (rule adds, adjudication) and for replay.
    _strip_nonverbatim_spans(final_result, note_text)

    # Fix 1 — Tag every code with its provenance and detect physician code replacements
    final_result = _tag_code_sources(final_result, physician_documented_codes or [], entities)

    logger.info(
        f"  Final: {len(final_result['icd10_codes'])} ICD, "
        f"{len(final_result['supporting_conditions'])} advisory, "
        f"{len(final_result['cpt_codes'])} CPT, "
        f"{len(final_result['hcpcs_codes'])} HCPCS, "
        f"{len(final_result['snomed_codes'])} SNOMED "
        f"(total tokens: {total_usage['total_tokens']})"
    )

    return final_result, total_usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_note_context(sections: dict, metadata: dict) -> str:
    return f"""## PATIENT METADATA
- Patient: {metadata.get('patient_name', 'Unknown')}
- DOB: {metadata.get('date_of_birth', 'Unknown')}
- DOS: {metadata.get('date_of_service', 'Unknown')}
- Provider: {metadata.get('provider', 'Unknown')}
- Insurance: {metadata.get('insurance', 'Unknown')}
- Note Type: {metadata.get('note_type', 'Unknown')}

## CLINICAL NOTE

### CHIEF COMPLAINT
{sections.get('chief_complaint', 'N/A')}

### HISTORY OF PRESENT ILLNESS
{sections.get('hpi', 'N/A')}

### PAST MEDICAL HISTORY / MEDICATIONS / ALLERGIES
{sections.get('pmh_medications_allergies', 'N/A')}

### PHYSICAL EXAMINATION
{sections.get('physical_examination', 'N/A')}

### IMAGING / DIAGNOSTICS
{sections.get('imaging_diagnostics', 'N/A')}

### ASSESSMENT / DIAGNOSES
{sections.get('assessment_diagnoses', 'N/A')}

### PLAN
{sections.get('plan', 'N/A')}"""


def _format_entities(entities: list[dict]) -> str:
    if not entities:
        return "No entities extracted."
    lines = []
    for e in entities:
        lat = f" [{e.get('laterality', '')}]" if e.get("laterality") else ""
        spec = f" — {e.get('specificity', '')}" if e.get("specificity") else ""
        ner = e.get("ner_source", "llm")
        # [G] = GLiNER-confirmed (biomedical NER validated), [L] = LLM-only
        ner_tag = "[G]" if ner == "gliner_confirmed" else "[L]"
        status = e.get("normalization_status", "not_evaluated")
        normalized = str(e.get("normalized_text") or "").strip()
        raw = str(e.get("text") or "").strip()
        terminology = ""
        if status == "accepted" and normalized and normalized.casefold() != raw.casefold():
            terminology = f", governed expansion: \"{normalized}\""
        elif status in {"ambiguous", "unresolved"}:
            terminology = ", unresolved shorthand: use the verbatim note only; do not guess"
        lines.append(
            f"- {ner_tag} [{e.get('category', '?').upper():>14}] {e.get('clinical_term', '')}{lat}{spec} "
            f"(section: {e.get('source_section', '?')}, text: \"{raw}\"{terminology})"
        )
    return "\n".join(lines)


def _build_billability_block(cpt_codes: list[dict], hcpcs_codes: list[dict], store) -> str:
    """Real billability status for every currently-assigned CPT/HCPCS code —
    computed fresh from compliance.db (store.not_separately_billable_reason),
    same pattern as _build_ncci_pair_block.

    Root cause this replaces: a wound culture with no billable CPT (correctly
    reasoned as "billed by the reference lab, not the podiatrist") got
    4261F substituted instead of no code at all — a CPT Category II
    performance-tracking code with zero RVU value by AMA design. Injecting
    the real billability status here lets Pass 4 catch it even if Pass 2
    already made the substitution.
    """
    if not store:
        return ""
    lines = []
    advisory_lines = []
    for c in list(cpt_codes) + list(hcpcs_codes):
        code = c.get("code", "")
        if not code:
            continue
        reason = store.not_separately_billable_reason(code)
        if reason:
            lines.append(f"  {code}: NOT SEPARATELY BILLABLE — {reason}")
            continue
        advisory = store.pfs_exclusion_advisory(code)
        if advisory:
            advisory_lines.append(f"  {code}: PFS-EXCLUDED (advisory) — {advisory}")
    if not lines and not advisory_lines:
        return ""
    block = [
        "## BILLABILITY STATUS (real data from compliance.db — authoritative)",
        "A code listed as NOT SEPARATELY BILLABLE is not separately payable under any payer,",
        "by AMA/CMS design — not a coding choice to reconsider. If it was added as a substitute",
        "for a service this provider doesn't bill for (e.g. reference lab processing), REMOVE",
        "it — the correct code set for that service is no code at all, not this placeholder.",
    ]
    block.extend(lines)
    if advisory_lines:
        block.append(
            "A code listed as PFS-EXCLUDED is real and payable, but under a DIFFERENT fee\n"
            "schedule (labs → CLFS, DME/supplies → DMEPOS) — keep it ONLY if THIS provider\n"
            "performs and bills that service (e.g. in-office CLIA lab, DMEPOS-enrolled\n"
            "supplier); if an outside lab/supplier bills it, remove it from this claim."
        )
        block.extend(advisory_lines)
    return "\n".join(block)


def _build_code_family_block(cpt_codes: list[dict], cpt_candidates: list[dict], db) -> str:
    """Real CPT code-family disambiguation — when an assigned CPT code
    shares its descriptor stem (the text before the first semicolon) with
    another code among the current candidates, show every family member's
    FULL real descriptor side by side, computed fresh from cpt_codes.json.

    Root cause this addresses: 28120/28122/28124 all share the stem
    "Partial excision ... bone (eg, osteomyelitis or bossing)", differing
    only in the anatomy clause after the semicolon (28120 = talus or
    calcaneus; 28122 = tarsal or metatarsal bone, EXCEPT talus or
    calcaneus; 28124 = phalanx of toe). Observed live: a navicular excision
    (a tarsal bone, not talus/calcaneus) was coded 28120 with reasoning
    that literally quoted 28120's real "talus or calcaneus" descriptor and
    then concluded it was a "verbatim match" for the navicular anyway — a
    reasoning failure, not a data-fabrication one (the real descriptor was
    already shown correctly). Unlike NCCI/billability/Excludes1 conflicts,
    which one CPT family member applies depends on which anatomy is
    documented — not decidable from structured data alone — so this can
    only be a real-data disambiguation aid, not an auto-correcting
    deterministic backstop; the actual match against documented anatomy is
    still the LLM's job, just with the full, real family laid out so a
    single member's descriptor can't be skimmed and misapplied.
    """
    if not db or not cpt_codes:
        return ""
    assigned_codes = {c.get("code", "") for c in cpt_codes if c.get("code")}
    candidate_codes = {c.get("code", "") for c in cpt_candidates if c.get("code")}
    relevant = assigned_codes | candidate_codes

    families: dict[str, list[str]] = {}
    for code in assigned_codes:
        rec = db.validate_cpt(code)
        desc = (rec or {}).get("long_description", "")
        if ";" not in desc:
            continue
        stem = desc.split(";")[0].strip()
        families.setdefault(stem, [])

    if not families:
        return ""

    # Second pass: find every relevant code (assigned or candidate) sharing
    # each stem found among assigned codes.
    lines: list[str] = []
    seen_stems: set[str] = set()
    for code in assigned_codes:
        rec = db.validate_cpt(code)
        desc = (rec or {}).get("long_description", "")
        if ";" not in desc:
            continue
        stem = desc.split(";")[0].strip()
        if stem in seen_stems:
            continue
        members = []
        for other in relevant:
            other_rec = db.validate_cpt(other)
            other_desc = (other_rec or {}).get("long_description", "")
            if other_desc.split(";")[0].strip() == stem:
                members.append((other, other_desc))
        if len(members) < 2:
            continue
        seen_stems.add(stem)
        lines.append(f'  Family "{stem}":')
        for m_code, m_desc in sorted(members):
            marker = " <- ASSIGNED" if m_code in assigned_codes else ""
            lines.append(f"    {m_code}: {m_desc}{marker}")

    if not lines:
        return ""
    return (
        "## CPT CODE FAMILY DISAMBIGUATION (real data from cpt_codes.json — authoritative)\n"
        "An assigned code below shares its descriptor stem with other real codes — they are\n"
        "NOT interchangeable. Read every family member's FULL descriptor, including any\n"
        '"except"/anatomy-restricting clause after the semicolon, and confirm the assigned\n'
        "code's specific clause — not just the shared stem — actually matches the documented\n"
        "anatomy. Do not assume the first-considered or most general-sounding family member is\n"
        "correct without checking whether a more specific member's clause fits better.\n"
        + "\n".join(lines)
    )


# ICD-10 category families above this size are excluded — real-world sizes
# range from single-digit (Z88.x: 10 drug-allergy-category codes) to
# thousands (max observed: 3096; median 9). A small family like Z88.x is a
# genuinely enumerable "pick one of N real options" choice worth showing in
# full; dumping dozens-to-thousands of siblings from a large category (e.g.
# M19: 52, E11: 87) would be noise, not disambiguation help, for a category
# that's usually distinguished by laterality/episode/severity rather than a
# fixed small enumeration.
_ICD_FAMILY_SIZE_CAP = 15


def _build_icd_family_block(icd_codes: list[dict], db) -> str:
    """Real ICD-10-CM category-family disambiguation — the ICD-10 analog of
    _build_code_family_block. Unlike CPT (siblings share a literal
    descriptor stem), ICD-10 siblings share only a category prefix (e.g.
    "Z88"); each sibling's full description differs entirely.

    Root cause this addresses: aspirin (an analgesic/NSAID) was coded
    Z88.5 ("Allergy status to narcotic agent") instead of Z88.6 ("...
    analgesic agent") — the assigned code's own real description was shown
    and stored correctly (not fabricated), but the LLM mapped the
    documented allergen to the wrong sibling in a 10-member category it
    never saw laid out side by side. Shown for EVERY assigned code's
    category, not filtered to RAG-retrieved candidates — the retrieval
    step itself can be the thing that anchored on the wrong sibling (e.g.
    if it only ever retrieved Z88.5 for an "aspirin allergy" query), so
    filtering to candidates would show nothing for exactly the cases where
    this matters most.
    """
    if not db:
        return ""
    lines: list[str] = []
    seen_prefixes: set[str] = set()
    for entry in icd_codes:
        code = entry.get("code", "")
        if not code:
            continue
        prefix = code.replace(".", "").strip().upper()[:3]
        if not prefix or prefix in seen_prefixes:
            continue
        siblings = db.icd10_siblings(prefix)
        if len(siblings) < 2 or len(siblings) > _ICD_FAMILY_SIZE_CAP:
            continue
        seen_prefixes.add(prefix)
        assigned_norm = code.replace(".", "").strip().upper()
        lines.append(f'  Category "{prefix}":')
        for s_code, s_desc in siblings:
            marker = " <- ASSIGNED" if s_code == assigned_norm else ""
            lines.append(f"    {s_code}: {s_desc}{marker}")

    if not lines:
        return ""
    return (
        "## ICD-10-CM CATEGORY FAMILY DISAMBIGUATION (real data — authoritative)\n"
        "An assigned code below belongs to a small category with other real, distinct sibling\n"
        "codes — they are NOT interchangeable, and each has its own complete, different\n"
        "description. Compare the documented specific term (e.g. a specific drug/allergen name)\n"
        "against every sibling's actual description, not just the assigned code's — a documented\n"
        "term can belong to a different sibling's category than the one currently assigned.\n"
        + "\n".join(lines)
    )


def _build_icd_excludes1_block(icd_codes: list[dict], store) -> str:
    """Real ICD-10-CM Type 1 Excludes ("not coded here") conflicts among the
    currently-assigned ICD-10 codes — computed fresh from compliance.db,
    same pattern as _build_ncci_pair_block/_build_billability_block.

    Root cause this replaces: M12.571 (Traumatic arthropathy) and M19.171
    (Post-traumatic osteoarthritis) coded together — M12.5's own Tabular
    List entry carries an explicit excludes1 note referencing M19.1;
    structurally mutually exclusive, not two similar codes to pick between.
    Which one is correct is a documentation-reading judgment call (per AHA
    Coding Clinic guidance on this exact pair: default to the
    osteoarthritis code unless documentation specifies a non-osteoarthritis
    traumatic arthropathy) — this block surfaces the real conflict so that
    judgment gets applied deliberately instead of both codes silently
    coexisting.
    """
    if not store or not icd_codes:
        return ""
    codes = [c.get("code", "") for c in icd_codes if c.get("code")]
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            key = tuple(sorted((codes[i], codes[j])))
            if key in seen:
                continue
            seen.add(key)
            if store.excludes1_conflict(codes[i], codes[j]):
                lines.append(
                    f"  {codes[i]} + {codes[j]}: TYPE 1 EXCLUDES CONFLICT — these are structurally "
                    f"mutually exclusive per the ICD-10-CM Tabular List; cannot both be coded on "
                    f"this claim"
                )
    if not lines:
        return ""
    return (
        "## ICD-10-CM EXCLUDES1 CONFLICTS (real data from compliance.db — authoritative)\n"
        'A pair listed here is a real Type 1 Excludes ("not coded here") relationship — CMS\'s\n'
        "own Tabular List says these two conditions cannot be coded together, not a stylistic\n"
        "choice between similar codes. Read the documentation to determine which condition is\n"
        "actually supported and remove the other; do not keep both.\n"
        + "\n".join(lines)
    )


def _build_ncci_pair_block(
    cpt_codes: list[dict], cpt_candidates: list[dict], store, dos=None,
) -> str:
    """Real, authoritative NCCI PTP pairwise status for every pair among the
    currently-assigned CPT codes, plus each assigned code against the top
    not-yet-assigned candidates — computed fresh from compliance.db every
    time, the same way global_period is already injected as [global=XXX] on
    CPT candidates (see _format_candidates_for_system), rather than a hand-
    written bundling rule per scenario.

    Root cause this replaces: a documented bone graft (20900) was silently
    dropped with the stated reason "bundled into the arthrodesis per
    standard NCCI convention" — store.ncci_pair('28730', '20900') returns
    None; no such edit exists. The LLM had no way to check this at
    generation time and pattern-matched a plausible-sounding but false
    claim. Injecting the real answer here removes the need to guess.
    """
    if not store or not cpt_codes:
        return ""
    if not dos or not store.ncci_data_available(dos):
        return (
            "## NCCI PAIR STATUS\n"
            "The applicable NCCI release is unavailable for this date of service. "
            "Do not infer that any pair is unedited; downstream release must remain on hold."
        )
    assigned = [c.get("code", "") for c in cpt_codes if c.get("code")]
    candidate_codes = [
        c.get("code", "") for c in cpt_candidates[:15]
        if c.get("code") and c.get("code") not in assigned
    ]

    seen: set[tuple[str, str]] = set()
    lines: list[str] = []

    def _check(c1: str, c2: str) -> None:
        if c1 == c2:
            return
        key = tuple(sorted((c1, c2)))
        if key in seen:
            return
        seen.add(key)
        edit = store.ncci_pair(c1, c2, dos)
        if edit:
            indicator = str(edit.get("modifier_indicator", ""))
            meaning = {
                "0": "hard bundle — NEVER separately billable together, regardless of modifier",
                "1": "bundled by default, but separately billable with modifier 59/XE/XS/XP/XU/25/57 if documented as distinct",
                "9": "edit does not restrict this pair",
            }.get(indicator, f"indicator={indicator}")
            lines.append(f"  {c1} + {c2}: NCCI EDIT EXISTS — {meaning}")
        else:
            lines.append(f"  {c1} + {c2}: NO NCCI edit in the real edit table — separately billable by default")

    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            _check(assigned[i], assigned[j])
    for a in assigned:
        for c in candidate_codes:
            _check(a, c)

    if not lines:
        return ""
    return (
        "## NCCI PAIR STATUS (real data from compliance.db — authoritative, not a guess)\n"
        "This is the actual NCCI PTP edit table result for these code pairs. Use it directly.\n"
        'Do NOT describe a pair as "bundled," "not separately billable," or "per NCCI\n'
        'convention" unless it is listed here as having an edit — a pair listed as "NO NCCI\n'
        "edit\" has no bundling relationship in the real data and should be coded separately\n"
        "if clinically documented, with modifier 59/RT/LT as appropriate.\n"
        + "\n".join(lines)
    )


def _format_candidates_for_system(rag_candidates: dict, system: str, store=None) -> str:
    candidates = rag_candidates.get(system, [])
    if not candidates:
        return "No candidates retrieved."
    lines = []
    for c in candidates[:25]:
        score = c.get("similarity_score", 0)
        code = c.get("code", "")
        desc = c.get("description", "") or c.get("long_description", "") or c.get("short_description", "")
        # Annotate CPT candidates with their actual global period from the data store.
        # This lets the LLM distinguish major (090) from minor (010/000) procedures at
        # coding time — the data-driven foundation for correct -25 vs -57 selection.
        global_tag = ""
        if system == "cpt" and store:
            glob = store.global_period(code)
            if glob:
                global_tag = f" [global={glob}]"
        max_desc = 500 if system == "hcpcs" else 150
        lines.append(f"  {code}{global_tag} (relevance: {score:.3f}) — {desc[:max_desc]}")
    return "\n".join(lines)


def _summarize_icd(icd_codes: list[dict]) -> str:
    if not icd_codes:
        return "No ICD-10-CM codes assigned."
    lines = []
    for c in icd_codes:
        lines.append(f"  {c.get('code', '?')} [{c.get('type', '?')}] — {c.get('description', '')[:80]}")
    return "\n".join(lines)


def _summarize_cpt(cpt_codes: list[dict], store=None) -> str:
    if not cpt_codes:
        return "No CPT codes assigned yet."
    lines = []
    for c in cpt_codes:
        mods = ", ".join(c.get("modifiers", [])) or "none"
        code = c.get("code", "?")
        global_tag = ""
        if store:
            glob = store.global_period(code)
            if glob:
                global_tag = f" [global={glob}]"
        lines.append(f"  {code}{global_tag} [{mods}] — {c.get('description', '')[:80]}")
    return "\n".join(lines)


def _safe_parse(raw: str, required_key: str) -> dict:
    try:
        # Every consumer expects list-of-dicts code arrays — normalize once
        # here so bare-string entries can't crash any pass (1-3 feed
        # _hard_db_gate, the verify pass feeds the enforcement helpers).
        return _normalize_code_arrays(json.loads(raw))
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response for {required_key}")
        # Empty dict, not {required_key: []} — every caller falls back via
        # .get(key, <pre-verification value>), and a failure dict that
        # already contains required_key: [] would make that fallback treat
        # a genuine parse FAILURE as if verification had legitimately
        # returned an empty list for that one field, skipping the fallback
        # for it while every other field correctly falls back.
        return {}


# ---------------------------------------------------------------------------
# Fix 6 — Hard Database Gate
# ---------------------------------------------------------------------------

def _hard_db_gate(entries: list[dict], code_system: str, db,
                  allowed_codes: set[str] | None = None) -> list[dict]:
    """Immediately remove codes that are NOT in the reference database.

    This prevents invalid/hallucinated codes from ever reaching the verification
    pass, and ensures every output code is defensible in an audit.
    """
    valid = []
    if db is None:
        logger.error(f"    [DB GATE] {code_system.upper()} reference database "
                     "is unavailable — removed every proposed code")
        return valid
    for entry in entries:
        code = entry.get("code", "").strip()
        if not code:
            continue
        if (allowed_codes is not None and
                _candidate_code(code_system, code) not in allowed_codes):
            logger.warning(
                f"    [CANDIDATE GATE] {code_system.upper()} {code!r} was "
                "not retrieved or physician-documented — removed")
            continue
        found = False
        if code_system == "icd10":
            found = bool(db.validate_icd10(code))
        elif code_system == "cpt":
            found = bool(db.validate_cpt(code))
        elif code_system == "hcpcs":
            found = bool(db.validate_hcpcs(code))
        if found:
            valid.append(entry)
        else:
            logger.warning(
                f"    [DB GATE] {code_system.upper()} {code!r} NOT FOUND in reference DB — removed"
            )
    return valid


# ---------------------------------------------------------------------------
# Fix 1 — Physician Code Source Tagging + Reconciliation
# ---------------------------------------------------------------------------

def _tag_code_sources(
    result: dict,
    physician_documented_codes: list[dict],
    entities: list[dict] | None = None,
) -> dict:
    """Tag every output code with its provenance and detect physician code replacements.

    Tags:
    - physician_documented  : physician explicitly wrote this code in the note
    - ai_confirmed          : AI assigned same code as physician (agreement), OR entity was
                              validated by GLiNER-BioMed biomedical NER
    - ai_replaced_physician : AI chose a DIFFERENT code in the same category as a physician code
    - ai_inferred           : AI derived this code with no external confirmation
    """
    if not physician_documented_codes:
        # No physician codes — tag everything as ai_inferred first, then upgrade via GLiNER
        for key in ("icd10_codes", "cpt_codes", "hcpcs_codes"):
            for e in result.get(key, []):
                e.setdefault("code_source", "ai_inferred")
        _upgrade_via_gliner(result, entities)
        return result

    # Every match below is keyed by (code_system, code) / (code_system, prefix),
    # never bare code/prefix — ICD-10 (stored undotted, e.g. "E1100") and HCPCS
    # ("E1100" = a real wheelchair code) can be byte-identical strings that
    # mean completely different things, so a bare-code match risks tagging an
    # AI-assigned code from one system as "physician_documented" because a
    # physician happened to write the same string in a different system, or
    # suppressing a genuinely missing physician code via a false cross-system
    # family match. physician_documented_codes carries its own "system" field
    # (icd10/cpt/hcpcs, from the extraction schema); AI-assigned codes get
    # their system from which result[key] list they're actually in.
    _KEY_TO_SYSTEM = {"icd10_codes": "icd10", "cpt_codes": "cpt", "hcpcs_codes": "hcpcs"}

    # Build maps: (system, exact code) → physician entry; (system, 3-char prefix) → physician entries
    phys_exact: dict[tuple[str, str], dict] = {}
    phys_prefix: dict[tuple[str, str], list[dict]] = {}
    for p in physician_documented_codes:
        code = p.get("code", "").strip().upper()
        system = str(p.get("system", "")).strip().lower()
        if not code or not system:
            continue
        phys_exact[(system, code)] = p
        prefix = code[:3]
        phys_prefix.setdefault((system, prefix), []).append(p)

    all_ai_codes: set[tuple[str, str]] = set()
    for key, system in _KEY_TO_SYSTEM.items():
        for e in result.get(key, []):
            all_ai_codes.add((system, e.get("code", "").strip().upper()))

    # Tag each code
    for key, system in _KEY_TO_SYSTEM.items():
        for e in result.get(key, []):
            code = e.get("code", "").strip().upper()
            if (system, code) in phys_exact:
                e["code_source"] = "physician_documented"
            else:
                # Check if a physician code in the same system + 3-char family was not used
                prefix = code[:3]
                same_family = phys_prefix.get((system, prefix), [])
                replaced = [p for p in same_family if (system, p.get("code", "").upper()) not in all_ai_codes]
                if replaced:
                    e["code_source"] = "ai_replaced_physician"
                    e["physician_code_note"] = (
                        f"Physician wrote {replaced[0].get('code')} "
                        f"({replaced[0].get('description', '')})"
                    )
                    logger.warning(
                        f"    [PHYSICIAN LOCK] AI assigned {code} but physician documented "
                        f"{replaced[0].get('code')} — flagged for review"
                    )
                else:
                    e.setdefault("code_source", "ai_inferred")

    # Detect physician codes completely absent from AI output
    missing = []
    for p in physician_documented_codes:
        code = p.get("code", "").strip().upper()
        system = str(p.get("system", "")).strip().lower()
        if code and system and (system, code) not in all_ai_codes:
            # Check if it wasn't replaced (already caught above)
            prefix = code[:3]
            ai_same_family = [c for s, c in all_ai_codes if s == system and c[:3] == prefix]
            if not ai_same_family:
                # Completely missing — not even a family replacement
                missing.append(p)
                logger.warning(
                    f"    [MISSING PHYSICIAN CODE] {code} ({p.get('description', '')}) "
                    f"was in physician notes but not in AI output"
                )

    result["missing_physician_codes"] = missing
    _upgrade_via_gliner(result, entities)
    return result


def _strip_invalid_cpt_modifiers(verified: dict, store=None) -> dict:
    """Remove modifiers that are not recognized modifiers AT ALL, per the
    merged AMA CPT Appendix A + CMS HCPCS Level II modifier reference
    (modifiers.json) — a hallucination gate, same role as _hard_db_gate for
    codes.

    Two deliberate design points, both learned from real failures:

    1. Gate on modifier_valid (recognized in either source), NOT
       modifier_valid_for_cpt (AMA's CPT-book cross-listing). CMS Level II
       modifiers absent from AMA's list — Q7/Q8/Q9 (routine-foot-care class
       findings), KX, GA/GX/GY/GZ, QW — are legitimately appended to CPT
       lines on CMS-1500 claims per CMS billing rules; AMA-book scope is the
       wrong authority for claim-form validity. (An earlier version gated on
       the CPT cross-listing and would have deleted a required Q8 from a
       covered 11720 routine-foot-care line.)

    2. Fail safe when the reference data is absent: if the modifier table has
       zero rows, every modifier looks "unrecognized" and a strip would
       delete ALL modifiers (-25/-57/RT/LT/TA…) from every claim — observed
       live when an ingestion bug left the table's systems column empty.
       Missing data means "cannot check", never "strip everything".
    """
    if store is None:
        return verified
    if store.modifier_count() == 0:
        logger.warning(
            "    [MODIFIER STRIP] modifier reference table is empty — "
            "skipping strip entirely (missing data must never remove modifiers)"
        )
        return verified
    for entry in verified.get("cpt_codes", []):
        raw = entry.get("modifiers", [])
        if not raw:
            continue
        valid = [m for m in raw if store.modifier_valid(str(m))]
        removed = [m for m in raw if not store.modifier_valid(str(m))]
        if removed:
            logger.warning(
                f"    [MODIFIER STRIP] CPT {entry.get('code')} — removed unrecognized modifiers: {removed}"
            )
            entry["modifiers"] = valid
    return verified


# First plausible code token in a free-text correction line — used only to
# recover the subject code when the LLM emits corrections_made as bare
# strings instead of the contracted dict shape.
_CORRECTION_CODE_RE = re.compile(r"\b([A-Z]\d{2}(?:\.[0-9A-Z]{1,4})?|\d{5}|[A-Z]\d{4})\b")


def _gate_verify_additions(final_result: dict, combined: dict, db, store=None,
                           allowed_codes: dict[str, set[str]] | None = None) -> dict:
    """Reference-DB gate for codes INTRODUCED by the verification pass.

    Passes 1-3 are hard-gated (_hard_db_gate), but Pass 4's output never
    was: a code the earlier gate removed could be re-added by the audit
    ("protected anchor" reasoning), or a wholly new line invented — both
    observed live in one batch (D48.1, a non-billable ICD category header,
    and CPT 20926, deleted from the code set years ago).

    Policy per entry not present before verification:
      * validates in its claimed system → keep (legitimate audit addition);
      * a real Tabular/code-set entry at the WRONG level or date (category
        header with billable children, or a code active on some other
        date) → keep — the specificity filter FAILs it with an actionable
        message ("assign the billable child" / "not active for DOS"),
        which serves reviewers better than silent disappearance;
      * known nowhere in any reference data → remove (hallucination).
    """
    if db is None:
        for key in ("icd10_codes", "supporting_conditions", "cpt_codes",
                    "hcpcs_codes"):
            final_result[key] = []
        return final_result
    checks = {
        "icd10_codes": ("ICD10", db.validate_icd10),
        "supporting_conditions": ("ICD10", db.validate_icd10),
        "cpt_codes": ("CPT", db.validate_cpt),
        "hcpcs_codes": ("HCPCS", db.validate_hcpcs),
    }
    allowed_by_key = {
        "icd10_codes": (allowed_codes or {}).get("icd10"),
        "supporting_conditions": (allowed_codes or {}).get("icd10"),
        "cpt_codes": (allowed_codes or {}).get("cpt"),
        "hcpcs_codes": (allowed_codes or {}).get("hcpcs"),
    }
    for key, (system, validate) in checks.items():
        pre = {
            (e.get("code") or "").upper()
            for e in combined.get(key, []) if isinstance(e, dict)
        }
        entries = final_result.get(key, [])
        kept = []
        for e in entries:
            code = (e.get("code") or "").strip()
            if not code:
                continue
            allowed = allowed_by_key.get(key)
            normalized_system = "icd10" if system == "ICD10" else system.lower()
            if (allowed is not None and
                    _candidate_code(normalized_system, code) not in allowed):
                logger.warning(
                    f"    [VERIFY CANDIDATE GATE] {key}: '{code}' was not "
                    "retrieved or physician-documented — removed")
                continue
            if code.upper() in pre or validate(code):
                kept.append(e)
                continue
            known_elsewhere = store is not None and (
                store.code_active_any_date(system, code)
                or store.children_exist(system, code)
            )
            if known_elsewhere:
                kept.append(e)  # real entry, wrong level/date — specificity flags it
                continue
            logger.warning(
                f"    [VERIFY GATE] {key}: '{code}' was introduced by the verification "
                f"pass but exists nowhere in the {system} reference data — removed "
                f"(likely hallucinated)"
            )
        if len(kept) != len(entries):
            final_result[key] = kept
    return final_result


def _inherit_dropped_fields(final_result: dict, combined: dict) -> None:
    """Backfill fields Pass 4 dropped from entries it re-emitted. The verify
    contract says 'return every entry in full', but the model often returns
    only the fields it thought about; anything omitted then collapses to the
    schema default (e.g. ICD type -> 'secondary', units -> 1), silently
    mutating claim data no correction ever mentioned. For each code present
    both pre- and post-verification, copy over any key the verified entry is
    missing. Keys the model explicitly returned (even falsy) are respected —
    an intentional change always survives; only omissions are repaired."""
    for key in ("icd10_codes", "supporting_conditions", "cpt_codes",
                "hcpcs_codes", "snomed_codes"):
        before = {e.get("code"): e for e in combined.get(key, [])
                  if isinstance(e, dict) and e.get("code")}
        for entry in final_result.get(key, []):
            if not isinstance(entry, dict):
                continue
            # A CHANGED-correction entry carries its pre-verification code so
            # inheritance can find the original: the slim verify entry (or an
            # in-place enforcement rewrite) holds the NEW code, which has no
            # pre-verification counterpart under that key.
            pre = entry.pop("_pre_verify_code", None)
            prior = before.get(entry.get("code")) or (before.get(pre) if pre else None)
            if not prior:
                continue
            for field, val in prior.items():
                if field not in entry and field != "code":
                    entry[field] = val


def _normalize_code_arrays(verified: dict) -> dict:
    """Coerce the verify pass's code arrays into the contracted
    list-of-dicts shape. Observed live: the model occasionally compacts an
    array to bare code strings ('icd10_codes': ['E11.42', ...]), crashing
    the first downstream .get() and aborting the note. Strings that look
    like codes are wrapped as minimal entries (descriptions/rationales are
    refreshed downstream by _enforce_real_descriptions); other non-dict
    garbage is dropped. A key that exists but isn't a list at all is
    removed so the caller's pre-verification fallback kicks in."""
    for key in ("icd10_codes", "supporting_conditions", "cpt_codes",
                "hcpcs_codes", "snomed_codes"):
        if key not in verified:
            continue  # missing key = fall back to pre-verification value
        raw = verified[key]
        if not isinstance(raw, list):
            logger.warning(f"    [ARRAY NORMALIZE] {key} is {type(raw).__name__}, "
                           f"not a list — dropped (pre-verification value will be used)")
            del verified[key]
            continue
        out = []
        for e in raw:
            if isinstance(e, dict):
                out.append(e)
            elif isinstance(e, str) and e.strip():
                out.append({"code": e.strip()})
                logger.warning(f"    [ARRAY NORMALIZE] {key}: bare string entry "
                               f"'{e.strip()}' wrapped as a code entry")
        verified[key] = out
    return verified


def _normalize_corrections(verified: dict) -> dict:
    """Coerce corrections_made into the contracted list-of-dicts shape.

    The verify pass occasionally emits corrections as bare strings
    ('Removed 11719 — not documented') instead of dicts — observed live
    crashing every downstream consumer (`'str' object has no attribute
    'get'`) and aborting the whole note. One malformed narrative entry must
    never kill a claim, so normalization happens once, here, before any
    consumer: dicts pass through, strings are wrapped as best-effort dicts
    (type inferred from the leading verb, code from the first code-shaped
    token), anything else is dropped."""
    raw = verified.get("corrections_made", [])
    if not isinstance(raw, list):
        verified["corrections_made"] = []
        return verified
    out = []
    for c in raw:
        if isinstance(c, dict):
            out.append(c)
            continue
        if isinstance(c, str) and c.strip():
            low = c.lower()
            kind = ("ADDED" if "add" in low.split(chr(32))[0] else
                    "REMOVED" if low.startswith(("remov", "delet", "drop")) else
                    "CHANGED" if low.startswith(("chang", "correct", "switch", "remap", "updat")) else
                    "OTHER")
            m = _CORRECTION_CODE_RE.search(c)
            entry = {"type": kind, "code": m.group(1) if m else "", "reason": c.strip()}
            out.append(entry)
            logger.warning(
                f"    [CORRECTIONS NORMALIZE] string entry coerced to dict: {entry['type']} "
                f"{entry['code'] or '(no code found)'}"
            )
    verified["corrections_made"] = out
    return verified


def _enforce_added_corrections(verified: dict, db=None) -> dict:
    """Guarantee every code noted as ADDED in corrections_made actually
    appears in its code array. The LLM sometimes writes 'ADDED J0702' (or
    any other code) in corrections_made but forgets to include the code in
    the actual array — silent billing/coding loss.

    Generalizes a prior J-code-only version of this rescuer (HCPCS drug
    codes specifically): the same "claimed but not applied" gap isn't
    J-code-shaped, it can happen to any code in any system, so scoping the
    fix to one regex pattern only rescued a fraction of real occurrences.

    System is resolved by validating the code against the real reference DB
    for each candidate system, not by code shape — CPT is unambiguous (5
    digits), but ICD-10 (stored undotted, e.g. "E1100") and HCPCS ("E1100"
    is a real wheelchair code) can be byte-identical strings across systems,
    so a shape-only guess (letter + 4 digits) is genuinely ambiguous. A code
    that validates as real in more than one system is left alone rather
    than guessed into the wrong array; a code that validates in neither is
    left alone too (likely hallucinated — the correction claims a code that
    doesn't exist, which is a separate problem this function shouldn't try
    to paper over).
    """
    import re
    if db is None:
        return verified
    corrections = verified.get("corrections_made", [])
    if not corrections:
        return verified

    icd_list = verified.get("icd10_codes", [])
    cpt_list = verified.get("cpt_codes", [])
    hcpcs_list = verified.get("hcpcs_codes", [])
    icd_existing = {c.get("code", "").upper() for c in icd_list}
    cpt_existing = {c.get("code", "").upper() for c in cpt_list}
    hcpcs_existing = {c.get("code", "").upper() for c in hcpcs_list}

    for correction in corrections:
        if correction.get("type", "").upper() != "ADDED":
            continue
        code = correction.get("code", "").strip().upper()
        if not code or code in icd_existing or code in cpt_existing or code in hcpcs_existing:
            continue

        is_cpt_shaped = bool(re.match(r"^\d{5}$", code))
        icd_rec = None if is_cpt_shaped else db.validate_icd10(code)
        hcpcs_rec = None if is_cpt_shaped else db.validate_hcpcs(code)
        cpt_rec = db.validate_cpt(code) if is_cpt_shaped else None

        base_entry = {
            "code": code,
            "description": correction.get("reason", "")[:100],
            "confidence": 0.85,
            "rationale": correction.get("reason", ""),
            "supporting_text": correction.get("evidence", ""),
            "code_source": "ai_inferred",
        }

        if icd_rec and hcpcs_rec:
            logger.warning(
                f"    [CORRECTION RESCUE] {code}: claimed ADDED but validates as a real code in "
                f"BOTH ICD-10 and HCPCS — genuinely ambiguous, not rescued (needs manual review)"
            )
        elif cpt_rec:
            cpt_list.append({**base_entry, "modifiers": [], "units": 1, "linked_diagnoses": []})
            cpt_existing.add(code)
            logger.info(f"    [CORRECTION RESCUE] {code} added to cpt_codes (was in corrections_made but missing)")
        elif hcpcs_rec:
            hcpcs_list.append({
                **base_entry, "modifiers": [], "units": 1, "linked_diagnoses": [],
                "needs_review": False, "review_reason": None,
            })
            hcpcs_existing.add(code)
            logger.info(f"    [CORRECTION RESCUE] {code} added to hcpcs_codes (was in corrections_made but missing)")
        elif icd_rec:
            icd_list.append({**base_entry, "type": "secondary"})
            icd_existing.add(code)
            logger.info(f"    [CORRECTION RESCUE] {code} added to icd10_codes (was in corrections_made but missing)")
        else:
            logger.warning(
                f"    [CORRECTION RESCUE] {code}: claimed ADDED but not found in any code system's "
                f"reference DB — not rescued (likely hallucinated)"
            )

    verified["icd10_codes"] = icd_list
    verified["cpt_codes"] = cpt_list
    verified["hcpcs_codes"] = hcpcs_list
    return verified


# Correction language that signals the CODE VALUE itself was replaced —
# used only to recover a missing to_code from the correction's own reason
# text (grammar of the audit narrative, not any medical-code list).
_CHANGED_TO_RE = re.compile(
    r"(?:changed|corrected|switched|remapped|updated)\s*(?:\w+\s+){0,4}?to\s+([A-Z][0-9][0-9A-Z]{1,5}(?:\.[0-9A-Z]{1,4})?)",
    re.IGNORECASE,
)


def _enforce_changed_corrections(verified: dict, db=None) -> dict:
    """Guarantee every code-value CHANGED correction is actually applied to
    its code array. The narrative-claims-fix-but-data-unchanged failure is
    not ADDED-shaped only: observed live, the audit pass wrote 'Z88.5
    changed to reflect analgesic-agent category (Z88.6)' in its narrative
    fields while the array entry still carried code=Z88.5 — the reasoning
    described the fix, the data didn't contain it (same failure class as
    the modifier-reasoning divergence fixed with structured ModifierClaim).

    Enforcement is deterministic: for each CHANGED correction carrying the
    old code and a replacement (structured `to_code`, else recovered from
    the correction's own 'changed/corrected ... to <code>' language), find
    the old code across ALL code arrays (including supporting_conditions —
    where sibling miscodes live), validate the replacement exists in the
    SAME code system's reference DB (a replacement that doesn't validate is
    likely hallucinated — left alone, existence checks flag it), then write
    the new code into the entry's code field. Descriptions refresh via
    _enforce_real_descriptions downstream.
    """
    if db is None:
        return verified

    arrays = {
        "icd10_codes": lambda c: db.validate_icd10(c),
        "supporting_conditions": lambda c: db.validate_icd10(c),
        "cpt_codes": lambda c: db.validate_cpt(c),
        "hcpcs_codes": lambda c: db.validate_hcpcs(c),
    }
    # Layer 1: explicit CHANGED entries in corrections_made. Layer 2 below
    # runs unconditionally — the observed live failure had NO correction
    # entry at all, only the code entry's own narrative.
    for correction in verified.get("corrections_made", []):
        if correction.get("type", "").upper() != "CHANGED":
            continue
        old = (correction.get("code") or "").strip().upper()
        new = (correction.get("to_code") or "").strip().upper()
        if not new:
            m = _CHANGED_TO_RE.search(correction.get("reason", "") or "")
            new = m.group(1).upper() if m else ""
        if not old or not new or old == new:
            continue
        for key, validate in arrays.items():
            for entry in verified.get(key, []):
                if (entry.get("code") or "").strip().upper() != old:
                    continue
                if not validate(new):
                    logger.warning(
                        f"    [CORRECTION ENFORCE] {old}→{new} claimed in corrections_made but "
                        f"{new} doesn't validate in the same code system — not applied"
                    )
                    continue
                entry.setdefault("_pre_verify_code", old)
                entry["code"] = new
                logger.info(
                    f"    [CORRECTION ENFORCE] {key}: {old} → {new} "
                    f"(CHANGED correction applied to the code field, not just the narrative)"
                )

    # Second layer: sibling switches declared only in an entry's OWN narrative
    # (no corrections_made entry at all — the observed Z88.5 case, where the
    # rationale argued 'aspirin allergy maps to the analgesic-agent sibling
    # (Z88.6)' and review_reason said 'corrected', while the code field still
    # read Z88.5). Scoped tightly to stay deterministic and safe:
    #   - only TARGET codes count: the code a directional phrase points AT
    #     ('from X to Y' → Y; 'maps to (Y)' → the nearest code after the
    #     verb). Direction matters both ways — when the entry's current code
    #     IS the target, the correction was already applied and the OLD code
    #     named in the same sentence must not flip it back (observed live:
    #     'corrected from Z88.5 to Z88.6' on an entry already carrying Z88.6
    #     was reverted to Z88.5 by a direction-blind version of this);
    #   - correction verb and candidate must share a SENTENCE with no refusal
    #     phrasing ('rather than', 'retained', ...), so 'retained X rather
    #     than switching to Y' never triggers;
    #   - the target must be a same-category sibling (shared 3-char category
    #     — the miscoded-sibling class the disambiguation block corrects)
    #     that validates in the entry's own code system;
    #   - exactly ONE distinct target, else leave it to existence/review.
    code_pat = r"[A-Z]\d{2,4}(?:\.[0-9A-Z]{1,4})?"
    pair_re = re.compile(
        rf"from\s+\(?({code_pat})\)?[^.;]{{0,60}}?\bto\s+\(?({code_pat})\)?", re.IGNORECASE)
    verb_re = re.compile(
        rf"\b(?:corrected|changed|switch(?:ed)?|remap(?:ped)?|should be|maps? to|use)\b"
        rf"[^.;]{{0,60}}?\b({code_pat})\b", re.IGNORECASE)
    neg_re = re.compile(r"\b(rather than|instead of|avoid(?:ed)?|declined|rejected|kept|retain(?:ed)?)\b",
                        re.IGNORECASE)
    for key, validate in arrays.items():
        for entry in verified.get(key, []):
            cur = (entry.get("code") or "").strip().upper()
            text = " ".join(filter(None, [entry.get("rationale"), entry.get("review_reason")]))
            if not cur or not text:
                continue
            targets, sources = set(), set()
            for sentence in re.split(r"[.;](?:\s|$)", text):
                if neg_re.search(sentence):
                    continue
                for m in pair_re.finditer(sentence):
                    sources.add(m.group(1).upper())
                    targets.add(m.group(2).upper())
                for m in verb_re.finditer(sentence):
                    c = m.group(1).upper()
                    if c not in sources:
                        targets.add(c)
            if cur in targets:
                continue  # correction already applied — never flip back to the old code
            candidates = {c for c in targets if c != cur and c[:3] == cur[:3] and validate(c)}
            if len(candidates) != 1:
                continue  # none, or ambiguous — leave to existence/review checks
            new = candidates.pop()
            entry.setdefault("_pre_verify_code", cur)
            entry["code"] = new
            logger.info(
                f"    [CORRECTION ENFORCE] {key}: {cur} → {new} (entry's own narrative declared "
                f"a sibling correction that was never applied to the code field)"
            )
    return verified


def _upgrade_via_gliner(result: dict, entities: list[dict] | None) -> None:
    """Upgrade ai_inferred → ai_confirmed for codes whose driving entity was GLiNER-validated."""
    if not entities:
        return
    confirmed_terms: set[str] = set()
    for e in entities:
        if e.get("ner_source") == "gliner_confirmed":
            for field in ("clinical_term", "text"):
                val = e.get(field, "").lower().strip()
                if len(val) >= 4:
                    confirmed_terms.add(val)
    if not confirmed_terms:
        return
    for key in ("icd10_codes", "cpt_codes", "hcpcs_codes"):
        for code_entry in result.get(key, []):
            if code_entry.get("code_source") != "ai_inferred":
                continue
            evidence = " ".join([
                code_entry.get("rationale", ""),
                code_entry.get("supporting_text", ""),
                code_entry.get("description", ""),
                " ".join(code_entry.get("evidence_spans", [])),
            ]).lower()
            if any(term in evidence for term in confirmed_terms):
                code_entry["code_source"] = "ai_confirmed"


def _build_anchor_block(
    assessment_text: str,
    pmh_text: str,
    vision_ctx: dict | None,
    prior_surgery_info: dict | None = None,
) -> str:
    lines = [
        "## PROTECTED ANCHORS — These MUST have corresponding codes in icd10_codes",
        "",
        "### Assessment Section Diagnoses (NEVER remove)",
    ]

    if assessment_text:
        for line in assessment_text.split("\n"):
            cleaned = line.strip().lstrip("•·-–—0123456789.) ").strip()
            if not cleaned or len(cleaned) <= 3:
                continue
            lines.append(f'  - ANCHOR: "{cleaned}"')
    else:
        lines.append("  (no assessment text available)")

    lines.append("")
    lines.append("### PMH Conditions (go into supporting_conditions — NOT icd10_codes)")
    if pmh_text:
        lines.append(f'  Source text: "{pmh_text[:400]}"')
        lines.append("  PMH comorbidities with active meds → supporting_conditions only.")

    if prior_surgery_info and prior_surgery_info.get("is_post_op_visit"):
        days = prior_surgery_info.get("days_post_op")
        desc = prior_surgery_info.get("prior_surgery_description", "prior surgery")
        lines.append("")
        lines.append("### Post-Operative Encounter Context (PROTECTED)")
        lines.append(f'  - ANCHOR: Post-operative follow-up after "{desc}"')
        lines.append(f"  - Days post-op: {days}")
        lines.append("  - Select an encounter diagnosis only from the supplied "
                     "authoritative candidates whose descriptor matches what occurred today; "
                     "do not infer device removal or another service.")

    if vision_ctx:
        procs = vision_ctx.get("procedures_performed_today", [])
        imgs = vision_ctx.get("imaging_performed_today", [])
        sups = vision_ctx.get("supplies_dispensed_today", [])

        if procs:
            lines.append("")
            lines.append("### Procedures Performed Today (must have CPT codes)")
            for p in procs:
                lines.append(f'  - ANCHOR: "{p}"')

        if imgs:
            lines.append("")
            lines.append("### Imaging Performed Today (must have CPT codes)")
            for i in imgs:
                lines.append(f'  - ANCHOR: "{i}"')

        if sups:
            lines.append("")
            lines.append("### Supplies Dispensed Today (must have HCPCS codes with laterality)")
            for s in sups:
                lines.append(f'  - ANCHOR: "{s}"')

        note_cat = vision_ctx.get("note_category", "")
        if note_cat and any(kw in note_cat for kw in ("visit", "followup", "follow_up", "post_op", "urgent")):
            lines.append("")
            lines.append("### E/M Visit Detection")
            lines.append(f"  Note category: {note_cat}")
            lines.append("  If this is a patient visit with HPI + exam + plan, an E/M MUST be coded.")

    return "\n".join(lines)


def _format_vision_context(ctx: dict) -> str:
    if not ctx:
        return ""
    parts = ["## VISION EXTRACTION CONTEXT (from intelligent PDF reading — ground truth)"]
    parts.append(f"- Note category: {ctx.get('note_category', 'unknown')}")
    procs = ctx.get("procedures_performed_today", [])
    parts.append(f"- Procedures PERFORMED today: {procs if procs else 'NONE'}")
    imgs = ctx.get("imaging_performed_today", [])
    parts.append(f"- Imaging PERFORMED today: {imgs if imgs else 'NONE'}")
    sups = ctx.get("supplies_dispensed_today", [])
    parts.append(f"- Supplies DISPENSED today: {sups if sups else 'NONE'}")
    parts.append("NOTE: Only code items listed above as performed/dispensed today.")
    return "\n".join(parts)


def _format_global_period_context(info: dict, store=None) -> str:
    if not info or not info.get("is_post_op_visit"):
        return ""
    days = info.get("days_post_op")
    desc = info.get("prior_surgery_description", "prior surgery")
    cpt = info.get("prior_surgery_cpt", "unknown")
    global_days = store.global_period(cpt) if store and cpt != "unknown" else None
    return (
        f"## GLOBAL SURGICAL PERIOD CONTEXT\n"
        f"- This is a POST-OPERATIVE FOLLOW-UP visit\n"
        f"- Prior surgery: {desc} (CPT {cpt})\n"
        f"- Days post-op: {days}\n"
        f"- Authoritative global-period value: {global_days or 'UNAVAILABLE'}\n"
        f"- Use the authoritative candidate descriptors and global-period data "
        f"to decide whether any encounter-reporting line is billable; if the "
        f"data is unavailable, flag review rather than guessing."
    )


def _add_usage(total: dict, new: dict):
    for k in total:
        total[k] += new.get(k, 0)


def _format_physician_codes(physician_codes: list[dict]) -> str:
    """Build a block for the verification prompt listing physician-documented codes."""
    if not physician_codes:
        return ""
    lines = [
        "## PHYSICIAN-DOCUMENTED CODES (explicitly written by the provider in this note)",
        "These codes were literally written by the physician — treat them with highest authority.",
        "If your output differs from any of these, you MUST flag it as needs_review=true and explain why.",
        "If a physician code is correct, include it in the output. If clinically wrong, flag for review — do NOT silently drop.",
        "",
    ]
    for p in physician_codes:
        code = p.get("code", "")
        desc = p.get("description", "")
        section = p.get("section", "")
        lines.append(f"  - {code} ({desc}) [from {section}]")
    return "\n".join(lines)
