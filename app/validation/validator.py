import math
import os
import re

from app.rag.code_reference import CodeReferenceDB
from app.models.schemas import ValidationIssue, DocumentationAudit, EncounterIntegrity
from app.core.logger import get_logger

logger = get_logger(__name__)

# E/M section of CPT — same structural-range pattern already used in
# medical_necessity.py (_is_em/_EM_SECTION), not a hand-picked code list, so
# it stays correct as CPT adds/removes individual E/M codes within the
# section. Any E/M subtype (office, hospital, ED, consult) needs -25/-57
# handling identically for this purpose, so the full section is the right
# scope, not just office-visit codes.
_EM_SECTION = range(99202, 99500)


def _is_em(code: str) -> bool:
    return code.isdigit() and int(code) in _EM_SECTION
# Global period days that actually denote a "procedure" for modifier -25/-57
# purposes. XXX/YYY/ZZZ/MMM (diagnostic tests, E/M codes, unlisted, etc.) do
# NOT trigger either — see _check_em_modifier25. Previously approximated via
# a CPT-section prefix list (70-79, "imaging"), which wrongly caught non-
# radiology diagnostic tests like the 93xxx vascular physiologic studies
# (also global=XXX, not a procedure).
#
# 000/010 (minor/intermediate) and 090 (major) are kept as separate sets, not
# one undifferentiated PROCEDURE_GLOBAL_DAYS — a same-day 090 procedure needs
# -57, not -25 (see the mutual-exclusivity comment in _check_em_modifier25).
# Treating them as one set meant a same-day major (090) procedure could still
# get -25 auto-added here, and -57 only got added later by an unrelated check
# (modifier_reasoning consistency, driven by the LLM's own stated reasoning)
# — with nothing to remove the now-contradictory -25 once -57 showed up.
MINOR_PROCEDURE_GLOBAL_DAYS = {"000", "010"}
MAJOR_PROCEDURE_GLOBAL_DAYS = {"090"}
# Language lexicon (no codes): how clinical notes phrase each imaging
# modality. Keys are matched against the guidance CPTs' own descriptors
# ("Fluoroscopic guidance for needle placement...", "Ultrasonic guidance...",
# "Computed tomography guidance...", "Magnetic resonance imaging guidance...")
# so the CODES come from the data; only the English synonyms live here.
MODALITY_LEXICON = {
    "fluoroscopic": ("fluoroscopic", "fluoroscopy", "c-arm"),
    "ultrasonic": ("ultrasound guided", "ultrasound-guided", "sonographic",
                   "us-guided", "under ultrasound", "ultrasound guidance"),
    "computed tomography": ("ct guided", "ct-guided", "under ct guidance",
                            "computed tomography guidance", "ct guidance"),
    "magnetic resonance": ("mri guided", "mri-guided", "under mri guidance",
                           "magnetic resonance guidance", "mri guidance"),
}
# Language lexicon (no codes): how notes phrase each Radiology-section
# modality when the SERVICE ITSELF was performed — matched against the billed
# CPT's own descriptor words ("Radiologic examination...", "Fluoroscopy...",
# "Ultrasound...") so the codes come from the data; only English synonyms
# live here. Used by _check_imaging_note_evidence.
IMAGING_MODALITY_LEXICON = {
    "radiologic": ("x-ray", "xray", "x ray", "radiograph", "views"),
    "fluoroscopy": ("fluoroscopy", "fluoroscopic", "c-arm"),
    "ultrasound": ("ultrasound", "sonograph", "duplex", "doppler"),
    "computed tomography": ("ct scan", "computed tomography", "ct of"),
    "magnetic resonance": ("mri", "magnetic resonance"),
}
# HCPCS L-code prefixes for unilateral equipment requiring RT/LT
UNILATERAL_L_PREFIXES = ("L1", "L2", "L3", "L4", "L5")

# Language lexicon (no codes): the patient risk factors the 2021 AMA MDM
# risk column names for "minor surgery WITH identified patient risk factors"
# (moderate). Matched against each claim diagnosis's OWN description text
# ("Type 2 diabetes mellitus with...", "Obesity, unspecified", "Long term
# (current) use of anticoagulants") — the CODES come from the claim itself;
# only the English condition words live here. Age >= 65 (also an AMA-named
# factor) is derived from the note's DOB + DOS, not a term.
EM_RISK_FACTOR_TERMS = ("diabetes mellitus", "obesity", "anticoagulant")

# Language lexicon (no codes): ICD-10-CM's onset/temporal qualifier axis —
# the words the classification itself splits sibling codes on ('Other ACUTE
# osteomyelitis' M86.1- vs 'Other CHRONIC osteomyelitis' M86.6-; acquired
# L60.2 vs CONGENITAL Q84.5). Guideline I.A/I.B: a qualifier the provider
# never documents must not be assigned — the Alphabetic Index's bare-term
# default applies instead. Only the English qualifier words live here; the
# code routing comes from the Index table itself.
ONSET_QUALIFIERS = frozenset({"acute", "chronic", "subacute", "congenital",
                              "hereditary"})

# Language lexicon (no codes): the radiographic projection names notes use
# when enumerating views ('AP/lateral views', 'oblique view obtained') —
# counted against the billed radiology descriptor's own view requirement
# ('2 views' vs 'complete, minimum of 3 views'). AP/PA abbreviations are
# matched word-bounded inside imaging sentences separately.
RADIOGRAPH_PROJECTION_TERMS = (
    "anteroposterior", "posteroanterior", "lateral", "oblique", "axial",
    "sunrise", "skyline", "sesamoid", "harris", "broden",
    "weightbearing", "weight-bearing",
)

# Language lexicon (no codes): the 2021 AMA MDM table's own HIGH problems
# row — '1+ chronic illnesses with SEVERE EXACERBATION, PROGRESSION, or side
# effects of treatment' / 'illness or injury that poses a THREAT to life or
# bodily function' — plus the threat conditions the row's CPT examples name
# (gangrene/sepsis-class). Matched adjacent to a billed diagnosis's own
# descriptor tokens, so the CODES and conditions come from the claim.
EM_HIGH_PROBLEM_RE = re.compile(
    r"\b(severe|exacerbat\w*|progress(?:ion|ive)|decompensat\w*|"
    r"deteriorat\w*|threat\w*|gangren\w*|sepsis|septic\w*|necrotizing|"
    r"unstable|critical)\b")

# Language lexicon (no codes): the 2021 AMA MDM table's own HIGH risk row —
# 'drug therapy requiring intensive monitoring for toxicity'. The canonical
# clinical shape is a parenteral agent from a toxicity-monitored class
# (glycopeptides/aminoglycosides/amphotericin/chemotherapy — the classes the
# AMA guideline examples name) being INITIATED at this encounter. All three
# elements must appear in one performed-context sentence: route + agent +
# initiation ('IV vancomycin initiated empirically per ID recommendation').
EM_HIGH_RISK_ROUTE_RE = re.compile(
    r"\b(?:iv|intravenous(?:ly)?|parenteral(?:ly)?)\b", re.IGNORECASE)
# Only agents whose therapy genuinely requires intensive toxicity
# monitoring (levels/troughs, renal/oto monitoring) — NOT generic
# 'antibiotics': initiating IV ceftriaxone is routine prescription drug
# management (moderate), and matching the generic word would over-floor it.
EM_HIGH_RISK_AGENT_RE = re.compile(
    r"\b(?:vancomycin|aminoglycoside\w*|gentamicin|tobramycin|amikacin|"
    r"amphotericin|chemotherap\w*|cytotoxic\w*)\b",
    re.IGNORECASE)
EM_HIGH_RISK_INITIATION_RE = re.compile(
    r"\b(?:initiat\w*|start\w*|began|begun|administer\w*|infus\w*)\b",
    re.IGNORECASE)

# The four MDM levels, lowest to highest — index+1 is the axis score the
# 2-of-3 recomputation uses throughout the E/M checks.
_MDM_LEVEL_NAMES = ("straightforward", "low", "moderate", "high")

# Language lexicon (no codes): how notes phrase each digit, keyed by the
# ordinal word the AMA/CMS modifier names themselves use ("Left foot, second
# digit" — see _digit_modifier_by_name). Values are the note-side spellings
# ('right 5th toe', 'right fifth toe', 'right first/great toe'); 'hallux' is
# handled as an extra alternative for the great toe.
DIGIT_ORDINALS = {
    "great": ("great", "first", "1st"),
    "second": ("second", "2nd"),
    "third": ("third", "3rd"),
    "fourth": ("fourth", "4th"),
    "fifth": ("fifth", "5th"),
}

# Surgical scheduling language that implies -57 is needed
SURGICAL_DECISION_KEYWORDS = [
    "patient elects", "will proceed with", "scheduled for", "consented for",
    "surgical correction", "will undergo", "elects surgical", "schedule surgery",
    "plan for surgery", "plan for bunionectomy", "plan for procedure",
]


class CodingValidator:
    def __init__(self, ref_db: CodeReferenceDB, compliance_store=None):
        self.db = ref_db
        self.store = compliance_store
        self.issues: list[ValidationIssue] = []
        self._bundled_codes_to_suppress: set[str] = set()
        self._non_billable_codes_to_suppress: set[str] = set()
        self._scrub_advisory_suppressions: list[dict] = []
        # Audit trail for validator advisories removed at source by an
        # advisory-suppression directive (see
        # _apply_validator_advisory_suppressions) — merged into
        # material_corrections so the clinical audit sees a reported
        # decision, never a silently vanished issue.
        self._advisory_suppression_corrections: list[dict] = []

    def validate(
        self,
        coding_result: dict,
        note_plan_text: str = "",
        note_full_text: str = "",
        physician_documented_codes: list[dict] | None = None,
        dos=None,
        note_category: str = "",
        patient_dob: str = "",
        payer_follows_medicare_coverage: bool = False,
        note_assessment_text: str = "",
    ) -> dict:
        self.issues = []
        self._bundled_codes_to_suppress = set()
        self._non_billable_codes_to_suppress = set()
        self._scrub_advisory_suppressions = []
        self._advisory_suppression_corrections = []
        # Payer context (parsed from the note's own insurance field via
        # payer_registry) — MUE is Medicare/NCCI policy, so the MUE-0
        # auto-suppression below applies only to payers bound to Original
        # Medicare's coverage floor (FFS + MA per 42 CFR 422.101).
        self._payer_follows_medicare = bool(payer_follows_medicare_coverage)

        icd = coding_result.get("icd10_codes", [])
        cpt = coding_result.get("cpt_codes", [])
        hcpcs = coding_result.get("hcpcs_codes", [])
        snomed = coding_result.get("snomed_codes", [])
        # supporting_conditions are advisory — not validated as billable codes

        # Snapshot the billable claim BEFORE any layer runs: material
        # corrections are derived by DIFFING this against the final arrays,
        # not by trusting layers to report on themselves. Measured live
        # (routine_00003): a demotion layer moved a load-bearing diagnosis
        # off the claim without recording a correction, so the clinical
        # audit never saw the decision it existed to check. The diff makes
        # unreported mutation structurally impossible.
        pre_claim = self._pre_validation_snapshot(icd, cpt, hcpcs)

        self._check_code_existence(icd, cpt, hcpcs, dos)
        # Runs before sequencing/linkage/pointer checks — it REMOVES subsumed
        # diagnosis codes, and those downstream checks must see the final list.
        self._check_icd_includes_subsumption(icd, cpt, hcpcs, coding_result)
        # After subsumption removal (both-billed case): when only the LOWER
        # member of a Tabular Includes chain is billed but the note documents
        # the higher member's own condition, upgrade to the ranked code.
        self._check_icd_includes_severity_upgrade(icd, cpt, hcpcs,
                                                  note_full_text)
        # MUE limits are published for HCPCS too (e.g. J3301 caps at 16 units
        # per line), not just CPT — check both code arrays against the table.
        self._check_mue(cpt + hcpcs)
        self._check_drug_units(hcpcs, note_full_text)
        self._check_timed_infusion_documentation(cpt, note_full_text)
        self._check_count_based_selection(cpt, note_full_text)
        self._check_duplicate_diagnoses(icd, coding_result)
        self._check_sequencing(icd)
        # Pointer integrity FIRST: a diagnosis corrected upstream (LLM verify
        # pass or _enforce_changed_corrections swapping a sibling) leaves the
        # service lines pointing at the OLD code — remap/drop dangling
        # pointers so the linkage/backfill/trim checks below see real ones.
        self._check_dx_pointer_integrity(icd, cpt, hcpcs)
        # Primary designation BEFORE the linkage backfill and pointer
        # reorder below: backfill orders pointers primary-first (it must
        # see the corrected designation), and the reorder auto-fix moves
        # the primary to every line's front — overwriting the raw
        # first-pointer evidence this check reads.
        self._check_primary_designation(icd, cpt)
        self._check_cpt_dx_linkage(cpt, icd)
        self._check_dx_pointers(icd, cpt, hcpcs)
        self._check_orphan_dx(icd, cpt, hcpcs)
        self._check_modifiers(cpt)
        # modifier_reasoning_consistency runs BEFORE the real-data-driven
        # modifier checks below (bilateral_modifier52, em_modifier25,
        # hcpcs/cpt_laterality) — not after. Found live: _check_em_modifier25
        # correctly removed a stale -25 in favor of -57 for a same-day major
        # procedure, but modifier_reasoning still carried the LLM's original
        # "-25 applied" claim (never updated to reflect the removal); running
        # reasoning-consistency afterward re-added -25 right back, leaving
        # both -25 and -57 present — the exact self-contradiction
        # verify_notes.py's -25/-57 mutual-exclusivity check exists to catch,
        # reintroduced by a check that runs after the authoritative decision.
        # Reasoning-consistency's job is resolving the LLM's OWN self-
        # contradiction (reasoning vs. array) — it has no authority over a
        # correction the real-data checks below are about to make, so it
        # must settle first and let those checks have the final, unretracted
        # word.
        self._check_modifier_reasoning_consistency(cpt, hcpcs)
        self._check_bilateral_modifier52(cpt)
        self._check_em_patient_status(cpt, note_category)
        # Axis floors BEFORE the level-consistency check: a floor may raise
        # the 2-of-3 median, and the consistency check below owns the
        # descriptor-driven sibling swap that realizes the corrected level.
        self._check_em_mdm_problems_floor(cpt, icd)
        self._check_em_mdm_risk_floor(cpt, icd, dos, patient_dob)
        self._check_em_mdm_risk_high_floor(cpt, note_full_text)
        self._check_em_mdm_data_floor(cpt, icd, note_full_text)
        # Ceiling after the floors (a floor may legitimately raise other
        # axes) and before the level-consistency swap that realizes it.
        self._check_em_mdm_problems_ceiling(cpt, icd, note_full_text)
        self._check_em_level_consistency(cpt)
        self._check_em_modifier25(cpt)
        self._check_modifier57(cpt, note_plan_text)
        # After the -25/-57 checks settle the modifier state: the NCCI
        # manual's own separately-identifiable test decides whether the E/M
        # line is billable AT ALL alongside a same-day minor procedure.
        self._check_em_minor_procedure_bundling(cpt, icd)
        # Before laterality/NCCI: may swap a debridement code for the family
        # member the documented tissue depth supports, and everything
        # downstream must see the final code.
        self._check_debridement_depth(cpt, note_full_text)
        # ...and the operative-field gate right after it: a debridement
        # whose only documentation is margin/edge preparation of a
        # same-claim surgery is included in the surgery (NCCI Ch.1) — the
        # suppression must land before the modifier/NCCI layers reason
        # about the line.
        self._check_operative_field_debridement(cpt, note_full_text)
        # Unbilled-descriptor match here, BEFORE the modifier/NCCI layers,
        # because its comprehensive-upgrade arm can change a line's CODE
        # (97597→11740-class): the digit/laterality normalizations and the
        # NCCI evaluation below must see the final code, and the upgraded
        # line must itself pass through them (measured live, note 004: an
        # upgraded line kept the RT its pre-upgrade code carried, flapping
        # against runs that billed the digit modifier).
        self._check_unbilled_descriptor_match(cpt, hcpcs, note_full_text)
        # Pathology-axis arbitration within a CPT family (same reasons as
        # above: it changes codes, so it runs before modifiers/NCCI). The
        # eg-parenthetical is the AMA's own statement of WHICH pathologies
        # a code exists for; a billed sibling whose distinguishing terms
        # the note never documents yields to the family member whose
        # eg-pathology is affirmatively documented.
        # Extent axis BEFORE pathology axis: when the operative wording
        # itself proves a partial removal, that swap settles the family and
        # the pathology arbitration must see the corrected code.
        self._check_excision_extent_axis(cpt, note_full_text)
        self._check_cpt_family_pathology_axis(cpt, note_full_text)
        self._check_image_guidance(cpt, note_full_text)
        # Dispensed-supply completion BEFORE the laterality/modifier layers:
        # an added supply line must pass through the same siding and
        # normalization as one the coder billed itself.
        self._check_dispensed_footwear_completion(hcpcs, cpt, icd,
                                                  note_full_text, dos,
                                                  patient_dob)
        self._check_hcpcs_laterality(cpt, hcpcs)
        self._check_cpt_laterality(cpt)
        # After the side-ADDING checks above, before anything that reads the
        # final modifier state: strips RT/LT made redundant by a digit
        # modifier, then strips 59/X where the PTP table shows no edit needs
        # bypassing. Both are pure normalizations — same claim facts, one
        # canonical modifier spelling — so independent runs of the same note
        # converge instead of flapping on optional decorations.
        self._check_redundant_laterality(cpt, hcpcs)
        # Guidance strip runs BEFORE the 59/X evaluation below: once RT/LT
        # is off a guidance line, that check correctly sees no anatomic
        # separation and judges 59 purely on the PTP table. Digit-supply
        # alignment likewise needs the procedures' final digit modifiers.
        self._check_guidance_laterality(cpt)
        # CPT digit upgrade BEFORE the supply alignment: once a procedure
        # line's RT/LT is upgraded to its true digit modifier, the supply
        # check can derive the same digit from the claim's own lines.
        self._check_cpt_digit_laterality(cpt, icd, note_full_text)
        # ...and its mirror: a digit modifier on a line whose descriptor and
        # dx linkage are only side-level gets normalized back to RT/LT.
        self._check_digit_modifier_scope(cpt)
        self._check_digit_supply_modifier(cpt, hcpcs, note_full_text)
        self._check_supply_laterality_strip(hcpcs)
        # With the digit modifiers final: identical exact-site designators on
        # an indicator-1 PTP pair prove same-site work — the column-2 line
        # bundles and is removed before the separation-modifier checks below
        # reason about what 59 may bypass.
        self._check_same_site_ptp_bundling(cpt)
        # Placement before the strip: once every separation modifier sits on
        # its pair's column-2 line, the strip judges necessity on the final
        # arrangement and removes whatever placement left decorative.
        self._check_separation_modifier_placement(cpt)
        self._check_unnecessary_separation_modifier(cpt)
        # After all modifier-mutating laterality checks above — needs the
        # final RT/LT state to compare against the linked diagnoses' sides.
        self._check_icd_cpt_laterality_agreement(cpt, icd)
        # NCCI runs after every modifier-mutating check above, not before —
        # its "was a separation modifier applied" test needs the final
        # modifiers array. Running it first meant it could evaluate against
        # a same-day 090-global E/M+procedure pair before -57 was even
        # added, always reporting "modifier exception allowed but not
        # applied" for a pair that resolved correctly moments later.
        self._check_ncci(cpt)
        self._check_billability(cpt, hcpcs)
        self._check_imaging_note_evidence(cpt, note_full_text)
        # Context gate after the presence gate: the modality IS in the note,
        # but every mention may be a prior-visit film, an ordered/future
        # study, or intraop confirmation bundled into a same-claim surgery.
        self._check_imaging_context(cpt, note_full_text)
        # View-count arbitration last: once a line survives both gates, the
        # descriptors' own view requirements decide WHICH family member the
        # documentation supports.
        self._check_radiograph_view_count(cpt, note_full_text)
        # Age correction FIRST — it may swap a code for its age-appropriate
        # sibling, and the evidence check below must assess the final code.
        self._check_hcpcs_age_range(hcpcs, dos, patient_dob)
        self._check_descriptor_variant_evidence(hcpcs, note_full_text)
        # Best-code-vs-note layers: each verifies the CHOSEN code's own
        # descriptor (or an unbilled sibling's) against the note's actual
        # words — the dimension the structural checks above can't see.
        self._check_cpt_descriptor_evidence(cpt, note_full_text)
        # (_check_unbilled_descriptor_match runs earlier, pre-modifier/NCCI —
        # its upgrade arm changes codes and everything below must see them.)
        self._check_icd_sibling_descriptor(icd, note_full_text)
        self._check_with_without_axis(icd, note_full_text)
        # Severity-tier arbitration alongside the sibling checks: final-
        # character axes (ulcer depth) are structurally excluded from the
        # token-diff sibling check (same-last-char constraint), so the
        # ordered-tier mechanic owns them.
        self._check_ulcer_severity_tier(icd, note_full_text)
        # Onset axis after the token-diff sibling checks: they own axes where
        # the descriptors differ on documented clinical attributes; this one
        # owns the acute/chronic/congenital qualifiers their ubiquity filters
        # deliberately exclude (df > 500), routed by the Alphabetic Index.
        self._check_onset_qualifier_axis(icd, note_full_text)
        # Residual-member downgrade after every specific-sibling arbitration
        # above (they own axes where a DOCUMENTED attribute picks the member;
        # this one owns the case where the specific member's qualifier is
        # documented nowhere and the family's own 'unspecified' code is the
        # only supportable spelling).
        self._check_undocumented_specific_sibling(icd, cpt, hcpcs,
                                                  note_full_text)
        self._check_injury_seventh_char(icd)
        self._check_measurement_companion(icd, note_full_text)
        self._check_redundant_dm_codes(icd)
        self._check_unjustified_zcodes(icd)
        # Mirror-image checks of the two above/below: instead of validating
        # what IS on the claim, they catch what the Tabular List's own
        # instructional notes say SHOULD be — a required companion code
        # (use additional code) or underlying etiology (code first) that the
        # note documents but the code arrays omit.
        # Assessment completion BEFORE the instructional-note family: an
        # added diagnosis may itself trigger (or satisfy) a Tabular
        # instruction, and the demotion pass further down must see it.
        self._check_assessment_dx_completion(icd, note_assessment_text)
        self._check_missing_use_additional_code(icd, coding_result, note_full_text)
        self._check_missing_code_first_etiology(icd, note_full_text)
        self._check_missing_code_also(icd, note_full_text)
        # With-convention completion alongside the instructional-note family:
        # the Index's 'with' subterms are presumed linked (guideline I.A.15),
        # so a diabetes + skin-ulcer claim must carry the combination code.
        self._check_diabetes_ulcer_combination(icd)
        # Second primary-designation pass: the completion checks above can
        # ADD the very codeFirst etiology the first pass (line ~185) could
        # not see. Measured live (note 004): runs where the LLM billed
        # E11.621 itself got it promoted over the L97.5- ulcer, runs where
        # the with-convention added it kept the ulcer primary — the claim's
        # final state must designate identically either way. Idempotent:
        # when the designation is already convention-correct it returns
        # without touching anything.
        self._check_primary_designation(icd, cpt)
        self._check_icd_excludes1(icd)
        # Demotion AFTER every check that can add or mandate a companion —
        # the mandated-companion exemption must see the final instruction
        # state — and BEFORE the pointer-integrity pass below, which drops
        # service-line pointers to the demoted codes.
        self._check_marginal_secondary_demotion(icd, coding_result,
                                                note_assessment_text)
        # Second pointer-integrity pass: the descriptor-evidence checks above
        # (_check_icd_sibling_descriptor, _check_with_without_axis,
        # _check_injury_seventh_char) can swap an ICD code, and excludes1 can
        # remove one — either leaves service lines pointing at a code no
        # longer on the claim. Same deterministic remap/drop as the first
        # pass, applied to the final diagnosis list.
        self._check_dx_pointer_integrity(icd, cpt, hcpcs)
        self._check_snomed_consistency(snomed)
        # Fix 1 — Physician code preservation checks
        all_codes = icd + cpt + hcpcs
        self._check_physician_code_preservation(all_codes, coding_result,
                                                physician_documented_codes or [], note_full_text)

        # Auto-generated declarative rules (flip-actuation pipeline): every
        # enabled pack rule marked auto_generated dispatches through its
        # template's generic executor. Runs BEFORE the suppression blocks
        # below so an auto rule's suppressions take effect this pass.
        self.rule_engine.run_auto_rules(icd, cpt, hcpcs, coding_result,
                                        note_full_text, note_assessment_text)

        # Conservation gate BEFORE the suppression blocks realize removals:
        # a documentation-mismatch removal must either substitute the family
        # member the documented work supports, or escalate loudly — never
        # drop documented work silently (measured live, routine_00001:
        # 27650's correct removal silently uncoded 27654's documented work).
        self._check_removal_conservation(cpt, note_full_text)

        # Remove bundled codes from CPT list (NCCI suppression)
        if self._bundled_codes_to_suppress:
            original_count = len(cpt)
            cpt[:] = [c for c in cpt if c.get("code", "") not in self._bundled_codes_to_suppress]
            removed = original_count - len(cpt)
            if removed:
                logger.info(f"  Suppressed {removed} NCCI-bundled CPT code(s): {self._bundled_codes_to_suppress}")
                coding_result["cpt_codes"] = cpt

        # Remove not-separately-billable codes from CPT/HCPCS lists. This is
        # the deterministic backstop for what the LLM's own audit narrative
        # (corrections_made / auto_coding_review_reasons) claims to have
        # done — a claim in those free-text fields is not itself a code
        # removal, and was observed to diverge from the actual arrays.
        if self._non_billable_codes_to_suppress:
            original_cpt_count = len(cpt)
            original_hcpcs_count = len(hcpcs)
            cpt[:] = [c for c in cpt if c.get("code", "") not in self._non_billable_codes_to_suppress]
            hcpcs[:] = [c for c in hcpcs if c.get("code", "") not in self._non_billable_codes_to_suppress]
            removed = (original_cpt_count - len(cpt)) + (original_hcpcs_count - len(hcpcs))
            if removed:
                logger.info(f"  Suppressed {removed} not-separately-billable code(s): {self._non_billable_codes_to_suppress}")
                coding_result["cpt_codes"] = cpt
                coding_result["hcpcs_codes"] = hcpcs

        # Source-level half of advisory suppression: validator WARNINGs
        # that assert the same adjudicated defect as a suppressed scrub
        # advisory are removed HERE, before the tier/report derivations
        # below read self.issues — so validation_issues, warnings,
        # pre_submission_audit_findings, and auto_coding_review_reasons
        # can never ship an advisory the scrub simultaneously suppresses
        # (observed live on routine_00003's Z79.01).
        self._apply_validator_advisory_suppressions()

        enc = self._encounter_integrity(icd)
        audit = self._documentation_audit(coding_result)
        tier, confidence, reasons = self._compute_tier(icd, cpt, hcpcs)

        critical_issues = [
            i for i in self.issues if i.severity in ("ERROR", "WARNING", "CRITICAL")
        ]

        return {
            "validation_issues": [i.model_dump() for i in self.issues],
            "encounter_integrity": enc.model_dump(),
            "documentation_audit": audit.model_dump(),
            "pre_submission_audit_findings": [i.model_dump() for i in critical_issues],
            "pre_submission_audit_score": self._audit_score(),
            "auto_coding_tier": tier,
            "auto_coding_confidence": confidence,
            "auto_coding_review_reasons": reasons,
            "auto_coding_summary": self._summary(tier, reasons),
            # Every claim-mutating action the layers took this pass — the
            # clinical-correctness audit (tools/clinical_auditor.py)
            # verifies these against the note and the authorities before
            # the claim can auto-verify into the registry. Self-reported
            # corrections come first; the pre/post claim diff then appends
            # a derived entry for every mutation no layer reported, so the
            # audit's field of view is the claim itself, not the layers'
            # own confessions.
            "material_corrections": self._material_corrections_with_derived(
                pre_claim, icd, cpt, hcpcs, coding_result),
            # Scrub-advisory suppressions recorded by declarative rules
            # (suppress_scrub_advisory): the scrubber runs AFTER validation
            # on the assembled result payload, so the instruction must ride
            # the report to reach it. WARN-only by contract (enforced again
            # at the scrubber — config can never suppress a FAIL).
            "scrub_advisory_suppressions":
                list(self._scrub_advisory_suppressions),
        }

    # --- Individual checks ---

    def _check_code_existence(self, icd, cpt, hcpcs, dos=None):
        """Existence is checked in two stages, not one: validate_*() (any
        point in time) first, then is_active_for_dos() (real
        effective_from/effective_to) only for codes that do exist — this
        distinguishes "not a real code" (ERROR, wrong code entirely) from
        "a real code, just not valid on this claim's date of service"
        (its own, more specific finding), rather than collapsing both into
        one generic "not found" message."""
        for entry in icd:
            code = entry.get("code", "")
            if not self.db.validate_icd10(code):
                # The curated reference set is a podiatry SUBSET; the store's
                # tabular table holds the entire CDC/NCHS code set. A code
                # present in the tabular but outside the subset is real
                # (D48.1 class, soft-tissue neoplasm) — route to review for
                # specialty relevance rather than erroring as nonexistent.
                tab = (self.store.icd10_tabular_description(code)
                       if self.store is not None else "")
                if tab:
                    entry["needs_review"] = True
                    entry.setdefault(
                        "review_reason",
                        f"real ICD-10-CM code ('{tab[:60]}') outside the curated "
                        f"specialty subset — verify relevance")
                    self._add(
                        "WARNING", code, "icd_outside_subset",
                        f"ICD-10-CM {code} ('{tab[:60]}') is a real FY2026 code but "
                        f"outside the curated specialty subset",
                        "Verify specialty relevance; keep if the documentation supports it",
                        denial_risk="LOW",
                    )
                else:
                    self._add(
                        "ERROR", code, "code_existence",
                        f"ICD-10-CM {code} not found in FY2026 code set",
                        "Verify code or use a valid alternative",
                        denial_risk="HIGH",
                    )
            elif dos and not self.db.is_active_for_dos("icd10", code, dos):
                self._add(
                    "ERROR", code, "code_not_active_for_dos",
                    f"ICD-10-CM {code} exists but is not effective on this claim's date of service ({dos})",
                    "Verify the date of service, or use the code valid for that date",
                    denial_risk="HIGH",
                )
            else:
                entry["s3_validated"] = True

        for entry in cpt:
            code = entry.get("code", "")
            if not self.db.validate_cpt(code):
                self._add(
                    "ERROR", code, "code_existence",
                    f"CPT {code} not found in code set",
                    "Verify code or use a valid alternative",
                    denial_risk="HIGH",
                )
            elif dos and not self.db.is_active_for_dos("cpt", code, dos):
                self._add(
                    "ERROR", code, "code_not_active_for_dos",
                    f"CPT {code} exists but is not effective on this claim's date of service ({dos})",
                    "Verify the date of service, or use the code valid for that date",
                    denial_risk="HIGH",
                )
            else:
                entry["ama_validated"] = True

        for entry in hcpcs:
            code = entry.get("code", "")
            if not code:
                continue
            if not self.db.validate_hcpcs(code):
                self._add(
                    "INFO", code, "code_existence",
                    f"HCPCS {code} not found in database (may still be valid — verify with payer)",
                    "Verify HCPCS code validity with payer",
                    denial_risk="MEDIUM",
                )
            elif dos and not self.db.is_active_for_dos("hcpcs", code, dos):
                self._add(
                    "ERROR", code, "code_not_active_for_dos",
                    f"HCPCS {code} exists but is not effective on this claim's date of service ({dos})",
                    "Verify the date of service, or use the code valid for that date",
                    denial_risk="HIGH",
                )

    def _check_ncci(self, cpt):
        codes = [c.get("code", "") for c in cpt if c.get("code")]
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                conflict = self.db.check_ncci(codes[i], codes[j])
                if not conflict:
                    continue

                mod_indicator = str(conflict.get("modifier", "")).strip()
                # CMS NCCI modifier indicators: 0 = hard bundle (no bypass),
                # 1 = separation modifier may bypass, 9 = the edit itself was
                # DELETED/is not applicable — indicator-9 pairs aren't
                # conflicts at all and must be skipped, not treated as
                # "modifier allowed" (which wrongly asked coders to add
                # 59/XE… to a pair CMS no longer edits).
                if mod_indicator == "9":
                    continue
                modifier_allowed = mod_indicator == "1"

                # 59/XE/XS/XP/XU separate two procedures from each other; 25/57
                # separate an E/M from a procedure — different purposes, so only
                # credit 25/57 when one side of the pair is actually an E/M code.
                # Previously both families were accepted for every pair, which
                # both under- and over-credited: a real 99215-25/28140 pair (25
                # correctly applied) still got flagged as "not applied" because
                # 25 wasn't in the accepted set at all, and a procedure/procedure
                # pair could have been wrongly satisfied by an unrelated 25/57
                # elsewhere on the claim.
                pair_is_em = _is_em(codes[i]) or _is_em(codes[j])
                sep_modifiers = {"25", "57"} if pair_is_em else {"59", "XE", "XS", "XP", "XU"}
                code_i_entry = next((c for c in cpt if c.get("code") == codes[i]), {})
                code_j_entry = next((c for c in cpt if c.get("code") == codes[j]), {})
                has_separator = (
                    bool(set(code_i_entry.get("modifiers", [])) & sep_modifiers)
                    or bool(set(code_j_entry.get("modifiers", [])) & sep_modifiers)
                )
                # Anatomic NCCI-associated modifiers (RT/LT, F/T digits,
                # eyelids, coronary — derived from the modifier reference
                # data): differing anatomic sets on the two lines document
                # distinct sites and bypass an indicator-1 edit like 59/X.
                # Same-side-on-both (RT vs RT) separates nothing.
                if not has_separator and not pair_is_em and self.store is not None:
                    anatomic = self.store.anatomic_modifiers()
                    sites_i = {str(m).strip().upper()
                               for m in code_i_entry.get("modifiers", [])} & anatomic
                    sites_j = {str(m).strip().upper()
                               for m in code_j_entry.get("modifiers", [])} & anatomic
                    if self._sites_distinct(sites_i, sites_j):
                        has_separator = True

                if modifier_allowed and has_separator:
                    self._add(
                        "INFO", f"{codes[i]}|{codes[j]}", "ncci_edit",
                        f"NCCI pair {codes[i]}/{codes[j]} — modifier exception applied",
                        "Verify distinct service is documented",
                        denial_risk="LOW",
                    )
                elif modifier_allowed and not has_separator:
                    suggestion = "25 or 57" if pair_is_em else "59/XE/XS/XP/XU"
                    self._add(
                        "WARNING", f"{codes[i]}|{codes[j]}", "ncci_edit",
                        f"NCCI conflict: {codes[i]} and {codes[j]} — modifier exception allowed but not applied",
                        f"Add modifier {suggestion} if services are distinct and documented",
                        denial_risk="MEDIUM",
                    )
                else:
                    if conflict.get("code1") == codes[i]:
                        bundled = codes[j]
                        primary = codes[i]
                    else:
                        bundled = codes[i]
                        primary = codes[j]
                    self._bundled_codes_to_suppress.add(bundled)
                    self._add(
                        "ERROR", f"{codes[i]}|{codes[j]}", "ncci_edit",
                        f"NCCI hard edit: {primary} and {bundled} are mutually exclusive. "
                        f"{bundled} is bundled into {primary} — suppressing {bundled}.",
                        f"Remove {bundled} from claim — it is included in {primary}",
                        denial_risk="HIGH",
                    )

    def _check_billability(self, cpt, hcpcs):
        if self.store is None:
            return
        for entry in cpt + hcpcs:
            code = entry.get("code", "")
            if not code:
                continue
            reason = self.store.not_separately_billable_reason(code)
            if reason is not None:
                # The code is deterministically REMOVED from the claim right
                # below (see _non_billable_codes_to_suppress) — a completed
                # auto-correction, reported as INFO like every other
                # AUTO-CORRECTED item. Leaving it at ERROR both misstated the
                # final claim's state (the problem is already gone) and
                # dragged the confidence tier down for a non-issue.
                self._non_billable_codes_to_suppress.add(code)
                self._add(
                    "INFO", code, "billability",
                    f"AUTO-CORRECTED: {code} removed — not separately billable ({reason}).",
                    "No action needed; the code was suppressed before submission",
                    denial_risk="LOW",
                )
                continue
            # HCPCS coverage code I/M/S — the alpha-numeric HCPCS file's own
            # Medicare non-coverage verdict (S-codes etc. never appear on
            # the PFS, so the billing-status check above can't see them).
            # Payer-gated exactly like the MUE-0 suppression: on a
            # Medicare-bound payer (FFS + MA per 42 CFR 422.101) the line
            # denies in any circumstance, so it is deterministically
            # suppressed — measured live (note 004, Humana MA), S8450
            # flapped present-in-2-of-3 runs while the billability agent
            # FAILed it every time it appeared; a never-payable line's
            # presence must not be a coin flip. Other payers keep the
            # review flag (they may reimburse under their own policy).
            cov_reason = (self.store.hcpcs_noncoverage_reason(code)
                          if entry in hcpcs else None)
            if cov_reason:
                if getattr(self, "_payer_follows_medicare", False):
                    self._non_billable_codes_to_suppress.add(code)
                    self._add(
                        "INFO", code, "billability",
                        f"AUTO-CORRECTED: {code} removed — HCPCS {cov_reason}, "
                        f"and this claim's payer follows Medicare coverage "
                        f"rules. If the supply is real, it is patient-billable "
                        f"with a signed ABN or billable to a payer that covers "
                        f"it, not on this claim.",
                        f"{code} suppressed per its own HCPCS coverage code",
                        denial_risk="LOW",
                    )
                else:
                    entry["needs_review"] = True
                    self._add(
                        "WARNING", code, "billability",
                        f"{code}: HCPCS {cov_reason} — this payer may still "
                        f"reimburse it under its own policy.",
                        f"Confirm this payer covers {code}; remove the line "
                        f"for Medicare-bound claims",
                        denial_risk="MEDIUM",
                    )
                continue
            # PFS status 'X' = excluded from the Physician Fee Schedule but
            # typically payable under another schedule (CLFS labs, DMEPOS
            # supplies) — a review-level routing question (who bills it,
            # under which schedule), never an auto-suppression. See
            # store.pfs_exclusion_advisory for why these are split.
            advisory = self.store.pfs_exclusion_advisory(code)
            if advisory:
                self._add(
                    "WARNING", code, "pfs_exclusion",
                    f"{code}: {advisory}.",
                    f"Confirm {code} is billed by the performing entity under its own fee "
                    f"schedule; if a reference lab/DME supplier bills it, remove it from this claim",
                    denial_risk="MEDIUM",
                )

    def _check_mue(self, lines):
        for entry in lines:
            code = entry.get("code", "")
            units = entry.get("units", 1)
            mue = self.db.get_mue(code)
            if mue is None:
                continue
            if mue == 0:
                # An MUE of 0 is CMS's "no units are ever payable" value —
                # every unit denies on a Medicare-bound claim. Found live:
                # 90389 (tetanus immune globulin) billed at 1 unit and slipping
                # through because this check only compared units when mue > 0.
                #
                # Payer-gated enforcement: for a payer bound to Medicare's
                # coverage floor the line denies at ANY quantity, so it is
                # auto-suppressed (same mechanism as NCCI/status-P
                # suppression) — this deterministically enacts the exact
                # recommendation the MUE agent was already issuing, and ends
                # the measured run-to-run flapping (A4570/A6545/L1940 supply
                # lines present in 2/3 independent runs of the same notes;
                # a never-payable line's presence should not be a coin
                # flip). Commercial/unrecognized payers may reimburse under
                # their own policy, so those claims keep the flag-and-review
                # behavior instead.
                if getattr(self, "_payer_follows_medicare", False):
                    self._non_billable_codes_to_suppress.add(code)
                    self._add(
                        "INFO", code, "mue_limit",
                        f"AUTO-CORRECTED: Removed {code} — its MUE is 0 on a "
                        f"professional claim (CMS pays zero units in any "
                        f"circumstance) and this claim's payer follows Medicare "
                        f"coverage rules. If the supply/service is real, it belongs "
                        f"on its payable channel (e.g. DMEPOS/DME MAC) or the "
                        f"payable equivalent code the documentation supports.",
                        f"{code} suppressed per its own MUE of 0",
                        denial_risk="LOW",
                    )
                else:
                    self._add(
                        "ERROR", code, "mue_limit",
                        f"{code} carries an MUE of 0 — CMS pays zero units of this code "
                        f"in any circumstance; all {units} unit(s) will deny on a "
                        f"Medicare-bound claim.",
                        f"Verify payer: remove {code} for Medicare, or confirm the payer "
                        f"reimburses it under its own policy",
                        denial_risk="HIGH",
                    )
                    entry["needs_review"] = True
            elif units > mue:
                self._add(
                    "ERROR", code, "mue_limit",
                    f"{code} units ({units}) exceeds MUE limit ({mue})",
                    f"Reduce to {mue} or document medical necessity for exception",
                    denial_risk="HIGH",
                )
            else:
                entry["mue_validated"] = True
                entry["mue_limit"] = mue

    # LCD medical-necessity coverage (routine foot care and everything else) is
    # handled by MedicalNecessityAgent (filter #5) against the generic
    # coverage_cpt/coverage_icd tables — see app/compliance/agents/
    # medical_necessity.py. This used to be duplicated here via a hardcoded
    # ROUTINE_FOOT_CARE_CPTS set; removed in favor of the one data-driven check.

    def _check_drug_units(self, hcpcs, note_full_text: str = ""):
        """Deterministic backstop for drug (J-code family) unit math — units
        previously defaulted to 1 with nothing verifying the prompt's own
        mg→units instruction (e.g. 'Kenalog 40mg' billed as J3301 x 1 when
        the descriptor is 'per 10 mg' → 4 units; a 4x underbill).

        Fully data-driven: the per-unit denomination comes from the code's
        OWN CMS long descriptor ('Injection, triamcinolone acetonide, ...,
        10 mg' = one billing unit per 10 mg), and the administered dose from
        the entry's own quoted evidence spans. A unit MISMATCH is flag-only
        (no auto-correct): a dose quote may legitimately describe a
        different quantity than what was drawn/billed (wastage, bilateral
        splits), so the resolution needs the note.

        A drug line with NO documented dose anywhere in the note is
        different: the code's own denomination makes the billing unit a
        dose statement, so a claim line whose units cannot be derived from
        any documentation is unbillable as written and is removed.
        Determinism layer, measured live (note 009): a J-code for 'IV
        vancomycin initiated empirically' — no dose, no quantity — flapped
        present-in-1-of-3 runs; the missing dose is the same claim fact in
        every run, so the line's absence must be too."""
        _, low_note = self._note_evidence(note_full_text or "")
        for entry in hcpcs:
            code = entry.get("code", "")
            info = self.db.validate_hcpcs(code) or {}
            desc = (info.get("long_description") or "").lower()
            # Only descriptors that denominate a dose: "..., 10 mg" / "per 1 mg" / "..., 6 mg"
            m = re.search(r"(?:,|per)\s*([\d.]+)\s*(mg|mcg|units?|ml)\s*$", desc.strip())
            if not m or not desc.startswith("injection"):
                continue
            per_unit, uom = float(m.group(1)), m.group(2).rstrip("s")
            if per_unit <= 0:
                continue
            evidence = " ".join(
                (entry.get("evidence_spans") or []) + [entry.get("supporting_text", "") or ""]
            ).lower()
            doses = re.findall(rf"([\d.]+)\s*{uom}\b", evidence)
            if not doses and low_note:
                # No dose in the entry's own evidence — the whole note gets
                # one chance: a quantity in a sentence naming the drug (the
                # descriptor's own agent word, e.g. 'vancomycin').
                parts = desc.split(",")
                agent = next(
                    (t for t in re.findall(r"[a-z]+", parts[1])
                     if len(t) >= 5), "") if len(parts) >= 2 else ""
                agent_doses: list[str] = []
                for sent in re.split(r"[.;]", low_note.replace("\n", " ")):
                    if agent and agent in sent:
                        agent_doses += re.findall(rf"([\d.]+)\s*{uom}\b", sent)
                        if uom == "mg":
                            # clinical notes dose some agents in grams
                            # ('vancomycin 1 g IV') — same fact, ×1000
                            agent_doses += [
                                str(float(g) * 1000) for g in
                                re.findall(r"([\d.]+)\s*g(?:ram|rams)?\b", sent)]
                if not agent_doses:
                    self._non_billable_codes_to_suppress.add(code)
                    self._add(
                        "WARNING", code, "drug_dose_undocumented",
                        f"AUTO-CORRECTED: {code} removed — its own descriptor "
                        f"denominates the billing unit in {uom} "
                        f"({per_unit:g} {uom} per unit), but neither the line's "
                        f"evidence nor any note sentence naming the drug "
                        f"documents an administered dose, so the units cannot "
                        f"be derived from the documentation.",
                        f"Document the administered dose (in {uom}) and re-bill "
                        f"{code} with the computed units",
                        denial_risk="HIGH",
                    )
                    continue
                doses = agent_doses
            if len(set(doses)) != 1:
                continue  # no dose or ambiguous doses documented — nothing to verify
            documented = float(doses[0])
            # float-safe ceiling: fractional denominations are real ('per
            # 0.5 mg', 'per 0.1 mg') and int() truncation of those divided
            # by zero live (note 013, integer modulo by zero)
            expected = max(1, math.ceil(documented / per_unit))
            billed = int(entry.get("units", 1) or 1)
            if billed != expected:
                self._add(
                    "WARNING", code, "drug_units_mismatch",
                    f"{code} billed with {billed} unit(s), but the documented dose "
                    f"({documented:g} {uom}) at the code's own denomination "
                    f"({per_unit:g} {uom} per unit) computes to {expected} unit(s). "
                    f"{'Underbilling' if billed < expected else 'Overbilling'} risk.",
                    f"Verify administered dose and set units to {expected} "
                    f"(or document wastage/JW if the difference is discarded drug)",
                    denial_risk="MEDIUM",
                )

    # A documented duration ('over 60 minutes', 'infused for 1 hour') or a
    # start/stop clock pair — the two forms CPT's time rule accepts.
    _INFUSION_DURATION_RE = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:min(?:ute)?s?|h(?:ou)?rs?)\b", re.IGNORECASE)
    _INFUSION_CONTEXT_RE = re.compile(
        r"\binfus|\bintravenous\b|\biv\b|\bdrip\b", re.IGNORECASE)

    def _check_timed_infusion_documentation(self, cpt, note_full_text: str):
        """CPT's own time rule for infusion administration codes: a
        descriptor denominated in time ('Intravenous infusion ... initial,
        up to 1 hour') makes documented infusion time an element OF the
        code — without a duration or start/stop times, the service as
        described was not documented and the line cannot be billed.

        Data-driven: the time gate comes from the billed code's OWN AMA
        descriptor (contains 'infusion' AND an hour/minute denomination);
        no list of administration codes exists anywhere. Evidence is a
        duration or clock-time pair in a negation-scrubbed note sentence
        that speaks infusion/IV language.

        Determinism layer, measured live (note 009): 96365 flapped
        present-in-1-of-3 runs on 'IV vancomycin initiated empirically per
        ID recommendation' — initiation per an external recommendation,
        with no site, no duration, no start/stop. The documentation is the
        same in every run, so the line's absence must be too."""
        if not note_full_text:
            return
        _, low_note = self._note_evidence(note_full_text)
        timed_documented = None
        for entry in list(cpt):
            code = entry.get("code", "")
            info = self.db.validate_cpt(code) or {}
            desc = (info.get("long_description")
                    or info.get("short_description") or "").lower()
            if "infusion" not in desc or not re.search(
                    r"\b(?:hour|minutes?)\b", desc):
                continue
            if timed_documented is None:
                timed_documented = any(
                    self._INFUSION_CONTEXT_RE.search(sent)
                    and (self._INFUSION_DURATION_RE.search(sent)
                         or len(re.findall(r"\b\d{1,2}:\d{2}\b", sent)) >= 2)
                    for sent in re.split(r"[.;]",
                                         low_note.replace("\n", " ")))
            if timed_documented:
                continue
            self._non_billable_codes_to_suppress.add(code)
            self._add(
                "WARNING", code, "infusion_time_undocumented",
                f"AUTO-CORRECTED: {code} removed — its own descriptor "
                f"denominates the service in time ('{desc[:90]}'), but no "
                f"note sentence documents an infusion duration or start/stop "
                f"times, so the time-based service as described was not "
                f"documented.",
                f"Document the infusion start/stop times or duration and "
                f"re-bill {code}",
                denial_risk="HIGH",
            )

    _COUNT_NOUNS = r"(?:nails?|lesions?|calluses?|corns?|digits?|toenails?)"

    def _check_count_based_selection(self, cpt, note_full_text: str = ""):
        """Deterministic backstop for count-ranged code families (nail
        debridement '1 to 5' vs '6 or more', lesion paring 'single'/'2 to
        4'/'more than 4') — the code CHOICE encodes the count, and nothing
        previously verified it against the documented count.

        Data-driven: the eligible range comes from the billed code's OWN AMA
        descriptor phrasing, and the documented count from the entry's
        evidence spans (falling back to the note). Codes whose descriptors
        carry no count range are skipped automatically — no hardcoded list
        of 'count-based codes' exists anywhere."""
        for entry in cpt:
            code = entry.get("code", "")
            info = self.db.validate_cpt(code) or {}
            desc = (info.get("long_description") or info.get("short_description") or "").lower()
            lo = hi = None
            m = re.search(r"\b(\d+)\s+to\s+(\d+)\b", desc)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
            elif (m := re.search(r"\b(\d+)\s+or\s+more\b", desc)):
                lo, hi = int(m.group(1)), None
            elif (m := re.search(r"\bmore\s+than\s+(\d+)\b", desc)):
                lo, hi = int(m.group(1)) + 1, None
            elif re.search(rf"\bsingle\s+{self._COUNT_NOUNS}", desc):
                lo, hi = 1, 1
            if lo is None:
                continue
            evidence = " ".join(
                (entry.get("evidence_spans") or []) + [entry.get("supporting_text", "") or ""]
            ).lower()
            counts = re.findall(rf"(\d+)\s*{self._COUNT_NOUNS}", evidence)
            if len(set(counts)) != 1:
                continue  # no count or conflicting counts documented — cannot verify
            documented = int(counts[0])
            in_range = documented >= lo and (hi is None or documented <= hi)
            if not in_range:
                range_txt = f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}–{hi}")
                self._add(
                    "WARNING", code, "count_range_mismatch",
                    f"{code}'s own descriptor covers {range_txt} per its AMA text, but the "
                    f"documentation quotes {documented} — the count determines which code in "
                    f"this family applies, so this selection looks wrong.",
                    f"Re-select the code in the same family whose descriptor range covers "
                    f"{documented}, or correct the documented count",
                    denial_risk="MEDIUM",
                )

    def _check_duplicate_diagnoses(self, icd, coding_result):
        """Remove literal duplicate ICD entries (same code listed twice).
        Observed live: Q66.89 emitted as both principal AND secondary — the
        MCE duplicate-of-principal edit then flags it, but the claim itself
        is malformed and the fix is purely mechanical. The entry with the
        'primary' designation wins (else the first occurrence); pointers are
        unaffected since they reference the code string."""
        seen: dict[str, dict] = {}
        for entry in icd:
            key = (entry.get("code") or "").replace(".", "").upper()
            if not key:
                continue
            if key not in seen:
                seen[key] = entry
            elif entry.get("type") == "primary" and seen[key].get("type") != "primary":
                seen[key] = entry
        removed = [e for e in icd if e is not seen.get((e.get("code") or "").replace(".", "").upper())]
        if removed:
            icd[:] = [e for e in icd if e not in removed]
            coding_result["icd10_codes"] = icd
            self._add(
                "INFO", removed[0].get("code", ""), "duplicate_diagnosis_removed",
                f"AUTO-CORRECTED: removed {len(removed)} duplicate diagnosis entr"
                f"{'y' if len(removed) == 1 else 'ies'} "
                f"({', '.join(e.get('code', '') for e in removed)}) — the same code was "
                f"listed more than once on the claim.",
                "No action needed",
                denial_risk="LOW",
            )

    def _check_sequencing(self, icd):
        primaries = [c for c in icd if c.get("type") == "primary"]
        if icd and not primaries:
            # Mirror image of the multi-primary demotion below: a claim with
            # NO first-listed diagnosis is rejected on transmission just as
            # surely as one with several. List order IS the claim's stated
            # sequencing, so the first-listed code takes the designation.
            icd[0]["type"] = "primary"
            self._add(
                "INFO", icd[0].get("code", ""), "sequencing",
                f"AUTO-CORRECTED: no ICD-10-CM code was marked primary — "
                f"promoted the first-listed ({icd[0].get('code')}) to primary; "
                f"a claim needs a first-listed diagnosis.",
                "Verify the first-listed diagnosis is the visit's principal reason",
                denial_risk="LOW",
            )
        if len(primaries) > 1:
            # Deterministic demotion: the claim's first-listed primary keeps
            # the designation (list order IS the claim's stated sequencing);
            # the rest become secondary. A claim transmitted with multiple
            # "primary" markers is malformed either way, and which code is
            # first-listed doesn't change — only the redundant markers do.
            for extra in primaries[1:]:
                extra["type"] = "secondary"
            self._add(
                "INFO", primaries[0].get("code", ""), "sequencing",
                f"AUTO-CORRECTED: {len(primaries)} ICD-10-CM codes were marked primary — "
                f"kept the first-listed ({primaries[0].get('code')}) and demoted "
                f"{', '.join(c.get('code', '') for c in primaries[1:])} to secondary.",
                "Verify the first-listed diagnosis is the visit's principal reason",
                denial_risk="LOW",
            )

    def _check_primary_designation(self, icd, cpt):
        """First-listed (primary) diagnosis from the claim's own procedure
        linkage. ICD-10-CM guideline IV.G: the first-listed diagnosis is
        the condition chiefly responsible for the services provided. On a
        procedure encounter the claim's structure already encodes that —
        each non-E/M service line's FIRST diagnosis pointer names the
        condition that procedure treated. Determinism layer, measured live
        (note 004): an identical note flipped which of two diagnoses was
        typed primary across independent runs, because designation was left
        to the LLM's discretion while the procedure linkage was stable.

        Fires only on the unambiguous shape: every non-E/M procedure line's
        first pointer names the SAME on-claim diagnosis, the current
        primary is not first-pointed by any of them, and the anchor is not
        a Chapter-20 external-cause or manifestation code. The
        etiology/manifestation convention (guideline I.A.13) outranks the
        pointer: when the anchored code carries a codeFirst instruction
        whose etiology is on the claim, the etiology takes the designation.
        Runs BEFORE _check_dx_pointers, whose reorder-primary-to-front
        auto-fix would overwrite exactly the evidence this check reads."""
        if not icd:
            return  # (empty cpt is fine — the codeFirst arm needs no lines)

        def _norm(c) -> str:
            return str(c or "").replace(".", "").strip().upper()

        # Unconditional arm first — the etiology/manifestation convention
        # (guideline I.A.13) is MANDATORY sequencing, independent of the
        # procedure-pointer shape below: when the CURRENT primary's own
        # Tabular codeFirst note names another billed code, that etiology
        # takes the designation. Measured live (note 004): one run of three
        # kept the L97.5- ulcer primary over its billed E11.621 diabetes
        # etiology because the pointer arm's unanimity gate never fired.
        if self.store is not None:
            current = next(
                (e for e in icd if e.get("type") == "primary"), None)
            if current is not None:
                refs = [_norm(r) for r in self.store.code_first_etiology_refs(
                    _norm(current.get("code")))]
                etio = next(
                    (e for e in icd if e is not current
                     and any(_norm(e.get("code")).startswith(r)
                             for r in refs if r)), None)
                if etio is not None:
                    current["type"] = "secondary"
                    current["needs_review"] = True
                    etio["type"] = "primary"
                    etio["needs_review"] = True
                    self._add(
                        "WARNING", etio.get("code", ""), "primary_designation",
                        f"AUTO-CORRECTED: {etio.get('code')} designated primary "
                        f"(was {current.get('code')}) — {current.get('code')}'s "
                        f"own Tabular 'code first' instruction names it as the "
                        f"underlying etiology, and the etiology/manifestation "
                        f"convention (ICD-10-CM guideline I.A.13) is mandatory "
                        f"sequencing.",
                        "Sequence the codeFirst etiology before its "
                        "manifestation code",
                        denial_risk="MEDIUM",
                    )
                    logger.info(
                        f"  Primary designation: {etio.get('code')} "
                        f"(codeFirst etiology of {current.get('code')})")

        # Suppressed lines don't vote: a line the validator already removed
        # (NCCI same-site bundling, billability) is not on the final claim,
        # so its pointer cannot anchor the first-listed designation.
        # Measured live (note 002): runs that billed a paronychia I&D — a
        # line the same-site PTP check then bundled away — kept the
        # paronychia primary via this arm's ambiguity gate, while runs
        # without that line anchored the ingrown-nail code; the FINAL claim
        # is identical either way, so the designation must be too.
        suppressed = (getattr(self, "_non_billable_codes_to_suppress", set())
                      | getattr(self, "_bundled_codes_to_suppress", set()))

        # Medical-necessity arm (payer-gated like the MUE-0 suppression:
        # LCD/Article coverage lists bind Original Medicare and MA per
        # 42 CFR 422.101). When the payer's own coverage policies for the
        # claim's procedure codes mark exactly one billed diagnosis as
        # qualifying — after collapsing a manifestation onto its billed
        # codeFirst etiology — that diagnosis IS the condition chiefly
        # responsible for the covered services (guideline IV.G), and the
        # payability of every governed line rides on it being pointed
        # first. Determinism layer, measured live: note 006 flipped
        # L60.1/S90.112A primary with the pointer arm below following the
        # LLM's own run-variant pointer order, while the coverage lists
        # (11730's nail-surgery LCDs qualify L60.1, never the contusion)
        # are identical in every run. Unlike the pointer arm, this reads
        # only run-invariant inputs: the billed code SETS and the policy
        # data.
        if (self.store is not None
                and getattr(self, "_payer_follows_medicare", False)):
            def _covered(policy_id: str, dx: str) -> bool:
                n = _norm(dx)
                row = self.store.conn.execute(
                    "SELECT 1 FROM coverage_icd WHERE policy_id=? AND "
                    "(icd_code=? OR ? LIKE icd_code || '%') LIMIT 1",
                    (policy_id, n, n)).fetchone()
                return row is not None

            qualifying: list = []
            for entry in cpt:
                pcode = entry.get("code", "")
                if not pcode or _is_em(pcode) or pcode in suppressed:
                    continue
                for pol in self.store.coverage_policies_for_cpt(pcode):
                    if not self.store.coverage_policy_has_dx_rules(pol):
                        continue
                    for e in icd:
                        if e not in qualifying and _covered(pol, e.get("code", "")):
                            qualifying.append(e)
            # collapse manifestation -> billed codeFirst etiology: the
            # etiology outranks it as first-listed (guideline I.A.13)
            collapsed = []
            for e in qualifying:
                refs = [_norm(r) for r in
                        self.store.code_first_etiology_refs(_norm(e.get("code")))]
                etio = next(
                    (o for o in icd if o is not e
                     and any(_norm(o.get("code")).startswith(r)
                             for r in refs if r)), None)
                target = etio if etio is not None else e
                if target not in collapsed:
                    collapsed.append(target)
            if len(collapsed) == 1:
                anchor_entry = collapsed[0]
                anchor_code = _norm(anchor_entry.get("code"))
                current = next(
                    (e for e in icd if e.get("type") == "primary"), None)
                if (anchor_code[:1] not in ("V", "W", "X", "Y")
                        and (current is None
                             or _norm(current.get("code")) != anchor_code)):
                    old_code = current.get("code", "") if current else "(none)"
                    if current is not None:
                        current["type"] = "secondary"
                        current["needs_review"] = True
                    anchor_entry["type"] = "primary"
                    anchor_entry["needs_review"] = True
                    self._add(
                        "WARNING", anchor_entry.get("code", ""),
                        "primary_designation",
                        f"AUTO-CORRECTED: {anchor_entry.get('code')} designated "
                        f"primary (was {old_code}) — it is the only billed "
                        f"diagnosis the payer's own coverage policies for this "
                        f"claim's procedures list as qualifying (medical "
                        f"necessity rides on it), making it the condition "
                        f"chiefly responsible for the services provided "
                        f"(ICD-10-CM guideline IV.G first-listed).",
                        "Verify the first-listed diagnosis is the coverage-"
                        "qualifying condition the procedures treated",
                        denial_risk="MEDIUM",
                    )
                    logger.info(
                        f"  Primary designation: {anchor_entry.get('code')} "
                        f"(coverage-qualifying anchor; was {old_code})")
                return  # this arm owns the designation when policies govern

        def _votes(entry) -> bool:
            code = entry.get("code", "")
            return (bool(code) and not _is_em(code)
                    and code not in suppressed
                    and bool(entry.get("linked_diagnoses")))

        first_ptrs = {
            _norm((entry.get("linked_diagnoses") or [None])[0])
            for entry in cpt if _votes(entry)
        }
        first_ptrs.discard("")
        if len(first_ptrs) != 1:
            return  # no procedure lines, or they disagree — ambiguous
        anchor = first_ptrs.pop()
        if anchor[:1] in ("V", "W", "X", "Y"):
            return  # external-cause codes are never first-listed (I.C.20)
        # etiology/manifestation: the anchored manifestation's codeFirst
        # etiology, when billed, is the true first-listed code
        etio_applied = False
        if self.store is not None:
            refs = [_norm(r) for r in
                    self.store.code_first_etiology_refs(anchor)]
            etio = next(
                (e for e in icd
                 if any(_norm(e.get("code")).startswith(r)
                        for r in refs if r)), None)
            if etio is not None:
                anchor = _norm(etio.get("code"))
                etio_applied = True
        anchor_entry = next(
            (e for e in icd if _norm(e.get("code")) == anchor), None)
        if anchor_entry is None:
            return
        current = next((e for e in icd if e.get("type") == "primary"), None)
        if current is not None and _norm(current.get("code")) == anchor:
            return
        # A current primary that is itself some procedure line's first
        # pointer keeps the designation (both diagnoses drive procedures —
        # sequencing between them is a judgment call, not this check's).
        # EXCEPT under the etiology convention: codeFirst is mandatory
        # sequencing, so a first-pointed manifestation still yields to its
        # billed etiology.
        if (current is not None and not etio_applied and any(
                _norm((entry.get("linked_diagnoses") or [None])[0])
                == _norm(current.get("code"))
                for entry in cpt if _votes(entry))):
            return
        old_code = current.get("code", "") if current else "(none)"
        if current is not None:
            current["type"] = "secondary"
            current["needs_review"] = True
        anchor_entry["type"] = "primary"
        anchor_entry["needs_review"] = True
        self._add(
            "WARNING", anchor_entry.get("code", ""), "primary_designation",
            f"AUTO-CORRECTED: {anchor_entry.get('code')} designated primary "
            f"(was {old_code}) — every non-E/M procedure line's first "
            f"diagnosis pointer names it, making it the condition chiefly "
            f"responsible for the services provided (ICD-10-CM guideline "
            f"IV.G first-listed), while the previous primary supported no "
            f"procedure line as its principal diagnosis.",
            "Verify the first-listed diagnosis is the visit's principal "
            "reason; the claim's own procedure linkage fixed it",
            denial_risk="MEDIUM",
        )
        logger.info(
            f"  Primary designation: {anchor_entry.get('code')} "
            f"(from procedure linkage; was {old_code})")

    def _check_dx_pointer_integrity(self, icd, cpt, hcpcs):
        """A service line's diagnosis pointer must reference a code that is
        actually on the claim — a dangling pointer is a transmission
        rejection. Found live three times in one audit: the claim's ICD was
        corrected to a sibling (S90.122A→S90.112A, I70.262→I70.245,
        Q69.2→Q69.9) but the lines kept pointing at the pre-correction code.
        Repair is deterministic and code-agnostic: a dangling pointer is
        remapped to the claim code sharing the longest category prefix
        (>=3 chars, the ICD-10-CM family) when that match is unique — i.e.
        its corrected sibling — otherwise dropped (the linkage backfill
        below restocks an emptied line from the claim's diagnoses)."""
        claim = [c.get("code", "") for c in icd if c.get("code")]
        claim_norm = {c.replace(".", "").upper(): c for c in claim}
        for entry in cpt + hcpcs:
            linked = entry.get("linked_diagnoses") or []
            if not linked:
                continue
            repaired, remaps, drops = [], [], []
            for ptr in linked:
                norm = str(ptr).replace(".", "").strip().upper()
                if norm in claim_norm:
                    repaired.append(claim_norm[norm])
                    continue
                best, best_len = [], 0
                for cn, orig in claim_norm.items():
                    n = len(os.path.commonprefix([norm, cn]))
                    if n >= 3:
                        if n > best_len:
                            best, best_len = [orig], n
                        elif n == best_len:
                            best.append(orig)
                if len(best) == 1 and best[0] not in repaired:
                    repaired.append(best[0])
                    remaps.append((ptr, best[0]))
                else:
                    drops.append(ptr)
            if not remaps and not drops:
                continue
            entry["linked_diagnoses"] = repaired
            code = entry.get("code", "")
            detail = "; ".join(
                ([f"remapped {a}→{b}" for a, b in remaps]) +
                ([f"dropped {p} (no unique family match)" for p in drops]))
            self._add(
                "INFO", code, "dx_pointer_integrity",
                f"AUTO-CORRECTED: {code}'s diagnosis pointer(s) referenced code(s) "
                f"not on the claim ({detail}) — a corrected diagnosis left the "
                f"service line pointing at its pre-correction value.",
                "Verify the remapped pointers support this line's medical necessity",
                denial_risk="LOW",
            )

    def _check_cpt_dx_linkage(self, cpt, icd):
        # billing_status 'B' (bundled/no-charge, e.g. 99024 post-op follow-up)
        # codes are legitimately billed without a linked ICD — queried from
        # real data (global_periods.json), not a hardcoded singleton for the
        # one code this used to be checked against. Same fix already applied
        # to the identical concept in _check_em_modifier25.
        claim_dx = [c.get("code", "") for c in icd if c.get("code")]
        primary_first = sorted(claim_dx, key=lambda d: next(
            (c.get("type") != "primary" for c in icd if c.get("code") == d), True))
        for entry in cpt:
            code = entry.get("code", "")
            if self.store is not None and self.store.billing_status(code) == "B":
                continue
            if not entry.get("linked_diagnoses"):
                if claim_dx:
                    # An empty box 24E is a rejection, not a judgment call —
                    # backfill deterministically: coverage-qualifying dxs for
                    # this code first (LCD/Article lists), then primary, then
                    # claim order, capped at the CMS-1500's 4 pointers.
                    coverage_dx = []
                    if self.store is not None:
                        pols = self.store.coverage_policies_for_cpt(code)
                        coverage_dx = [dx for dx in primary_first if any(
                            self.store.coverage_icd_covered(p, dx) for p in pols)]
                    ordered = coverage_dx + [d for d in primary_first if d not in coverage_dx]
                    entry["linked_diagnoses"] = ordered[:4]
                    self._add(
                        "INFO", code, "cpt_icd_linkage",
                        f"AUTO-CORRECTED: CPT {code} had no linked diagnosis (box 24E empty "
                        f"= rejection) — linked {', '.join(ordered[:4])} from the claim's "
                        f"diagnosis list.",
                        "Verify the linked diagnoses support this procedure's medical necessity",
                        denial_risk="LOW",
                    )
                else:
                    self._add(
                        "WARNING", code, "cpt_icd_linkage",
                        f"CPT {code} has no linked diagnosis code and the claim carries no "
                        f"diagnoses to link",
                        "Link at least one ICD-10-CM code for medical necessity",
                        denial_risk="MEDIUM",
                    )

    def _check_dx_pointers(self, icd, cpt, hcpcs):
        """CMS-1500 diagnosis-pointer structural rules, previously unenforced:

        * Box 24E allows at most 4 diagnosis pointers per service line — a
          5th+ linked diagnosis silently fails electronic submission (or the
          clearinghouse drops an ARBITRARY pointer). Auto-trim to the 4 most
          defensible pointers with a deterministic, explainable ranking:
          primary diagnosis first, then diagnoses that satisfy a coverage
          policy governing this code, then clinical-condition codes over
          Z-chapter status codes (chapter grammar, not a medical-rule list),
          then original claim order. The dropped codes stay on the claim's
          diagnosis list — only this line's pointers change.
        * The line's primary-supporting diagnosis should be pointer 1. When
          the encounter's primary diagnosis is among a line's linked
          diagnoses but not FIRST, reorder it to the front (auto-fix: pure
          reordering never changes WHICH diagnoses support the line, only
          their pointer sequence, so it cannot alter medical meaning).
        """
        primary_codes = [c.get("code", "") for c in icd if c.get("type") == "primary"]
        primary = primary_codes[0] if primary_codes else None
        for entry in cpt + hcpcs:
            linked = entry.get("linked_diagnoses") or []
            if len(linked) > 4:
                code = entry.get("code", "")
                coverage_dx = set()
                if self.store is not None:
                    pols = self.store.coverage_policies_for_cpt(code)
                    coverage_dx = {
                        dx for dx in linked
                        if any(self.store.coverage_icd_covered(p, dx) for p in pols)
                    }
                ranked = sorted(
                    enumerate(linked),
                    key=lambda p: (p[1] != primary,
                                   p[1] not in coverage_dx,
                                   p[1].upper().startswith("Z"),
                                   p[0]),
                )
                keep = {dx for _, dx in ranked[:4]}
                dropped = [dx for dx in linked if dx not in keep]
                entry["linked_diagnoses"] = [dx for dx in linked if dx in keep]
                linked = entry["linked_diagnoses"]
                self._add(
                    "WARNING", code, "dx_pointer_overflow",
                    f"AUTO-CORRECTED: {code} linked {len(dropped) + 4} diagnoses, but CMS-1500 "
                    f"box 24E carries at most 4 diagnosis pointers per service line — kept "
                    f"{', '.join(linked)} (ranked: primary > coverage-qualifying > clinical "
                    f"condition > Z-chapter status > claim order) and unlinked "
                    f"{', '.join(dropped)} from this line. The dropped code(s) remain on the "
                    f"claim's diagnosis list.",
                    "Review the kept pointers; repoint manually if a dropped diagnosis is a "
                    "stronger driver of this line's medical necessity",
                    denial_risk="MEDIUM",
                )
            if primary and primary in linked and linked[0] != primary:
                linked.remove(primary)
                linked.insert(0, primary)
                entry["linked_diagnoses"] = linked
                self._add(
                    "INFO", entry.get("code", ""), "dx_pointer_reordered",
                    f"AUTO-CORRECTED: moved primary diagnosis {primary} to pointer 1 on "
                    f"{entry.get('code')} — CMS-1500 expects the line's principal supporting "
                    f"diagnosis first.",
                    "Primary diagnosis reordered to pointer 1",
                    denial_risk="LOW",
                )

    # Minimum length for a descriptor token to count as a distinguishing
    # attribute — shorter tokens are abbreviations/particles, not attributes.
    _VARIANT_MIN_TOKEN_LEN = 4

    def _hcpcs_variant_index(self) -> dict:
        """code → [(own_token, sibling_token, sibling_code), ...] for HCPCS
        codes whose CMS long descriptors are identical except ONE
        distinguishing word — e.g. Q4038 "...short leg cast, adult,
        fiberglass" vs Q4037 "...short leg cast, adult, plaster". Derived
        entirely from the codes' own descriptors (no curated pairs list):
        two codes are variant siblings when removing one word from each
        leaves the same token set. Built once per validator instance."""
        if getattr(self, "_variant_idx", None) is not None:
            return self._variant_idx
        sig_map: dict[frozenset, list[tuple[str, str]]] = {}
        for code, info in (getattr(self.db, "hcpcs", {}) or {}).items():
            desc = (info.get("long_description") or info.get("description") or "").lower()
            toks = frozenset(re.findall(r"[a-z]+", desc))
            if len(toks) < 3:
                continue  # too short to define a family (would over-pair)
            for t in toks:
                if len(t) >= self._VARIANT_MIN_TOKEN_LEN:
                    sig_map.setdefault(toks - {t}, []).append((code, t))
        idx: dict[str, list[tuple[str, str, str]]] = {}
        for members in sig_map.values():
            # small families only — a signature shared by many codes is a
            # generic phrase, not a this-or-that attribute choice
            if not (2 <= len(members) <= 6):
                continue
            for code, tok in members:
                for code2, tok2 in members:
                    if code2 != code and tok2 != tok:
                        idx.setdefault(code, []).append((tok, tok2, code2))
        self._variant_idx = idx
        return idx

    # Age qualifier embedded in CMS HCPCS long descriptors — "adult (11
    # years +)", "pediatric (0-10 years)", "infant (0-9 months)". A range
    # is only actionable when it has an upper bound or an explicit "+".
    _AGE_RANGE_RE = re.compile(r"\((\d+)(?:\s*-\s*(\d+))?\s*(year|month)s?\s*(\+)?\)")

    def _descriptor_age_range(self, code: str):
        """(min_years, max_years) parsed from the code's own CMS descriptor,
        or None when the descriptor carries no age qualifier."""
        info = self.db.validate_hcpcs(code) or {}
        m = self._AGE_RANGE_RE.search(info.get("long_description") or "")
        if not m:
            return None
        lo, hi, unit, plus = int(m[1]), m[2], m[3], m[4]
        if hi is None and not plus:
            return None  # "(1 year)" style durations are not age ranges
        scale = 1 / 12 if unit == "month" else 1
        return (lo * scale, float(hi) * scale if hi is not None else float("inf"))

    def _check_hcpcs_age_range(self, hcpcs, dos, patient_dob: str):
        """Enforce the age qualifier that CMS bakes into HCPCS descriptors.
        Found live on note 031: Q4039 — 'Cast supplies, short leg cast,
        pediatric (0-10 years), plaster' — was billed for a 70-year-old.
        The age is fully determined by the note's DOB + DOS, and the
        correct code is the variant sibling that differs ONLY on the age
        attribute, so this auto-corrects (deterministic, data-driven from
        the descriptors themselves) rather than merely flagging."""
        from app.compliance.engine import _parse_date
        dob = _parse_date(patient_dob or "")
        if not dob or not dos:
            return
        age = (dos - dob).days / 365.25
        if age < 0:
            return
        idx = self._hcpcs_variant_index()
        for entry in hcpcs:
            code = entry.get("code", "")
            rng = self._descriptor_age_range(code)
            if rng is None or rng[0] <= age <= rng[1]:
                continue
            # unique sibling whose descriptor matches the patient's age
            fits = sorted({
                sib for _own, _sib_tok, sib in idx.get(code) or []
                if (r := self._descriptor_age_range(sib)) and r[0] <= age <= r[1]
            })
            if len(fits) == 1:
                new_code = fits[0]
                new_info = self.db.validate_hcpcs(new_code) or {}
                old = code
                entry["code"] = new_code
                entry["description"] = new_info.get("description", entry.get("description", ""))
                self._add(
                    "WARNING", new_code, "hcpcs_age_range_mismatch",
                    f"AUTO-CORRECTED: {old} is age-restricted by its own CMS descriptor "
                    f"('{(self.db.validate_hcpcs(old) or {}).get('long_description', '')}') but "
                    f"the patient is {int(age)} at the date of service — switched to {new_code} "
                    f"('{new_info.get('long_description', '')}'), the variant sibling matching "
                    f"the patient's age.",
                    "Verify the corrected supply code",
                    denial_risk="HIGH",
                )
            else:
                self._add(
                    "ERROR", code, "hcpcs_age_range_mismatch",
                    f"{code} is age-restricted by its own CMS descriptor "
                    f"('{(self.db.validate_hcpcs(code) or {}).get('long_description', '')}') but "
                    f"the patient is {int(age)} at the date of service, and no single "
                    f"age-appropriate variant sibling exists to substitute.",
                    "Replace with the age-appropriate code for this supply",
                    denial_risk="HIGH",
                )

    def _check_descriptor_variant_evidence(self, hcpcs, note_full_text: str):
        """A code chosen among near-identical variant siblings must have its
        distinguishing attribute documented. Found live on note 031: the
        note said only "well-padded cast materials", but Q4038 (fiberglass)
        was billed — the material was assumed, and if it's actually plaster
        the correct code is the plaster sibling. Flag-only (needs review):
        the right resolution requires confirming the fact with the provider,
        not swapping to another equally unevidenced sibling."""
        if not note_full_text:
            return
        note_words = set(re.findall(r"[a-z]+", note_full_text.lower()))
        idx = self._hcpcs_variant_index()
        for entry in hcpcs:
            code = entry.get("code", "")
            gaps, contradicted, seen = [], False, set()
            for own_tok, sib_tok, sib_code in idx.get(code) or []:
                if own_tok in note_words or own_tok in seen:
                    continue
                own_rng = self._descriptor_age_range(code)
                sib_rng = self._descriptor_age_range(sib_code)
                if own_rng and sib_rng and own_rng != sib_rng:
                    # age-qualified pair (adult vs pediatric): decided by the
                    # note's DOB + DOS in _check_hcpcs_age_range, never by
                    # whether the prose happens to say "adult"
                    continue
                seen.add(own_tok)
                if sib_tok in note_words:
                    contradicted = True
                gaps.append(f"'{own_tok}' (vs '{sib_tok}' → {sib_code})")
            if not gaps:
                continue
            self._add(
                "WARNING", code, "descriptor_variant_unverified",
                f"{code} was selected among near-identical variant codes, but the note does "
                f"not document the distinguishing attribute(s): {'; '.join(gaps[:3])}."
                + (" The note documents the SIBLING's attribute — the selected variant may "
                   "be wrong." if contradicted else ""),
                "Confirm the attribute (e.g. actual material/size/age category) with the "
                "provider before submission; switch to the sibling code if it is the "
                "documented one",
                denial_risk="HIGH" if contradicted else "MEDIUM",
            )
            entry["needs_review"] = True
            prev = (entry.get("review_reason") or "").strip()
            note = f"Variant attribute unverified: {'; '.join(gaps[:3])}."
            entry["review_reason"] = f"{prev} {note}".strip()

    # ---- Best-code-vs-note layers ------------------------------------
    # Shared tokenizer for descriptor-vs-note comparisons. Hyphenated
    # compounds are joined first ('post-traumatic' -> 'posttraumatic') so
    # the compound is one token on both the descriptor and note side —
    # otherwise the fragment 'post' becomes an "attribute" that any
    # post-op note appears to document.
    @staticmethod
    def _tokens(text: str) -> set:
        joined = re.sub(r"(?<=[a-z])-(?=[a-z])", "", (text or "").lower())
        return set(re.findall(r"[a-z]+", joined))

    # Light suffix stemmer so descriptor "excision" matches the note's
    # "excised", "rupture" matches "ruptured", etc. Linguistic
    # normalization only — no clinical content.
    @staticmethod
    def _stem(token: str) -> str:
        if token.endswith("ies") and len(token) >= 7:
            return token[:-3] + "y"   # deformities -> deformity
        for suf in ("ations", "ation", "ition", "sions", "sion", "tion",
                    "ing", "ed", "es", "s", "al", "ic", "y"):
            # '-al' bridges the noun/verb registers descriptors and op notes
            # split across: descriptor 'removal' vs note 'removed' both stem
            # to 'remov' (measured live, note 001: the '...for permanent
            # removal' descriptor sat undocumented against a note that wrote
            # 'nail plate removed'). The len guard keeps short words (oral,
            # renal) whole.
            # '-ic'/'-y' bridge the Greek noun/adjective registers the same
            # way: descriptor 'dystrophy' vs note 'dystrophic' both stem to
            # 'dystroph' (measured live, note routine_00003: 'periodic
            # debridement of DYSTROPHIC, thickened toenails' never counted
            # as documentation of L60.3 'Nail DYSTROPHY', so the specificity
            # demotion overrode the expert reviewer's ruling on every replay
            # for four convergence cycles). Same pattern: atrophy/atrophic,
            # hypertrophy/hypertrophic (and cystic/cyst gets the same
            # bridge). The len guard keeps short words (bony, toxic)
            # whole, exactly as it does for oral/renal on '-al'.
            if token.endswith(suf) and len(token) - len(suf) >= 4:
                return token[: -len(suf)]
        return token

    # Latin <-> English anatomical synonym groups used interchangeably in
    # op notes vs. code descriptors — a language lexicon (like
    # MODALITY_LEXICON above), not a code list. Bidirectional: any member
    # present in the note makes the whole group documented.
    _TERM_EQUIV_GROUPS = (
        frozenset({"hallux", "halluces", "great"}),
        frozenset({"interdigital", "intermetatarsal", "interspace", "webspace"}),
        frozenset({"tendinitis", "tendonitis"}),  # accepted spelling variants
        # standard clinical abbreviations for diabetes mellitus — notes write
        # 'T2DM'/'DM' where descriptors spell 'Type 2 diabetes mellitus'
        frozenset({"t2dm", "t1dm", "dm", "diabetes", "diabetic", "mellitus"}),
        # a benign bony outgrowth: notes write 'exostosis' where AMA
        # descriptors write 'bossing' (28124 '...eg, osteomyelitis or
        # bossing') — same entity, different register
        frozenset({"exostosis", "bossing", "osteochondroma"}),
        # grammatical forms of the same bone: descriptors alternate
        # 'phalanges of foot' / 'phalanx of toe' / 'phalangeal base' for
        # the identical anatomy — a token diff between two descriptors
        # must never treat these as a distinguishing attribute
        frozenset({"phalanx", "phalanges", "phalangeal"}),
        # releasing a fluid collection: descriptors write 'evacuation'
        # (11740 'Evacuation of subungual hematoma') where op notes write
        # 'decompression'/'trephination'/'drainage' — the same intervention
        frozenset({"evacuation", "decompression", "trephination", "drainage"}),
    )

    # Negation cues: a finding mentioned only to deny it ('No lymphangitis.',
    # 'denies fever', 'without erythema') is NOT documentation of the finding.
    # Allergy/avoidance cues likewise: 'ALLERGIES: Aspirin (GI bleed —
    # avoided)' is documentation that the drug is NOT in use. The scrubbed
    # span runs to the next punctuation/section boundary.
    # 'below threshold for fasciotomy' / 'not indicated' are threshold-negation
    # constructions: the named intervention was evaluated and NOT performed.
    _NEGATION_RE = re.compile(
        r"\b(?:no|denies|denied|without|negative\s+for|absent|non-?tender|"
        r"ruled?\s+out|free\s+of|allerg(?:y|ies|ic)|avoided|discontinued|"
        r"below\s+(?:the\s+)?threshold\s+for|not\s+indicated|"
        r"criteria\s+not\s+met\s+for)\b[^.;:\n()]*",
        re.IGNORECASE)

    # Futurity/deferral cues: a sentence about a procedure that was
    # 'discussed as surgical option', 'deferred pending clearance',
    # 'planned', or 'recommended' documents intent, not performance. Used
    # by checks that ask 'was this SERVICE rendered today' (the unbilled
    # dedicated-code match), NOT by diagnosis-evidence checks — a condition
    # discussed in the plan is still a documented condition.
    _FUTURITY_RE = re.compile(
        r"\b(?:deferred|planned|discussed|scheduled|recommend(?:ed|s)?|"
        r"consider(?:ed|ing)?|option[s]?|elective|referr(?:ed|al)|"
        r"if\s+(?:unsuccessful|symptoms|pain|no\s+improvement)|future|"
        r"pending|will\s+(?:discuss|schedule|consider))\b",
        re.IGNORECASE)

    def _performed_context(self, note_full_text: str) -> str:
        """The note minus sentences that talk about future/deferred care."""
        parts = re.split(r"(?<=[.;:\n])", note_full_text or "")
        return " ".join(p for p in parts if not self._FUTURITY_RE.search(p))

    def _note_evidence(self, note_full_text: str) -> tuple:
        """(word set incl. stems and terminology equivalents, lowercase text).
        Negated spans are scrubbed first — 'No lymphangitis' must not make
        'lymphangitis' count as documented."""
        low = self._NEGATION_RE.sub(" ", (note_full_text or "").lower())
        words = self._tokens(low)
        words |= {self._stem(w) for w in set(words)}
        for group in self._TERM_EQUIV_GROUPS:
            if group & words:
                words.update(group)
                words.update(self._stem(g) for g in group)
        return words, low

    def _one_sentence_rare_anchor(self, rare_toks, low_text: str) -> bool:
        """>=2 of a descriptor's rare tokens inside ONE sentence: the note
        names the dedicated act itself ('partial nail avulsion ...') rather
        than grazing its vocabulary across the document. The completion arm
        of the unbilled-descriptor check requires this — scattered hits
        qualify a candidate for an advisory WARNING, never for adding a
        claim line."""
        toks = [t for t in rare_toks if t]
        if len(toks) < 2:
            return False
        for sent in re.split(r"[.;\n]", low_text or ""):
            sw = self._tokens(sent)
            sw |= {self._stem(w) for w in set(sw)}
            n = sum(1 for t in toks
                    if t in sw or self._stem(t) in sw
                    or (len(t) >= 5 and any(w.startswith(t) for w in sw)))
            if n >= 2:
                return True
        return False

    def _desc_documented(self, token: str, note_words: set, note_text: str) -> bool:
        """Word-boundary, stem, or compound-prefix match — deliberately NOT
        substring: the note's 'microfracture' must never satisfy a
        descriptor's 'fracture' (that exact collision hid a fracture-code
        misuse). Compound-prefix covers closed compounds the other way:
        descriptor 'hammer toe' IS documented by the note's 'hammertoe'."""
        if token in note_words or self._stem(token) in note_words:
            return True
        if len(token) >= 5:
            return any(w.startswith(token) for w in note_words)
        return False

    # Incidental-context markers: operative-logistics language whose anatomy
    # words name equipment placement, positioning, or prep — never pathology.
    # Measured live (note routine_00001): 'a well-padded THIGH tourniquet
    # inflated to 300 mmHg' made 'thigh' count as documentation of a
    # thigh-deformity sibling, and the swap relocated a heel diagnosis to
    # the thigh. Language lexicon (no medical codes): these are surgical-
    # workflow words, not clinical entities.
    _INCIDENTAL_CONTEXT_RE = re.compile(
        r"\b(?:tourniquet|esmarch|exsanguinat\w*|prepped|draped|drapes?|"
        r"positioned|positioning|prone|supine|time-?out|timeout|"
        r"anesthesi\w*|anesthet\w*|sedation|sedated|"
        r"chlorhexidine|betadine|povidone|sterile fashion)\b")

    def _clinical_note_view(self, note_full_text: str) -> str:
        """The note minus incidental-context sentences. Evidence that
        CHANGES a claim (a sibling swap's distinguishing attribute, a
        completion's descriptor match) must be clinically assertive — a
        body part mentioned only while describing tourniquet placement,
        positioning, prep/drape, or anesthesia is not documentation of a
        condition there. Protective evidence (what keeps a billed code)
        deliberately still reads the full note: blocking a change on broad
        evidence is the safe direction, making one on it is not."""
        text = note_full_text or ""
        kept = [s for s in re.split(r"(?<=[.;\n])", text)
                if not self._INCIDENTAL_CONTEXT_RE.search(s.lower())]
        return "".join(kept)

    def _clinical_evidence(self, note_full_text: str) -> tuple:
        """_note_evidence over the clinical view — the evidence set every
        claim-mutating comparison must use (cached per note text)."""
        cache = getattr(self, "_clin_ev_cache", None)
        key = hash(note_full_text or "")
        if cache is not None and cache[0] == key:
            return cache[1]
        ev = self._note_evidence(self._clinical_note_view(note_full_text))
        self._clin_ev_cache = (key, ev)
        return ev

    # Descriptor stopwords: grammar/qualifier words that never distinguish
    # one clinical entity from another (NOT a medical-code list).
    _DESC_STOPWORDS = frozenset(
        "a an and any are as at by each for from in including includes of on or "
        "other others specified unspecified than the to with without when performed "
        "eg ie etc not more less all both same single multiple initial subsequent "
        "encounter sequela right left foot feet toe toes acquired except".split())

    _EG_PAREN_RE = re.compile(r"\((?:eg|e\.g\.)[^)]*\)")

    def _anatomy_lexicon(self) -> set:
        """Anatomy/site vocabulary mined from ICD-10-CM category headings:
        the words AFTER the 'of/at' pivot name sites ('Fracture OF foot and
        toe', 'Dislocation and sprain OF joints and ligaments...'), the ones
        before it name conditions.         Tokens recurring in >=2 heading tails are
        site words — they must never be treated as the clinical entity a
        code turns on."""
        if getattr(self, "_anat_lex", None) is not None:
            return self._anat_lex
        self._anat_lex = {t for t in self._site_lexicon() if len(t) >= 5}
        return self._anat_lex

    def _site_lexicon(self) -> set:
        """The anatomy mining WITHOUT the len>=5 floor: short site words
        ('hip', 'rib', 'jaw', 'arm') are body sites too. Kept separate from
        _anatomy_lexicon because its consumers use the length floor as a
        noise filter for entity-vs-site judgments; the sibling-swap site
        guard needs the complete site vocabulary — measured live, a swap
        blocked at 'shoulder' simply landed on 'hip' next, which the
        len>=5 lexicon cannot see."""
        if getattr(self, "_site_lex", None) is not None:
            return self._site_lex
        cnt: dict[str, int] = {}
        cats = (self.store.icd10_category_descriptions()
                if self.store is not None else [])
        for _c, desc in cats:
            parts = re.split(r"\b(?:of|at|involving)\b", desc.lower(), maxsplit=1)
            if len(parts) < 2:
                continue
            for t in self._tokens(parts[1]):
                if len(t) >= 3 and t not in self._DESC_STOPWORDS:
                    cnt[t] = cnt.get(t, 0) + 1
        self._site_lex = {t for t, n in cnt.items() if n >= 2}
        return self._site_lex

    def _icd_condition_lexicon(self) -> set:
        """Clinical-condition vocabulary mined from the ICD-10-CM code set's
        own descriptions: tokens (len >= 6) appearing in at least 25 distinct
        code descriptions are condition/entity words (fracture, dislocation,
        melanoma, cellulitis...). Derived from data, not curated."""
        if getattr(self, "_cond_lex", None) is not None:
            return self._cond_lex
        df: dict[str, int] = {}
        for entry in (getattr(self.db, "icd10", {}) or {}).values():
            for t in self._tokens(entry.get("description", "")):
                if len(t) >= 3 and t not in self._DESC_STOPWORDS:
                    df[t] = df.get(t, 0) + 1
        # df counts EVERY token (>=3 chars) so ubiquity filters see short
        # anatomy words too — 'ankle'/'joint' appear in thousands of
        # descriptions and must never pass a rare-attribute gate. The
        # condition lexicon itself keeps the len>=6 floor.
        self._cond_lex = {t for t, n in df.items() if n >= 25 and len(t) >= 6}
        self._icd_token_df = df
        return self._cond_lex

    def _injury_entity_lexicon(self) -> set:
        """Injury-entity vocabulary mined from the ICD-10-CM S/T-chapter
        CATEGORY headings ('Fracture of foot and toe...', 'Dislocation and
        sprain of joints...'): the words those headings share across many
        categories are the injury entities themselves (fracture,
        dislocation, sprain, wound, burn...), while anatomy words appear in
        only a few headings each and are dropped by the frequency floor.
        Authoritative and curated-list-free."""
        if getattr(self, "_injury_lex", None) is not None:
            return self._injury_lex
        cnt: dict[str, int] = {}
        cats = (self.store.icd10_category_descriptions(("S", "T"))
                if self.store is not None else [])
        for _c, desc in cats:
            # ICD headings lead with the entity noun ('Fracture of...',
            # 'Dislocation and sprain of...'); anatomy and positional words
            # ('...at lower leg level') trail. Only the head phrase counts.
            head = re.split(r"\b(?:of|at|involving)\b", desc.lower(), maxsplit=1)[0]
            for t in self._tokens(head):
                if len(t) >= 5 and t not in self._DESC_STOPWORDS:
                    cnt[t] = cnt.get(t, 0) + 1
        # an entity word recurs across many category headings (fracture: 30+,
        # dislocation: 15+); anatomy appears in a handful
        self._injury_lex = {t for t, n in cnt.items() if n >= 8}
        return self._injury_lex

    def _check_cpt_descriptor_evidence(self, cpt, note_full_text: str):
        """The billed CPT's own descriptor is defined by an injury entity
        the encounter never documents. Found live: 27766 (medial malleolus
        FRACTURE) billed for an intentional osteotomy approach, and 11012
        (debridement at an OPEN FRACTURE site) with no fracture anywhere on
        the claim or in the note. The entity vocabulary is mined from the
        ICD-10 S/T-chapter category headings; evidence is the note text
        plus the line's linked diagnoses' official descriptions."""
        if not note_full_text:
            return
        lex = self._injury_entity_lexicon()
        if not lex:
            return
        for entry in cpt:
            code = entry.get("code", "")
            info = self.db.validate_cpt(code) or {}
            desc = self._EG_PAREN_RE.sub(" ", (info.get("long_description")
                                               or info.get("description") or "").lower())
            # main clause only: text after ';' is a variant qualifier
            # ('...; superficial'), not the condition the code treats
            cond_toks = [t for t in self._tokens(desc.split(";")[0]) if t in lex]
            if not cond_toks:
                continue
            linked_txt = " ".join(
                (self.db.validate_icd10(dx) or {}).get("description", "")
                for dx in entry.get("linked_diagnoses") or [])
            # Word-boundary/stem matching only — 'microfracture' in the note
            # must NOT satisfy a descriptor's 'fracture' (that collision is
            # what hid the 27766 misuse).
            evidence, _ = self._note_evidence(note_full_text + " " + linked_txt)
            if any(self._desc_documented(t, evidence, "") for t in cond_toks):
                continue
            self._add(
                "WARNING", code, "descriptor_condition_undocumented",
                f"{code}'s own descriptor is defined by injury condition(s) the encounter "
                f"never documents: {', '.join(sorted(cond_toks)[:4])}. The procedure "
                f"performed may be a different one that this code does not describe (e.g. "
                f"an intentional osteotomy billed with a fracture-treatment code).",
                "Verify the documented procedure matches this code's descriptor; if the "
                "condition named by the descriptor was not present, select the code for "
                "the procedure actually performed (or the unlisted code).",
                denial_risk="HIGH",
            )
            entry["needs_review"] = True

    def _procedure_corpus_df(self) -> dict:
        """Token document frequency across the CPT+HCPCS descriptor corpus."""
        if getattr(self, "_proc_df", None) is None:
            df: dict[str, int] = {}
            for src in ("cpt", "hcpcs"):
                for info in (getattr(self.db, src, {}) or {}).values():
                    for t in self._tokens(info.get("long_description")
                                          or info.get("description") or ""):
                        df[t] = df.get(t, 0) + 1
            self._proc_df = df
        return self._proc_df

    # Contrasting-qualifier pairs (autograft/allograft class): two tokens
    # from the procedure-code corpus sharing a >=5-char suffix with short
    # differing prefixes are variant qualifiers of one attribute. Both must
    # be uncommon in the corpus (df <= 150 — autograft 81, allograft 78):
    # ubiquitous near-twins name distinct procedures, not variants.
    def _qualifier_axes(self) -> list:
        if getattr(self, "_qual_axes", None) is not None:
            return self._qual_axes
        df = self._procedure_corpus_df()
        vocab = {t for t, n in df.items()
                 if len(t) >= 8 and n <= 150 and t not in self._DESC_STOPWORDS}
        axes = []
        all_tokens = set(df)
        by_suffix: dict[str, list[str]] = {}
        for t in vocab:
            by_suffix.setdefault(t[-5:], []).append(t)
        for members in by_suffix.values():
            for a in members:
                for b in members:
                    if not (a < b and len(a) == len(b)
                            and a[: len(a) - 5] != b[: len(b) - 5]
                            and 3 <= len(a) - 5 <= 4):
                        continue
                    # The full shared suffix must itself be a standalone
                    # corpus word ('graft' for autograft/allograft) — a real
                    # attribute the prefixes qualify. Coincidental rhymes
                    # (selection/infection share 'ection', which names
                    # nothing) are excluded.
                    common = 0
                    while common < len(a) and a[len(a) - 1 - common] == b[len(b) - 1 - common]:
                        common += 1
                    shared = a[len(a) - common:]
                    if any(len(shared[i:]) >= 5 and shared[i:] in all_tokens
                           for i in range(len(shared) - 4)):
                        axes.append((a, b))
        self._qual_axes = axes
        return axes

    def _check_qualifier_contradiction(self, entry, low_note: str):
        """The note repeatedly documents one variant qualifier (allograft)
        while the billed descriptor specifies its counterpart (autograft).
        Requires >=2 note occurrences of the contradicting qualifier and 0
        of the billed one — a deliberate documentation choice, not a
        stray word. Found live: 28446 (osteochondral AUTOgraft, talus)
        billed for a documented ALLOgraft transplant — a payable-vs-
        unlisted distinction."""
        code = entry.get("code", "")
        info = self.db.validate_cpt(code) or self.db.validate_hcpcs(code) or {}
        desc_toks = self._tokens(self._EG_PAREN_RE.sub(
            " ", (info.get("long_description") or info.get("description") or "").lower()))
        for a, b in self._qualifier_axes():
            for own, other in ((a, b), (b, a)):
                if own in desc_toks and other not in desc_toks \
                        and low_note.count(other) >= 2 and low_note.count(own) == 0:
                    self._add(
                        "WARNING", code, "descriptor_qualifier_contradicted",
                        f"{code}'s descriptor specifies '{own}', but the note documents "
                        f"'{other}' ({low_note.count(other)}x) and never '{own}' — the "
                        f"documented variant appears to be the one this code does NOT describe.",
                        f"Confirm which variant was performed; if '{other}', select the code "
                        f"that describes it (an unlisted code if none exists).",
                        denial_risk="HIGH",
                    )
                    entry["needs_review"] = True
                    return

    def _check_unbilled_descriptor_match(self, cpt, hcpcs, note_full_text: str):
        """A dedicated code exists whose descriptor the note itself spells
        out, but a generic code was billed instead. Found live: trephination
        of a subungual hematoma billed as 10140 (generic I&D of hematoma)
        when CPT carries 11740 ('Evacuation of subungual hematoma') — and
        10140 pays more, so the generic choice reads as upcoding. Detection
        is corpus-driven: an unbilled code is surfaced when at least half of
        its significant descriptor tokens appear in the note INCLUDING >=2
        rare tokens (document frequency <=25 across the CPT+HCPCS corpus) —
        rare tokens are what make a descriptor dedicated rather than
        generic."""
        if not note_full_text or not cpt:
            return
        # Full note for qualifier contradictions (allograft vs autograft is
        # about what WAS done); performed-only context for the unbilled
        # match — a procedure 'discussed as surgical option' or 'deferred
        # pending clearance' was not rendered and must not be surfaced as
        # an unbilled service.
        _, low_note = self._note_evidence(note_full_text)
        note_words, low_perf = self._note_evidence(
            self._performed_context(note_full_text))
        billed = {e.get("code", "") for e in cpt} | {e.get("code", "") for e in hcpcs}
        # qualifier-contradiction sub-check rides along on the same pass
        for entry in cpt + hcpcs:
            self._check_qualifier_contradiction(entry, low_note)
        low_note = low_perf
        df = self._procedure_corpus_df()
        billed_list = [c for c in billed if c]
        candidates = []  # (rare-hit count, documented fraction, code, info, rare_hits)
        for code, info in (getattr(self.db, "cpt", {}) or {}).items():
            if code in billed:
                continue
            # NCCI-bundled INTO a billed code → the billed comprehensive
            # service already includes it (11730 nail avulsion inside 11750
            # matrixectomy); the reverse direction — a billed code bundled
            # into the candidate — is precisely the finding (64776 is the
            # component of 28080) and stays.
            if any((e := self.db.check_ncci(b, code)) and e.get("code2") == code
                   for b in billed_list):
                continue
            # Category II (xxxxF) codes are $0.00 performance-tracking codes,
            # never a 'dedicated alternative' to a payable service — and the
            # pipeline auto-suppresses them elsewhere for the same reason.
            if code.endswith("F") and code[:-1].isdigit():
                continue
            desc = self._EG_PAREN_RE.sub(" ", (info.get("long_description") or "").lower())
            sig = [t for t in self._tokens(desc)
                   if len(t) >= self._VARIANT_MIN_TOKEN_LEN and t not in self._DESC_STOPWORDS]
            if not 2 <= len(sig) <= 8:
                continue  # very long descriptors match everything; 1-token ones anything
            hits = [t for t in sig if self._desc_documented(t, note_words, low_note)]
            rare = [t for t in sig if df.get(t, 99) <= 25]
            rare_hits = [t for t in hits if df.get(t, 99) <= 25]
            # EVERY distinctive token must be documented — one undocumented
            # rare token (21510's 'thorax', 11719's 'nondystrophic') means
            # the descriptor is about something else. Two qualifying shapes:
            # (a) two rare tokens + half the descriptor documented, or (b) a
            # near-unique token (df<=5, 'subungual' has df 1) with the ENTIRE
            # descriptor documented — the note spells the code out verbatim.
            if len(rare_hits) != len(rare):
                continue
            unique_full = (any(df.get(t, 99) <= 5 for t in hits)
                           and len(hits) == len(sig))
            if (len(rare_hits) >= 2 and len(hits) * 2 >= len(sig)) or unique_full:
                candidates.append((len(rare_hits), len(hits) / len(sig), code, info,
                                   rare_hits or hits))
                continue
            # Third shape, NCCI-anchored: the candidate is the COLUMN-1
            # comprehensive service of a billed line per the PTP table. The
            # structural relationship replaces the 2-rare-token requirement
            # (descriptors like 'Excision of nail and nail matrix ... for
            # permanent removal' carry only one rare token), but the note
            # must still document a strict majority of the descriptor with
            # at least one rare token among the hits.
            if (rare_hits and len(hits) * 2 > len(sig) and len(hits) >= 3
                    and any((e := self.db.check_ncci(code, b))
                            and e.get("code1") == code and e.get("code2") == b
                            for b in billed_list)):
                candidates.append((len(rare_hits), len(hits) / len(sig), code, info,
                                   rare_hits or hits))
        candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
        # The MUTATION arms below (upgrade / completion) evaluate EVERY
        # qualified candidate — their structural guards (single NCCI
        # component line, kindred/anchor conditions) are the selectivity.
        # The advisory WARNING keeps the top-2 cap: a long op note grazes
        # many descriptors, and only the strongest grazes are worth a
        # coder's attention. Measured live (note 001): the true
        # comprehensive code ranked below two incidental matches, so a cap
        # applied before the mutation arms silently skipped the upgrade.
        mutated: set = set()
        for _n, _frac, code, info, rare_hits in candidates:
            # NCCI comprehensive upgrade: the documented-but-unbilled code
            # is the COLUMN-1 comprehensive service of exactly one billed
            # line (the billed code is its component per the PTP table).
            # The note documents the comprehensive service's own descriptor
            # yet the claim bills only the piece NCCI bundles into it —
            # under-coding that also flaps run to run (measured live, note
            # 001: a documented phenol matrixectomy billed as plain nail
            # avulsion in 1 of 3 runs; note 006: a documented partial
            # avulsion billed as hematoma evacuation alone). The upgrade is
            # the PTP table's own semantics: performing the comprehensive
            # service IS performing the component, so the line converges on
            # the code whose descriptor the note spells out. Modifiers,
            # units, and pointers ride along unchanged; review-flagged.
            def _kindred(billed_code: str) -> bool:
                """The billed line and the candidate must describe the SAME
                service family — sharing at least one significant descriptor
                token ('hematoma' ties 10140 I&D-of-hematoma to 11740
                evacuation-of-subungual-hematoma). An NCCI edit alone is
                same-session bundling POLICY, not identity of work: measured
                live (note 004), a performed wound debridement (97597) was
                rewritten into a nail avulsion the note explicitly deferred,
                and (note 009) into a nail-unit biopsy, purely because both
                are PTP column-2 of those codes. Zero descriptor overlap
                means the billed work is a different procedure that happens
                to bundle — never upgrade it."""
                def _sig_toks(d: str) -> set:
                    low_d = self._EG_PAREN_RE.sub(" ", (d or "").lower())
                    return {t for t in self._tokens(low_d)
                            if len(t) >= self._VARIANT_MIN_TOKEN_LEN
                            and t not in self._DESC_STOPWORDS}
                btoks = _sig_toks((self.db.validate_cpt(billed_code) or {})
                                  .get("long_description", ""))
                return bool(btoks & _sig_toks(info.get("long_description", "")))

            comp_all = [
                e for e in cpt
                if e.get("code")
                # E/M lines are PTP column-2 of most procedures (the -25
                # relationship) — that edit expresses same-day bundling
                # policy, not that the procedure 'includes' the E/M work,
                # so an E/M line must never be rewritten into a surgery.
                and not _is_em(e.get("code", ""))
                and e.get("code") not in self._non_billable_codes_to_suppress
                and (edit := self.db.check_ncci(code, e.get("code")))
                and edit.get("code1") == code
                and edit.get("code2") == e.get("code")
            ]
            comp_kin = [e for e in comp_all if _kindred(e.get("code", ""))]
            already_billed = any(e.get("code") == code for e in cpt)
            if len(comp_kin) == 1 and not already_billed:
                entry = comp_kin[0]
                old = entry.get("code", "")
                entry["code"] = code
                entry["description"] = info.get("long_description", "")
                entry["needs_review"] = True
                self._add(
                    "WARNING", old, "component_upgraded_to_comprehensive",
                    f"AUTO-CORRECTED: {old} upgraded to {code} "
                    f"('{(info.get('long_description') or '')[:70]}') — the note "
                    f"documents the comprehensive service's own descriptor "
                    f"({', '.join(rare_hits[:4])}), and the NCCI PTP table lists "
                    f"{old} as a component bundled into {code}; billing only the "
                    f"component under-codes the documented work.",
                    f"Verify the documented service against CPT {code}; the "
                    f"comprehensive code includes the billed component.",
                    denial_risk="MEDIUM",
                )
                logger.info(
                    f"  Comprehensive upgrade: {old} → {code} (NCCI component, "
                    f"descriptor documented: {', '.join(rare_hits[:4])})")
                mutated.add(code)
                continue
            # Comprehensive COMPLETION: same PTP column-1 relationship, but
            # the billed component is a DIFFERENT act (zero descriptor
            # overlap — kindred fails), and the candidate's own act is
            # affirmatively documented as performed. Measured live (note
            # 006): 'trephination x2 ... partial nail avulsion of separated
            # portion' billed as hematoma evacuation alone in 1 of 3 runs —
            # both acts were performed and documented, and the runs that
            # billed both converged (the PTP edit then bundles the
            # evacuation into the avulsion). Rewriting the evacuation line
            # would erase real work, so the candidate is ADDED instead,
            # inheriting the component line's own site modifiers and
            # pointers; the same-site bundling check downstream then applies
            # the PTP table's verdict to the pair exactly as it does when
            # the model bills both. The documentation bar is strict: >=2 of
            # the candidate's RARE descriptor tokens inside ONE
            # performed-context sentence — the note names the dedicated act
            # itself ('partial NAIL AVULSION'), not scattered token grazes.
            # Deferral override: the performed-context filter drops futurity
            # sentences, but a deferred act can ALSO be named in the plan
            # without a futurity cue ('Bilateral partial nail avulsion with
            # chemical matrixectomy' — measured live, note 004, where the
            # body says 'Formal avulsion deferred pending vascular
            # clearance'). When any of the candidate's own dedicated terms
            # sits in a deferral/futurity sentence, the act was not rendered
            # today and must never be added.
            def _act_deferred(toks) -> bool:
                for sent in re.split(r"(?<=[.;:\n])", note_full_text or ""):
                    if not self._FUTURITY_RE.search(sent):
                        continue
                    sw = self._tokens(sent)
                    sw |= {self._stem(w) for w in set(sw)}
                    if any(t in sw or self._stem(t) in sw for t in toks):
                        return True
                return False

            if (len(comp_all) == 1 and not comp_kin and not already_billed
                    and int(comp_all[0].get("units") or 1) == 1
                    and self._one_sentence_rare_anchor(rare_hits, low_note)
                    and not _act_deferred(rare_hits)):
                comp = comp_all[0]
                anat = (self.store.anatomic_modifiers()
                        if self.store is not None else set())
                new_entry = {
                    "code": code,
                    "description": info.get("long_description", ""),
                    "modifiers": [m for m in comp.get("modifiers") or []
                                  if str(m).strip().upper() in anat],
                    "units": 1,
                    "linked_diagnoses": list(comp.get("linked_diagnoses") or []),
                    "needs_review": True,
                    "source_section": "validator:comprehensive_completion",
                }
                cpt.append(new_entry)
                self._add(
                    "WARNING", code, "comprehensive_completion_added",
                    f"AUTO-ADDED: {code} "
                    f"('{(info.get('long_description') or '')[:70]}') — a single "
                    f"performed-procedure sentence documents this service's own "
                    f"dedicated terms ({', '.join(rare_hits[:4])}), and the NCCI "
                    f"PTP table lists billed {comp.get('code')} as its component; "
                    f"omitting the comprehensive service under-codes the "
                    f"documented work. The PTP edit between the two lines is "
                    f"then applied as usual.",
                    f"Verify the documented service against CPT {code}; if only "
                    f"the component was performed, remove the added line.",
                    denial_risk="MEDIUM",
                )
                logger.info(
                    f"  Comprehensive completion: added {code} alongside "
                    f"component {comp.get('code')} (dedicated act documented: "
                    f"{', '.join(rare_hits[:4])})")
                mutated.add(code)
                continue
        for _n, _frac, code, info, rare_hits in candidates[:2]:
            if code in mutated:
                continue
            self._add(
                "WARNING", code, "dedicated_code_unbilled",
                f"The note documents the descriptor of unbilled CPT {code} "
                f"('{(info.get('long_description') or '')[:70]}') — distinctive terms "
                f"documented: {', '.join(rare_hits[:4])}. If a generic code was billed "
                f"for this service instead, the dedicated code is the accurate (and "
                f"audit-safe) choice.",
                f"Compare the billed line(s) against CPT {code}; bill the dedicated "
                f"code if it describes the documented service.",
                denial_risk="MEDIUM",
            )

    # Language lexicon (no codes): how operative notes phrase a PARTIAL bone
    # excision — cutting THROUGH the bone (transection) is partial removal by
    # definition (a complete excision disarticulates at the joint), plus the
    # technique words the AMA partial-excision descriptors themselves name
    # in their parentheticals (craterization/saucerization/sequestrectomy/
    # diaphysectomy).
    _EXCISION_PARTIAL_RE = re.compile(
        r"\b(?:partial(?:ly)?|transect\w*|craterizat\w*|saucerizat\w*|"
        r"sequestrectom\w*|diaphysectom\w*|hemisect\w*)\b")
    _EXCISION_COMPLETE_RE = re.compile(
        r"\b(?:entire|complete(?:ly)?|total(?:ly)?|disarticulat\w*|in\s+toto)\b")

    def _check_excision_extent_axis(self, cpt, note_full_text: str):
        """Extent-axis arbitration between an excision code and its 'Partial
        excision' NCCI partner. The AMA writes the axis into the descriptors
        themselves: 28124 'PARTIAL excision ... bone; phalanx of toe' vs
        28150 'Phalangectomy, toe' (the complete procedure). When the billed
        code is the COMPLETE member, the note's own operative wording
        documents a partial removal (bone transected, craterized,
        saucerized... — the techniques the partial descriptor's parenthetical
        names), and nothing documents a complete removal, the partial member
        is the accurate code.

        Measured live (note 010): 'Distal phalangectomy — bone transected at
        metaphysis proximal to infected cortex' flapped 28150 (1 of 3 runs)
        vs 28124. The pathology-axis layer could not settle it because the
        note's word 'phalangectomy' counts as documentation of the complete
        code's distinguishing term — but transection THROUGH the bone is
        partial removal by definition, and the extent axis outranks the
        procedure's colloquial name. Everything is data: the 'partial
        excision' descriptor grammar, the NCCI PTP table, the numeric family
        prefix, shared bone-structure vocabulary, and the note's own
        sentences. No code names appear here."""
        if not note_full_text or not cpt or not getattr(self.db, "cpt", None):
            return
        _, low_note = self._note_evidence(
            self._performed_context(note_full_text))

        def _struct_stems(desc: str) -> set:
            # 6-char stems of long tokens: 'phalanx'/'phalangectomy' meet at
            # 'phalan' — the bone structure survives the -ectomy compound
            return {t[:6] for t in self._tokens((desc or "").lower())
                    if len(t) >= 6 and t not in self._DESC_STOPWORDS}

        billed_codes = {e.get("code", "") for e in cpt}
        for entry in cpt:
            code = entry.get("code", "")
            if (not code or _is_em(code)
                    or code in self._non_billable_codes_to_suppress):
                continue
            own = self.db.validate_cpt(code)
            if not own:
                continue
            own_desc = (own.get("long_description") or "").lower()
            if "partial" in own_desc:
                continue  # already the partial member — nothing to arbitrate
            own_stems = _struct_stems(own_desc)
            if not own_stems:
                continue

            # the note's own operative wording must decide the axis: a
            # partial marker in a sentence naming the shared structure,
            # with no complete-removal wording anywhere. Newlines are NOT
            # sentence boundaries here — PDF extraction hard-wraps lines
            # mid-sentence ('Distal phalangectomy left hallux\n— bone
            # transected...'), which severed the structure from its own
            # transection clause when \n split the sentence.
            partial_sent = ""
            for sent in re.split(r"[.;]", low_note.replace("\n", " ")):
                if not self._EXCISION_PARTIAL_RE.search(sent):
                    continue
                sent_stems = {t[:6] for t in self._tokens(sent) if len(t) >= 6}
                if own_stems & sent_stems:
                    partial_sent = sent.strip()
                    break
            if not partial_sent or self._EXCISION_COMPLETE_RE.search(low_note):
                continue

            candidates = []
            for cand_code, cand_info in self.db.cpt.items():
                if (cand_code == code or cand_code in billed_codes
                        or cand_code[:3] != code[:3]):
                    continue
                cand_desc = (cand_info.get("long_description") or "").lower()
                if not cand_desc.startswith("partial excision"):
                    continue
                if not (own_stems & _struct_stems(cand_desc)):
                    continue  # different structure, not this code's axis
                edit = (self.db.check_ncci(code, cand_code)
                        or self.db.check_ncci(cand_code, code))
                if not edit:
                    continue  # not the mutually-exclusive same-structure pair
                candidates.append((cand_code, cand_info))
            if len(candidates) != 1:
                continue  # ambiguous family — a human question, not a swap
            cand_code, cand_info = candidates[0]

            entry["code"] = cand_code
            entry["description"] = (cand_info.get("long_description")
                                    or cand_info.get("short_description")
                                    or entry.get("description", ""))
            entry["needs_review"] = True
            entry["review_reason"] = (
                f"Swapped from {code} on the partial-excision axis — confirm "
                f"the operative wording ('{partial_sent[:80]}') describes a "
                f"partial removal")
            billed_codes.discard(code)
            billed_codes.add(cand_code)
            self._add(
                "WARNING", code, "excision_extent_axis",
                f"AUTO-CORRECTED: {code} → {cand_code}. The note's own operative "
                f"wording documents a PARTIAL excision ('{partial_sent[:100]}') — "
                f"transection/craterization through the bone removes part of it by "
                f"definition — and nothing documents a complete removal; "
                f"{cand_code}'s own AMA descriptor ('{(cand_info.get('long_description') or '')[:70]}') "
                f"is the partial member of this NCCI-paired family.",
                f"Bill {code} only when the entire structure is excised "
                f"(disarticulated); document and bill the partial-excision code "
                f"for transection-level removal",
                denial_risk="MEDIUM",
            )

    def _check_cpt_family_pathology_axis(self, cpt, note_full_text: str):
        """Pathology-axis arbitration between CPT family members. The AMA
        writes a code's qualifying pathologies into its own descriptor as an
        '(eg, ...)' parenthetical — 28124 'Partial excision ... bone (eg,
        osteomyelitis or bossing); phalanx of toe'. When a billed code and a
        same-family alternative are NCCI PTP partners (mutually exclusive
        same-session work on the same structure) and the note affirmatively
        documents the ALTERNATIVE's eg-pathology while never documenting a
        single one of the billed code's own distinguishing terms, the billed
        code describes a pathology the encounter didn't have.

        Measured live (note 008): a documented subungual EXOSTOSIS resection
        flapped between 28108 ('bone cyst or benign tumor' — no cyst and no
        tumor anywhere in the note) and 28124 (whose eg-parenthetical names
        bossing, the descriptor register for exostosis; CPT's own reference
        note under the cyst/tumor code directs partial excision of bossing/
        exostosis of a phalanx to this code). The prompt's family tie-break
        states the same rule; this layer enforces it deterministically.

        Everything is data: the eg-parenthetical grammar, the NCCI PTP
        table, the numeric family prefix, descriptor token overlap, and the
        note's own words. No code names appear here."""
        if not note_full_text or not cpt or not getattr(self.db, "cpt", None):
            return
        note_words, low_note = self._note_evidence(
            self._performed_context(note_full_text))

        def _canon(t: str) -> str:
            for group in self._TERM_EQUIV_GROUPS:
                if t in group:
                    return min(group)
            return t

        def _sig(desc: str) -> set:
            d = self._EG_PAREN_RE.sub(" ", (desc or "").lower())
            return {_canon(t) for t in self._tokens(d)
                    if len(t) >= 4 and t not in self._DESC_STOPWORDS}

        _EG_TERMS_RE = re.compile(r"\((?:eg|e\.g\.)[,.]?\s*([^)]*)\)")

        def _eg_terms(desc: str) -> list:
            terms = []
            for m in _EG_TERMS_RE.finditer((desc or "").lower()):
                terms.extend(t for t in self._tokens(m.group(1))
                             if len(t) >= 4 and t not in self._DESC_STOPWORDS)
            return terms

        billed_codes = {e.get("code", "") for e in cpt}
        for entry in cpt:
            code = entry.get("code", "")
            if (not code or _is_em(code)
                    or code in self._non_billable_codes_to_suppress):
                continue
            own = self.db.validate_cpt(code)
            if not own:
                continue
            own_desc = own.get("long_description", "")
            # the billed code's own eg-pathology documented → it is the
            # right family member; nothing to arbitrate
            if any(self._desc_documented(t, note_words, low_note)
                   for t in _eg_terms(own_desc)):
                continue
            own_sig = _sig(own_desc)
            qualifying = []  # (eg_hits, overlap, cand_code, cand_info)
            for cand_code, cand_info in self.db.cpt.items():
                if (cand_code == code or cand_code in billed_codes
                        or cand_code[:3] != code[:3]):
                    continue
                cand_desc = cand_info.get("long_description", "")
                eg_hits = [t for t in _eg_terms(cand_desc)
                           if self._desc_documented(t, note_words, low_note)]
                if not eg_hits:
                    continue
                # mutually exclusive same-session pair per the PTP table —
                # the structural statement that these are alternative
                # spellings of work on the same structure
                edit = (self.db.check_ncci(code, cand_code)
                        or self.db.check_ncci(cand_code, code))
                if not edit:
                    continue
                cand_sig = _sig(cand_desc)
                own_only = own_sig - cand_sig
                # the billed code's every distinguishing term is absent
                # from the note — the note never says what would make the
                # billed code the right one
                if any(self._desc_documented(t, note_words, low_note)
                       for t in own_only):
                    continue
                overlap = len(own_sig & cand_sig)
                if overlap < 2:
                    continue  # different work, not a family axis
                qualifying.append((len(eg_hits), overlap, cand_code, cand_info))
            if not qualifying:
                continue
            qualifying.sort(key=lambda q: (-q[0], -q[1], q[2]))
            # strict argmax: an ambiguous family (two members equally
            # documented and equally close) is a human call, not a swap
            if (len(qualifying) > 1
                    and qualifying[0][:2] == qualifying[1][:2]):
                continue
            _hits, _ov, new_code, new_info = qualifying[0]
            old = code
            entry["code"] = new_code
            entry["description"] = new_info.get("long_description", "")
            entry["needs_review"] = True
            self._add(
                "WARNING", old, "cpt_pathology_axis",
                f"AUTO-CORRECTED: {old} swapped to {new_code} "
                f"('{(new_info.get('long_description') or '')[:70]}') — the note "
                f"documents the pathology {new_code}'s own descriptor names in "
                f"its '(eg, ...)' parenthetical, while none of {old}'s "
                f"distinguishing descriptor terms appear anywhere in the note; "
                f"the NCCI PTP table lists the two as mutually exclusive "
                f"same-session services.",
                f"Verify the documented pathology against CPT {new_code}'s "
                f"descriptor; the billed code described a pathology the note "
                f"never documents.",
                denial_risk="MEDIUM",
            )
            billed_codes.discard(old)
            billed_codes.add(new_code)
            logger.info(
                f"  Pathology-axis swap: {old} → {new_code} (eg-pathology "
                f"documented; billed code's distinguishing terms absent)")

    def _check_icd_sibling_descriptor(self, icd, note_full_text: str):
        """Mirror of the HCPCS descriptor-variant check for ICD-10: the
        billed code was selected among category siblings that differ on a
        clinical attribute, and the note documents the SIBLING's attribute,
        not the billed one. Found live: documented 'paronychia' (a
        cellulitis, L03.03-) billed as L03.04- (acute LYMPHANGITIS — never
        documented); documented left-hallux subungual hematoma billed as
        S90.122A (lesser toe, without nail damage). Evidence includes the
        Tabular List's own inclusion terms (B35.1's 'onychomycosis' supports
        B35.1 even when 'tinea unguium' never appears in the note),
        restricted to levels BELOW the siblings' common ancestor — a shared
        parent's terms describe both variants and prove neither. Data-driven
        from the code set's own descriptions.

        When the evidence is DECISIVE — the sibling's distinguishing
        attribute is affirmatively documented AND the billed code's own
        attribute never appears — the entry is auto-swapped to the sibling
        (flagged for review). Measured live: runs flapped E11.40/E11.42 on a
        note documenting 'polyneuropathy'; the note is identical every run,
        so the arbitration must be too. Weaker evidence still only flags.

        Evidence asymmetry (measured live, routine_00001): the SIBLING's
        attribute — the evidence that DRIVES a swap — is read from the
        clinical note view (incidental-context sentences removed): 'a
        well-padded THIGH tourniquet' must never document a thigh
        condition. The billed code's OWN attribute — the evidence that
        PROTECTS the billed code — still reads the full note: blocking a
        swap on broad evidence is the safe direction."""
        if not note_full_text or not icd:
            return
        self._icd_condition_lexicon()
        note_words, low_note = self._note_evidence(note_full_text)
        clin_words, clin_low = self._clinical_evidence(note_full_text)

        def _incl_terms(c: str, min_level: int) -> set:
            """Tabular inclusion terms below a level — a shared parent's
            term ('Paronychia' under L03.0 'Cellulitis and acute
            lymphangitis...') describes both siblings and proves neither, so
            sibling comparisons restrict to levels below the common
            ancestor."""
            if self.store is None:
                return set()
            return {t.lower() for t in
                    self.store.icd10_inclusion_terms(c, min_level=min_level)}

        def _index_terms(c: str) -> set:
            """Alphabetic Index phrases along the code's whole stem chain.
            The Index is the authoritative synonym→code routing (paronychia →
            Cellulitis, digit → L03.03-); phrases shared by both siblings'
            chains cancel in the caller's set difference, so only the
            phrases the Index routed to ONE branch discriminate."""
            if self.store is None:
                return set()
            norm_c = c.replace(".", "")
            terms = set()
            for ln in range(3, len(norm_c) + 1):
                rows = self.store.conn.execute(
                    "SELECT term FROM icd10_index_term WHERE code=?",
                    (norm_c[:ln],)).fetchall()
                terms.update(r[0].lower() for r in rows)
            return terms

        def _any_term_documented(terms, words=None, low=None) -> bool:
            words = note_words if words is None else words
            low = low_note if low is None else low
            for term in terms:
                toks = [t for t in self._tokens(term) if t not in self._DESC_STOPWORDS]
                if not toks:
                    continue
                # full-phrase match, OR any of the term's signature tokens —
                # 'Morton's metatarsalgia' is documented by a note that says
                # 'Morton's neuroma' ('morton' is unique to this entity even
                # though 'metatarsalgia' never appears).
                if all(self._desc_documented(t, words, low) for t in toks):
                    return True
                if any(self._icd_token_df.get(t, 0) <= 25
                       and self._desc_documented(t, words, low)
                       for t in toks):
                    return True
            return False

        def _terms_documented(c: str, min_level: int) -> bool:
            return _any_term_documented(_incl_terms(c, min_level) | _index_terms(c))

        sites = self._site_lexicon()

        def _side_words(desc: str) -> frozenset:
            # from the RAW text: 'left'/'right' are descriptor stopwords
            return frozenset(re.findall(r"\b(left|right)\b", desc.lower()))

        def _axis_toks(desc: str) -> set:
            """Descriptor tokens eligible as axis attributes. The df<=500
            ubiquity cut removes severity/temporal qualifiers — but site
            words are EXEMPT from it: a site is part of the axis and its
            documentation must be proven like any other attribute. Measured
            live (note 008): 'ankle' (df 1490) and 'shoulder' (df 1637)
            both vanished under the cut, so M25.771 'Osteophyte, right
            ankle' vs M25.511 'Pain in right SHOULDER' looked like a pure
            osteophyte-vs-pain axis and the swap moved the diagnosis to a
            different joint no podiatry note documents. With sites in the
            sets, an undocumented site on the sibling ('shoulder') blocks
            the swap via the site guard below."""
            return {t for t in self._tokens(desc)
                    if len(t) >= 3 and t not in self._DESC_STOPWORDS
                    and (self._icd_token_df.get(t, 0) <= 500 or t in sites)}

        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            own = self.db.validate_icd10(code)
            if not own:
                continue
            # Ubiquitous tokens (df > 400 across the ICD corpus: 'acute',
            # 'chronic', 'unspecified'...) are excluded when BUILDING the
            # sets — they are severity/temporal qualifiers, not the
            # this-or-that clinical attribute a sibling axis turns on.
            own_toks = _axis_toks(own.get("description", ""))
            own_side = _side_words(own.get("description", ""))
            norm = code.replace(".", "")
            fam = [(c, d) for c, d in self.db.icd10_siblings(code[:3])
                   if c != norm and len(c) == len(norm) and c[-1] == norm[-1]]
            best = None   # (coverage, sib_code, sib_desc, sib_only, own_only)
            undoc_axis = None
            for sib_code, sib_desc in fam:
                # SIDE is an invariant of a sibling swap, never the axis:
                # left/right are stopwords, so opposite-side siblings have
                # identical token sets and can only be 'chosen' by
                # iteration order. Measured live (note 006): S90.122A
                # ('LEFT lesser toe') swapped to S90.111A ('RIGHT great
                # toe') on a left-hallux note because the right-side
                # sibling sorted first. Laterality corrections belong to
                # _check_icd_cpt_laterality_agreement, which validates the
                # side against the claim's own procedure modifiers.
                if _side_words(sib_desc) != own_side:
                    continue
                sib_toks = _axis_toks(sib_desc)
                own_only = own_toks - sib_toks
                sib_only = sib_toks - own_toks
                # tight axes only: 1-2 differing attribute tokens per side
                if not (1 <= len(sib_only) <= 2 and len(own_only) <= 2):
                    continue
                # A swap must never MOVE the diagnosis: when EITHER side of
                # the axis contains a site word, the pair differs on
                # anatomy, and relocating a condition is only defensible
                # when the target site is affirmatively documented AND the
                # billed site is not (the note names the other site, the
                # code doesn't). Any other shape — target site undocumented,
                # or both sites documented (ambiguous) — is skipped: no
                # synonym/Index rescue may relocate a diagnosis. Measured
                # live (note 008): M25.771 'Osteophyte, right ankle' was
                # swapped to M25.511 'Pain in right SHOULDER' on a hallux
                # note; with shoulder blocked, the next candidate was
                # M25.551 'Pain in right HIP' — the guard must hold for
                # every site pair, not just the first one caught.
                own_site_toks = {t for t in own_only if t in sites}
                sib_site_toks = {t for t in sib_only if t in sites}
                if own_site_toks or sib_site_toks:
                    # target site must be documented CLINICALLY (relocation-
                    # driving); the billed site's protection reads the full
                    # note — asymmetry is deliberate (see docstring)
                    sib_site_doc = all(
                        self._desc_documented(t, clin_words, clin_low)
                        for t in sib_site_toks)
                    own_site_doc = any(
                        self._desc_documented(t, note_words, low_note)
                        for t in own_site_toks)
                    if not sib_site_doc or (own_site_toks and own_site_doc):
                        continue
                # Synonym evidence must DISCRIMINATE between the two
                # siblings. Inclusion terms: only levels below the common
                # ancestor. Index phrases: the full chains, minus whatever
                # both share ('hematoma → contusion' lands on the whole
                # family and proves neither; 'paronychia → Cellulitis,
                # digit' lands on one branch and decides it).
                common = 0
                while (common < min(len(norm), len(sib_code))
                       and norm[common] == sib_code[common]):
                    common += 1
                own_ix, sib_ix = _index_terms(norm), _index_terms(sib_code)
                own_syn = _incl_terms(norm, common + 1) | (own_ix - sib_ix)
                sib_syn = _incl_terms(sib_code, common + 1) | (sib_ix - own_ix)
                # Condition-entity tokens (rare, lexicon-grade: 'lymphangitis')
                # outrank qualifier tokens ('acute'): if the axis contains an
                # entity, ITS documentation decides support — otherwise a
                # ubiquitous qualifier the note happens to contain ('acute'
                # paronychia) masks an undocumented billed entity.
                own_entity = [t for t in own_only
                              if t in self._cond_lex
                              and self._icd_token_df.get(t, 0) <= 150]
                decisive = own_entity or own_only
                own_documented = (
                    any(self._desc_documented(t, note_words, low_note) for t in decisive)
                    or _any_term_documented(own_syn))
                # swap-driving evidence: clinical view only (incidental
                # tourniquet/positioning/prep anatomy never drives a swap)
                sib_documented = (
                    all(self._desc_documented(t, clin_words, clin_low) for t in sib_only)
                    or _any_term_documented(sib_syn, clin_words, clin_low))
                if sib_documented and not own_documented:
                    coverage = sum(1 for t in sib_toks
                                   if self._desc_documented(t, clin_words, clin_low))
                    if best is None or coverage > best[0]:
                        best = (coverage, sib_code, sib_desc, sib_only, own_only)
                elif own_entity and not own_documented and undoc_axis is None:
                    # Billed code's OWN distinguishing attribute is a rare
                    # clinical entity the note never mentions (lymphangitis
                    # billed on a documented paronychia) — reviewable even
                    # when no sibling is affirmatively documented either.
                    # Anatomy/site words (plantar, cartilage, joints) are
                    # excluded: a site is implied by the documented procedure
                    # and exam, not something a note re-names explicitly.
                    # ANY-ancestor inclusion terms rescue the code here: for
                    # this weaker claim, a documented parent-level synonym
                    # ('Paronychia' under L03.0) is evidence the billed child
                    # is plausibly right, so no flag.
                    # Compound refinements of a documented entity are also
                    # rescued: 'polyneuropathy' when the note documents
                    # 'neuropathy' — the billed token linguistically CONTAINS
                    # the documented condition, so this weaker any-sibling
                    # claim has no footing (the sibling branch above still
                    # applies when a sibling is affirmatively documented).
                    def _refines_documented(tok: str) -> bool:
                        return any(len(w) >= 6 and tok != w and tok.endswith(w)
                                   for w in note_words)
                    entity_only = [t for t in own_entity
                                   if t not in self._anatomy_lexicon()
                                   and not _refines_documented(t)]
                    if (entity_only and len(entity_only) == len(own_entity)
                            and not _terms_documented(norm, 3)):
                        undoc_axis = (sorted(own_entity), sib_code, sib_desc)

            def _dotted(c):
                return c if "." in c else (c[:3] + "." + c[3:] if len(c) > 3 else c)

            if best is not None:
                _, sib_code, sib_desc, sib_only, own_only = best
                target = _dotted(sib_code)
                already_billed = any(
                    (e.get("code") or "").strip().upper().replace(".", "") == sib_code
                    for e in icd if e is not entry)
                if not already_billed:
                    entry["code"] = target
                    entry["description"] = sib_desc
                    self._add(
                        "WARNING", code, "sibling_matches_note_better",
                        f"AUTO-CORRECTED: {code} ('{own.get('description', '')[:60]}') "
                        f"replaced with {target} ('{sib_desc[:60]}') — the note documents "
                        f"'{', '.join(sorted(sib_only))}', the sibling's distinguishing "
                        f"attribute, and never documents this code's own attribute(s) "
                        f"({', '.join(sorted(own_only)) or 'n/a'}).",
                        f"Verify against the note: {target} is the documented variant.",
                        denial_risk="HIGH",
                    )
                else:
                    self._add(
                        "WARNING", code, "sibling_matches_note_better",
                        f"{code} ('{own.get('description', '')[:60]}') was billed, but the "
                        f"note documents '{', '.join(sorted(sib_only))}' — the distinguishing "
                        f"attribute of sibling {target} ('{sib_desc[:60]}'), already on the "
                        f"claim — and never documents this code's own attribute(s) "
                        f"({', '.join(sorted(own_only)) or 'n/a'}).",
                        f"Verify against the note: {target} appears to be the "
                        f"documented variant; this line may be a duplicate.",
                        denial_risk="HIGH",
                    )
                entry["needs_review"] = True
            elif undoc_axis is not None:
                axis_toks, sib_code, sib_desc = undoc_axis
                self._add(
                    "WARNING", code, "billed_attribute_undocumented",
                    f"{code}'s distinguishing attribute ('{', '.join(axis_toks)}') is a "
                    f"specific clinical entity the note never documents, and sibling codes "
                    f"exist on this axis (e.g. {_dotted(sib_code)} '{sib_desc[:60]}'). The "
                    f"billed variant may not be the documented condition.",
                    f"Verify the documented condition against the {code[:3]} family and "
                    f"select the variant the note actually supports.",
                    denial_risk="MEDIUM",
                )
                entry["needs_review"] = True

    def _check_with_without_axis(self, icd, note_full_text: str):
        """Arbitrate ICD sibling pairs whose descriptors differ ONLY by
        'with X' vs 'without X' ('Contusion of left lesser toe(s) WITH
        damage to nail' S90.212A vs '...WITHOUT damage to nail' S90.122A).
        The token-diff sibling check above cannot see this axis — 'with'
        and 'without' are stopwords, so the two descriptors' token sets are
        identical — yet the note decides it deterministically:
          * X affirmatively documented (negation-scrubbed) → the 'with X'
            variant is the documented one;
          * X never documented, or documented only inside a negated span
            ('no nail damage') → the 'without X' variant.
        Determinism layer: measured live (note 006), runs flapped
        S90.122A/S90.212A on an identical note. Swaps are flagged for
        review; a swap is skipped when the counterpart is already billed."""
        if not note_full_text or not icd:
            return
        note_words, low_note = self._note_evidence(note_full_text)
        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            own = self.db.validate_icd10(code)
            if not own:
                continue
            own_desc = own.get("description", "")
            m = re.search(r"\b(with|without)\s+([^,;(]+)", own_desc, re.IGNORECASE)
            if not m:
                continue
            keyword = m.group(1).lower()
            phrase = m.group(2).strip()
            toks = [t for t in self._tokens(phrase)
                    if len(t) >= 3 and t not in self._DESC_STOPWORDS]
            if not toks:
                continue
            counterpart_kw = "without" if keyword == "with" else "with"
            want = re.sub(rf"\b{keyword}\b", counterpart_kw, own_desc,
                          count=1, flags=re.IGNORECASE)
            want_norm = " ".join(want.lower().split())
            norm = code.replace(".", "")
            partners = [
                (c, d) for c, d in self.db.icd10_siblings(code[:3])
                if c != norm and len(c) == len(norm)
                and (len(norm) < 7 or c[-1] == norm[-1])
                and " ".join(d.lower().split()) == want_norm
            ]
            if len(partners) != 1:
                continue
            sib_code, sib_desc = partners[0]
            documented = all(self._desc_documented(t, note_words, low_note)
                             for t in toks)
            # billed variant already matches the note's evidence → nothing to do
            if (keyword == "with") == documented:
                continue
            target = sib_code if "." in sib_code else (
                sib_code[:3] + "." + sib_code[3:] if len(sib_code) > 3 else sib_code)
            if any((e.get("code") or "").strip().upper().replace(".", "") == sib_code
                   for e in icd if e is not entry):
                continue
            entry["code"] = target
            entry["description"] = sib_desc
            entry["needs_review"] = True
            evidence_why = (
                f"the note documents '{phrase}' outside any negated span"
                if documented else
                f"the note never documents '{phrase}' (or only inside a negated span)")
            self._add(
                "WARNING", code, "with_without_axis_corrected",
                f"AUTO-CORRECTED: {code} ('{own_desc[:60]}') replaced with {target} "
                f"('{sib_desc[:60]}') — the two codes differ only on the "
                f"'{counterpart_kw} {phrase}' axis, and {evidence_why}.",
                f"Verify against the note: the '{counterpart_kw} {phrase}' variant is "
                f"the documented one.",
                denial_risk="HIGH",
            )

    def _index_rows(self) -> list:
        """(term, code, code-with-trailing-zeros-stripped) for the entire
        ICD-10-CM Alphabetic Index table, cached. The strip-0 form makes
        stem routing comparable: the Index files 'osteomyelitis chronic'
        under M8660, which covers the whole M866x family (M86.672 etc.)."""
        if getattr(self, "_index_rows_cache", None) is None:
            rows = (self.store.conn.execute(
                "SELECT term, code FROM icd10_index_term").fetchall()
                if self.store is not None else [])
            out = []
            for term, code in rows:
                stripped = code.rstrip("0")
                if len(stripped) < 3:
                    stripped = code[:3]
                out.append((term, code, stripped))
            self._index_rows_cache = out
        return self._index_rows_cache

    def _index_term_map(self) -> dict:
        """{index term -> set of codes} for exact-term lookups."""
        if getattr(self, "_index_term_map_cache", None) is None:
            m: dict = {}
            for term, code, _s in self._index_rows():
                m.setdefault(term, set()).add(code)
            self._index_term_map_cache = m
        return self._index_term_map_cache

    def _check_onset_qualifier_axis(self, icd, note_full_text: str):
        """ICD-10-CM's onset/temporal qualifier axis, arbitrated by the
        note + the Alphabetic Index's own routing. The classification
        splits sibling families on qualifiers the provider must actually
        document ('Other ACUTE osteomyelitis' M86.1- / 'Other CHRONIC
        osteomyelitis' M86.6- / bare 'osteomyelitis' → M86.9 per the
        Index; acquired 'onychauxis' → L60.2 / 'onychauxis congenital' →
        Q84.5). Guideline I.A/I.B: an axis the record never states must
        not be assigned.

        Deterministic arbitration, all routing from the Index table:
          * billed qualifier documented (negation-scrubbed, sentence-
            adjacent to the condition's own terms) → billed code stands;
          * a COUNTERPART qualifier documented instead → swap within the
            axis (same trailing site characters transplanted when the
            counterpart family has the billable form);
          * NEITHER documented → swap to the Index's bare-term default
            (the code the Index gives the unqualified condition), flagged
            for review.

        Safety gates: a qualifier that appears in the 3-character CATEGORY
        description is definitional for the whole family (every L97 code
        is a 'chronic ulcer'; every N18 code 'chronic kidney disease') —
        not a sibling axis — and is skipped (congenital/hereditary
        exempt: chapter Q is definitionally congenital and acquired-vs-
        congenital is exactly the axis to arbitrate). A code any of whose
        Index routes is fully documented is supported and never touched.

        Determinism layer: measured live (note 010), independent runs
        split M86.172 (acute) / M86.672 (chronic) on a note that documents
        only 'osteomyelitis'; note 007 added congenital Q84.5 for
        documented acquired nail hypertrophy in 1 of 3 runs."""
        if self.store is None or not note_full_text or not icd:
            return
        note_words, low_note = self._note_evidence(note_full_text)
        sent_words = []
        for s in re.split(r"[.;\n]", low_note):
            w = self._tokens(s)
            sent_words.append(w | {self._stem(t) for t in w})

        def _adjacent(qual: str, others) -> bool:
            qs = self._stem(qual)
            for words in sent_words:
                if qual not in words and qs not in words:
                    continue
                if any(o in words or self._stem(o) in words for o in others):
                    return True
            return False

        def _content(term: str) -> list:
            return [t for t in self._tokens(term)
                    if len(t) >= 3 and t not in self._DESC_STOPWORDS
                    and t not in ONSET_QUALIFIERS]

        def _majority_documented(toks) -> bool:
            if not toks:
                return False
            n = sum(1 for t in toks
                    if self._desc_documented(t, note_words, low_note))
            return n * 2 > len(toks)

        def _dotted(c: str) -> str:
            return c if "." in c else (c[:3] + "." + c[3:] if len(c) > 3 else c)

        tmap = self._index_term_map()
        to_remove = []
        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            norm = code.replace(".", "")
            if not norm or norm[0] in ("S", "T"):
                continue  # injury chapter — 7th-char machinery owns onset
            own = self.db.validate_icd10(code) or {}
            own_desc = (own.get("description")
                        or self.store.icd10_tabular_description(norm))
            if not own_desc:
                continue
            cat_toks = self._tokens(self.store.icd10_tabular_description(norm[:3]))
            quals = (self._tokens(own_desc) | cat_toks) & ONSET_QUALIFIERS
            if not quals:
                continue
            routes = [(t, c, s) for t, c, s in self._index_rows()
                      if norm.startswith(s)]
            if not routes:
                continue
            # any fully documented route (qualifiers sentence-adjacent to the
            # route's own condition terms) supports the billed code as-is
            supported = False
            for term, _c, _s in routes:
                content = _content(term)
                if not content or not _majority_documented(content):
                    continue
                r_quals = self._tokens(term) & ONSET_QUALIFIERS
                if all(_adjacent(q, content) for q in r_quals):
                    supported = True
                    break
            if supported:
                continue

            fired = False
            for q in sorted(quals):
                if fired:
                    break
                if q not in ("congenital", "hereditary") and q in cat_toks:
                    continue  # definitional for the whole category, not an axis
                q_re = re.compile(rf"\b{q}\b")
                for term, rcode, rstrip in routes:
                    if not q_re.search(term):
                        continue
                    base = re.sub(r"\s+", " ", q_re.sub(" ", term)).strip(" ,")
                    content = _content(base)
                    if not content or not _majority_documented(content):
                        continue  # the underlying condition itself isn't documented
                    if _adjacent(q, content):
                        continue  # billed qualifier is documented after all
                    target = why = None
                    # counterpart qualifier documented → transplant within axis
                    for q2 in sorted(ONSET_QUALIFIERS - {q}):
                        codes2 = tmap.get(f"{base} {q2}") or tmap.get(f"{q2} {base}")
                        if not codes2 or not _adjacent(q2, content):
                            continue
                        stem2 = sorted(codes2)[0].rstrip("0") or sorted(codes2)[0][:3]
                        cand = _dotted(stem2 + norm[len(rstrip):])
                        if self.db.validate_icd10(cand):
                            target = cand
                            why = (f"the note documents '{q2}' with the "
                                   f"condition, never '{q}'")
                            break
                    if target is None:
                        # neither qualifier documented → Index bare-term default
                        for bc in sorted(tmap.get(base, set())):
                            if norm.startswith(bc.rstrip("0") or bc[:3]):
                                continue  # routes back to the billed family
                            billable = self.store.icd10_billable_under(bc)
                            if billable and billable[0][0] == bc:
                                target = _dotted(bc)
                                why = (f"neither '{q}' nor a counterpart "
                                       f"qualifier is documented with the "
                                       f"condition — the Index's bare "
                                       f"'{base}' default applies")
                                break
                    if target is None:
                        continue
                    tdesc = (self.db.validate_icd10(target) or {}).get(
                        "description", "")
                    already = any(
                        (e.get("code") or "").strip().upper() == target
                        for e in icd if e is not entry)
                    if already:
                        to_remove.append(entry)
                        action = (f"line removed — {target} (already on the "
                                  f"claim) is the documented spelling")
                    else:
                        entry["code"] = target
                        entry["description"] = tdesc or entry.get("description", "")
                        entry["needs_review"] = True
                        entry["review_reason"] = (
                            f"Onset-qualifier axis: confirm {target} — '{q}' "
                            f"was never documented for the condition")
                        action = f"swapped {code} → {target}"
                    self._add(
                        "WARNING", code, "onset_qualifier_undocumented",
                        f"AUTO-CORRECTED: {code} ('{own_desc[:60]}') carries the "
                        f"'{q}' qualifier, but {why}; {action}. Routing per the "
                        f"ICD-10-CM Alphabetic Index and guideline I.A (assign "
                        f"only what the record states).",
                        f"Query the provider if '{q}' applies; the documented "
                        f"default is {target} ('{tdesc[:60]}')",
                        denial_risk="MEDIUM",
                    )
                    fired = True
                    break
        if to_remove:
            icd[:] = [e for e in icd if e not in to_remove]

    def _check_injury_seventh_char(self, icd):
        """ICD-10-CM guideline I.B.10: when a claim carries a late-effect
        condition (a non-injury code whose own description says
        'post-traumatic', e.g. M19.171 post-traumatic osteoarthritis), the
        causal S/T-chapter injury code is reported with 7th character S
        (sequela), not D (active routine healing). Found live: an old
        Lisfranc injury driving a post-traumatic arthrodesis carried
        S93.321D. Corrects D→S only ('A' alongside a post-traumatic code
        can be a genuinely new same-site injury — not touched), and only
        when the S-variant exists in the code set."""
        has_late_effect = any(
            not (e.get("code") or "").strip().upper().startswith(("S", "T"))
            and "post-traumatic" in (self.db.validate_icd10(e.get("code", "")) or {})
            .get("description", "").lower()
            for e in icd)
        if not has_late_effect:
            return
        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            if not code.startswith(("S", "T")) or len(code.replace(".", "")) != 7:
                continue
            if code[-1] != "D":
                continue
            s_variant = code[:-1] + "S"
            ref = self.db.validate_icd10(s_variant)
            if not ref:
                continue
            entry["code"] = s_variant
            entry["description"] = ref.get("description", "")
            self._add(
                "INFO", s_variant, "seventh_char_sequela",
                f"AUTO-CORRECTED: {code} → {s_variant}. The claim carries a late-effect "
                f"condition (post-traumatic, per the coexisting code's own ICD-10 "
                f"description), so the causal injury is reported with 7th character 'S' "
                f"(sequela), not 'D' (guideline I.B.10).",
                "Verify the injury is historical (not an active healing encounter)",
                denial_risk="LOW",
            )

    def _check_orphan_dx(self, icd, cpt, hcpcs):
        """Flag ICD codes (billable) not linked to any procedure — WARNING level.

        On a CMS-1500 claim every diagnosis is available to every procedure on the claim.
        The "orphan" concept only matters when an ICD code has ZERO CPT/HCPCS codes to
        hang off — i.e. the whole encounter has no procedures.  When any billable CPT or
        HCPCS code is present, secondary diagnoses (comorbidities, BMI, etc.) are
        implicitly supported by the encounter and do NOT need individual linkage.
        """
        if cpt or hcpcs:
            # Any procedure/supply present → all diagnoses are valid secondary DX
            return

        # Only reaches here if the LLM produced ICD codes with NO procedures at all
        for entry in icd:
            self._add(
                "WARNING", entry.get("code", ""), "ORPHAN_DIAGNOSIS",
                f"{entry.get('code')} coded but no CPT/HCPCS procedure found — entire encounter may lack medical necessity",
                "Ensure at least one billable procedure or E/M code is present",
                denial_risk="HIGH",
            )

    def _check_modifiers(self, cpt):
        if self.store is None:
            return
        for entry in cpt:
            for mod in entry.get("modifiers", []):
                if not self.store.modifier_valid(mod):
                    self._add(
                        "WARNING", entry.get("code", ""), "modifier_validity",
                        f"Modifier '{mod}' is not a recognized modifier for CPT {entry.get('code')}",
                        "Verify modifier is appropriate for this service",
                        denial_risk="MEDIUM",
                    )

    def _check_bilateral_modifier52(self, cpt):
        """A code whose own AMA description says "bilateral" (e.g. 93923) performed
        on only one side needs modifier 52 (reduced services) — the inverse of
        modifier 50 (a normally-unilateral code performed on both sides)."""
        for entry in cpt:
            code = entry.get("code", "")
            laterality = str(entry.get("laterality", "")).strip().lower()
            if laterality not in ("right", "left"):
                continue  # bilateral-documented, or unknown — not this scenario
            cpt_info = self.db.validate_cpt(code) if self.db else None
            if not cpt_info:
                continue
            description = (cpt_info.get("long_description") or cpt_info.get("short_description") or "").lower()
            if "bilateral" not in description:
                continue
            # "unilateral or bilateral" descriptors (the AMA's own phrasing,
            # e.g. many radiology codes) mean EITHER extent is full service —
            # one-sided performance is not "reduced services" and adding 52
            # would under-bill. Only bilateral-ONLY descriptors qualify.
            if "unilateral" in description:
                continue
            mods = entry.setdefault("modifiers", [])
            if "52" in mods:
                continue
            mods.append("52")
            self._add(
                "INFO", code, "modifier_52_added",
                f"AUTO-CORRECTED: Added modifier -52 to {code} — code is defined as bilateral "
                f"by its own AMA description but documentation shows only {laterality}-side testing.",
                "Modifier -52 auto-added (reduced services)",
                denial_risk="MEDIUM",
            )

    def _check_em_patient_status(self, cpt, note_category: str):
        """Deterministic new-vs-established E/M cross-check.

        Two already-real data points are compared, no medical rules invented:
        - the vision extractor's note_category (new_patient_visit /
          established_patient_visit), which reads the note's own header/HPI
          ("NEW PATIENT", "returns to clinic", etc.), and
        - the billed E/M code's OWN AMA descriptor text, which states
          "new patient" or "established patient" verbatim.
        A mismatch (e.g. 99213 "established patient" on a new_patient_visit
        note) is a guaranteed denial or downcoded payment — CMS matches
        against the 3-year established-patient rule. Flag for review rather
        than auto-swap: picking the level-equivalent code in the other
        family is a coding decision the LLM passes should make with the
        documentation in front of them, not a string transform.
        """
        expected = None
        if "new_patient" in (note_category or ""):
            expected = "new patient"
        elif "established_patient" in (note_category or ""):
            expected = "established patient"
        if expected is None:
            return
        wrong = "established patient" if expected == "new patient" else "new patient"
        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            info = self.db.validate_cpt(code) or {}
            desc = f"{info.get('short_description', '')} {info.get('long_description', '')}".lower()
            if wrong in desc and expected not in desc:
                self._add(
                    "ERROR", code, "em_patient_status",
                    f"E/M {code} is a '{wrong}' code per its own AMA descriptor, but the note "
                    f"is categorized as a {expected} visit ({note_category}). New-vs-established "
                    f"mismatches deny or downcode under CMS's 3-year rule.",
                    f"Replace {code} with the documentation-supported E/M level from the "
                    f"'{expected}' family (verify the 3-year rule against patient history)",
                    denial_risk="HIGH",
                )

    def _mdm_claimed_level(self, mdm: dict) -> str:
        """Normalize mdm_details.mdm_level to one of the four bare AMA levels
        and write the normalized value back.

        The structured-output schema now enum-forces the bare level, but
        results generated before that fix (and any non-schema path) can carry
        the whole 2-of-3 derivation sentence as the level — observed live on
        note 009: 'high (problems) / moderate (data, risk) → overall MDM
        moderate by 2-of-3 rule'. Every floor/ceiling/consistency layer gates
        on claimed == recomputed, so an unnormalized sentence silently
        disabled ALL of them at once. Extraction is deterministic: an exact
        level string wins; else the level named by an 'overall …' conclusion;
        else a single unambiguous level word; else "" (the layers skip, and
        em_level_consistency's ERROR path owns the contradiction)."""
        raw = str(mdm.get("mdm_level", "")).strip().lower()
        if not raw:
            return ""
        level = ""
        if raw in _MDM_LEVEL_NAMES:
            level = raw
        else:
            m = re.search(
                r"overall\s+(?:mdm\s+|level\s+)?(?:is\s+|=\s*)?"
                r"(straightforward|low|moderate|high)\b"
                r"|\boverall\b[^.;]*?\b(straightforward|low|moderate|high)\b",
                raw)
            if m:
                level = m.group(1) or m.group(2)
            else:
                found = {w for w in _MDM_LEVEL_NAMES
                         if re.search(rf"\b{w}\b", raw)}
                if len(found) == 1:
                    level = found.pop()
        if level:
            mdm["mdm_level"] = level
        return level

    def _check_em_mdm_problems_floor(self, cpt, icd):
        """Deterministic floor for the MDM problems axis, from the 2021 AMA
        MDM table's own moderate row: '2 or more stable chronic illnesses'
        (and '1 or more chronic illnesses with exacerbation' — so 2+ chronic
        diagnoses on the claim reach moderate on EITHER branch, stable or
        not). Chronicity is a classification-level fact, not judgment: the
        AHRQ HCUP CCIR file flags every ICD-10-CM code chronic/not-chronic,
        ingested into icd10_chronic — never a keyword guess.

        Determinism layer: measured across independent runs of the same
        notes, E/M presence flips (99213 one run, 99214 the next) were the
        single largest disagreement class (15 of 93 billing flips on the
        first 27-note new-layer batch), and the risk floor alone never fired
        because many flip notes carry no same-claim minor surgery. The
        diagnoses ARE on the claim in every run — a coded diagnosis is by
        definition addressed/managed at the encounter (ICD-10-CM reporting
        guideline IV.J) — so the problems axis has a run-invariant floor.

        Floor only — never lowers. Same internal-consistency guard as the
        risk floor: only applied when the coder's stated mdm_level agrees
        with its own axis scores; contradictions belong to
        _check_em_level_consistency's ERROR path."""
        if self.store is None:
            return
        chronic = sorted({
            e.get("code") for e in icd
            if self.store.icd10_is_chronic(e.get("code", "")) is True
        })
        if len(chronic) < 2:
            return

        level_names = ("straightforward", "low", "moderate", "high")
        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            mdm = entry.get("mdm_details") or {}
            claimed = self._mdm_claimed_level(mdm)
            if not claimed:
                continue
            try:
                scores = {k: int(mdm.get(k)) for k in
                          ("problems_score", "data_score", "risk_score")}
            except (TypeError, ValueError):
                continue
            if not all(1 <= v <= len(level_names) for v in scores.values()):
                continue
            if scores["problems_score"] >= 3:
                continue
            if claimed != level_names[sorted(scores.values())[1] - 1]:
                continue  # level/axes already contradict — em_level_consistency flags it
            mdm["problems_score"] = 3
            new_level = level_names[
                sorted([3, scores["data_score"], scores["risk_score"]])[1] - 1
            ]
            level_changed = new_level != claimed
            if level_changed:
                mdm["mdm_level"] = new_level
            self._add(
                "INFO", code, "em_mdm_problems_floor",
                f"AUTO-CORRECTED: MDM problems axis raised from "
                f"{scores['problems_score']} to 3 (moderate) for E/M {code} — the "
                f"claim itself carries {len(chronic)} chronic illnesses per the "
                f"AHRQ CCIR classification ({', '.join(chronic[:4])}), meeting the "
                f"2021 AMA MDM table's moderate problems row ('2 or more stable "
                f"chronic illnesses', or '1+ chronic with exacerbation' if not stable)."
                + (f" 2-of-3 MDM recomputes to '{new_level}'." if level_changed else ""),
                "E/M problems axis floored per the AMA MDM table's chronic-illnesses row",
                denial_risk="LOW",
            )

    def _check_em_mdm_risk_floor(self, cpt, icd, dos, patient_dob):
        """Deterministic floor for the MDM risk axis, from the 2021 AMA MDM
        table's own moderate-risk row: 'minor surgery with identified patient
        or procedure risk factors'. Both halves of that condition are plain
        claim facts, not judgment: 'minor surgery' is a same-claim CPT with
        CMS global period 000/010 (store data), and the named patient risk
        factors are read from the claim's OWN diagnosis descriptions
        (EM_RISK_FACTOR_TERMS: diabetes mellitus / obesity / anticoagulant
        use) plus patient age >= 65 from the note's DOB + DOS.

        Determinism layer: measured across independent runs of the same
        notes, the risk axis was the E/M flap point — one run scored 'minor
        in-office procedure on an 83-year-old' as moderate risk (correctly
        citing the AMA age factor), the next scored the identical facts low,
        flipping 99214/99213. The facts don't change between runs, so the
        floor shouldn't either.

        Floor only — never lowers a score. Applied only when the coder's
        stated mdm_level agrees with its own axis scores (internally
        consistent): the bump then flows into the recomputed 2-of-3 median
        and mdm_level, and _check_em_level_consistency (which runs next)
        performs the actual code swap through the same descriptor-driven
        sibling machinery. When the coder's level and axes already
        contradict each other, that check's ERROR path owns the problem —
        bumping an axis inside an inconsistent structure settles nothing."""
        if self.store is None:
            return
        has_minor_surgery = any(
            not _is_em(c.get("code", ""))
            and self.store.global_period(c.get("code", "")) in MINOR_PROCEDURE_GLOBAL_DAYS
            for c in cpt
        )
        if not has_minor_surgery:
            return

        factors = []
        for e in icd:
            desc = ((self.db.validate_icd10(e.get("code", "")) or {}).get("description") or "").lower()
            hit = next((t for t in EM_RISK_FACTOR_TERMS if t in desc), None)
            if hit:
                factors.append(f"{e.get('code')} ('{hit}')")
        from app.compliance.engine import _parse_date
        dob = _parse_date(patient_dob or "")
        if dob and dos:
            age = (dos - dob).days / 365.25
            if age >= 65:
                factors.append(f"patient age {int(age)} (>= 65)")
        if not factors:
            return

        level_names = ("straightforward", "low", "moderate", "high")
        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            mdm = entry.get("mdm_details") or {}
            claimed = self._mdm_claimed_level(mdm)
            if not claimed:
                continue
            try:
                scores = {k: int(mdm.get(k)) for k in
                          ("problems_score", "data_score", "risk_score")}
            except (TypeError, ValueError):
                continue
            if not all(1 <= v <= len(level_names) for v in scores.values()):
                continue
            if scores["risk_score"] >= 3:
                continue
            if claimed != level_names[sorted(scores.values())[1] - 1]:
                continue  # level/axes already contradict — em_level_consistency flags it
            mdm["risk_score"] = 3
            new_level = level_names[
                sorted([scores["problems_score"], scores["data_score"], 3])[1] - 1
            ]
            level_changed = new_level != claimed
            if level_changed:
                mdm["mdm_level"] = new_level
            self._add(
                "INFO", code, "em_mdm_risk_floor",
                f"AUTO-CORRECTED: MDM risk axis raised from {scores['risk_score']} to 3 "
                f"(moderate) for E/M {code} — the claim itself documents minor surgery "
                f"(global 000/010 procedure) with AMA-identified patient risk factors "
                f"({'; '.join(factors)}), the 2021 AMA MDM table's own moderate-risk row."
                + (f" 2-of-3 MDM recomputes to '{new_level}'." if level_changed else ""),
                "E/M risk axis floored per the AMA MDM table's minor-surgery-with-risk-factors row",
                denial_risk="LOW",
            )

    def _check_em_mdm_risk_high_floor(self, cpt, note_full_text: str):
        """Deterministic floor for the MDM risk axis to HIGH (4), from the
        2021 AMA MDM table's own high-risk row: 'drug therapy requiring
        intensive monitoring for toxicity'. The note answers this
        deterministically when one negation-scrubbed, performed-context
        sentence documents all three elements of the canonical clinical
        shape: parenteral route + a toxicity-monitored agent class + the
        therapy being initiated/administered at this encounter ('IV
        vancomycin initiated empirically per ID recommendation').

        Determinism layer: measured live (note 009), the risk axis was the
        residual 99214/99215 flap point after the data floor — two runs
        scored the identical IV-vancomycin initiation as high risk (4),
        one as moderate (3). The facts don't change between runs, so the
        floor shouldn't either. Floor only — never lowers; same internal-
        consistency guard as the other floors, with the code swap owned by
        _check_em_level_consistency."""
        if not note_full_text:
            return
        _, low_note = self._note_evidence(note_full_text)
        perf = self._performed_context(low_note)
        hit_sentence = ""
        for sent in re.split(r"[.;\n]", perf):
            if (EM_HIGH_RISK_ROUTE_RE.search(sent)
                    and EM_HIGH_RISK_AGENT_RE.search(sent)
                    and EM_HIGH_RISK_INITIATION_RE.search(sent)):
                hit_sentence = sent.strip()
                break
        if not hit_sentence:
            return

        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            mdm = entry.get("mdm_details") or {}
            claimed = self._mdm_claimed_level(mdm)
            if not claimed:
                continue
            try:
                scores = {k: int(mdm.get(k)) for k in
                          ("problems_score", "data_score", "risk_score")}
            except (TypeError, ValueError):
                continue
            if not all(1 <= v <= len(_MDM_LEVEL_NAMES) for v in scores.values()):
                continue
            if scores["risk_score"] >= 4:
                continue
            if claimed != _MDM_LEVEL_NAMES[sorted(scores.values())[1] - 1]:
                continue  # level/axes contradict — em_level_consistency owns it
            mdm["risk_score"] = 4
            new_level = _MDM_LEVEL_NAMES[
                sorted([scores["problems_score"], scores["data_score"], 4])[1] - 1
            ]
            level_changed = new_level != claimed
            if level_changed:
                mdm["mdm_level"] = new_level
            self._add(
                "INFO", code, "em_mdm_risk_high_floor",
                f"AUTO-CORRECTED: MDM risk axis raised from {scores['risk_score']} "
                f"to 4 (high) for E/M {code} — the note itself documents initiation "
                f"of parenteral drug therapy requiring intensive monitoring for "
                f"toxicity ('{hit_sentence[:120]}'), the 2021 AMA MDM table's own "
                f"high-risk row."
                + (f" 2-of-3 MDM recomputes to '{new_level}'." if level_changed else ""),
                "E/M risk axis floored per the AMA MDM table's "
                "toxicity-monitored-drug-therapy row",
                denial_risk="LOW",
            )

    # AMA 2021 MDM data column, Extensive row — Category 3 evidence: a
    # documented discussion of management with an external physician/QHP.
    # Language cues, not codes: a consult was placed AND the note records
    # acting on (or receiving) the consultant's input.
    _EM_DATA_CONSULT_RE = re.compile(
        r"\bconsult(?:ed|ation|s)?\b", re.IGNORECASE)
    _EM_DATA_DISCUSSION_RE = re.compile(
        r"\bper\s+(?:\w+[\s.]+){0,3}?recommendation|\bdiscuss(?:ed|ion)\s+with\b",
        re.IGNORECASE)
    # Category 1 evidence: unique quantified test results ('ESR 78',
    # 'CRP 42', 'HbA1c 10.8', 'ABI 0.72') — a clinical acronym (>=2
    # capitals) followed by its value; a trailing dose unit means a
    # MEDICATION amount ('NPH 40 units'), not a test result.
    _EM_DATA_TEST_RE = re.compile(
        r"\b((?:[A-Z]{2,6}|[A-Z][a-z]?[A-Z][A-Za-z0-9]{0,4}))\s*[:=]?\s*"
        r"\d+(?:\.\d+)?(?!\s*(?:mg|mcg|ml|mL|units?|tabs?|%))\b")

    def _check_em_mdm_data_floor(self, cpt, icd, note_full_text: str):
        """Deterministic floor for the MDM data axis, from the 2021 AMA MDM
        table's own Extensive row: 'must meet the requirements of at least
        2 of the 3 categories'. Two of those categories are plain note
        facts:
          Category 1 — review of unique test results: >= 3 distinct
            quantified results documented (each acronym+value pair,
            excluding medication doses);
          Category 3 — discussion of management with an external
            physician/QHP: a consult documented together with the
            consultant's input being received or acted on ('IV vancomycin
            initiated per ID recommendation').

        Determinism layer: measured live (note 009), the data axis was the
        99214/99215 flap point — two runs scored the identical note's five
        quantified results (ESR/CRP/WBC/HbA1c/ABI) plus ID and vascular
        consults as extensive (4), one as moderate (3). The facts don't
        change between runs, so the floor shouldn't either. Floor only —
        never lowers; same internal-consistency gating as the risk floor,
        with the code swap owned by _check_em_level_consistency."""
        if not note_full_text:
            return
        perf = self._performed_context(note_full_text)
        acronyms = {m.group(1) for m in
                    self._EM_DATA_TEST_RE.finditer(note_full_text)}
        cat1 = len(acronyms) >= 3
        cat3 = bool(self._EM_DATA_CONSULT_RE.search(perf)
                    and self._EM_DATA_DISCUSSION_RE.search(note_full_text))
        if not (cat1 and cat3):
            return
        level_names = ("straightforward", "low", "moderate", "high")
        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            mdm = entry.get("mdm_details") or {}
            claimed = self._mdm_claimed_level(mdm)
            if not claimed:
                continue
            try:
                scores = {k: int(mdm.get(k)) for k in
                          ("problems_score", "data_score", "risk_score")}
            except (TypeError, ValueError):
                continue
            if not all(1 <= v <= len(level_names) for v in scores.values()):
                continue
            if scores["data_score"] >= 4:
                continue
            if claimed != level_names[sorted(scores.values())[1] - 1]:
                continue  # level/axes contradict — em_level_consistency owns it
            mdm["data_score"] = 4
            new_level = level_names[
                sorted([scores["problems_score"], 4, scores["risk_score"]])[1] - 1
            ]
            level_changed = new_level != claimed
            if level_changed:
                mdm["mdm_level"] = new_level
            self._add(
                "INFO", code, "em_mdm_data_floor",
                f"AUTO-CORRECTED: MDM data axis raised from {scores['data_score']} "
                f"to 4 (extensive) for E/M {code} — the note itself meets 2 of the "
                f"AMA Extensive-data categories: {len(acronyms)} unique quantified "
                f"test results ({', '.join(sorted(acronyms)[:6])}) and a documented "
                f"discussion of management with an external physician (consult with "
                f"recommendation acted on)."
                + (f" 2-of-3 MDM recomputes to '{new_level}'." if level_changed else ""),
                "E/M data axis floored per the AMA MDM table's Extensive-data row",
                denial_risk="LOW",
            )

    def _check_em_mdm_problems_ceiling(self, cpt, icd, note_full_text: str):
        """Mirror of the problems FLOOR above, for the opposite failure: a
        problems axis scored HIGH (4) on a note that documents none of the
        2021 AMA MDM table's own high-row prerequisites — 'chronic illness
        with severe exacerbation/progression' or an 'illness or injury that
        poses a threat to life or bodily function'. The high row is an
        evidence question the note answers deterministically: either a
        threat/exacerbation term appears tied to a billed diagnosis, or it
        doesn't.

        'Tied to a billed diagnosis' means one of:
          * a billed diagnosis's OWN descriptor contains a threat term
            (I70.262 '...with gangrene' — the classification itself says
            the condition threatens the limb), or
          * a negation-scrubbed note sentence contains a high-row term AND
            a content token of some billed diagnosis's descriptor ('severe
            exacerbation of COPD'). A stray 'severely involuted nail' in a
            sentence naming no billed condition supports nothing.

        Determinism layer: measured live (note 004), one run of three
        scored problems=4/risk=4 → 99215 while the others scored 3/3 →
        99214 on an identical note with no severe-exacerbation or threat
        documentation. Ceiling only — never raises. Same internal-
        consistency guard as the floors; the recomputed level flows into
        _check_em_level_consistency's descriptor-driven sibling swap."""
        if not note_full_text:
            return
        level_names = ("straightforward", "low", "moderate", "high")
        dx_toks: set = set()
        dx_descriptor_threat = False
        for e in icd:
            desc = ((self.db.validate_icd10(e.get("code", "")) or {})
                    .get("description", "")).lower()
            if not desc:
                continue
            if EM_HIGH_PROBLEM_RE.search(desc):
                dx_descriptor_threat = True
            dx_toks |= {t for t in self._tokens(desc)
                        if len(t) >= 4 and t not in self._DESC_STOPWORDS}
        high_supported = dx_descriptor_threat
        if not high_supported:
            _, low_note = self._note_evidence(note_full_text)
            for sent in re.split(r"[.;\n]", low_note):
                if not EM_HIGH_PROBLEM_RE.search(sent):
                    continue
                words = self._tokens(sent)
                words |= {self._stem(w) for w in set(words)}
                if any(t in words or self._stem(t) in words for t in dx_toks):
                    high_supported = True
                    break
        if high_supported:
            return

        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            mdm = entry.get("mdm_details") or {}
            claimed = self._mdm_claimed_level(mdm)
            if not claimed:
                continue
            try:
                scores = {k: int(mdm.get(k)) for k in
                          ("problems_score", "data_score", "risk_score")}
            except (TypeError, ValueError):
                continue
            if not all(1 <= v <= len(level_names) for v in scores.values()):
                continue
            if scores["problems_score"] < 4:
                continue
            if claimed != level_names[sorted(scores.values())[1] - 1]:
                continue  # level/axes already contradict — em_level_consistency flags it
            mdm["problems_score"] = 3
            new_level = level_names[
                sorted([3, scores["data_score"], scores["risk_score"]])[1] - 1
            ]
            level_changed = new_level != claimed
            if level_changed:
                mdm["mdm_level"] = new_level
            self._add(
                "INFO", code, "em_mdm_problems_ceiling",
                f"AUTO-CORRECTED: MDM problems axis lowered from 4 to 3 (moderate) "
                f"for E/M {code} — the 2021 AMA MDM table's high problems row "
                f"requires a chronic illness with severe exacerbation/progression "
                f"or an illness posing a threat to life or bodily function, and "
                f"neither the note nor any billed diagnosis's own descriptor "
                f"documents one."
                + (f" 2-of-3 MDM recomputes to '{new_level}'." if level_changed else ""),
                "E/M problems axis capped per the AMA MDM table's high-row "
                "documentation requirements",
                denial_risk="LOW",
            )

    def _check_em_level_consistency(self, cpt):
        """Deterministic E/M level cross-check — the highest-volume upcoding/
        downcoding audit target, previously verified by nothing at all (only
        new-vs-established was checked).

        Data-driven the same way as _check_em_patient_status: each office
        E/M code's OWN AMA descriptor states its MDM level verbatim ("...and
        low level of medical decision making" for x3, "moderate" for x4,
        "high" for x5). The coder's Pass-2 output carries its claimed
        mdm_details.mdm_level with the 2-of-3 axis rationale.

        When the coder's own MDM evidence is internally CONSISTENT — the
        three axis scores' 2-of-3 result (AMA's own algorithm: the level at
        least two axes reach, i.e. the median) equals the stated mdm_level —
        but the BILLED code's descriptor states a different level, the code
        is the odd one out and is deterministically swapped to the same-
        family sibling whose descriptor matches (found by descriptor
        structure via store.em_level_sibling, never a code table). Measured
        live: this exact pattern was the single largest run-to-run
        instability — 99214/99215 flapping across independent runs of the
        same note while the axis scores stayed put.

        When the claimed level and the axes DISAGREE with each other, there
        is no deterministic winner — whether the scoring or the code choice
        is wrong requires reading the documentation — so it stays a flag
        for review, not an auto-swap."""
        level_names = ("straightforward", "low", "moderate", "high")
        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            mdm = entry.get("mdm_details") or {}
            claimed = self._mdm_claimed_level(mdm)
            if not claimed:
                continue
            info = self.db.validate_cpt(code) or {}
            desc = (info.get("long_description") or info.get("short_description") or "").lower()
            m = re.search(
                r"(straightforward|low|moderate|high)\s+(?:level\s+of\s+)?medical decision making",
                desc,
            )
            if not m:
                continue  # descriptor carries no MDM level (e.g. 99211) — nothing to compare
            descriptor_level = m.group(1)
            if claimed == descriptor_level:
                continue

            axes_level = None
            try:
                axes = sorted(int(mdm.get(k)) for k in
                              ("problems_score", "data_score", "risk_score"))
                if all(1 <= a <= len(level_names) for a in axes):
                    axes_level = level_names[axes[1] - 1]  # 2-of-3 = median axis
            except (TypeError, ValueError):
                pass

            sibling = (self.store.em_level_sibling(code, claimed)
                       if self.store is not None and axes_level == claimed else None)
            if sibling and sibling != code:
                sib_info = self.db.validate_cpt(sibling) or {}
                entry["code"] = sibling
                if sib_info.get("long_description") or sib_info.get("short_description"):
                    entry["description"] = (sib_info.get("long_description")
                                            or sib_info.get("short_description"))
                self._add(
                    "INFO", sibling, "em_level_corrected",
                    f"AUTO-CORRECTED: E/M {code} → {sibling}. The coder's own MDM axes "
                    f"(2-of-3 = '{claimed}') and stated level agree with each other but "
                    f"contradicted {code}'s descriptor ('{descriptor_level}'); {sibling} is "
                    f"the same-family code whose own AMA descriptor states '{claimed}' MDM.",
                    f"E/M level aligned to the documented MDM ('{claimed}')",
                    denial_risk="LOW",
                )
                continue

            self._add(
                "ERROR", code, "em_level_mismatch",
                f"E/M {code} requires '{descriptor_level}' MDM per its own AMA descriptor, "
                f"but the coding rationale scored this visit's MDM as '{claimed}'. "
                f"One of the two is wrong — level upcoding/downcoding is a primary audit target.",
                f"Re-derive MDM from the documented problems/data/risk axes; either correct the "
                f"E/M code to the '{claimed}'-level code in the same family or fix the MDM scoring",
                denial_risk="HIGH",
            )

    def _check_em_modifier25(self, cpt):
        """Two-directional modifier -25/-57 check — auto-corrects missing or
        contradictory -25/-57 based on the same-day procedure's real global
        period (000/010 -> -25; 090 -> -57; diagnostic tests, global=XXX,
        trigger neither).

        Both directions are handled deterministically here — not just -25,
        with -57 left to a separate text-keyword check or to whatever the
        LLM's own reasoning happened to state — because a same-day major
        (090) procedure previously still got -25 auto-added by the old
        undifferentiated PROCEDURE_GLOBAL_DAYS check, and nothing removed it
        once -57 was added later (by the reasoning-consistency check, which
        runs after this one) — leaving both modifiers present at once,
        directly contradicting the note's own "-57, not -25" reasoning.
        """
        em_entry = None
        has_minor_procedure = False
        has_major_procedure = False
        # A same-day code with global=XXX (diagnostic test/study, e.g.
        # 93923 vascular physiologic study) is neither a minor nor major
        # procedure — but that does NOT mean modifier -25 is automatically
        # wrong the way "no procedure at all" does. Whether -25 applies
        # for a same-day diagnostic test turns on the E/M's own MDM
        # (significant work documented beyond the test itself), a
        # judgment call this deterministic check has no real data to
        # settle either way — see code_assigner.py's "Modifier -25 —
        # SAME-DAY DIAGNOSTIC TEST" prompt rule. Found live: -25 was
        # legitimately claimed and added (reasoning-consistency) for a
        # 99215 alongside a 93923 diagnostic test with genuine documented
        # MDM complexity, then unconditionally stripped back out here
        # because has_minor_procedure was False — indistinguishable, by
        # the two flags that existed before this one, from "no other code
        # on the claim at all," which IS a real reason to remove -25.
        has_diagnostic_test = False

        for c in cpt:
            code = c.get("code", "")
            # billing_status 'B' (bundled/no-charge, e.g. 99024 post-op follow-up)
            # is neither the E/M line for modifier purposes nor a procedure —
            # queried from real data (global_periods.json), not a hardcoded
            # singleton for the one code this used to be checked against.
            if self.store is not None and self.store.billing_status(code) == "B":
                continue
            if _is_em(code):
                em_entry = c
            elif self.store is not None:
                gp = self.store.global_period(code)
                if gp in MINOR_PROCEDURE_GLOBAL_DAYS:
                    has_minor_procedure = True
                elif gp in MAJOR_PROCEDURE_GLOBAL_DAYS:
                    has_major_procedure = True
                elif gp == "XXX":
                    has_diagnostic_test = True

        if em_entry is None:
            return

        em_code = em_entry.get("code", "")
        mods = em_entry.setdefault("modifiers", [])
        has_mod25 = "25" in mods
        has_mod57 = "57" in mods

        # A same-day major (090) procedure requires -57, never -25 — settle
        # this first and return, regardless of what has_mod25/has_mod57
        # looked like coming in.
        if has_major_procedure:
            if has_mod25:
                mods.remove("25")
                self._add(
                    "INFO", em_code, "modifier_25_removed_major_procedure",
                    f"AUTO-CORRECTED: Removed modifier -25 from {em_code} — a same-day major "
                    f"procedure (global=090) requires modifier -57, not -25.",
                    "Modifier -25 removed (superseded by -57)",
                    denial_risk="LOW",
                )
            if not has_mod57:
                mods.append("57")
                self._add(
                    "INFO", em_code, "modifier_57_added",
                    f"AUTO-CORRECTED: Added modifier -57 to {em_code} — same-day major procedure "
                    f"(global=090) detected. Without -57, payer bundles the E/M into the "
                    f"surgical global period.",
                    "Modifier -57 auto-added",
                    denial_risk="LOW",
                )
            return

        # -57 present without a same-day major procedure detected here (e.g.
        # the decision was for surgery on a later date, so no 090-global CPT
        # appears on today's claim) — trust it, don't second-guess an
        # explicit -57 the coder already determined was correct.
        if has_mod57:
            if has_mod25:
                mods.remove("25")
                self._add(
                    "INFO", em_code, "modifier_25_removed_57_present",
                    f"AUTO-CORRECTED: Removed modifier -25 from {em_code} — modifier -57 is "
                    f"present (major surgery decision). Modifier -25 and -57 are mutually "
                    f"exclusive; -57 takes precedence.",
                    "Modifier -25 removed (superseded by -57)",
                    denial_risk="LOW",
                )
            return  # Do not apply any -25 logic when -57 is in play

        # Only auto-remove when there's genuinely nothing on the claim that
        # could justify -25 — not a minor/intermediate procedure to
        # separate the E/M from, AND not even a same-day diagnostic test.
        # A diagnostic test alone is a real, undecided case (judgment call
        # on the E/M's own MDM, not this check's to make either way) —
        # conflating it with "nothing else on the claim at all" previously
        # stripped a legitimately-claimed and reasoning-consistency-added
        # -25 back out whenever the only same-day code was a diagnostic
        # test, unconditionally, regardless of documented MDM complexity.
        if has_mod25 and not has_minor_procedure and not has_diagnostic_test:
            mods.remove("25")
            self._add(
                "INFO", em_code, "modifier_25_removed",
                f"AUTO-CORRECTED: Removed modifier -25 from {em_code} — no same-day minor/"
                f"intermediate procedure or diagnostic test found to justify it.",
                "Modifier -25 auto-removed",
                denial_risk="LOW",
            )

        if has_minor_procedure and not has_mod25:
            mods.append("25")
            self._add(
                "INFO", em_code, "modifier_25_added",
                f"AUTO-CORRECTED: Added modifier -25 to {em_code} — same-day minor/intermediate "
                f"procedure detected. Without -25, payer bundles the E/M into the procedure "
                f"global period.",
                "Modifier -25 auto-added",
                denial_risk="LOW",
            )

    def _check_modifier57(self, cpt, note_plan_text: str = ""):
        """Flag when E/M lacks -57 but plan text indicates surgical decision was made today."""
        if not note_plan_text:
            return

        em_entry = next((c for c in cpt if _is_em(c.get("code", ""))), None)
        if em_entry is None:
            return

        plan_lower = note_plan_text.lower()
        has_surgical_decision = any(kw in plan_lower for kw in SURGICAL_DECISION_KEYWORDS)
        if not has_surgical_decision:
            return

        em_code = em_entry.get("code", "")
        has_mod57 = "57" in em_entry.get("modifiers", [])
        if has_mod57:
            return

        # -57 exists to protect the E/M from the surgical global period, and
        # per CMS Ch.12 §30.6.6 that bundling only happens for an E/M on the
        # DAY OF or DAY BEFORE a major (090-global) procedure. Whether any
        # code on TODAY'S claim carries a 090 global is checkable against the
        # PFS data; surgery merely PLANNED for a future date creates no
        # bundling risk today, so demanding -57 from keyword evidence alone
        # was a false ERROR (observed live: 'discussed surgical correction'
        # in a plan with only 000-global codes billed).
        has_major_today = self.store is not None and any(
            (self.store.global_period(c.get("code", "")) or "").strip() == "090"
            for c in cpt if c.get("code")
        )
        if has_major_today:
            self._add(
                "ERROR", em_code, "modifier_57_missing",
                f"E/M {em_code} billed at the visit where the decision for major surgery was made "
                f"AND a 90-day-global procedure is on today's claim, but modifier -57 is absent — "
                f"the E/M will bundle into the surgical global period.",
                "Add modifier -57 to protect the E/M from bundling into the surgical global period",
                denial_risk="HIGH",
            )
        else:
            self._add(
                "INFO", em_code, "modifier_57_future_surgery",
                f"Plan documents a decision for surgery, but no 90-day-global procedure is on "
                f"today's claim — modifier -57 belongs on the E/M of the day of/day before the "
                f"surgery, not today's visit. No action needed on this claim.",
                "If the surgery happens today or tomorrow under this E/M, append -57 then",
                denial_risk="LOW",
            )

    def _check_em_minor_procedure_bundling(self, cpt, icd):
        """NCCI Policy Manual Ch. 1: 'The decision to perform a minor
        surgical procedure [global 000/010] is included in the payment for
        the minor surgical procedure and shall not be reported separately as
        an E&M service. However, a significant and separately identifiable
        E&M service UNRELATED to the decision to perform the minor surgical
        procedure is separately reportable.'

        The manual ALSO says the two services 'do not require different
        diagnoses' — so diagnosis overlap alone cannot justify an automatic
        suppression. What overlap DOES mean is that nothing on the claim
        itself evidences the separately identifiable service; the -25 claim
        then rests entirely on the note's documentation. This check is
        therefore OBSERVE-ONLY (INFO): it surfaces every established-patient
        E/M-25 whose diagnoses are all procedure-addressed, so audits and
        the denial-feedback loop can quantify the practice's -25 exposure.
        ENFORCE=True turns it into a deterministic suppression (same
        mechanism as NCCI PTP bundling) if the practice adopts the strict
        posture. Kept silent when a diagnosis unaddressed by any procedure
        is linked (e.g. Z79.4 long-term-medication management), or for
        new-patient E/M (per the code's own descriptor), or a -57 major-
        surgery context.

        Measured live (notes 003/007): independent runs flapped on whether
        to bill 99214-25 alongside a nail procedure with identical
        diagnoses; 001/002/005 bill the same pattern unanimously. The
        presence judgment is documentation-driven, so the convergence fix
        lives in the coding prompt's explicit E/M policy — this check is
        the claim-level audit trail for it.
        """
        ENFORCE = False
        if self.store is None:
            return
        minor_on_claim = any(
            not _is_em(c.get("code", ""))
            and (self.store.global_period(c.get("code", "")) or "").strip()
            in MINOR_PROCEDURE_GLOBAL_DAYS
            for c in cpt if c.get("code"))
        if not minor_on_claim:
            return
        major_on_claim = any(
            (self.store.global_period(c.get("code", "")) or "").strip()
            in MAJOR_PROCEDURE_GLOBAL_DAYS
            for c in cpt if c.get("code"))

        def _norm_dx(d) -> str:
            return str(d).replace(".", "").strip().upper()

        proc_dx = {
            _norm_dx(d)
            for c in cpt if c.get("code") and not _is_em(c.get("code", ""))
            for d in c.get("linked_diagnoses") or []
        }
        for entry in cpt:
            code = entry.get("code", "")
            if not _is_em(code):
                continue
            desc = ((self.db.validate_cpt(code) or {})
                    .get("long_description", "")).lower()
            if "new patient" in desc:
                continue
            # -57 context: the E/M carrying the decision for a same-day
            # MAJOR procedure is protected by -57, not judged here.
            mods = {str(m).strip().upper() for m in entry.get("modifiers") or []}
            if "57" in mods and major_on_claim:
                continue
            em_dx = [d for d in entry.get("linked_diagnoses") or []]
            separate = [d for d in em_dx if _norm_dx(d) not in proc_dx]
            if separate:
                continue
            if ENFORCE:
                self._non_billable_codes_to_suppress.add(code)
            self._add(
                "INFO", code, "em_minor_procedure_bundled",
                (f"AUTO-CORRECTED: {code} removed — " if ENFORCE else
                 f"-25 exposure: {code} is kept, but note that ")
                + f"every diagnosis linked to this established-patient E/M "
                f"({', '.join(em_dx) or 'none'}) is also linked to a same-day minor "
                f"(000/010-global) procedure line. Per NCCI Policy Manual Ch. 1 the "
                f"pre-procedure evaluation is included in the procedure's payment; "
                f"with no separately linked diagnosis, the -25 claim rests entirely "
                f"on the note documenting a significant, separately identifiable "
                f"service.",
                "Ensure the note's E/M documentation stands apart from the procedure "
                "workup; link any separately managed diagnosis to the E/M line",
                denial_risk="MEDIUM",
            )

    def _check_imaging_note_evidence(self, cpt, note_full_text: str):
        """A Radiology-section service (CPT 70010-79999) is billable only
        when the note documents the imaging was PERFORMED at this encounter
        — CMS Claims Processing Manual Ch. 13 (the professional component
        requires a rendered service with a report; reviewing outside films
        is part of the E/M, not a radiology charge).

        Enforcement is descriptor-driven: the billed code's own descriptor
        names the modality ('Radiologic examination...', 'Fluoroscopy...'),
        and IMAGING_MODALITY_LEXICON maps that modality word to how notes
        phrase it. If the negation-scrubbed note contains no phrasing of the
        billed modality at all, the line is suppressed.

        Determinism layer: measured live (note 008), single runs added
        73630/73660/76000 X-ray/fluoroscopy lines the other runs never
        billed — the note's own words, not per-run mood, decide whether an
        imaging service happened."""
        if not note_full_text:
            return
        _, low_note = self._note_evidence(note_full_text)
        for entry in cpt:
            code = (entry.get("code") or "").strip()
            if not (len(code) == 5 and code.isdigit() and code[0] == "7"):
                continue
            desc = ((self.db.validate_cpt(code) or {})
                    .get("long_description", "")).lower()
            if not desc:
                continue
            synonyms: tuple = ()
            for modality_word, phrasings in IMAGING_MODALITY_LEXICON.items():
                if modality_word in desc:
                    synonyms = phrasings + (modality_word,)
                    break
            if not synonyms:
                continue
            if any(s in low_note for s in synonyms):
                continue
            self._non_billable_codes_to_suppress.add(code)
            self._add(
                "INFO", code, "imaging_not_documented",
                f"AUTO-CORRECTED: {code} ('{desc[:60]}') removed — the note never "
                f"documents this imaging modality being performed at this encounter "
                f"(no mention of {', '.join(synonyms[:3])}). A radiology charge "
                f"requires a rendered, documented service.",
                "If imaging was performed, document the study and its findings, "
                "then re-bill",
                denial_risk="LOW",
            )

    @property
    def rule_engine(self):
        """Declarative rule engine (lazy) — the generic executors behind the
        config-authored rules in data/rules/validator_rules.json."""
        if getattr(self, "_rule_engine", None) is None:
            from app.validation.rule_engine import RuleEngine
            self._rule_engine = RuleEngine(self)
        return self._rule_engine

    def _modality_synonyms(self, code: str) -> tuple:
        """Note-side phrasings for the billed radiology code's own modality
        word, from IMAGING_MODALITY_LEXICON (shared with the presence gate)."""
        desc = ((self.db.validate_cpt(code) or {})
                .get("long_description", "")).lower()
        for modality_word, phrasings in IMAGING_MODALITY_LEXICON.items():
            if modality_word in desc:
                return phrasings + (modality_word,)
        return ()

    def _check_imaging_context(self, cpt, note_full_text: str):
        """A radiology line that survived the presence gate (the modality IS
        in the note) is still not separately billable when EVERY mention of
        that modality is a non-billable context: a prior-visit film, an
        ordered/future study, or intraoperative/post-procedure confirmation
        imaging bundled into a same-claim surgery.

        Sentences are classified on the RAW note text — negation of findings
        ('no fracture identified on AP/lateral views') does not negate the
        study having been performed, so the negation scrub must not eat the
        evidence here.

        Determinism layer: measured live (note 008), minority runs billed
        73620/76000 off a note whose only imaging mentions were 'X-ray at
        prior visit', 'post-op X-ray confirms adequate bony resection',
        'intraoperative fluoroscopy used to confirm', and 'post-op X-ray at
        6 weeks' — none a billable diagnostic study rendered today.

        Declarative: rule 'imaging-context-gate' in the rule pack (context
        regexes, surgery range, message text all live there)."""
        self.rule_engine.context_gate("imaging-context-gate", cpt,
                                      note_full_text)

    _VIEWS_SPEC_RE = re.compile(r"(?:minimum\s+of\s+)?(\d+)\s+views?")

    def _check_radiograph_view_count(self, cpt, note_full_text: str):
        """Arbitrate radiology sibling codes that differ ONLY by view count
        — the AMA structures these families in the descriptors themselves
        ('Radiologic examination, foot; 2 views' 73620 vs '...; complete,
        minimum of 3 views' 73630). The billable variant is a documentation
        question the note answers deterministically: the views the note
        names (or counts) either meet the billed descriptor's minimum or
        they don't. When the note documents no views at all, only the
        family's FEWEST-views code is supportable (billing 'minimum of 3
        views' with zero views documented cannot survive an audit).

        Fully descriptor-driven: the family is every CPT sharing the billed
        code's descriptor stem before the ';' with a parseable view count —
        no code list. The projection vocabulary (AP, lateral, oblique...)
        is a language lexicon; explicit 'N views' phrasing also counts.

        Determinism layer: measured live (note 008), independent runs split
        73620/73630/none on a note whose imaging section names no views —
        the note's own words, not per-run mood, decide the view level."""
        if not note_full_text:
            return
        _, low_note = self._note_evidence(note_full_text)

        documented_views = 0
        for sent in re.split(r"[.;\n]", low_note):
            if not re.search(r"\b(view|views|x-?ray|xray|radiograph\w*|film)\b",
                             sent):
                continue
            named = {t for t in RADIOGRAPH_PROJECTION_TERMS
                     if re.search(rf"\b{t}\b", sent)}
            named |= set(re.findall(r"\b(ap|pa)\b", sent))
            m = re.search(r"\b(\d+)\s+views?\b", sent)
            explicit = int(m.group(1)) if m else 0
            documented_views = max(documented_views, len(named), explicit)

        def _fam_key(desc: str) -> str | None:
            """Family identity = the descriptor with the view-count phrase
            (and its 'complete'/'minimum of' qualifiers) removed. Keying on
            the head before ';' is NOT enough: for 'Radiologic examination;
            toe(s), minimum of 2 views' the anatomy lives in the TAIL, and a
            head-only key ('Radiologic examination') matched every anatomy —
            measured live, 73660 (toe) was swapped to 71120 (sternum)."""
            dl = desc.lower()
            if not self._VIEWS_SPEC_RE.search(dl):
                return None
            dl = re.sub(r"(?:complete,?\s*)?(?:minimum\s+of\s+)?\d+\s+views?",
                        " ", dl)
            return re.sub(r"[^a-z0-9()]+", " ", dl).strip()

        def _family(code: str):
            info = self.db.validate_cpt(code) or {}
            desc = info.get("long_description") or ""
            m = self._VIEWS_SPEC_RE.search(desc.lower())
            key = _fam_key(desc)
            if not m or key is None:
                return None, None
            members = []
            for c, i in (getattr(self.db, "cpt", {}) or {}).items():
                d = i.get("long_description") or ""
                if _fam_key(d) != key:
                    continue
                m2 = self._VIEWS_SPEC_RE.search(d.lower())
                if m2:
                    members.append((int(m2.group(1)), c))
            return int(m.group(1)), sorted(members)

        billed_codes = {e.get("code", "") for e in cpt}
        for entry in cpt:
            code = (entry.get("code") or "").strip()
            if not (len(code) == 5 and code.isdigit() and code[0] == "7"):
                continue
            if code in self._non_billable_codes_to_suppress:
                continue  # a context/presence gate already removed this line
            billed_views, members = _family(code)
            if billed_views is None or len(members) < 2:
                continue
            if documented_views >= billed_views:
                continue  # documentation meets the billed minimum
            # largest family member the documented views support; fewest-views
            # member when nothing is documented
            supported = [(n, c) for n, c in members if n <= documented_views]
            target_views, target = (max(supported) if supported else members[0])
            if target == code:
                continue
            if target in billed_codes:
                self._non_billable_codes_to_suppress.add(code)
                action = (f"line removed — {target} (already on the claim) is "
                          f"the supportable member of this family")
            else:
                old = code
                entry["code"] = target
                entry["needs_review"] = True
                billed_codes.add(target)
                info_t = self.db.validate_cpt(target) or {}
                entry["description"] = (info_t.get("long_description")
                                        or entry.get("description", ""))
                action = f"swapped {old} → {target}"
            self._add(
                "INFO", code, "radiograph_view_count",
                f"AUTO-CORRECTED: {code} requires {billed_views} views per its own "
                f"descriptor, but the note documents "
                f"{documented_views or 'no'} view(s) — {action}. The view-count "
                f"family comes from the descriptors themselves "
                f"({', '.join(f'{c}={n}v' for n, c in members)}).",
                "If more views were actually obtained, document each projection "
                "and re-bill the complete-study code",
                denial_risk="MEDIUM",
            )

    # Language lexicon (no codes): how notes phrase a supply being handed to
    # the patient AT the encounter — dispensing verbs only; 'prescribed' /
    # 'recommended' / 'ordered' footwear was not furnished today and is not
    # a billable supply line.
    _DISPENSE_RE = re.compile(
        r"\b(?:provided|dispensed|issued|furnished|fitted)\b")
    _FOOTWEAR_RE = re.compile(r"\b(?:shoe|boot)\b")

    def _check_dispensed_footwear_completion(self, hcpcs, cpt, icd,
                                             note_full_text: str, dos,
                                             patient_dob):
        """Add the surgical boot/shoe supply line the note itself documents
        as furnished. A dispensed DMEPOS item is a billable supply; when a
        performed-context sentence states protective footwear was provided
        ('Hard-soled shoe provided') and no footwear line is on the claim,
        the claim under-reports the encounter.

        Determinism layer: measured live (note 005), the L-code for a
        documented 'hard-soled shoe provided' flapped present-in-2-of-3
        runs — the sentence is identical every run, so the line must be too.

        No codes live here: the family is discovered from the HCPCS
        reference's own descriptors (head phrase 'surgical boot'), and the
        member is arbitrated on the family's own age axis (infant/child/
        junior vs the ageless adult member) from the claim's DOB + DOS.
        Adds only the ageless member for an adult patient; a pediatric
        patient's bracket boundaries aren't in the data, so that case stays
        with the human. The added line then flows through the same
        laterality/modifier normalizations as a coder-billed one."""
        if self.db is None or not note_full_text:
            return
        _, low_note = self._note_evidence(note_full_text)
        perf = self._performed_context(low_note)
        hit_sentence = next(
            (s.strip() for s in re.split(r"[.;]", perf.replace("\n", " "))
             if self._DISPENSE_RE.search(s) and self._FOOTWEAR_RE.search(s)),
            "")
        if not hit_sentence:
            return
        # already on the claim? any billed HCPCS whose descriptor is footwear
        for e in hcpcs:
            info = self.db.validate_hcpcs(e.get("code", "")) or {}
            d = (info.get("long_description") or info.get("description") or "")
            if self._FOOTWEAR_RE.search(d.lower()):
                return
        # adult patient only — the pediatric members' brackets aren't data
        from app.compliance.engine import _parse_date
        dob = _parse_date(patient_dob or "")
        if not (dob and dos and (dos - dob).days / 365.25 >= 18):
            return
        age_words = ("infant", "child", "junior")
        candidates = [
            (c, (i.get("long_description") or i.get("description") or ""))
            for c, i in (getattr(self.db, "hcpcs", {}) or {}).items()
            if (i.get("long_description") or i.get("description") or "")
            .lower().startswith("surgical boot")
            and not any(w in (i.get("long_description") or "").lower()
                        for w in age_words)
        ]
        if len(candidates) != 1:
            return  # ambiguous family — a human question, not an add
        code, desc = candidates[0]
        primary = next((c.get("code") for c in icd
                        if c.get("type") == "primary" and c.get("code")), None)
        hcpcs.append({
            "code": code,
            "description": desc,
            "modifiers": [],
            "units": 1,
            "linked_diagnoses": [primary] if primary else [],
            "confidence": 0.9,
            "source": "validator:dispensed_supply_completion",
            "needs_review": True,
            "review_reason": (
                f"Auto-added from the note's own dispensing statement "
                f"('{hit_sentence[:80]}') — confirm the item furnished"),
            "evidence_spans": [hit_sentence[:160]],
        })
        self._add(
            "WARNING", code, "dispensed_supply_completion",
            f"AUTO-ADDED: {code} ('{desc[:60]}') — the note's performed-"
            f"context documents protective footwear furnished at this "
            f"encounter ('{hit_sentence[:100]}') and no footwear supply line "
            f"was billed; a dispensed DMEPOS item is a billable supply.",
            f"Confirm {code} and the payer's DMEPOS billing route for "
            f"dispensed footwear",
            denial_risk="MEDIUM",
        )
        logger.info(f"  Added dispensed supply {code} ('{hit_sentence[:60]}')")

    def _check_supply_laterality_strip(self, hcpcs):
        """Strip RT/LT from HCPCS A-code lines — the A chapter is materials
        and supplies ('Splint', 'Cast supplies...'), not sided devices; CMS's
        RT/LT definition ('procedures performed on one side of the body')
        attaches to procedures and fitted DMEPOS items (the L chapter, which
        _check_hcpcs_laterality already sides via UNILATERAL_L_PREFIXES),
        not to consumed materials. A digit modifier derived by the
        supply-descriptor check is specific siting and is preserved.

        Determinism layer: measured live (note 005), A4570 flapped between
        [] and ['RT'] across runs of an identical note — then, with RT
        stripped, between [] and ['T6'] — the canonical spelling of a
        materials line has no site designator at all, so runs converge on
        it. The one exception is the descriptor's OWN instruction: a supply
        whose descriptor says 'specify digit by use of modifier' (S8450
        class) mandates the digit, and _check_digit_supply_modifier owns
        that spelling — those lines keep their digit here."""
        for entry in hcpcs:
            code = (entry.get("code") or "").strip().upper()
            if not code.startswith("A"):
                continue
            mods = entry.get("modifiers") or []
            info = self.db.validate_hcpcs(code) or {}
            desc = ((info.get("long_description")
                     or info.get("description") or "")).lower()
            digit_mandated = bool(re.search(r"specify\s+digit", desc))
            stripped = [
                m for m in mods
                if str(m).strip().upper() in ("RT", "LT")
                or (not digit_mandated
                    and str(m).strip().upper() in self._digit_site_mods(mods))
            ]
            if not stripped:
                continue
            for m in stripped:
                mods.remove(m)
            self._add(
                "INFO", code, "supply_laterality_removed",
                f"AUTO-CORRECTED: Removed {'/'.join(str(m) for m in stripped)} from "
                f"{code} — an A-code is a materials/supply line, not a sided service; "
                f"side and digit designators attach to procedures and fitted devices "
                f"(or to supplies whose own descriptor mandates a digit modifier), "
                f"not consumed materials.",
                "No action needed; supply lines carry no site designator",
                denial_risk="LOW",
            )

    def _guidance_cpts(self) -> dict[str, str]:
        """{modality key -> guidance CPT} discovered from the CPT reference's
        own descriptors: every code whose long description reads
        '<modality> guidance for needle placement' (76942 ultrasonic, 77002
        fluoroscopic, 77012 CT, 77021 MRI as of 2026 — but derived, not
        listed, so AMA additions/deletions flow through the data refresh)."""
        if getattr(self, "_guidance_map", None) is not None:
            return self._guidance_map
        out: dict[str, str] = {}
        for code, info in (getattr(self.db, "cpt", {}) or {}).items():
            desc = (info.get("long_description") or info.get("description") or "").lower()
            m = re.match(r"(.+?)\s+guidance for needle placement", desc)
            if not m:
                continue
            head = m.group(1)
            for modality in MODALITY_LEXICON:
                if modality in head:
                    out[modality] = code
        self._guidance_map = out
        return out

    def _debridement_family(self) -> dict:
        """{code: (depth rank, descriptor)} for primary wound-debridement
        codes — derived from the CPT descriptors by the rule engine's
        tier-family builder per the 'debridement-depth' rule's descriptor
        grammar (prefix 'Debridement', requires 'includes'/'open wound',
        excludes each-additional add-ons, strips the includes-parenthetical
        because it names the SHALLOWER layers a code subsumes)."""
        return self.rule_engine._tier_family("debridement-depth")

    def _check_debridement_depth(self, cpt, note_full_text: str):
        """CPT's own instruction for wound debridement: report the code for
        the DEEPEST tissue level removed — and an audit pays only the level
        the note documents. The family is structured in the descriptors
        themselves (97597 open wound at epidermis/dermis; 11042
        subcutaneous; 11043 muscle/fascia; 11044 bone), so which member is
        billable is a documentation question the note answers
        deterministically: the deepest tissue word appearing in a
        debridement sentence, or — when no tissue level is documented at
        all — only the shallowest member survives review.

        Determinism layer: measured live (note 009), runs split
        11042/97597 on a note documenting only 'deep wound debridement
        performed' with no tissue level named. Swaps flag for review with
        a query recommendation; a 'bone biopsy' sentence never upgrades a
        debridement (different service, own code).

        Declarative: rule 'debridement-depth' in the rule pack (descriptor
        grammar, anatomy tier lexicon, message text all live there)."""
        self.rule_engine.tiered_family_arbitration("debridement-depth", cpt,
                                                   note_full_text)

    def _check_operative_field_debridement(self, cpt, note_full_text: str):
        """NCCI Policy Manual Ch.1: debridement of the operative site —
        preparing wound margins/edges during a same-claim surgical procedure
        — is integral to the surgery and never separately reportable. When
        EVERY debridement mention in the note is margin/edge preparation and
        the claim carries a surgical procedure, the standalone debridement
        line is suppressed.

        Determinism layer: measured live (note 010), 'wound margins debrided
        to bleeding tissue' inside a phalangectomy op note produced a 97597
        line in 1 of 3 runs, decorated with a 59 that NCCI's own manual
        forbids for same-site same-session work.

        Declarative: rule 'operative-field-debridement-gate' in the rule
        pack (context grammar, message text live there)."""
        self.rule_engine.context_gate("operative-field-debridement-gate",
                                      cpt, note_full_text)

    def _check_ulcer_severity_tier(self, icd, note_full_text: str):
        """ICD-10-CM's L97-style severity axis: the final character encodes
        the deepest tissue layer involved ('limited to breakdown of skin' <
        'fat layer exposed' < 'necrosis of muscle' < 'necrosis of bone'),
        and an audit pays only the depth the documentation supports. The
        billed member is arbitrated against the note's own ulcer sentences;
        with no tier evidence, only the shallowest member is supportable.

        Determinism layer: measured live (note 004), a 'wound depth 2mm,
        Wagner grade 1' ulcer flapped L97.511/L97.512 across runs — the
        note documents no fat-layer exposure, so the skin-breakdown tier is
        the only supportable member, identically every run.

        Declarative: rule 'ulcer-severity-tier' in the rule pack (tier
        lexicon, evidence grammar, message text live there)."""
        self.rule_engine.icd_tiered_axis("ulcer-severity-tier", icd,
                                         note_full_text)

    def _check_image_guidance(self, cpt, note_full_text: str = ""):
        """Flag a missing needle-placement image-guidance code when the note
        documents the procedure was performed under imaging. Both sides are
        data-driven: the guidance codes come from the CPT reference's own
        'guidance for needle placement' descriptors (_guidance_cpts), and
        procedure eligibility from CPT 77002's real NCCI Add-On Code (AOC)
        edit table (its 72 valid primary codes). 76942/77012/77021 aren't
        themselves AMA-flagged as add-on codes, but cover the same underlying
        needle-placement procedures as alternative modalities, so the same
        eligible-code set gates every branch."""
        if not note_full_text or self.store is None:
            return

        guidance = self._guidance_cpts()
        fluoro_code = guidance.get("fluoroscopic")
        if not fluoro_code:
            return
        cpt_codes = {c.get("code", "") for c in cpt}
        guidance_eligible_codes = {e.get("code2") for e in self.store.ncci_aoc_edits(fluoro_code)}
        if not (cpt_codes & guidance_eligible_codes):
            return

        note_lower = note_full_text.lower()
        for modality, phrases in MODALITY_LEXICON.items():
            g_code = guidance.get(modality)
            if not g_code or g_code in cpt_codes:
                continue
            if not any(kw in note_lower for kw in phrases):
                continue
            g_desc = ((self.db.validate_cpt(g_code) or {}).get("long_description")
                      or "").split(";")[0]
            self._add(
                "ERROR", g_code, "image_guidance_missing",
                f"Procedure performed under {modality} guidance per the note, but CPT "
                f"{g_code} ({g_desc}) is missing. {g_code} is separately billable "
                f"whenever this guidance modality is documented.",
                f"Add CPT {g_code} to the claim alongside the procedure code",
                denial_risk="HIGH",
            )

    # _check_global_period (the E/M-during-prior-surgery's-global-window
    # check) was removed — it duplicated, less correctly, a concept the
    # 13-filter compliance scrubber's GlobalPeriodAgent already covers as
    # the authoritative gate. This version's only real-data source,
    # CodeReferenceDB.get_global_period(), returns 0 for BOTH "genuinely
    # 000-day global" and "code not found in data" (collapsing XXX/YYY/ZZZ/
    # MMM to 0 too, per its own docstring), and the check treated any 0 as
    # "no global period, skip" — silently missing a real violation whenever
    # prior_cpt wasn't recognized, instead of flagging "can't confirm,
    # needs review" the way GlobalPeriodAgent does (it queries
    # store.global_period()'s raw string and explicitly WARNs on missing
    # data). Keeping both meant a case validator.py's cruder logic silently
    # cleared could still surface via the scrubber, or vice versa, with two
    # different messages for the same fact — pipeline.py's
    # _apply_scrub_verdict already makes the scrubber authoritative for
    # tier/confidence, so this duplicate added risk without adding coverage.

    def _check_hcpcs_laterality(self, cpt, hcpcs):
        """Auto-correct HCPCS L-codes missing RT/LT when procedure laterality is known.

        The side is derived from each CPT modifier's own name in the AMA/CMS
        modifier reference (store.modifier_laterality: 'Left foot, great toe'
        → LT), not a hand-written toe-modifier map — a prior hardcoded map
        here had TA/T1–T4 (left foot digits) and T5–T9 (right foot digits)
        INVERTED, which would have auto-added the WRONG side to L-codes.
        All CPT lines are scanned (no first-match-wins): if the claim's
        procedures span BOTH sides, no single side can be inferred for an
        unsided L-code, so it's flagged for manual assignment instead of
        guessing."""
        sides_seen: set[str] = set()
        if self.store is not None:
            for c in cpt:
                for m in c.get("modifiers", []):
                    side = self.store.modifier_laterality(str(m))
                    if side:
                        sides_seen.add(side)
        procedure_side = sides_seen.pop() if len(sides_seen) == 1 else None
        mixed_sides = len(sides_seen) >= 1 and procedure_side is None

        for entry in hcpcs:
            code = entry.get("code", "")
            if not code.startswith(UNILATERAL_L_PREFIXES):
                continue

            mods = entry.setdefault("modifiers", [])
            has_laterality = "RT" in mods or "LT" in mods or "50" in mods

            if not has_laterality:
                if procedure_side:
                    mods.append(procedure_side)
                    self._add(
                        "INFO", code, "hcpcs_laterality_added",
                        f"AUTO-CORRECTED: Added {procedure_side} modifier to HCPCS {code} — "
                        f"inferred from CPT procedure side. CMS requires laterality on unilateral L-codes.",
                        f"Laterality {procedure_side} auto-added",
                        denial_risk="LOW",
                    )
                else:
                    why = (
                        "the claim's CPT procedures span BOTH sides, so no single side can be inferred"
                        if mixed_sides else "could not infer side from CPT codes"
                    )
                    self._add(
                        "ERROR", code, "hcpcs_laterality",
                        f"HCPCS {code} (L-code) is missing a laterality modifier (RT or LT). "
                        f"CMS requires laterality on unilateral DME/orthotic L-codes — "
                        f"claims are rejected without it ({why}).",
                        "Manually add RT or LT modifier to match the dispensed side",
                        denial_risk="HIGH",
                    )

    def _check_icd_cpt_laterality_agreement(self, cpt, icd=None):
        """Cross-check each sided procedure against the sides its own linked
        diagnoses encode — a right-sided ICD pointing at an LT procedure is a
        classic denial pattern that nothing previously caught (procedure-side
        checks and ICD-side specificity were each validated alone, never
        against each other).

        Data-driven both ways: the procedure side comes from the line's
        modifiers via the AMA/CMS modifier reference (modifier_laterality),
        and the diagnosis side from the ICD code's OWN CMS description text
        ('Morton's neuroma, right foot' / '...left foot'). Diagnoses whose
        descriptions state no side, or state both, are skipped.

        When the CLAIM is unilaterally consistent — every sided procedure
        line carries the same side (all left, say) — a lone contralateral
        diagnosis is corrected to its opposite-side sibling, provided a
        sibling whose descriptor is the side-swapped text exists in the code
        set. The procedure side is the stronger signal: it was itself
        validated against the note by the modifier checks upstream.
        Determinism layer: measured live (note 006), one run billed
        S90.111A (RIGHT great toe contusion) on a claim whose every
        procedure was TA (left great toe) — runs converge on the side the
        claim itself operates on. Ambiguous claims (mixed sides, no exact
        sibling) keep the WARNING, never a guess."""
        if self.store is None:
            return
        # claim-wide procedure side, for the auto-swap arm
        claim_sides = {
            s for c in cpt for m in c.get("modifiers", []) or []
            if (s := self.store.modifier_laterality(str(m)))
        }
        uni_side = claim_sides.pop() if len(claim_sides) == 1 else None
        if uni_side is not None and icd:
            uni_word = "right" if uni_side == "RT" else "left"
            wrong_word = "left" if uni_side == "RT" else "right"
            for dx_entry in icd:
                code = (dx_entry.get("code") or "").strip().upper()
                info = self.db.validate_icd10(code)
                if not info:
                    continue
                desc = info.get("description", "")
                low = desc.lower()
                if "bilateral" in low or wrong_word not in low or uni_word in low:
                    continue
                want = " ".join(
                    re.sub(rf"\b{wrong_word}\b", uni_word, low).split())
                norm = code.replace(".", "")
                partners = [
                    (c, d) for c, d in self.db.icd10_siblings(code[:3])
                    if c != norm and len(c) == len(norm)
                    and " ".join(d.lower().split()) == want
                ]
                if len(partners) != 1:
                    continue
                sib_code, sib_desc = partners[0]
                if any((e.get("code") or "").strip().upper().replace(".", "")
                       == sib_code for e in icd if e is not dx_entry):
                    continue
                target = sib_code if "." in sib_code else (
                    sib_code[:3] + "." + sib_code[3:] if len(sib_code) > 3
                    else sib_code)
                dx_entry["code"] = target
                dx_entry["description"] = sib_desc
                dx_entry["needs_review"] = True
                self._add(
                    "WARNING", code, "icd_laterality_corrected",
                    f"AUTO-CORRECTED: {code} ('{desc[:55]}') replaced with {target} "
                    f"('{sib_desc[:55]}') — every sided procedure on this claim is "
                    f"{uni_side} ({uni_word}), and the modifier checks validated that "
                    f"side against the note; a lone {wrong_word}-side diagnosis "
                    f"contradicts the claim's own operative side.",
                    f"Verify the documented side; the claim's procedures fix it as "
                    f"{uni_word}",
                    denial_risk="HIGH",
                )
        for entry in cpt:
            sides = {
                s for m in entry.get("modifiers", [])
                if (s := self.store.modifier_laterality(str(m)))
            }
            if len(sides) != 1:
                continue  # unsided or explicitly bilateral procedure line
            proc_side = sides.pop()
            proc_word = "right" if proc_side == "RT" else "left"
            other_word = "left" if proc_side == "RT" else "right"
            for dx in entry.get("linked_diagnoses", []) or []:
                info = self.db.validate_icd10(str(dx)) or {}
                desc = (info.get("description") or "").lower()
                if not desc or "bilateral" in desc:
                    continue
                if other_word in desc and proc_word not in desc:
                    self._add(
                        "WARNING", entry.get("code", ""), "icd_cpt_laterality_conflict",
                        f"{entry.get('code')}-{proc_side} is a {proc_word}-side procedure per its "
                        f"modifier, but linked diagnosis {dx} is '{info.get('description')}' — "
                        f"a {other_word}-side condition. Side mismatches between the diagnosis "
                        f"pointer and the procedure deny.",
                        f"Verify the documented side: correct the {proc_side} modifier, the "
                        f"diagnosis laterality, or the linkage so they agree",
                        denial_risk="HIGH",
                    )

    def _check_cpt_laterality(self, cpt):
        """Auto-correct CPT codes missing RT/LT/50 when CMS's bilateral-surgery
        indicator (bilat_surg='1', from global_periods.json — see
        ComplianceDataStore.bilat_surg) says a laterality modifier is
        expected on this code, and the coder's own structured `laterality`
        field already states the side.

        This is a different failure mode from
        _check_modifier_reasoning_consistency: that check only catches a
        mismatch when modifier_reasoning actually contains a structured
        claim entry for the modifier (e.g. {"modifier": "RT", "status":
        "applied"}). Observed live case: reasoning text said "Right-sided
        procedure per documentation" for four consecutive surgical CPT
        codes with no RT claim entry at all — laterality was documented
        in prose, not as a modifier claim, so there was nothing for that
        check to catch. bilat_surg + the structured laterality field are
        both already-real, already-populated data — this closes the gap
        without parsing prose or hardcoding which CPT codes require
        laterality.
        """
        if self.store is None:
            return
        for entry in cpt:
            code = entry.get("code", "")
            if not code or self.store.bilat_surg(code) != "1":
                continue
            mods = entry.setdefault("modifiers", [])
            if "50" in mods:
                continue
            # Any modifier whose own AMA/CMS name states a side satisfies the
            # laterality requirement — that includes RT/LT themselves AND the
            # site-specific digit modifiers (T5 'Right foot, great toe'). A
            # line already carrying T5 documents the side MORE specifically
            # than RT would; adding RT on top of it is the redundancy
            # _check_redundant_laterality exists to remove.
            if any(self.store.modifier_laterality(str(m)) for m in mods):
                continue
            laterality = str(entry.get("laterality") or "").strip().upper()
            side = {"RIGHT": "RT", "LEFT": "LT", "BILATERAL": "50"}.get(laterality)
            if side:
                mods.append(side)
                self._add(
                    "INFO", code, "cpt_laterality_added",
                    f"AUTO-CORRECTED: Added {side} modifier to CPT {code} — CMS bilateral-surgery "
                    f"indicator requires a laterality modifier on this code, and the coder's own "
                    f"structured laterality field ('{laterality}') was never reflected in the "
                    f"modifiers array.",
                    f"Laterality {side} auto-added",
                    denial_risk="LOW",
                )
            else:
                self._add(
                    "ERROR", code, "cpt_laterality",
                    f"CPT {code} requires a laterality modifier (RT/LT/50) per CMS bilateral-surgery "
                    f"indicator, but neither the modifiers array nor the laterality field states a side.",
                    "Confirm and add the correct laterality modifier (RT, LT, or 50)",
                    denial_risk="HIGH",
                )

    def _check_redundant_laterality(self, cpt, hcpcs):
        """Strip RT/LT from a line that already carries a site-specific digit
        modifier whose own AMA/CMS name states the same side (T5 'Right foot,
        great toe' makes RT redundant); flag a CONTRADICTION when the generic
        side and the digit modifier's side disagree.

        CMS/MAC billing guidance is to use the most specific site modifier —
        the T/F digit designators already encode side + digit, and carrier
        edits reject or ignore a generic RT/LT stacked on top of one. The
        side of each digit modifier comes from modifier_laterality (the
        modifier's own name in the reference data), never a hand-typed map.

        This is also a determinism layer: measured across independent runs
        of the same notes, RT flapped on/off next to a stable T5/T6 on five
        surgical lines — the LLM re-deciding per run a question that has
        exactly one data-driven answer (the digit modifier wins).
        """
        if self.store is None:
            return
        for entry in list(cpt) + list(hcpcs):
            code = entry.get("code", "")
            mods = entry.get("modifiers") or []
            digit_sides = {
                s for m in mods
                if str(m).strip().upper() not in ("RT", "LT")
                and (s := self.store.modifier_laterality(str(m)))
            }
            if len(digit_sides) != 1:
                continue  # no digit-side info, or digits span both sides
            digit_side = next(iter(digit_sides))
            for generic in [m for m in list(mods) if str(m).strip().upper() in ("RT", "LT")]:
                if str(generic).strip().upper() == digit_side:
                    mods.remove(generic)
                    self._add(
                        "INFO", code, "redundant_laterality_removed",
                        f"AUTO-CORRECTED: Removed {generic} from {code} — the line's own "
                        f"digit modifier already designates the {digit_side} side more "
                        f"specifically; CMS site-modifier guidance is to report the most "
                        f"specific designator, not both.",
                        f"Redundant {generic} removed; digit modifier retained",
                        denial_risk="LOW",
                    )
                else:
                    self._add(
                        "ERROR", code, "laterality_contradiction",
                        f"{code} carries {generic} but its digit modifier designates the "
                        f"{digit_side} side per the modifier's own AMA/CMS name — the two "
                        f"side designators on one line contradict each other.",
                        "Verify the documented site; correct either the RT/LT or the digit modifier",
                        denial_risk="HIGH",
                    )

    def _check_guidance_laterality(self, cpt):
        """Strip RT/LT from imaging-guidance lines — a guidance code is not a
        sided service. Both halves of that judgment come from CMS's own data:
        the code's descriptor states it is guidance ('Ultrasonic guidance for
        needle placement...', 'Fluoroscopic guidance...'), and its PFS
        bilateral-surgery indicator is 0 ('the concept of bilateral does not
        apply'). The anatomic site of the encounter is documented by the
        GUIDED procedure line's own modifiers, which carry it authoritatively.

        Determinism layer: measured across independent runs of the same
        notes, RT/LT flapped on/off on 76942 in three separate claims while
        the guided injection's own site modifier never moved — the LLM
        re-deciding per run a decoration CMS's indicators say means nothing
        on this line.

        The descriptor must BE a guidance service ('<modality> guidance
        for...' as its leading clause), not merely mention guidance: 0232T
        'Injection(s), platelet rich plasma, ... including image guidance'
        is a sided injection whose descriptor bundles the guidance in — a
        plain substring match stripped a legitimate RT from it live."""
        if self.store is None:
            return
        for entry in cpt:
            code = entry.get("code", "")
            desc = ((self.db.validate_cpt(code) or {}).get("long_description")
                    or (self.db.validate_cpt(code) or {}).get("short_description") or "").lower()
            if not re.match(r"[a-z() ]*\bguidance\b", desc) or self.store.bilat_surg(code) != "0":
                continue
            mods = entry.get("modifiers") or []
            stripped = [m for m in list(mods) if str(m).strip().upper() in ("RT", "LT")]
            if not stripped:
                continue
            for m in stripped:
                mods.remove(m)
            self._add(
                "INFO", code, "guidance_laterality_removed",
                f"AUTO-CORRECTED: Removed {'/'.join(str(m) for m in stripped)} from {code} — "
                f"its own descriptor states it is an imaging-guidance service and its CMS "
                f"bilateral-surgery indicator is 0 (bilateral concept does not apply); the "
                f"guided procedure line's own site modifiers document the anatomic side.",
                "No action needed; laterality is carried by the guided procedure line",
                denial_risk="LOW",
            )

    def _check_digit_supply_modifier(self, cpt, hcpcs, note_full_text: str = ""):
        """Enforce a supply descriptor's OWN modifier instruction: HCPCS
        descriptors like S8450 'Splint, prefabricated, digit (specify digit
        by use of modifier)' mandate a digit designator (TA/T1-T9, FA/F1-F9),
        not a generic RT/LT. The digit is derived from real claim/note data,
        never guessed:
          1. the unique digit modifier already on the claim's procedure
             lines (the splinted digit is the treated digit), else
          2. the unique digit the note itself names ('right hallux',
             'left second toe'), resolved to the modifier whose own AMA/CMS
             name matches ('Right foot, great toe' -> T5) — the same
             name-driven resolution modifier_laterality uses.

        Determinism layer: measured live, S8450 flapped RT/T5/T5 across
        independent runs of a note documenting a single 'right hallux'
        ulcer — one data-driven answer exists (T5), and the descriptor
        itself says a generic side is the wrong spelling of it. A side
        CONTRADICTION (supply says LT, derived digit is right-sided) is
        flagged, never silently rewritten."""
        if self.store is None:
            return
        applicable = [
            e for e in hcpcs
            if re.search(r"specify\s+digit", (((self.db.validate_hcpcs(e.get("code", "")) or {})
                         .get("long_description")
                         or (self.db.validate_hcpcs(e.get("code", "")) or {})
                         .get("description") or "")).lower())
        ]
        if not applicable:
            return

        def _is_digit_mod(m: str) -> bool:
            if m in ("RT", "LT", "50"):
                return False
            name = (self.store.modifier_name(m) or "").lower()
            return bool(self.store.modifier_laterality(m)) and (
                "digit" in name or "toe" in name or "finger" in name)

        claim_digits = {
            mu for c in cpt for m in c.get("modifiers") or []
            if _is_digit_mod(mu := str(m).strip().upper())
        }
        digit = claim_digits.pop() if len(claim_digits) == 1 else None

        if digit is None:
            # unique digit named by the note itself -> the modifier whose own
            # AMA/CMS name states that side + digit
            _, low_note = self._note_evidence(note_full_text or "")
            mentions = set()
            for side_word in ("right", "left"):
                for ordinal, spellings in DIGIT_ORDINALS.items():
                    alt = "|".join(spellings)
                    m = re.search(
                        rf"\b{side_word}\s+(?:(?:{alt})\s+(?P<noun>toe|digit)"
                        + (r"|(?P<hallux>hallux)" if ordinal == "great" else "")
                        + r")\b",
                        low_note,
                    )
                    if m:
                        # 'toe'/'hallux' pins the body part to foot — needed
                        # because 'fifth digit' names BOTH T9 (foot) and F9
                        # (hand) in the modifier reference
                        noun = m.groupdict().get("noun") or "toe"
                        mentions.add((side_word, ordinal,
                                      "foot" if noun in ("toe",) or m.groupdict().get("hallux")
                                      else None))
            if len(mentions) == 1:
                side_word, ordinal, part = mentions.pop()
                digit = self._digit_modifier_by_name(side_word, ordinal, part)

        for entry in applicable:
            code = entry.get("code", "")
            mods = entry.setdefault("modifiers", [])
            has_digit = any(_is_digit_mod(str(m).strip().upper()) for m in mods)
            if has_digit:
                continue
            generic = [m for m in list(mods) if str(m).strip().upper() in ("RT", "LT")]
            if digit is None:
                self._add(
                    "WARNING", code, "digit_modifier_required",
                    f"{code}'s own descriptor instructs 'specify digit by use of modifier', "
                    f"but the line carries {('only generic ' + '/'.join(str(m) for m in generic)) if generic else 'no digit designator'} "
                    f"and no single digit could be derived from the claim's procedure "
                    f"modifiers or the note text.",
                    "Add the digit modifier (TA/T1-T9) for the treated digit",
                    denial_risk="MEDIUM",
                )
                continue
            digit_side = self.store.modifier_laterality(digit)
            conflict = [m for m in generic if str(m).strip().upper() != digit_side]
            if conflict:
                self._add(
                    "ERROR", code, "digit_modifier_side_conflict",
                    f"{code} carries {'/'.join(str(m) for m in conflict)} but the treated digit "
                    f"resolves to {digit} ('{self.store.modifier_name(digit)}') — the two side "
                    f"designators contradict each other.",
                    "Verify the documented side; correct the supply's modifier to the treated digit",
                    denial_risk="HIGH",
                )
                continue
            for m in generic:
                mods.remove(m)
            mods.append(digit)
            self._add(
                "INFO", code, "digit_modifier_applied",
                f"AUTO-CORRECTED: {code}'s own descriptor instructs 'specify digit by use of "
                f"modifier' — set {digit} ('{self.store.modifier_name(digit)}')"
                + (f" replacing generic {'/'.join(str(m) for m in generic)}" if generic else "")
                + ", derived from the claim's own procedure modifiers/note documentation.",
                "Digit modifier applied per the supply descriptor's own instruction",
                denial_risk="LOW",
            )

    def _digit_modifier_by_name(self, side_word: str, ordinal: str,
                                part: str | None = None) -> str | None:
        """The unique modifier whose own AMA/CMS name contains the given side
        word and digit ordinal ('right' + 'great' -> T5 'Right foot, great
        toe'), optionally constrained to a body part word ('foot' — a note's
        'fifth toe' otherwise names both T9 'Right foot, fifth digit' and F9
        'Right hand, fifth digit'). None when zero or several match — never
        a guess."""
        if self.store is None:
            return None
        matches = set()
        for mod in self.store.anatomic_modifiers():
            name = (self.store.modifier_name(mod) or "").lower()
            if part and part not in name:
                continue
            if side_word in name and ordinal in name and (
                    "toe" in name or "digit" in name or "finger" in name):
                matches.add(mod)
        return matches.pop() if len(matches) == 1 else None

    def _check_cpt_digit_laterality(self, cpt, icd, note_full_text: str = ""):
        """Upgrade a procedure line's generic RT/LT to the specific digit
        modifier when the line's OWN linked diagnosis pins the procedure to
        a digit.         Two data-driven derivations, never a guess:
          1. the dx descriptor names the digit outright ('Contusion of right
             great toe...' → T5 'Right foot, great toe');
          2. the line is digit-scoped — its linked dx descriptor names the
             toe ('Cellulitis of right toe' L03.031) OR the CPT's own
             descriptor does ('Resection... phalangeal base, each toe'
             28126) — but not WHICH toe; then the note's UNIQUE same-side
             digit mention ('right great toe paronychia') supplies the
             ordinal. Lines billing multiple units of an 'each toe' code
             span several digits and are skipped.
        NCCI's anatomic-modifier guidance prefers the most specific site
        designator, and the claim's own dx-to-line linkage (or the code's
        own descriptor) restricts which lines are digit procedures at all.

        Determinism layer: measured live (note 002), an I&D of a toe abscess
        flapped between ['RT'] and ['T5', ...] across runs — the linked
        diagnosis (right-toe cellulitis) and the note's 'right great toe'
        were identical every time. Only fires when exactly ONE digit
        modifier is derivable AND its side agrees with the line's existing
        RT/LT; a side contradiction is flagged, never rewritten."""
        if self.store is None or self.db is None:
            return
        for entry in cpt:
            code = entry.get("code", "")
            if not code or _is_em(code):
                continue
            mods = entry.setdefault("modifiers", [])
            norm_mods = [str(m).strip().upper() for m in mods]
            generic = [m for m in norm_mods if m in ("RT", "LT")]
            # zero RT/LT is handled too (the add-missing arm below): a
            # digit-scoped line with NO side designator at all flapped
            # against its sided twin across runs (note 004, 11730 ['RT']
            # vs []) — both shapes must converge on the same digit
            # modifier, not just the RT-carrying one.
            if len(generic) > 1:
                continue
            has_digit = any(
                (self.store.modifier_name(m) or "").lower()
                and self.store.modifier_laterality(m)
                and any(w in (self.store.modifier_name(m) or "").lower()
                        for w in ("digit", "toe", "finger"))
                for m in norm_mods if m not in ("RT", "LT", "50")
            )
            if has_digit:
                continue

            digits = set()
            digit_scoped = False
            sides_from_dx = set()
            sides_any_dx = set()
            own_desc = ((self.db.validate_cpt(code) or {})
                        .get("long_description", "")).lower()
            # Nail-structure procedures are digit procedures too (a nail is
            # on exactly one digit) — but only single-site descriptors:
            # multi-nail codes ('trimming ... any number', 'debridement of
            # nail(s) ... 1 to 5') span digits and take no digit modifier.
            # 'phalan\w*' not 'phalang\w*': the singular is 'phalanx' —
            # 'phalang' misses it (measured live, note 010: 20240 'Biopsy,
            # bone ... phalanx' was treated as not digit-scoped)
            if (re.search(r"\b(toe|toes|hallux|phalan\w*|\w*ungual|nail)\b",
                          own_desc)
                    and not re.search(
                        r"\b(?:any number|\d+\s+to\s+\d+|\d+\s+or\s+more)\b",
                        own_desc)
                    and int(entry.get("units") or 1) == 1):
                digit_scoped = True
            for dx in entry.get("linked_diagnoses") or []:
                info = self.db.validate_icd10(str(dx))
                desc = (info or {}).get("description", "").lower()
                if not desc:
                    continue
                for side_word in ("right", "left"):
                    if re.search(rf"\b{side_word}\b", desc):
                        sides_any_dx.add(side_word)
                if not re.search(r"\b(toe|toes|hallux)\b", desc):
                    continue
                digit_scoped = True
                for side_word in ("right", "left"):
                    if not re.search(rf"\b{side_word}\b", desc):
                        continue
                    sides_from_dx.add(side_word)
                    for ordinal, spellings in DIGIT_ORDINALS.items():
                        alt = "|".join(spellings)
                        if re.search(
                                rf"\b(?:(?:{alt})\s+(?:toe|digit)"
                                + (r"|hallux" if ordinal == "great" else "")
                                + r")\b", desc):
                            d = self._digit_modifier_by_name(side_word, ordinal, "foot")
                            if d:
                                digits.add(d)
            if not digit_scoped:
                continue
            if not digits and note_full_text:
                # dx pins side + toe-scope but not WHICH toe — the note's
                # unique same-side digit mention resolves the ordinal. With
                # no RT/LT on the line, the side comes from the linked
                # diagnoses' own descriptors instead (and must be unique).
                if generic:
                    side_words = [{"RT": "right", "LT": "left"}[generic[0]]]
                elif len(sides_from_dx) == 1:
                    side_words = list(sides_from_dx)
                elif len(sides_any_dx) == 1:
                    # no RT/LT and no toe-naming dx — the side comes from
                    # the linked diagnoses' own descriptors when they agree
                    # on one ('...right foot' L97.511)
                    side_words = list(sides_any_dx)
                else:
                    side_words = []
                for side_word in side_words:
                    if sides_from_dx and side_word not in sides_from_dx:
                        continue
                    _, low_note = self._note_evidence(note_full_text)
                    ordinals = set()
                    for ordinal, spellings in DIGIT_ORDINALS.items():
                        alt = "|".join(spellings)
                        if re.search(
                                rf"\b{side_word}\s+(?:(?:{alt})\s+(?:toe|digit)"
                                + (r"|hallux" if ordinal == "great" else "")
                                + r")\b", low_note):
                            ordinals.add(ordinal)
                    if len(ordinals) == 1:
                        d = self._digit_modifier_by_name(side_word, ordinals.pop(), "foot")
                        if d:
                            digits.add(d)
            if len(digits) != 1:
                continue
            digit = digits.pop()
            digit_side = self.store.modifier_laterality(digit)
            if generic and generic[0] != digit_side:
                self._add(
                    "ERROR", code, "digit_modifier_side_conflict",
                    f"{code} carries {generic[0]} but its own linked diagnosis names the "
                    f"{self.store.modifier_name(digit)} ({digit}) — the side designators "
                    f"contradict each other.",
                    "Verify the documented side; align the line's modifier with the linked diagnosis",
                    denial_risk="HIGH",
                )
                continue
            for m in list(mods):
                if str(m).strip().upper() in ("RT", "LT"):
                    mods.remove(m)
            mods.append(digit)
            origin = (f"generic {generic[0]} upgraded to {digit}" if generic
                      else f"missing site designator supplied as {digit}")
            self._add(
                "INFO", code, "digit_modifier_applied",
                f"AUTO-CORRECTED: {code}'s {origin} "
                f"('{self.store.modifier_name(digit)}') — the line's own linked diagnosis "
                f"names that digit, and NCCI prefers the most specific anatomic designator.",
                "Digit modifier derived from the line's own linked diagnosis",
                denial_risk="LOW",
            )

    def _digit_site_mods(self, mods) -> set:
        """The exact digit-site designators (great/lesser toe, finger) among
        a line's modifiers — identified by the modifier's OWN AMA/CMS name
        (has a laterality AND names a digit), never a hardcoded list."""
        if self.store is None:
            return set()
        out = set()
        for m in mods or []:
            mm = str(m).strip().upper()
            name = (self.store.modifier_name(mm) or "").lower()
            # AMA names spell great toes as 'toe' and lesser ones as
            # 'digit' ('Right foot, second digit' T6); hands use 'digit'
            # and 'thumb'
            if (self.store.modifier_laterality(mm)
                    and re.search(r"\b(toe|finger|thumb|digit)\b", name)):
                out.add(mm)
        return out

    def _sites_distinct(self, sites_a: set, sites_b: set) -> bool:
        """Do two lines' anatomic-modifier sets document genuinely DIFFERENT
        sites? NCCI separation requires a different site, and a generic side
        designator is not a site point: RT/LT names a whole side, and every
        specific site modifier on that same side lies WITHIN it. Measured
        live (note 010): a bone biopsy carrying LT was treated as a
        different site from the same toe's phalangectomy carrying TA
        ('Left foot, great toe'), bypassing the PTP edit — LT vs TA on the
        same side distinguishes nothing."""
        if not sites_a or not sites_b or sites_a == sites_b:
            return False
        generic = {"RT", "LT"}
        spec_a, spec_b = sites_a - generic, sites_b - generic
        if spec_a and spec_b:
            return spec_a != spec_b
        if not spec_a and not spec_b:
            return True  # {RT} vs {LT}
        gen_set, spec_set = (sites_a, spec_b) if not spec_a else (sites_b, spec_a)
        gen_sides = gen_set & generic
        spec_sides = {self.store.modifier_laterality(m) for m in spec_set
                      if self.store is not None
                      and self.store.modifier_laterality(m)}
        return bool(spec_sides) and not (gen_sides & spec_sides)

    def _check_same_site_ptp_bundling(self, cpt):
        """An indicator-1 NCCI PTP pair where both lines' anatomic modifiers
        fail to designate DISTINCT sites: NCCI Policy Manual Ch.1 allows
        59/X{EPSU} only for a different session, site/organ, or separate
        injury, and the claim's own anatomic designators are its only
        site documentation — identical digit modifiers (T5 vs T5) or a
        specific digit lying within the other line's same-side generic
        (T5 vs RT) document the same operative field, exactly the
        _sites_distinct standard the NCCI bypass credit above already
        applies. A generic-only pair (RT vs RT) is deliberately excluded:
        a whole side has many sites, so a matching generic proves nothing
        either way. The column-2 line (the PTP table fixes which) is
        therefore not separately billable and is removed, regardless of
        any 59 the model appended.

        Determinism layer: measured live — note 005 flapped 29550 (strapping,
        T6+59) against 11740 (T6), note 006 flapped 11730/11740 (both TA)
        with a wandering 59, and note 004 flapped 29550 (T5+59) against
        97597 (RT, same right hallux); in every case the pair, the
        indicator, and the non-distinct site modifiers were claim facts
        present in all runs."""
        if self.db is None or self.store is None:
            return
        anatomic = self.store.anatomic_modifiers()
        for entry in cpt:
            code = entry.get("code", "")
            if (not code or _is_em(code)
                    or code in self._non_billable_codes_to_suppress):
                continue
            if int(entry.get("units") or 1) != 1:
                continue  # multi-unit lines can span several digits
            sites = {str(m).strip().upper()
                     for m in (entry.get("modifiers") or [])} & anatomic
            if not sites:
                continue
            for other in cpt:
                other_code = other.get("code", "")
                if (other is entry or not other_code or other_code == code
                        or _is_em(other_code)
                        or other_code in self._non_billable_codes_to_suppress):
                    continue
                if int(other.get("units") or 1) != 1:
                    continue
                other_sites = {str(m).strip().upper()
                               for m in (other.get("modifiers") or [])} & anatomic
                if not other_sites or self._sites_distinct(sites, other_sites):
                    continue
                if not (sites - {"RT", "LT"}) and not (other_sites - {"RT", "LT"}):
                    continue  # generic-only pair proves nothing about the site
                conflict = self.db.check_ncci(code, other_code)
                if (not conflict
                        or str(conflict.get("modifier", "")).strip() != "1"):
                    continue
                if conflict.get("code2") != code.strip().upper():
                    continue  # only the column-2 line bundles
                site = ", ".join(sorted(sites | other_sites))
                self._non_billable_codes_to_suppress.add(code)
                self._add(
                    "WARNING", code, "same_site_ptp_bundled",
                    f"AUTO-CORRECTED: {code} removed — it is the column-2 "
                    f"code of an NCCI PTP edit with {other_code}, and the two "
                    f"lines' anatomic modifiers do not designate distinct sites "
                    f"({site}). NCCI Policy Manual Ch.1 permits a separation "
                    f"modifier only for a different session, site, or injury; "
                    f"the claim's own anatomic designators do not document "
                    f"distinct sites, so the service is included in "
                    f"{other_code}.",
                    f"Bill only {other_code} for same-site same-session work; "
                    f"if the services were truly at different sites, correct "
                    f"the digit modifiers instead",
                    denial_risk="HIGH",
                )
                break

    def _check_digit_modifier_scope(self, cpt):
        """Mirror image of _check_cpt_digit_laterality: that layer UPGRADES
        a generic RT/LT to the digit designator when the line's own linked
        diagnosis or descriptor pins a digit; this one DOWNGRADES a digit
        modifier to the plain side modifier when nothing on the line is
        digit-scoped — the CPT descriptor names no digit/nail structure and
        no linked diagnosis does either. The claim's own adjudicable record
        (procedure descriptor + diagnosis linkage) is then foot-level, and
        a toe designator overstates it.

        Determinism layer: measured live (note 004), 97597 (open-wound
        debridement, linked to L97.5- 'ulcer of other part of FOOT')
        flapped between [RT] and [T5] across runs — descriptor and linkage
        are foot-level claim facts identical in every run."""
        if self.db is None or self.store is None:
            return
        scope_re = re.compile(
            r"\b(toe|toes|hallux|phalan\w*|digit\w*|\w*ungual|nail\w*)\b")
        for entry in cpt:
            code = entry.get("code", "")
            if not code or _is_em(code):
                continue
            mods = entry.setdefault("modifiers", [])
            digits = self._digit_site_mods(mods)
            if len(digits) != 1:
                continue
            own = ((self.db.validate_cpt(code) or {})
                   .get("long_description", "")).lower()
            if scope_re.search(own):
                continue  # the code itself is a digit/nail procedure
            dx_scoped = False
            for dx in entry.get("linked_diagnoses") or []:
                desc = ((self.db.validate_icd10(str(dx)) or {})
                        .get("description", "")).lower()
                if desc and scope_re.search(desc):
                    dx_scoped = True
                    break
            if dx_scoped:
                continue
            digit = digits.pop()
            side = self.store.modifier_laterality(digit)
            if not side:
                continue
            norm_mods = [str(m).strip().upper() for m in mods]
            new_mods = [m for m in mods
                        if str(m).strip().upper() != digit]
            if side not in norm_mods:
                new_mods.append(side)
            entry["modifiers"] = new_mods
            entry["needs_review"] = True
            self._add(
                "INFO", code, "digit_modifier_descoped",
                f"AUTO-CORRECTED: {code}'s {digit} "
                f"('{self.store.modifier_name(digit)}') normalized to {side} "
                f"— neither the procedure's own descriptor nor any linked "
                f"diagnosis names a digit or nail structure, so the claim's "
                f"adjudicable record supports side-level, not digit-level, "
                f"specificity. NCCI expects the anatomic modifier to match "
                f"the site the claim itself documents.",
                "If the service was digit-specific, link a digit-level "
                "diagnosis and re-bill with the digit modifier",
                denial_risk="LOW",
            )

    def _check_separation_modifier_placement(self, cpt):
        """Move a separation modifier (59/X{EPSU}) to the COLUMN-2 line of
        the NCCI PTP pair that needs it — CMS's own instruction (NCCI Policy
        Manual Ch. 1 §E: 'the modifier should be appended to the Column Two
        code'). The PTP table fixes which code of a pair is column 1 vs
        column 2, so placement is a claim fact, not a judgment.

        Determinism layer: measured live (note 006), independent runs all
        billed 11730+11740 with one separation modifier but flapped on WHICH
        line carried it — same pair, same table, two spellings. Runs before
        _check_unnecessary_separation_modifier, which then strips whatever
        placement left unnecessary."""
        if self.db is None:
            return
        sep_set = {"59", "XE", "XS", "XP", "XU"}
        anatomic = self.store.anatomic_modifiers() if self.store is not None else set()

        def _seps(mods):
            return [m for m in mods if str(m).strip().upper() in sep_set]

        for entry in cpt:
            code = entry.get("code", "")
            mods = entry.get("modifiers") or []
            seps = _seps(mods)
            if not code or not seps or _is_em(code):
                continue
            sites_self = {str(m).strip().upper() for m in mods} & anatomic
            norm_code = code.strip().upper().replace(".", "")

            def _live_pairs():
                """(other, conflict) for every indicator-1, non-bypassed PTP
                pair this line forms with another claim line."""
                for other in cpt:
                    other_code = other.get("code", "")
                    if other is entry or not other_code or other_code == code:
                        continue
                    if _is_em(other_code):
                        continue
                    conflict = self.db.check_ncci(code, other_code)
                    if not conflict or str(conflict.get("modifier", "")).strip() != "1":
                        continue
                    sites_other = {str(m).strip().upper()
                                   for m in other.get("modifiers") or []} & anatomic
                    if self._sites_distinct(sites_self, sites_other):
                        continue  # anatomically bypassed — the strip check owns it
                    yield other, conflict

            # if this line is the COLUMN-2 code of ANY live pair, its
            # separation modifier is already where CMS wants it — leave it
            if any(c.get("code2") == norm_code for _, c in _live_pairs()):
                continue
            other = next((o for o, c in _live_pairs()
                          if c.get("code1") == norm_code), None)
            if other is None:
                continue
            other_code = other.get("code", "")
            other_mods = other.setdefault("modifiers", [])
            if not _seps(other_mods):
                other_mods.append(seps[0])
            for s in seps:
                mods.remove(s)
            self._add(
                "INFO", code, "separation_modifier_moved",
                f"AUTO-CORRECTED: Moved {'/'.join(seps)} from {code} to {other_code} — "
                f"the NCCI PTP table lists {code} as the column-1 (comprehensive) code "
                f"and {other_code} as column 2, and the NCCI manual's own instruction "
                f"is to append the separation modifier to the column-2 code.",
                "No action needed; separation modifier moved to the column-2 line per NCCI",
                denial_risk="LOW",
            )

    def _check_unnecessary_separation_modifier(self, cpt):
        """Remove 59/XE/XS/XP/XU from lines where it can do nothing — NCCI
        Policy Manual Ch. 1: modifier 59 (and the X{EPSU} refinements) exists
        solely to bypass a PTP edit, must not be appended when no edit
        requires it, and 'should not be used when a more descriptive modifier
        is available' (the anatomic site modifiers CMS itself associates with
        PTP bypass).

        A separation modifier is kept only when some OTHER line on the claim
        forms an indicator-1 PTP pair with this one that is NOT already
        bypassed by differing anatomic-modifier sets. Everything else is a
        deterministic strip:
          * no PTP edit at all with any other line → 59 is decoration;
          * indicator-0 pair → the edit cannot be bypassed by any modifier;
          * indicator-1 pair already separated by anatomic modifiers → the
            site modifiers are the CMS-preferred documentation of
            distinctness, and 59 on top is the exact misuse the manual
            calls out.

        Determinism layer: measured across independent runs, 59 flapped
        on/off on lines whose PTP relationships never changed — the PTP
        table, not the LLM's per-run mood, decides where 59 belongs.
        """
        if self.db is None:
            return
        sep_set = {"59", "XE", "XS", "XP", "XU"}
        anatomic = self.store.anatomic_modifiers() if self.store is not None else set()
        for entry in cpt:
            code = entry.get("code", "")
            mods = entry.get("modifiers") or []
            present_seps = [m for m in mods if str(m).strip().upper() in sep_set]
            if not code or not present_seps:
                continue
            sites_self = {str(m).strip().upper() for m in mods} & anatomic
            needed_for = None
            anatomic_bypassed_pair = None
            # E/M-involved pairs are separated by 25/57, never 59 (same
            # split _check_ncci applies) — an E/M line's 59 is always
            # stripped, and an E/M partner is never a reason to keep one.
            if not _is_em(code):
                for other in cpt:
                    other_code = other.get("code", "")
                    if other is entry or not other_code or other_code == code:
                        continue
                    if _is_em(other_code):
                        continue
                    conflict = self.db.check_ncci(code, other_code)
                    if not conflict or str(conflict.get("modifier", "")).strip() != "1":
                        continue
                    sites_other = {str(m).strip().upper()
                                   for m in other.get("modifiers") or []} & anatomic
                    if self._sites_distinct(sites_self, sites_other):
                        # already separated by CMS anatomic modifiers
                        anatomic_bypassed_pair = other_code
                        continue
                    needed_for = other_code
                    break
            if needed_for:
                continue
            for sep in present_seps:
                mods.remove(sep)
            why = (
                f"its indicator-1 PTP pair with {anatomic_bypassed_pair} is already "
                f"separated by the two lines' differing anatomic site modifiers — the "
                f"NCCI manual's own instruction is to use the more descriptive site "
                f"modifier, not 59, in that case"
                if anatomic_bypassed_pair else
                "no indicator-1 NCCI PTP edit with any other line on this claim "
                "requires a separation modifier here"
            )
            self._add(
                "INFO", code, "unnecessary_separation_modifier_removed",
                f"AUTO-CORRECTED: Removed {'/'.join(present_seps)} from {code} — {why}.",
                "No action needed; separation modifier was unnecessary per the PTP table",
                denial_risk="LOW",
            )

    def _check_modifier_reasoning_consistency(self, cpt, hcpcs):
        """The coder's modifier_reasoning is a structured list of
        {modifier, status, reason} claims (see schemas.py's ModifierClaim)
        — it must stay consistent with what's actually billed. A claim
        with status="applied" for a modifier missing from the array, or
        status="not_applicable" for a modifier still present, is the
        LLM's own output contradicting itself, not a coding judgment
        call, so both directions are auto-corrected here.

        This replaces a regex-based free-text parser (cue words, negation
        windows) that had to guess a sentence's polarity from prose — that
        guessing kept finding new failure modes as new phrasings appeared
        (a removed/unnecessary modifier wrongly read as present; later a
        retained modifier wrongly read as absent; an unrelated clinical
        negation elsewhere in the sentence wrongly vetoing a real claim).
        status is now an explicit field, not inferred from text, so there
        is nothing left to parse.
        """
        for entry in list(cpt) + list(hcpcs):
            code = entry.get("code", "")
            mods = entry.setdefault("modifiers", [])
            for claim in entry.get("modifier_reasoning", []) or []:
                if not isinstance(claim, dict):
                    continue
                modifier = str(claim.get("modifier", "")).strip().upper()
                status = str(claim.get("status", "")).strip().lower()
                reason = claim.get("reason", "")
                if not modifier:
                    continue
                if status == "applied" and modifier not in mods:
                    if self.store is not None and not self.store.modifier_valid(modifier):
                        continue  # don't inject an unrecognized modifier code
                    mods.append(modifier)
                    self._add(
                        "INFO", code, "modifier_reasoning_mismatch",
                        f"AUTO-CORRECTED: Added modifier -{modifier} to {code} — the coder's own "
                        f"reasoning claims it applied (\"{reason}\") but it was missing from the "
                        f"modifiers array.",
                        f"Modifier -{modifier} auto-added to match stated reasoning",
                        denial_risk="MEDIUM",
                    )
                elif status == "not_applicable" and modifier in mods:
                    mods.remove(modifier)
                    self._add(
                        "INFO", code, "modifier_reasoning_mismatch",
                        f"AUTO-CORRECTED: Removed modifier -{modifier} from {code} — the coder's "
                        f"own reasoning states it is not applicable (\"{reason}\") but it was "
                        f"still present in the modifiers array.",
                        f"Modifier -{modifier} auto-removed to match stated reasoning",
                        denial_risk="MEDIUM",
                    )

    # A Tabular note that names a measurement gives its clinical acronym in
    # parens: 'code to identify body mass index (BMI), if known...'. The
    # acronym is how the note text will reference the value ('BMI 36.2').
    _MEASURE_ACRONYM_RE = re.compile(r"\(([A-Za-z]{2,6})\)")
    _DESC_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    _DESC_FLOOR_RE = re.compile(r"(\d+(?:\.\d+)?)\s+or\s+greater", re.IGNORECASE)

    def _check_measurement_companion(self, icd, note_full_text: str):
        """Replaces the old hardcoded obesity→BMI check with zero code
        literals: when a billed diagnosis's own Tabular 'use additional
        code' note names a measurement by acronym ('...body mass index
        (BMI), if known (Z68.-)') and the note documents a value ('BMI
        36.2'), the companion is required — and the exact code is selected
        by parsing the numeric range each candidate's own description
        carries ('Body mass index [BMI] 36.0-36.9, adult'). Every element —
        the carrier/companion relationship, the acronym, the candidate
        codes, their ranges — comes from the reference data."""
        if self.store is None or not note_full_text:
            return
        present = {c.get("code", "").replace(".", "").upper() for c in icd if c.get("code")}
        low = note_full_text.lower()
        handled: set[str] = set()
        for entry in icd:
            code = entry.get("code", "")
            if not code:
                continue
            for carrier, refs in self.store.use_additional_code_groups(code):
                if carrier in handled:
                    continue
                handled.add(carrier)
                # A measurement note's refs are the RANGE ENDPOINTS of the
                # scale ('(Z68.1-Z68.45)' parses to Z681 and Z6845): the real
                # candidate set is every billable code under the endpoints'
                # shared category stem, and a claim satisfies the note if it
                # carries ANY code from that category.
                ref_codes = [r for r, _ in refs]
                stem = os.path.commonprefix(ref_codes) if ref_codes else ""
                if len(stem) < 3:
                    continue
                if any(p.startswith(stem) for p in present):
                    continue  # a code from the measurement scale is already on the claim
                for ref, note_line in refs:
                    acronyms = self._MEASURE_ACRONYM_RE.findall(note_line or "")
                    if not acronyms:
                        continue
                    value = None
                    for acr in acronyms:
                        m = re.search(
                            rf"\b{re.escape(acr.lower())}\b[^0-9%]{{0,20}}(\d+(?:\.\d+)?)(?!\s*%)",
                            low)
                        if m:
                            value = float(m.group(1))
                            break
                    if value is None:
                        continue  # 'if known' — value not documented, silence is correct
                    exact = None
                    for cand, desc in self.store.icd10_billable_under(stem):
                        rng = self._DESC_RANGE_RE.search(desc or "")
                        if rng and float(rng.group(1)) <= value <= float(rng.group(2)):
                            exact = (self._redot_icd(cand), desc)
                            break
                        floor = self._DESC_FLOOR_RE.search(desc or "")
                        if floor and value >= float(floor.group(1)):
                            exact = (self._redot_icd(cand), desc)
                            break
                    target = exact[0] if exact else self._redot_icd(ref)
                    detail = (f" — the documented value {value:g} falls in {exact[0]}'s own "
                              f"range ('{exact[1]}')" if exact else "")
                    self._add(
                        "WARNING", target, "measurement_companion_missing",
                        f"{code}'s Tabular List entry instructs 'use additional {note_line.strip()}' "
                        f"and the note documents the value ({acronyms[0]} {value:g}), but no "
                        f"matching code is on the claim{detail}.",
                        f"Add {target} to the billed diagnoses per the use-additional-code "
                        f"instruction",
                        denial_risk="LOW",
                    )
                    break

    def _check_redundant_dm_codes(self, icd):
        """Flag a DM 'without complications'/remission code alongside a more
        specific DM combination code. "Generic" vs "specific" is derived
        from each code's own real ICD-10-CM description (icd10cm_codes.json)
        rather than a hardcoded prefix list: every diabetes category's
        unspecified/remission code is literally described as "...diabetes
        mellitus without complications" (verified against all 260 real
        E10/E11/E13 codes — exactly 4 match: E10.9, E11.9, E11.A, E13.9;
        E11.A was missing from the old hardcoded 3-code tuple entirely, even
        though it's the same kind of "no active complications" state as
        E11.9), and every specific complication code's description contains
        "with" (verified: 100% of the other 256 codes).
        """
        if self.db is None:
            return
        generic_code = None
        has_specific = False
        for c in icd:
            code = c.get("code", "")
            info = self.db.validate_icd10(code)
            if not info:
                continue
            desc = (info.get("description") or "").lower()
            if "diabetes mellitus" not in desc:
                continue
            if "without complications" in desc:
                generic_code = code
            elif "with" in desc:
                has_specific = True

        if generic_code is not None and has_specific:
            self._add(
                "ERROR", generic_code, "redundant_dm_code",
                f"{generic_code} (unspecified/no-active-complications DM) coded alongside a more "
                f"specific DM combination code. Per ICD-10-CM guidelines, do not code both — the "
                f"combination code captures the DM.",
                f"Remove {generic_code} — the specific DM combination code already captures the diabetes",
                denial_risk="MEDIUM",
            )

    def _check_unjustified_zcodes(self, icd):
        """Flag a Z79.x (long-term drug therapy) code with no real linkage
        to any other coded condition on the claim, via ICD-10-CM's own
        'use additional code' guidance (icd10cm_instructional_notes.json,
        parsed from CDC/NCHS's icd10cm-tabular-2026.xml).

        Replaces a prior hardcoded 4-code "inappropriate for podiatry" list.
        That premise doesn't survive contact with real data: Z79.84 (long-
        term oral hypoglycemics) is literally the CDC/CMS-recommended
        'use additional code' companion for E11.x (type 2 diabetes) — the
        old check would have flagged the single most common, clinically
        correct Z79/diabetes pairing as "inappropriate." The real,
        generalizable compliance concern isn't any specific Z-code value,
        it's an ORPHANED Z-code with no supporting condition on the claim —
        Z79 (all 32 codes, "Long term (current) drug therapy") is a real,
        stable CMS structural category, not a proxy code list.
        """
        if self.store is None:
            return
        codes = [c.get("code", "") for c in icd]
        z79_codes = [c for c in codes if c.replace(".", "").upper().startswith("Z79")]
        if not z79_codes:
            return

        recommended_refs = set()
        for c in codes:
            recommended_refs.update(self.store.use_additional_code_refs(c))

        for z in z79_codes:
            znorm = z.replace(".", "").upper()
            justified = any(
                znorm.startswith(ref) or ref.startswith(znorm)
                for ref in recommended_refs
            )
            if not justified:
                self._add(
                    "WARNING", z, "unjustified_zcode",
                    f"{z} (long-term drug therapy) has no linked condition code on this claim "
                    f"recommending it, per ICD-10-CM's 'use additional code' guidance.",
                    f"Verify {z} is clinically supported, or link it to the condition it treats",
                    denial_risk="LOW",
                )

    # Generic-language filter for documentation-evidence matching: common
    # English function words plus the boilerplate vocabulary that appears in
    # thousands of ICD-10-CM descriptions ("other specified", "unspecified",
    # "classified elsewhere", "personal history of", ...). Matching on these
    # would connect virtually any description to any note. Linguistic filler,
    # not a medical code list — the codes themselves all come from the store.
    _EVIDENCE_STOPWORDS = frozenset("""
        about after against agent agents applicable associated because before
        both care chronic classified code codes complicating condition
        conditions current disease diseases disorder disorders drugs during
        elsewhere encounter episode factors following history identify
        including involving level long manifestations other patient personal
        presence problems related right left specified status syndrome term
        therapy through under underlying unspecified using which with without
    """.split())

    @classmethod
    def _distinctive_terms(cls, text: str) -> set[str]:
        """Clinically distinctive words of a code description — the terms
        whose presence in a note is real evidence the condition is
        documented (e.g. 'dialysis', 'insulin', 'tobacco'), after dropping
        generic coding/English filler."""
        words = re.findall(r"[a-z]+", (text or "").lower())
        return {w for w in words if len(w) >= 5 and w not in cls._EVIDENCE_STOPWORDS}

    @classmethod
    def _documented(cls, note_words: set[str], *texts: str) -> set[str] | None:
        """Matched terms if ANY of the given texts passes the evidence bar
        against the note: a strict majority of its distinctive terms must
        appear. Single-word overlap misfires ('renal' alone matched a note
        explicitly documenting NO renal replacement; 'kidney' alone matched
        plain CKD with no transplant), and requiring every term misses real
        documentation ('ESRD on dialysis' vs Z99.2's full 'Dependence on
        renal dialysis'). Multiple texts because the Tabular gives two
        vocabularies for the same condition — the ref code's formal
        description AND the instructional note line's clinical wording
        ('dialysis status') — and the note only needs to speak one of them."""
        for text in texts:
            terms = cls._distinctive_terms(text)
            matched = terms & note_words
            if terms and len(matched) * 2 > len(terms):
                return matched
        return None

    @staticmethod
    def _redot_icd(code: str) -> str:
        """Normalized (dotless) ICD-10-CM code back to display format."""
        c = code.replace(".", "").strip().upper()
        return f"{c[:3]}.{c[3:]}" if len(c) > 3 else c

    def _check_missing_use_additional_code(self, icd, coding_result, note_full_text):
        """The mirror image of _check_unjustified_zcodes: that check asks
        'is this companion code justified by a condition on the claim?';
        this one asks 'is a REQUIRED companion code missing?'. Found live:
        N18.6 (ESRD) billed for a patient documented as 'ESRD on dialysis'
        with no Z99.2 (dependence on renal dialysis) anywhere on the claim —
        N18.6's own Tabular List entry instructs 'use additional code to
        identify dialysis status (Z99.2)'.

        Fully data-driven: the companion refs come from the billed code's
        real useAdditionalCode notes (CDC/NCHS Tabular XML), candidate codes
        from the billable code set, and the documentation gate from matching
        the candidate's own description terms against the note text. Many
        useAdditionalCode notes are conditional ('if applicable', 'if
        known'), so a missing companion is only flagged when the note
        actually documents it — silence is correct when the condition isn't
        documented.

        When the evidence-backed candidate is a single specific billable
        code, it is auto-added to the BILLED diagnosis list (secondary,
        flagged for review) — not to advisory supporting_conditions. The
        Tabular instruction is 'use additional CODE': a companion the
        instruction mandates and the note documents belongs on the claim,
        and parking it in an advisory array left the claim non-compliant
        with the instruction this check exists to enforce. Determinism
        layer too, measured live: Z79.84 under an E11.x carrier flapped in/
        out of icd_codes across independent runs — promotion lands every
        run on the same instruction-compliant array."""
        if self.store is None or not note_full_text:
            return
        note_words = set(re.findall(r"[a-z]+", note_full_text.lower()))
        supporting = coding_result.get("supporting_conditions") or []
        present = {
            c.get("code", "").replace(".", "").upper()
            for c in list(icd) + list(supporting)
            if c.get("code")
        }
        handled_carriers: set[str] = set()
        for entry in icd:
            code = entry.get("code", "")
            if not code:
                continue
            for carrier, refs in self.store.use_additional_code_groups(code):
                if carrier in handled_carriers:
                    continue
                handled_carriers.add(carrier)
                # a note's refs are alternatives — any one present satisfies it
                if any(p.startswith(r) or r.startswith(p) for r, _ in refs for p in present):
                    continue
                # Evidence bar (see _documented): strict-majority term match
                # against the candidate's description OR the Tabular note
                # line's own wording for the ref ('dialysis status'). Category
                # refs (N18 -> E88 'family of disorders') additionally need
                # >=2 matched terms and are flagged for the coder rather than
                # auto-added — picking one child of a category off a single
                # shared word misfired live (E88.811 'Insulin resistance
                # syndrome' matched a med list's 'insulin' that actually
                # evidences Z79.4).
                candidates = []
                for ref, note_line in refs:
                    if code.replace(".", "").upper().startswith(ref):
                        continue  # self-referential ref — nothing to add
                    billable = self.store.icd10_billable_under(ref)
                    ref_is_exact = bool(billable) and billable[0][0] == ref
                    for cand, desc in billable:
                        text = desc or self.store.icd10_tabular_description(cand)
                        is_exact = ref_is_exact and cand == ref
                        # the note line describes the REF's condition — only
                        # valid evidence for the exact ref code, not for
                        # picking one child out of a category
                        matched = (self._documented(note_words, text, note_line)
                                   if is_exact else self._documented(note_words, text))
                        via_acro = False
                        if not matched or (not is_exact and len(matched) < 2):
                            # Official-acronym arm: the Tabular's own
                            # inclusion terms spell the acronym in
                            # parentheses ('Methicillin resistant
                            # staphylococcus aureus (MRSA) infection...') —
                            # a note speaking the acronym documents the full
                            # phrase. Measured live (note 010): 'MRSA as
                            # cause of disease' in the assessment never
                            # majority-matched B95.62's five spelled-out
                            # terms, so the mandated companion flapped with
                            # the LLM instead of being added every run.
                            acro_hit = next(
                                (a for term in
                                 self.store.icd10_inclusion_terms(cand)
                                 for a in re.findall(r"\(([A-Z]{3,10})\)", term)
                                 if a.lower() in note_words), None)
                            if acro_hit is None:
                                continue
                            matched, via_acro = {acro_hit.lower()}, True
                        candidates.append((cand, text, matched, is_exact, via_acro))
                if not candidates:
                    continue
                best = max(candidates, key=lambda t: (t[3], len(t[2]), -len(t[0])))
                best_code = self._redot_icd(best[0])
                evidence = ", ".join(sorted(best[2]))
                # An acronym match is an exact official synonym for ONE
                # specific child — unlike scattered-word matches, it is safe
                # to auto-add from a category-level ref, but only when it
                # selects a single candidate (a shared ancestor acronym
                # describing several siblings decides nothing).
                acro_unique = best[4] and sum(1 for c in candidates if c[4]) == 1
                if best[3] or acro_unique:  # Tabular-designated or unique official synonym — auto-add
                    icd.append({
                        "code": best_code,
                        "description": best[1],
                        "type": "secondary",
                        "rationale": (
                            f"AUTO-ADDED: {self._redot_icd(carrier)}'s ICD-10-CM Tabular List entry "
                            f"instructs 'use additional code' referencing {best_code}, the note "
                            f"documents it ('{evidence}'), and no code on the claim satisfies the "
                            f"instruction."
                        ),
                        "source_section": "validator:use_additional_code",
                        "needs_review": True,
                        "review_reason": f"Confirm {best_code}, added per the Tabular's use-additional-code instruction",
                    })
                    present.add(best[0])
                    action = f"added {best_code} to the billed diagnoses (secondary) for review"
                else:
                    action = f"best documented candidate is {best_code} — not auto-added (category-level ref)"
                self._add(
                    "WARNING", best_code, "missing_use_additional_code",
                    f"{code}'s Tabular List entry ({self._redot_icd(carrier)}) instructs 'use "
                    f"additional code' and the note documents the companion condition "
                    f"('{evidence}'), but no matching code is on the claim — {action}.",
                    f"Confirm and add {best_code} to the billed diagnoses per the "
                    f"use-additional-code instruction",
                    denial_risk="MEDIUM",
                )
                logger.info(
                    f"  Missing companion code: {code} requires {best_code} "
                    f"(use additional code, evidence: {evidence})"
                )

    def _check_diabetes_ulcer_combination(self, icd):
        """ICD-10-CM's 'with' convention (guideline I.A.15) as a completion
        rule: terms under 'with' in the Alphabetic Index are PRESUMED linked
        — a claim carrying a diabetes code and a skin-ulcer code must also
        carry the Index's diabetes-with-ulcer combination code (I.C.4.a:
        'assign as many codes from the category as needed to identify all
        of the associated conditions'). Everything is derived: the diabetes
        carrier is any code whose own category description says 'diabetes
        mellitus', the ulcer any L-chapter code whose descriptor says
        'ulcer', and the combination code comes from the Index rows linking
        'diabetes' and 'ulcer' — matched to the billed ulcer's own site
        words, transposed into the billed carrier's family (E10/E11/E13).

        Determinism layer: measured live (note 004), E11.621 appeared in
        1 of 3 runs of a claim that carried E11.40 + L97.511 every run —
        the convention mandates it every run.

        Declarative: rule 'diabetes-ulcer-with-convention' in the rule pack
        (carrier/trigger selectors, Index query, message text all there)."""
        self.rule_engine.companion_completion(
            "diabetes-ulcer-with-convention", icd)

    def _check_assessment_dx_completion(self, icd, note_assessment_text: str):
        """Mirror image of the marginal-secondary demotion below: that layer
        removes billed codes the assessment never documents; this one adds
        the assessment-listed diagnosis the code arrays OMIT — ICD-10-CM
        guideline IV.J requires coding all documented conditions the
        encounter addresses, and an assessment line is the provider's own
        statement that it was addressed.

        Fires only on the unambiguous shape, everything from real data:
          * an assessment segment's head term (before its qualifiers) is an
            exact Alphabetic Index term — matched whole or as an order-free
            token set ('subungual hematoma' vs the Index's 'hematoma,
            subungual');
          * the Index routes that term to exactly ONE code, billable per
            the FY code set;
          * no billed code shares its 3-char category (the condition is
            genuinely absent, not differently specified);
          * symptom-chapter (R), status (Z), and external-cause codes are
            excluded — their billability turns on rules of their own, not
            on assessment listing.

        Determinism layer: measured live (note 006), 'Onycholysis, left
        hallux' sat verbatim in the assessment while L60.1 appeared in 2
        of 3 runs — the omission run under-coded a documented condition."""
        if self.store is None or self.db is None or not note_assessment_text:
            return
        present = {str(e.get("code") or "").replace(".", "").strip().upper()
                   for e in icd if e.get("code")}
        # exact-term and order-free token-set lookups over the Index
        if getattr(self, "_index_lookup_cache", None) is None:
            exact: dict = {}
            tokset: dict = {}
            for term, code, _s in self._index_rows():
                t = term.lower().strip()
                exact.setdefault(t, set()).add(code)
                toks = frozenset(self._tokens(t))
                if 1 <= len(toks) <= 4:
                    tokset.setdefault(toks, set()).add(code)
            self._index_lookup_cache = (exact, tokset)
        exact, tokset = self._index_lookup_cache
        low = self._NEGATION_RE.sub(" ", note_assessment_text.lower())
        for seg in re.split(r"[;\n]", low):
            head = seg.split(",")[0]
            head = re.sub(r"\([^)]*\)", " ", head)
            head = re.sub(r"\s+", " ", head).strip(" .:-—")
            if len(head) < 4:
                continue
            codes = set(exact.get(head, ()))
            if not codes:
                codes = set(tokset.get(frozenset(self._tokens(head)), ()))
            if len(codes) != 1:
                continue  # ambiguous or unknown term — not this check's shape
            code = codes.pop().upper()
            if code[:1] in ("R", "Z", "V", "W", "X", "Y"):
                continue
            if any(p[:3] == code[:3] for p in present):
                continue
            dotted = code[:3] + "." + code[3:] if len(code) > 3 else code
            info = self.db.validate_icd10(dotted)
            if not info:
                continue  # Index points at a non-billable header — needs axes
            icd.append({
                "code": dotted,
                "description": info.get("description", ""),
                "type": "secondary",
                "rationale": (
                    f"AUTO-ADDED: the assessment lists '{head}' verbatim and "
                    f"the Alphabetic Index routes that term to exactly "
                    f"{dotted}; guideline IV.J requires coding all documented "
                    f"conditions the encounter addresses."),
                "source_section": "validator:assessment_completion",
                "needs_review": True,
                "review_reason": (
                    f"Added from the assessment's own wording ('{head}') — "
                    f"confirm the condition was addressed this encounter"),
            })
            present.add(code)
            self._add(
                "WARNING", dotted, "assessment_dx_omitted",
                f"AUTO-ADDED: {dotted} ('{info.get('description', '')[:60]}') — "
                f"the assessment lists '{head}', the Alphabetic Index maps that "
                f"term to exactly this code, and no billed diagnosis shares its "
                f"category. Guideline IV.J: code all documented conditions "
                f"addressed at the encounter.",
                f"Confirm and keep {dotted}, or document why the condition "
                f"was not addressed",
                denial_risk="LOW",
            )
            logger.info(f"  Assessment completion: added {dotted} ('{head}')")

    def _check_marginal_secondary_demotion(self, icd, coding_result,
                                           note_assessment_text: str):
        """The billable-vs-advisory split, enforced deterministically at the
        end of the ICD checks: a NON-primary residual-category diagnosis
        ('unspecified' / 'other specified' — the classification's own
        markers for codes without a specific documented entity) stays on
        the claim only when the encounter's assessment/imaging documents it
        or a Tabular instruction of another billed code mandates it.
        Otherwise it is demoted to supporting_conditions — still recorded,
        not billed.

        Exemptions (all claim/data facts): primary diagnoses; codes the
        validator itself added (instruction enforcement); companions named
        by any billed code's useAdditionalCode/codeFirst/codeAlso notes;
        external-cause codes accompanying a billed injury. Documentation is
        matched against the code's own descriptor AND its Alphabetic Index
        synonyms ('exostosis' documents M89.8X7 'Other specified disorders
        of bone' even though no descriptor token appears).

        Determinism layer: measured live, marginal secondaries were the
        second-largest flip class (I73.9 'Peripheral vascular disease,
        unspecified' present in 1 of 3 runs off a PMH mention; M25.871
        'Other specified joint disorders' likewise) — the assessment is
        identical every run, so billability must be too.

        Declarative: rule 'marginal-secondary-demotion' in the rule pack
        (residual markers, anchor evidence thresholds, message text)."""
        self.rule_engine.residual_secondary_demotion(
            "marginal-secondary-demotion", icd, coding_result,
            note_assessment_text)

    def _check_missing_code_also(self, icd, note_full_text):
        """Third member of the instructional-note family, alongside
        use-additional-code and code-first: 'Code also' notes say a second
        code may be required to fully describe the condition, with
        sequencing left to the encounter (vs the other two, which mandate
        order). Same evidence gate; flag-only, no auto-add — 'code also' is
        'if applicable' by definition, so whether it applies is a coder
        decision."""
        if self.store is None or not note_full_text:
            return
        note_words = set(re.findall(r"[a-z]+", note_full_text.lower()))
        present = {c.get("code", "").replace(".", "").upper() for c in icd if c.get("code")}
        for entry in icd:
            code = entry.get("code", "")
            if not code:
                continue
            for carrier, refs in self.store.code_also_groups(code):
                if any(p.startswith(r) or r.startswith(p) for r, _ in refs for p in present):
                    continue  # a companion satisfying the note is on the claim
                for ref, note_line in refs:
                    if code.replace(".", "").upper().startswith(ref):
                        continue
                    text = self.store.icd10_tabular_description(ref)
                    matched = self._documented(note_words, text, note_line)
                    if not matched:
                        continue
                    ref_disp = self._redot_icd(ref)
                    self._add(
                        "WARNING", code, "missing_code_also",
                        f"{code}'s Tabular List entry ({self._redot_icd(carrier)}) carries a "
                        f"'code also' note referencing {ref_disp} ({text}), the note documents "
                        f"it ('{', '.join(sorted(matched))}'), but no {ref_disp} code is on "
                        f"the claim.",
                        f"Confirm whether {ref_disp} applies and add it per the code-also note",
                        denial_risk="LOW",
                    )
                    break  # one flag per codeAlso note is enough

    def _check_missing_code_first_etiology(self, icd, note_full_text):
        """Mirror image of the codeFirst sequencing check in
        _encounter_integrity: that one only fires when manifestation AND
        etiology are both coded but mis-ordered; this one catches the
        etiology being ABSENT entirely while the note documents it. No
        auto-add — a codeFirst ref is usually a category/range and choosing
        the specific etiology code is a clinical decision, so this flags
        for the coder instead of guessing."""
        if self.store is None or not note_full_text:
            return
        note_words = set(re.findall(r"[a-z]+", note_full_text.lower()))
        present = {c.get("code", "").replace(".", "").upper() for c in icd if c.get("code")}
        for entry in icd:
            code = entry.get("code", "")
            if not code:
                continue
            for carrier, refs in self.store.code_first_groups(code):
                if any(p.startswith(r) or r.startswith(p) for r, _ in refs for p in present):
                    continue  # etiology on claim — ordering handled by _encounter_integrity
                for ref, note_line in refs:
                    text = self.store.icd10_tabular_description(ref)
                    # same strict-majority evidence bar as the
                    # use-additional-code check, for the same reason
                    matched = self._documented(note_words, text, note_line)
                    if not matched:
                        continue
                    ref_disp = self._redot_icd(ref)
                    self._add(
                        "WARNING", code, "missing_code_first_etiology",
                        f"{code}'s Tabular List entry ({self._redot_icd(carrier)}) carries a "
                        f"'code first' note referencing {ref_disp} ({text}), the note documents "
                        f"it ('{', '.join(sorted(matched))}'), but no {ref_disp} etiology code "
                        f"is on the claim.",
                        f"Code the underlying {ref_disp} etiology and sequence it before {code}",
                        denial_risk="MEDIUM",
                    )
                    break  # one flag per codeFirst note is enough

    def _check_undocumented_specific_sibling(self, icd, cpt, hcpcs,
                                             note_full_text: str):
        """Downgrade a billed diagnosis to its own family's 'unspecified'
        sibling when the specific member's distinguishing condition is
        never documented. ICD-10-CM guideline I.B says code to the highest
        specificity DOCUMENTED — a specific qualifier the provider never
        wrote (guideline I.A: the Index's bare term routes to unspecified)
        may not be assigned.

        Measured live (note 004): 'peripheral neuropathy' / 'diabetic
        neuropathy' documentation flapped between E11.40 (with diabetic
        neuropathy, UNSPECIFIED — 2 of 3 runs) and E11.42 (with diabetic
        POLYNEUROPATHY — 1 of 3), a word the note never contains. The
        documentation is identical every run, so the pick must be too.

        Everything is data: the sibling family (same category, same length,
        differing only at the final character), the residual marker in the
        sibling's own descriptor (an entity-level ', unspecified' ending —
        attribute-level residuals like L97's 'with unspecified severity'
        belong to the tiered-axis rules), the condition lexicon mined from
        the code set, the Tabular inclusion terms and the Alphabetic
        Index's member-specific phrases (both are official synonyms that
        count as documentation — the Index IS the clinician-term-to-code
        routing, guideline I.B.1), and the negation-scrubbed note.
        Downgrade only — never invents specificity; the shared base
        condition must itself be documented or the entry is other checks'
        problem."""
        if self.db is None or not note_full_text:
            return
        note_words, low_note = self._note_evidence(note_full_text)
        cond_lex = self._icd_condition_lexicon()
        site_lex = self._site_lexicon()

        def _toks(desc: str) -> set:
            return {t for t in self._tokens((desc or "").lower())
                    if len(t) >= 4 and t not in self._DESC_STOPWORDS}

        swapped_map: dict[str, str] = {}
        billed_norms = {str(e.get("code") or "").replace(".", "").upper()
                        for e in icd}
        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            info = self.db.validate_icd10(code)
            if not info:
                continue
            own_desc = info.get("description", "")
            if "unspecified" in own_desc.lower():
                continue  # already the residual member
            norm_c = code.replace(".", "")
            sib = next(
                ((sc, sd) for sc, sd in self.db.icd10_siblings(code[:3])
                 if sc != norm_c and len(sc) == len(norm_c)
                 and sc[:-1] == norm_c[:-1]
                 and sd.strip().lower().endswith("unspecified")),
                None)
            if sib is None:
                continue
            sib_code, sib_desc = sib
            if sib_code in billed_norms:
                continue
            own_toks, sib_toks = _toks(own_desc), _toks(sib_desc)
            distinguishing = {t for t in own_toks - sib_toks
                              if t in cond_lex and t not in site_lex}
            if not distinguishing:
                continue
            if any(self._desc_documented(t, note_words, low_note)
                   for t in distinguishing):
                continue  # the specific qualifier IS documented — keep it
            # the Tabular's own inclusion terms are official synonyms —
            # a documented synonym supports the specific code too
            inc_terms = (self.store.icd10_inclusion_terms(code)
                         if self.store is not None else [])
            if any(toks and all(self._desc_documented(t, note_words, low_note)
                                for t in toks)
                   for toks in (_toks(term) for term in inc_terms)):
                continue
            # The Alphabetic Index's own routing is official synonym
            # evidence with equal standing: a note documenting a phrase
            # the Index resolves to THIS exact member supports the
            # specific code even when the descriptor's own word never
            # appears — 'onychauxis' routes to L60.3 Nail dystrophy, so a
            # note documenting onychauxis supports L60.3 without the word
            # 'dystrophy' (ICD-10-CM guideline I.B.1: the Alphabetic
            # Index is step one of code assignment). Discrimination is
            # enforced structurally: min_level=4 drops phrases the Index
            # attaches to the 3-char category (they describe the whole
            # family), and phrases the Index also routes to the
            # unspecified sibling prove neither member, so they cancel.
            # Measured live (routine_00003): the expert reviewer's L60.x
            # ruling was overridden on every replay because this check
            # could not see Index-phrase evidence the sibling-swap check
            # (_check_billed_vs_sibling) had accepted for years.
            if self.store is not None:
                sib_ix = {t.lower() for t in
                          self.store.icd10_index_terms(sib_code,
                                                       min_level=4)}
                own_ix = {t.lower() for t in
                          self.store.icd10_index_terms(code, min_level=4)
                          } - sib_ix
                if any(toks and all(self._desc_documented(t, note_words,
                                                          low_note)
                                    for t in toks)
                       for toks in (_toks(term) for term in own_ix)):
                    continue
            shared = {t for t in own_toks & sib_toks
                      if t in cond_lex and t not in site_lex}
            if shared and not any(self._desc_documented(t, note_words, low_note)
                                  for t in shared):
                continue  # base condition undocumented — not a specificity axis
            dotted = (sib_code[:3] + "." + sib_code[3:]
                      if len(sib_code) > 3 else sib_code)
            old = entry.get("code", "")
            entry["code"] = dotted
            entry["description"] = sib_desc
            entry["needs_review"] = True
            entry["review_reason"] = (
                f"Downgraded from {old}: '{', '.join(sorted(distinguishing))}' "
                f"is not documented — confirm or have the provider document "
                f"the specific condition")
            swapped_map[old] = dotted
            billed_norms.discard(norm_c)
            billed_norms.add(sib_code)
            self._add(
                "WARNING", old, "undocumented_specific_sibling",
                f"AUTO-CORRECTED: {old} → {dotted} ('{sib_desc[:70]}'). "
                f"{old}'s own descriptor requires "
                f"'{', '.join(sorted(distinguishing))}', which neither the "
                f"note, the code's Tabular inclusion terms, nor its "
                f"Alphabetic Index phrases document; "
                f"ICD-10-CM guideline I.B permits only the specificity the "
                f"documentation supports, so the family's unspecified member "
                f"is the accurate code.",
                f"Bill {old} only when the provider documents "
                f"'{', '.join(sorted(distinguishing))}'",
                denial_risk="MEDIUM",
            )
            logger.info(f"  Downgraded ICD {old} → {dotted} "
                        f"(undocumented: {', '.join(sorted(distinguishing))})")
        if not swapped_map:
            return
        for line in cpt + hcpcs:
            linked = line.get("linked_diagnoses") or []
            if not any(d in swapped_map for d in linked):
                continue
            new_linked = []
            for d in linked:
                d2 = swapped_map.get(d, d)
                if d2 not in new_linked:
                    new_linked.append(d2)
            line["linked_diagnoses"] = new_linked

    def _check_icd_includes_severity_upgrade(self, icd, cpt, hcpcs,
                                             note_full_text: str):
        """Upgrade a billed diagnosis to the sibling the Tabular List itself
        ranks above it, when the note affirmatively documents that sibling's
        own distinguishing condition. The ICD-10-CM Includes chain IS the
        ranking: I70.26-'s Tabular entry states it includes 'any condition
        classifiable to I70.21- through I70.25-' — gangrene CAPTURES the
        ulceration/rest-pain/claudication manifestations, so when gangrene
        is documented, the gangrene code is the only correct member
        (guideline I.B, code to the highest specificity documented).

        Measured live (note 010): 'Gangrene developing distally' with
        critical limb ischemia flapped I70.242 (ulceration, 1 of 3 runs) vs
        I70.262 (gangrene, 2 of 3). The subsumption REMOVAL check above only
        fires when both are billed; this layer settles the single-code
        choice deterministically. Everything is data: the icd10_includes
        chain (which designates the higher member), the sibling's own
        descriptor tokens (condition-lexicon words only, site vocabulary
        excluded), the negation-scrubbed note, and a same-final-character
        guard so laterality never changes. Strict argmax: two qualifying
        upgrades is a human question, not a swap."""
        if self.store is None or self.db is None or not note_full_text:
            return
        note_words, low_note = self._note_evidence(note_full_text)
        cond_lex = self._icd_condition_lexicon()
        site_lex = self._site_lexicon()

        def _toks(desc: str) -> set:
            return {t for t in self._tokens((desc or "").lower())
                    if len(t) >= 4 and t not in self._DESC_STOPWORDS}

        swapped_map: dict[str, str] = {}
        billed_norms = {str(e.get("code") or "").replace(".", "").upper()
                        for e in icd}
        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            info = self.db.validate_icd10(code)
            if not info:
                continue
            norm_c = code.replace(".", "")
            own_toks = _toks(info.get("description", ""))
            upgrades = []
            for sib_code, sib_desc in self.db.icd10_siblings(code[:3]):
                if (sib_code == norm_c or len(sib_code) != len(norm_c)
                        or sib_code[-1] != norm_c[-1]
                        or sib_code in billed_norms):
                    continue
                if not self.store.includes_subsumption(sib_code, norm_c):
                    continue  # the Tabular chain doesn't rank sib above code
                distinguishing = {
                    t for t in _toks(sib_desc) - own_toks
                    if t in cond_lex and t not in site_lex
                }
                if not distinguishing:
                    continue
                if all(self._desc_documented(t, note_words, low_note)
                       for t in distinguishing):
                    upgrades.append((sib_code, sib_desc, distinguishing))
            if len(upgrades) != 1:
                continue
            sib_code, sib_desc, documented = upgrades[0]
            dotted = (sib_code[:3] + "." + sib_code[3:]
                      if len(sib_code) > 3 else sib_code)
            entry["code"] = dotted
            entry["description"] = sib_desc
            entry["needs_review"] = True
            entry["review_reason"] = (
                f"Upgraded from {code} per the Tabular Includes chain — "
                f"confirm the documented {', '.join(sorted(documented))}")
            swapped_map[code] = dotted
            billed_norms.discard(norm_c)
            billed_norms.add(sib_code)
            self._add(
                "WARNING", code, "icd_includes_severity_upgrade",
                f"AUTO-CORRECTED: {code} → {dotted} ('{sib_desc[:70]}'). The note "
                f"affirmatively documents {', '.join(sorted(documented))}, and the "
                f"Tabular List's own Includes note ranks {dotted} above {code} — "
                f"it captures any condition classifiable to {code}, so the "
                f"documented higher manifestation is the only correct member "
                f"(guideline I.B: code to the highest specificity documented).",
                f"Report {dotted}; {code}'s manifestation is captured within it "
                f"per the Tabular Includes chain",
                denial_risk="MEDIUM",
            )
            logger.info(f"  Upgraded ICD {code} → {dotted} "
                        f"(documented {', '.join(sorted(documented))})")
        if not swapped_map:
            return
        for line in cpt + hcpcs:
            linked = line.get("linked_diagnoses") or []
            if not any(d in swapped_map for d in linked):
                continue
            new_linked = []
            for d in linked:
                d2 = swapped_map.get(d, d)
                if d2 not in new_linked:
                    new_linked.append(d2)
            line["linked_diagnoses"] = new_linked

    def _check_icd_includes_subsumption(self, icd, cpt, hcpcs, coding_result):
        """Remove a diagnosis the Tabular List's own Includes note subsumes
        into another billed diagnosis. Found live: I70.221 (rest pain) and
        I70.235 (ulceration) billed together for the same leg — I70.23-'s
        real Tabular entry says it includes 'any condition classifiable to
        I70.211 and I70.221', i.e. when ulceration is documented the rest-
        pain manifestation is captured within the ulceration code and only
        that one is reported.

        Unlike Excludes1 (symmetric — the data can't say which code the
        documentation supports, so that check only flags), subsumption is
        directional: the data itself designates the keeper (the note-carrying,
        more specific code), so removal is deterministic, mirroring NCCI
        bundling suppression. Fully data-driven from CDC/NCHS's Tabular XML
        (icd10_includes + the useAdditionalCode guard for combination
        categories that REQUIRE their companion code) — no code lists here.
        """
        if self.store is None:
            return
        removed_map: dict[str, str] = {}  # subsumed code -> keeper code
        for keeper in icd:
            kc = keeper.get("code", "")
            if not kc or kc in removed_map:
                continue
            for other in icd:
                oc = other.get("code", "")
                if not oc or oc == kc or oc in removed_map:
                    continue
                carrier = self.store.includes_subsumption(kc, oc)
                if not carrier:
                    continue
                removed_map[oc] = kc
                # the subsumed code may have been the designated primary —
                # the keeper inherits that role, not an arbitrary survivor
                if other.get("type") == "primary" and keeper.get("type") != "primary":
                    keeper["type"] = "primary"
                self._add(
                    "WARNING", oc, "icd_includes_subsumption",
                    f"{oc} removed: ICD-10-CM Tabular List entry {carrier} states it includes "
                    f"any condition classifiable to {oc} — the condition is captured within "
                    f"{kc}, so billing both is redundant.",
                    f"Report only {kc}; {oc} is subsumed per the Tabular List Includes note",
                    denial_risk="MEDIUM",
                )
                logger.info(f"  Suppressed subsumed ICD {oc} (captured within {kc} per Tabular Includes)")
        if not removed_map:
            return
        icd[:] = [c for c in icd if c.get("code", "") not in removed_map]
        coding_result["icd10_codes"] = icd
        # Relink service lines: a pointer to the removed dx becomes a pointer
        # to its keeper (deduplicated, order preserved).
        for entry in cpt + hcpcs:
            linked = entry.get("linked_diagnoses") or []
            if not any(d in removed_map for d in linked):
                continue
            new_linked = []
            for d in linked:
                d2 = removed_map.get(d, d)
                if d2 not in new_linked:
                    new_linked.append(d2)
            entry["linked_diagnoses"] = new_linked

    def _check_icd_excludes1(self, icd):
        """Flag a pair of ICD-10-CM codes that are Type 1 Excludes ("not
        coded here") per the real Tabular List data — structurally mutually
        exclusive, not a stylistic choice between similar codes. Found live:
        M12.571 (Traumatic arthropathy) and M19.171 (Post-traumatic
        osteoarthritis) coded together — M12.5's own Tabular entry excludes
        M19.1 explicitly.

        Flags only — does not auto-remove either code. Unlike an NCCI pair
        (where the real data itself designates a primary/component code),
        an Excludes1 conflict is symmetric: CMS's data says these two
        conditions can't coexist, but not which one the documentation
        actually supports. That requires reading the note (e.g. AHA Coding
        Clinic guidance on M12 vs M19 turns on whether documentation
        specifies a non-osteoarthritis traumatic arthropathy or defaults to
        osteoarthritis) — a judgment call for the coder/reviewer, not
        something this structural check should guess at.
        """
        if self.store is None:
            return
        codes = [c.get("code", "") for c in icd if c.get("code")]
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                if self.store.excludes1_conflict(codes[i], codes[j]):
                    self._add(
                        "ERROR", f"{codes[i]}|{codes[j]}", "icd_excludes1_conflict",
                        f"{codes[i]} and {codes[j]} are ICD-10-CM Type 1 Excludes of each other "
                        f"('not coded here') — structurally mutually exclusive per the Tabular "
                        f"List, not codeable together on the same claim.",
                        f"Review documentation to determine which of {codes[i]}/{codes[j]} is "
                        f"clinically supported and remove the other",
                        denial_risk="HIGH",
                    )

    def _check_snomed_consistency(self, snomed):
        """Label drift detection + root concept detection."""
        seen_ids: dict[str, str] = {}
        _LATERALITY = {"right", "left", "bilateral", "rt", "lt", "r ", "l "}

        def _is_bilateral_pair(a: str, b: str) -> bool:
            """Return True when a and b are the same clinical concept but differ by laterality."""
            a_lower, b_lower = a.lower(), b.lower()
            a_has_side = any(w in a_lower for w in _LATERALITY)
            b_has_side = any(w in b_lower for w in _LATERALITY)
            if not (a_has_side and b_has_side):
                return False
            # Strip laterality words and compare core text
            for w in ("right", "left", "bilateral", " rt", " lt"):
                a_lower = a_lower.replace(w, "").strip()
                b_lower = b_lower.replace(w, "").strip()
            return a_lower == b_lower

        for entry in snomed:
            concept_id = str(entry.get("concept_id", "")).strip()
            entity_text = entry.get("entity_text", "")
            description = entry.get("description", "")

            if concept_id in seen_ids:
                if seen_ids[concept_id] != entity_text and not _is_bilateral_pair(seen_ids[concept_id], entity_text):
                    # INFO — SNOMED consistency issue; does not affect billing or claims
                    self._add(
                        "INFO", concept_id, "snomed_label_drift",
                        f"SNOMED concept {concept_id} ({description}) assigned to two different terms: "
                        f'"{seen_ids[concept_id]}" and "{entity_text}". One mapping may be incorrect.',
                        "Review both SNOMED mappings and correct the wrong one",
                        denial_risk="LOW",
                    )
            else:
                seen_ids[concept_id] = entity_text

            if self.db.is_snomed_root(concept_id):
                root_label = self.db.get_snomed_root_label(concept_id)
                current_conf = entry.get("confidence", 1.0)
                if current_conf > self.db.snomed_root_confidence_cap:
                    entry["confidence"] = self.db.snomed_root_confidence_cap
                    entry["is_root_concept"] = True
                # INFO (not WARNING) — SNOMED quality issue only; does not affect billing or claim
                self._add(
                    "INFO", concept_id, "snomed_root_concept",
                    f"SNOMED {concept_id} is a top-level parent concept ({root_label}) — "
                    f"too generic for clinical coding. A specific descendant should be used.",
                    f"Find a more specific SNOMED concept for '{entity_text}'",
                    denial_risk="LOW",
                )

    # --- Encounter integrity ---

    def _encounter_integrity(self, icd) -> EncounterIntegrity:
        """Flag a manifestation code sequenced before its required etiology
        code, per ICD-10-CM's real Tabular List codeFirst notes (parsed from
        CDC/NCHS's own icd10cm-tabular-2026.xml) — not a hardcoded prefix
        pairing. Checked against real data, every one of the 4 previously
        hardcoded MANIFESTATION_PREFIXES ({H36,G63,N08,M14}) turned out not
        to actually pair with the 5 hardcoded ETIOLOGY_PREFIXES
        ({E10,E11,E13,I70,I73}) at all — e.g. H36's real codeFirst refs are
        lipid storage disorders (E75) and sickle-cell disorders (D57), not
        diabetes; diabetic retinopathy is a self-contained combination code
        (E11.3x). The check never could have fired correctly. The genuinely
        podiatry-relevant real pairing this data surfaces is L97 (chronic
        lower-limb ulcer) requiring specific diabetic/PVD etiology sub-codes
        (e.g. E11.621/E11.622, I70.23/I70.24) sequenced first — not covered
        by the old hardcoded set at all.
        """
        issues = []
        if self.store is not None:
            codes = [entry.get("code", "") for entry in icd]
            for idx, code in enumerate(codes):
                etiology_refs = self.store.code_first_etiology_refs(code)
                if not etiology_refs:
                    continue
                norm_refs = [r.replace(".", "").upper() for r in etiology_refs]
                etio_idx = next(
                    (j for j, other in enumerate(codes)
                     if j != idx and other.replace(".", "").upper().startswith(tuple(norm_refs))),
                    None,
                )
                if etio_idx is not None and idx < etio_idx:
                    issues.append({
                        "type": "SEQUENCING_ERROR",
                        "severity": "WARNING",
                        "message": (
                            f"ICD {code} (manifestation) is sequenced before "
                            f"etiology ({codes[etio_idx]})"
                        ),
                    })
        errors = sum(1 for i in issues if i.get("severity") == "ERROR")
        warns = sum(1 for i in issues if i.get("severity") == "WARNING")
        return EncounterIntegrity(encounter_issues=issues, error_count=errors, warning_count=warns)

    # --- Documentation audit ---

    def _documentation_audit(self, coding_result: dict) -> DocumentationAudit:
        entries = []
        for key in ["icd10_codes", "cpt_codes", "hcpcs_codes"]:
            for entry in coding_result.get(key, []):
                support = []
                rationale = entry.get("rationale", "")
                if rationale:
                    support.append(f"Rationale: {rationale}")

                src = entry.get("source_section", "") or entry.get("source", "")
                if src:
                    support.append(f"Section: {src}")

                mdm = entry.get("mdm_details", {})
                if mdm and mdm.get("mdm_level"):
                    p = mdm.get("problems_score", mdm.get("problem_score", "?"))
                    d = mdm.get("data_score", "?")
                    r = mdm.get("risk_score", "?")
                    support.append(f"MDM: {mdm.get('mdm_level')} (P:{p}/D:{d}/R:{r})")
                    linked = entry.get("linked_diagnoses", [])
                    if linked:
                        support.append(f"Linked DX: {', '.join(linked)}")

                supporting_text = entry.get("supporting_text", "")
                if supporting_text:
                    support.append(f"Supporting text: {supporting_text}")

                # "gaps" must be reachable while still "supported", or
                # partially_supported below is dead code by construction
                # (supported and gaps were previously mutually exclusive —
                # gaps only got populated in the *not* support case). A code
                # with some support but no direct supporting_text (the
                # strongest evidence — a verbatim excerpt from the note, vs.
                # a rationale/MDM score derived from it) is the real
                # "partial" case: weaker than fully documented, stronger
                # than undocumented.
                gaps = []
                if not support:
                    gaps.append("No documentation evidence found for this code")
                elif not supporting_text:
                    gaps.append("No direct supporting text captured from the note (other evidence present)")

                entries.append({
                    "code": entry.get("code", ""),
                    "description": entry.get("description", ""),
                    "documentation_support": support,
                    "documentation_gaps": gaps,
                    "supported": len(support) > 0,
                })

        total = len(entries)
        fully_supported = sum(1 for e in entries if e["supported"] and not e["documentation_gaps"])
        partially_supported = sum(1 for e in entries if e["supported"] and e["documentation_gaps"])
        unsupported = sum(1 for e in entries if not e["supported"])
        with_gaps = sum(1 for e in entries if e["documentation_gaps"])

        return DocumentationAudit(
            audit_entries=entries,
            total_codes=total,
            fully_supported=fully_supported,
            partially_supported=partially_supported,
            unsupported=unsupported,
            codes_with_gaps=with_gaps,
            documentation_score=round((fully_supported + 0.5 * partially_supported) / total, 2) if total else 1.0,
        )

    # --- Tier scoring ---

    def _audit_score(self) -> float:
        errors = sum(1 for i in self.issues if i.severity == "ERROR")
        warns = sum(1 for i in self.issues if i.severity == "WARNING")
        infos = sum(1 for i in self.issues if i.severity == "INFO")
        return max(0.0, round(1.0 - errors * 0.15 - warns * 0.05 - infos * 0.01, 2))

    def _check_physician_code_preservation(
        self,
        all_codes: list[dict],
        coding_result: dict,
        physician_documented_codes: list[dict],
        note_full_text: str = "",
    ):
        """Fix 1 — Enforce physician code lock.

        Any code the AI replaced or dropped that the physician explicitly documented
        must be flagged as REVIEW with clear audit trail.
        """
        # Flag codes tagged as ai_replaced_physician
        for entry in all_codes:
            if entry.get("code_source") == "ai_replaced_physician":
                self._add(
                    "WARNING",
                    entry.get("code", ""),
                    "physician_code_replaced",
                    f"AI assigned {entry.get('code')} but physician documented a different code in this "
                    f"category ({entry.get('physician_code_note', 'see note')}). "
                    f"Verify with provider before billing — cannot be AUTO-approved.",
                    "Review with provider: confirm AI correction or restore physician-specified code",
                    denial_risk="HIGH",
                )

        # Flag physician codes completely missing from output — UNLESS the
        # physician's own wording marks the code as a ruled-out differential.
        # ICD-10-CM guideline IV.H (outpatient): do not code diagnoses
        # documented as "rule out"/"ruled out"/differential — omitting them
        # is CORRECT coding, not a dropped code (observed live: 'L84 —
        # differential ruled out by black dot visualization' flagged as a
        # HIGH-risk omission when the AI had rightly excluded it).
        _RULED_OUT_RE = re.compile(
            r"\b(ruled?\s+out|rule[- ]?out|r/o|differential|less\s+likely|excluded|unlikely)\b",
            re.IGNORECASE,
        )
        for p in coding_result.get("missing_physician_codes", []):
            raw = p.get("raw_text", "") or ""
            if _RULED_OUT_RE.search(raw):
                self._add(
                    "INFO",
                    p.get("code", ""),
                    "physician_code_ruled_out",
                    f"Physician-documented code {p.get('code')} ({p.get('description', '')}) was "
                    f"omitted because the note marks it as a ruled-out differential "
                    f"('{raw[:80]}') — correct per ICD-10-CM guideline IV.H (do not code "
                    f"ruled-out diagnoses).",
                    "No action needed",
                    denial_risk="LOW",
                )
                continue
            self._add(
                "WARNING",
                p.get("code", ""),
                "physician_code_missing",
                f"Physician-documented code {p.get('code')} ({p.get('description', '')}) "
                f"from the {p.get('section', 'note')} section was not included in the coding output. "
                f"A physician-specified code must either appear in output or be explicitly flagged.",
                "Add physician code to output, or document clinical reason it was excluded",
                denial_risk="HIGH",
            )

        # Flag laterality modifiers that are NOT defensible from the source
        # document. Previously keyed only on the LLM's self-reported
        # code_source ('ai_inferred'), which ignored the note entirely — 27
        # of 49 batch results carried this WARNING even though the side was
        # plainly written in the note ('right hallux', 'left heel'). The
        # audit-defensibility question is answerable deterministically: does
        # the note text actually contain the side word (or a side-specific
        # toe/finger modifier's own description terms)? Only silence in the
        # note makes the modifier indefensible.
        note_lower = (note_full_text or "").lower()
        side_words = {"RT": "right", "LT": "left"}
        for entry in all_codes:
            code = entry.get("code", "")
            mods = entry.get("modifiers", [])
            for m in mods:
                side = side_words.get(m)
                if side is None:
                    continue
                documented = bool(re.search(rf"\b{side}\b", note_lower))
                if not documented:
                    self._add(
                        "WARNING",
                        code,
                        "laterality_not_in_note",
                        f"{code} carries laterality modifier {m}, but the word '{side}' appears "
                        f"nowhere in the note — the side is not defensible from the source "
                        f"document if audited.",
                        "Verify laterality with the provider and document it before billing",
                        denial_risk="MEDIUM",
                    )
                elif entry.get("code_source") == "ai_inferred":
                    self._add(
                        "INFO",
                        code,
                        "laterality_confirmed_in_note",
                        f"{code}'s {m} modifier was AI-selected but the note documents "
                        f"'{side}' — defensible from the source document.",
                        "No action needed",
                        denial_risk="LOW",
                    )

    def _compute_tier(self, icd, cpt, hcpcs) -> tuple[str, float, list[str]]:
        """Fix 5 — Penalty-based confidence scoring.

        Confidence starts at 1.0 and is reduced by each signal of uncertainty.
        This makes the score discriminative — it reflects actual coding quality,
        not just a uniform number.
        """
        errors = [i for i in self.issues if i.severity == "ERROR"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]
        infos = [i for i in self.issues if i.severity == "INFO"]

        # --- Tier from issues ---
        if errors:
            tier = "REJECT"
        elif warnings:
            tier = "REVIEW"
        elif infos:
            tier = "AUTO"
        else:
            tier = "AUTO"

        # --- Penalty-based confidence (Fix 5) ---
        confidence = 1.0
        all_codes = icd + cpt + hcpcs

        for entry in all_codes:
            src = entry.get("code_source", "ai_inferred")
            llm_conf = entry.get("confidence", 0.5)

            if src == "ai_replaced_physician":
                confidence -= 0.25  # replaced a physician code
            elif src == "ai_inferred":
                confidence -= 0.05  # not physician-documented
            # physician_documented or ai_confirmed → no penalty

            if llm_conf < 0.80:
                confidence -= 0.05  # LLM itself was uncertain

        # Penalties from validation issues
        for issue in errors:
            cat = issue.category
            if cat == "code_existence":
                confidence -= 0.20  # invalid code — major issue
            elif cat == "modifier_57_missing":
                confidence -= 0.10
            else:
                confidence -= 0.08

        for issue in warnings:
            cat = issue.category
            if cat in ("physician_code_replaced", "physician_code_missing"):
                confidence -= 0.15  # audit risk
            elif cat == "laterality_not_in_note":
                confidence -= 0.10
            elif cat == "lcd_coverage":
                confidence -= 0.08
            else:
                confidence -= 0.05

        for _ in infos:
            confidence -= 0.01

        # Clamp confidence to tier-appropriate range
        if tier == "REJECT":
            confidence = min(confidence, 0.45)
        elif tier == "REVIEW":
            confidence = min(confidence, 0.84)
        else:
            confidence = max(confidence, 0.85)  # AUTO is always ≥0.85

        confidence = round(max(0.0, min(1.0, confidence)), 2)

        reasons = []
        for i in errors:
            reasons.append(f"[ERROR] {i.code}: {i.message}")
        for i in warnings:
            reasons.append(f"[WARNING] {i.code}: {i.message}")

        return tier, confidence, reasons

    def _summary(self, tier: str, reasons: list[str]) -> str:
        if tier == "AUTO" and not reasons:
            return "All validation checks passed. Ready for submission."
        if tier == "AUTO":
            return f"Auto-codeable with {len(reasons)} informational note(s). Review before submission."
        if tier == "REVIEW":
            return f"Coder review required — {len(reasons)} issue(s) need resolution before submission."
        return f"Claim rejected — {len(reasons)} critical error(s) must be corrected before submission."

    # ------------------------------------------------------------------
    # Removal conservation: documented work must never fall off a claim
    # silently. Measured live (routine_00001): 27650 was correctly removed
    # (descriptor requires a RUPTURED Achilles; none documented) but the
    # documented work — insertional detachment, degenerative-tendon
    # excision, anchor reattachment — was 27654's, and no layer owned the
    # substitution, so ~$1000 of documented surgery vanished without a
    # trace. This check runs at suppression time, over each CPT line queued
    # for removal by a documentation/descriptor-mismatch layer (never a
    # structural one: NCCI/MUE/billability removals are the claim's correct
    # final state). It searches the code's own descriptor family for the
    # member whose distinguishing attributes ARE documented:
    #   - exactly one such member  -> substitute it on the line (flagged),
    #   - none provable but the family's shared work IS documented
    #     -> escalate loudly: WARNING that documented work may be uncoded.
    # Either way, the removal can no longer be silent.
    # ------------------------------------------------------------------

    _DOC_MISMATCH_CAT_RE = re.compile(
        r"undocumented|indication|descriptor|documentation|evidence",
        re.IGNORECASE)
    _STRUCTURAL_CAT_RE = re.compile(
        r"mue|ncci|billab|bundl|global|frequen|existence|not_active|"
        r"age_range|pos_|laterality|modifier|pointer|linkage",
        re.IGNORECASE)
    # Interpretive layer decisions: grounded in READING the note's language
    # (sibling arbitration, axis swaps, completion elections, primary
    # re-designation, severity tiers) — the class the clinical audit must
    # verify. Data-structural decisions (table lookups: NCCI, MUE,
    # descriptor denominations) carry their authority in the data itself.
    _INTERPRETIVE_CAT_RE = re.compile(
        r"undocumented|indication|descriptor|documentation|evidence|"
        r"sibling|axis|completion|match|severity|designation|demotion|"
        r"conservation|assessment|depth|note", re.IGNORECASE)

    def _cpt_desc_tokens(self, desc: str) -> set:
        return {t for t in self._tokens(self._EG_PAREN_RE.sub(" ", desc or ""))
                if len(t) >= 4 and t not in self._DESC_STOPWORDS}

    def _check_removal_conservation(self, cpt, note_full_text: str):
        if not note_full_text or not self._non_billable_codes_to_suppress:
            return
        clin_words, clin_low = self._clinical_evidence(note_full_text)
        billed = {(e.get("code") or "").strip().upper() for e in cpt}
        for entry in cpt:
            code = (entry.get("code") or "").strip().upper()
            if code not in self._non_billable_codes_to_suppress:
                continue
            cats = {i.category for i in self.issues
                    if i.code == code and i.severity in ("ERROR", "WARNING")}
            doc_cats = {c for c in cats if self._DOC_MISMATCH_CAT_RE.search(c)
                        and not self._STRUCTURAL_CAT_RE.search(c)}
            struct_cats = {c for c in cats if self._STRUCTURAL_CAT_RE.search(c)}
            if not doc_cats or struct_cats:
                continue  # structural removals stand; nothing to conserve
            own = self.db.validate_cpt(code)
            if not own or not code.isdigit():
                continue
            own_toks = self._cpt_desc_tokens(own.get("description", ""))
            if not own_toks:
                continue
            # Shared work documented? The family's common tokens (the
            # operation itself: 'repair', 'achilles', 'tendon') must be
            # spoken by the note before any conservation claim is made.
            candidates = []
            shared_documented = False
            for c2, info in (getattr(self.db, "cpt", {}) or {}).items():
                if c2 == code or not c2.isdigit() \
                        or abs(int(c2) - int(code)) > 30:
                    continue
                desc2 = info.get("long_description") or info.get(
                    "description") or ""
                toks2 = self._cpt_desc_tokens(desc2)
                shared = own_toks & toks2
                if len(shared) < 2:
                    continue
                doc_shared = [t for t in shared if self._desc_documented(
                    t, clin_words, clin_low)]
                if len(doc_shared) < 2:
                    continue
                shared_documented = True
                sib_only = toks2 - own_toks
                if (not sib_only or c2 in billed
                        or c2 in self._non_billable_codes_to_suppress
                        or c2 in self._bundled_codes_to_suppress):
                    continue  # never resurrect a code another layer removed
                # substitution requires EVERY distinguishing attribute of
                # the sibling affirmatively documented in the clinical view
                if all(self._desc_documented(t, clin_words, clin_low)
                       for t in sib_only):
                    candidates.append((len(shared), c2, desc2))
            if not shared_documented:
                continue  # the work itself isn't documented — removal stands
            candidates.sort(reverse=True)
            if candidates and (len(candidates) == 1
                               or candidates[0][0] > candidates[1][0]):
                _, target, tdesc = candidates[0]
                entry["code"] = target
                entry["description"] = tdesc
                entry["needs_review"] = True
                self._non_billable_codes_to_suppress.discard(code)
                billed.discard(code)
                billed.add(target)
                self._add(
                    "WARNING", target, "removal_conservation",
                    f"AUTO-CORRECTED: {code} was removed for a documentation "
                    f"mismatch, but the documented work matches family member "
                    f"{target} ('{tdesc[:60]}') — every distinguishing "
                    f"attribute of {target}'s descriptor is documented, so "
                    f"the line was substituted instead of dropped.",
                    "Verify the substituted code matches the documented work",
                    denial_risk="MEDIUM",
                )
                logger.info(f"  Removal conservation: {code} → {target} "
                            f"(documented work preserved)")
            else:
                self._add(
                    "WARNING", code, "removal_conservation",
                    f"{code} was removed for a documentation mismatch, but "
                    f"the note DOES document the family's work — documented "
                    f"surgical work may now be uncoded on this claim, and no "
                    f"single family member's distinguishing attributes are "
                    f"provable deterministically.",
                    "Coder must select the family member the documented "
                    "work supports (or confirm the removal)",
                    denial_risk="HIGH",
                )
                logger.info(f"  Removal conservation: {code} removed but "
                            f"documented work may be uncoded — escalated")

    # Material corrections: every AUTO-CORRECTED action and suppression a
    # layer took this pass, extracted for the clinical-correctness audit
    # (tools/clinical_auditor.py). 'interpretive' marks decisions grounded
    # in note-text interpretation (the class that produced the thigh-
    # tourniquet regression) vs. authoritative-data lookups (NCCI/MUE/
    # descriptor tables), whose authority is the data itself.
    def _material_corrections(self) -> list[dict]:
        def _interp(cat: str) -> bool:
            return bool(self._INTERPRETIVE_CAT_RE.search(cat)
                        and not self._STRUCTURAL_CAT_RE.search(cat))

        # AUTO-ADDED lines (completion layers electing a new code from the
        # note) are claim mutations exactly like AUTO-CORRECTED swaps — a
        # wrongly added code is as billable-wrong as a wrongly changed one.
        out = [{
            "category": i.category, "code": i.code,
            "action": ("auto_addition"
                       if str(i.message).startswith("AUTO-ADDED")
                       else "auto_correction"),
            "interpretive": _interp(i.category),
            "message": i.message,
        } for i in self.issues
            if str(i.message).startswith(("AUTO-CORRECTED", "AUTO-ADDED"))]

        suppressed = (self._non_billable_codes_to_suppress
                      | self._bundled_codes_to_suppress)
        for code in sorted(suppressed):
            about = [i for i in self.issues if i.code == code
                     and i.severity in ("ERROR", "WARNING")]
            lead = about[0] if about else None
            out.append({
                "category": lead.category if lead else "suppression",
                "code": code,
                "action": "removal",
                "interpretive": _interp(lead.category) if lead else False,
                "message": (lead.message if lead
                            else "line suppressed from the claim"),
            })
        # Validator advisories removed at source by advisory-suppression
        # directives — reported so the removal is auditable (the removed
        # issue no longer exists in self.issues to derive from).
        out.extend(self._advisory_suppression_corrections)
        return out

    # ------------------------------------------------------------------
    # Diff-derived corrections — the structural fix for self-reporting.
    # Measured live (routine_00003): _check_marginal_secondary_demotion
    # moved R26.2 (the claim's coverage-pathway diagnosis) off the billed
    # arrays without emitting an AUTO-CORRECTED issue, so the clinical
    # audit — whose input was the layers' own reports — never saw the
    # decision and vacuously upheld the claim. The snapshot/diff below
    # derives the mutation record FROM THE CLAIM STATE ITSELF: any code
    # added, removed, re-typed, re-united or re-modified between entry and
    # exit of validate() that no recorded correction accounts for becomes
    # a derived correction, tagged interpretive (unknown provenance MUST
    # be audited — fail closed, never open). No layer can act unseen.
    # ------------------------------------------------------------------

    @staticmethod
    def _pre_validation_snapshot(icd, cpt, hcpcs) -> dict:
        def rows(arr):
            return [{
                "code": str(e.get("code") or "").strip().upper(),
                "type": e.get("type"),
                "modifiers": sorted(str(m).upper()
                                    for m in (e.get("modifiers") or [])),
                "units": e.get("units"),
            } for e in arr if isinstance(e, dict) and e.get("code")]
        return {"icd_codes": rows(icd), "cpt_codes": rows(cpt),
                "hcpcs_codes": rows(hcpcs)}

    # A recorded correction "accounts for" a diffed code when it names the
    # code as its subject, or its message states a swap/substitution/
    # displacement the code participates in ("28308 → 28309", "27650 was
    # removed", "designated primary (was M77.31)"). These patterns are OUR
    # OWN layers' message grammars — matching them never encodes any
    # medical knowledge. A pointer-remap message that merely MENTIONS a
    # dropped diagnosis ("dropped R26.2") is deliberately not a match: the
    # pointer correction reports a pointer action, not the diagnosis
    # mutation itself (the exact hole measured live).
    @staticmethod
    def _correction_accounts_for(recorded: list[dict], code: str) -> bool:
        cu = str(code).upper()
        for m in recorded:
            if str(m.get("code") or "").upper() == cu:
                return True
            msg = str(m.get("message") or "").upper()
            if re.search(rf"\b{re.escape(cu)}\s*(?:→|->)", msg) or \
                    re.search(rf"(?:→|->)\s*{re.escape(cu)}\b", msg) or \
                    re.search(rf"\b{re.escape(cu)}\s+WAS\s+REMOVED", msg) or \
                    re.search(rf"\(WAS\s+{re.escape(cu)}\)", msg):
                return True
        return False

    def _material_corrections_with_derived(self, pre_claim: dict,
                                           icd, cpt, hcpcs,
                                           coding_result: dict) -> list[dict]:
        recorded = self._material_corrections()
        post = self._pre_validation_snapshot(icd, cpt, hcpcs)

        # Where do demoted diagnoses land? supporting_conditions — carry
        # the layer's stated reason into the derived record so the audit
        # judges the actual rationale, not a reconstruction.
        demoted_reasons = {
            str(e.get("code") or "").upper():
                str(e.get("review_reason") or "")
            for e in (coding_result.get("supporting_conditions") or [])
            if isinstance(e, dict) and e.get("code")}

        derived: list[dict] = []
        for arr in ("icd_codes", "cpt_codes", "hcpcs_codes"):
            before = {r["code"]: r for r in pre_claim.get(arr) or []}
            after = {r["code"]: r for r in post.get(arr) or []}
            for code in sorted(set(before) - set(after)):
                if self._correction_accounts_for(recorded, code):
                    continue
                reason = demoted_reasons.get(code, "")
                derived.append({
                    "category": "unreported_mutation",
                    "code": code, "action": "derived_removal",
                    "interpretive": True, "derived": True,
                    "message": (f"DERIVED: {code} was on the billed claim "
                                f"when validation began and is gone from it "
                                f"now, with no layer reporting the removal"
                                + (f" — demoted to supporting_conditions "
                                   f"with reason: {reason[:220]}"
                                   if reason else "")),
                })
            for code in sorted(set(after) - set(before)):
                if self._correction_accounts_for(recorded, code):
                    continue
                derived.append({
                    "category": "unreported_mutation",
                    "code": code, "action": "derived_addition",
                    "interpretive": True, "derived": True,
                    "message": (f"DERIVED: {code} was added to the billed "
                                f"claim during validation with no layer "
                                f"reporting the addition"),
                })
            for code in sorted(set(before) & set(after)):
                b, a = before[code], after[code]
                changes = [f"{k}: {b[k]!r} -> {a[k]!r}"
                           for k in ("type", "modifiers", "units")
                           if b[k] != a[k]]
                if not changes or self._correction_accounts_for(
                        recorded, code):
                    continue
                derived.append({
                    "category": "unreported_mutation",
                    "code": code, "action": "derived_modification",
                    "interpretive": True, "derived": True,
                    "message": (f"DERIVED: {code} was modified during "
                                f"validation with no layer reporting it "
                                f"({'; '.join(changes)})"),
                })
        return recorded + derived

    def _add(self, severity, code, category, message, recommendation, denial_risk=None):
        self.issues.append(ValidationIssue(
            severity=severity,
            code=str(code),
            category=category,
            message=message,
            recommendation=recommendation,
            denial_risk=denial_risk,
        ))

    def suppress_scrub_advisory(self, filter_id: str, code: str,
                                rule_id: str = "", authority: str = "",
                                note: str = "", clause: str = "",
                                validator_categories: list | tuple = ()) -> None:
        """Record that a compliance-scrubber ADVISORY (a WARN finding from
        the named filter about the named code) must not fire on this claim
        — the deterministic rule engine's bridge into the scrubber's
        advisory layer. The suppression rides the validation report into
        the scrub payload; ClaimScrubber honors it for WARN findings ONLY
        (a FAIL — the clean-claim gate — can never be config-suppressed)
        and records the suppression as its own PASS finding, so the audit
        trail shows a rule decision, not a vanished check. Grown for
        advisory-shaped audit disputes: an advisory whose recommendation
        the authorities contradict for a documented fact pattern (e.g. a
        coverage advisory keyed to one pathway when the note satisfies a
        distinct, authority-recognized pathway).

        clause: the finding CLAUSE the suppression was verified against —
        the scrubber matches it exactly, so a rule can never retire a
        sibling assertion the same filter makes about the same code (see
        engine._apply_advisory_suppressions). "" matches only findings
        that themselves carry no clause.

        validator_categories: validator-issue categories that assert the
        SAME defect this suppression retires (e.g. the scrubber's
        long-term-medication advisory and the validator's
        'unjustified_zcode' WARNING are two emissions of one claim about
        one code). Matching WARNING issues on the same code are removed
        at report build (_apply_validator_advisory_suppressions) so the
        record never ships an advisory both suppressed and active —
        observed live (routine_00003): the Z79.01 advisory was suppressed
        in the scrub yet still shipped in validation_issues, warnings,
        pre_submission_audit_findings, and auto_coding_review_reasons."""
        fid = str(filter_id or "").strip()
        c = str(code or "").strip().upper()
        if not fid or not c:
            return
        cats = [str(v).strip() for v in (validator_categories or ())
                if str(v or "").strip()]
        entry = {"filter_id": fid, "code": c,
                 "rule_id": str(rule_id or ""),
                 "authority": str(authority or "")[:400],
                 "note": str(note or "")[:400],
                 "clause": str(clause or "").strip(),
                 "validator_categories": cats}
        if entry not in self._scrub_advisory_suppressions:
            self._scrub_advisory_suppressions.append(entry)

    def _apply_validator_advisory_suppressions(self) -> None:
        """Remove validator WARNING issues that a recorded advisory
        suppression declares to be the same adjudicated defect (matched on
        category ∈ validator_categories AND same code) — the source-level
        half of advisory suppression. Runs BEFORE _compute_tier and the
        report build, so every downstream mirror (validation_issues,
        warnings, pre_submission_audit_findings, auto_coding_review_
        reasons) is consistent by construction instead of annotated after
        the fact. Removals are recorded as corrections so the clinical
        audit sees a reported action, never a silent vanishing. WARNING
        only — an ERROR/CRITICAL issue is never config-suppressible,
        mirroring the scrubber's WARN-only contract."""
        directives = [s for s in self._scrub_advisory_suppressions
                      if s.get("validator_categories")]
        if not directives or not self.issues:
            return
        keyed = {}
        for s in directives:
            for cat in s["validator_categories"]:
                keyed[(cat, s["code"])] = s
        kept = []
        for issue in self.issues:
            s = (keyed.get((issue.category,
                            str(issue.code or "").strip().upper()))
                 if issue.severity == "WARNING" else None)
            if s is None:
                kept.append(issue)
                continue
            self._advisory_suppression_corrections.append({
                "category": "validator_advisory_suppressed",
                "code": issue.code,
                "action": "advisory_removed",
                "interpretive": False,
                "message": (f"Validator advisory '{issue.category}' on "
                            f"{issue.code} removed by rule "
                            f"{s.get('rule_id') or '(unnamed)'} — same "
                            f"adjudicated defect as the suppressed scrub "
                            f"advisory ({s.get('filter_id')}). Authority: "
                            f"{(s.get('authority') or '')[:160]}"),
            })
            logger.info(f"  Validator advisory '{issue.category}' on "
                        f"{issue.code} suppressed at source by rule "
                        f"{s.get('rule_id')!r}")
        self.issues[:] = kept
