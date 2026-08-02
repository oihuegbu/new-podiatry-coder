"""Filter #5 — Medical necessity (LCD/NCD ICD↔CPT linkage).

A procedure can be perfectly coded and still deny as "not medically necessary"
if the linked diagnosis isn't on the policy's covered list. CMS publishes NCDs
(national) and MACs publish LCDs/Billing & Coding Articles listing which ICD-10
codes justify a given CPT.

This agent is fully data-driven off the `coverage_cpt` / `coverage_icd` tables,
populated from the full CMS Coverage API dataset (hundreds of LCDs/Articles
across every specialty) via `store.load_coverage_articles(...)`.

E/M codes are excluded from the coverage-policy check even when a real LCD/
Article happens to mention one (e.g. "Cognitive Assessment and Care Plan
Service" bills using a standard E/M code) — an E/M visit's medical necessity
is established by its documented MDM, not by LCD code-pairing, which is a
procedure/DME/supply concept. Without this, any E/M code incidentally named
in an unrelated specialty's billing article would false-positive here.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

# CPT section boundary (code-system grammar, not a medical-rule list — same
# pattern as _EM_SECTION in modifiers.py/global_period.py/ncci_ptp.py).
_EM_SECTION = range(99202, 99500)

# Medicare routine-foot-care class-findings modifiers (Q7 = one class A
# finding, Q8 = two class B, Q9 = one class B + two class C) and the ABN/
# noncovered routing modifiers that legitimately replace them. Modifier
# references are structural (allowed per the no-hardcoding guard); WHICH
# CPT codes need them is derived from the governing policy's own CMS title
# ("routine foot care") — never a hardcoded procedure list.
_CLASS_FINDINGS_MODIFIERS = {"Q7", "Q8", "Q9"}
_ABN_ROUTING_MODIFIERS = {"GA", "GX", "GY", "GZ"}
_ROUTINE_FOOT_CARE_TITLE = "routine foot care"


def _is_em(code: str) -> bool:
    return code.isdigit() and int(code) in _EM_SECTION


class MedicalNecessityAgent(ComplianceAgent):
    filter_id = "MEDICAL_NECESSITY"
    filter_name = "Medical necessity (LCD/NCD coverage)"

    def _policy_coverage(self, policy_id: str, line_code: str,
                         dxs: list[str]) -> dict:
        """Evaluate ONE policy's coverage for one line against a diagnosis
        set, honoring the policy's own covered-ICD group grammar (roles
        parsed from the MCD export's group paragraphs — see
        store.coverage_groups / parsers.group_role_from_paragraph).

        full    — coverage established: a diagnosis hit a standalone
                  (unspecified-role) group or flat row, OR a primary-
                  eligible hit is accompanied by a required-secondary hit
                  (the claim-COMPOSITION rule, e.g. A57193: "B35.1 ...
                  must be reported as primary, with the diagnosis
                  representing the patient's symptom reported as the
                  secondary" — the corresponding coverage edit denies
                  automatically when the secondary is absent).
        partial — a primary-eligible diagnosis is on the claim but no
                  required-secondary accompanies it: the precise,
                  repairable composition gap.
        hits    — role → first matching diagnosis (evidence for messages).

        Policies ingested without group grammar evaluate exactly as
        before groups existed (any covered dx = full), never stricter."""
        meta = {g["group_id"]: g for g in self.store.coverage_groups(policy_id)}
        hits: dict[str, str] = {}
        for dx in dxs:
            for gid in self.store.coverage_dx_groups(policy_id, dx):
                g = meta.get(gid)
                if g and g["cpt_scope"] and line_code not in g["cpt_scope"]:
                    continue  # group scoped to other procedure codes
                role = (g or {}).get("role") or "unspecified"
                hits.setdefault(role, dx)
        full = ("unspecified" in hits
                or ("primary_eligible" in hits and "required_secondary" in hits))
        return {"full": full,
                "partial": "primary_eligible" in hits and not full,
                "hits": hits}

    def _secondary_group_provenance(self, policy_id: str) -> str:
        """Excerpt of the required-secondary group's own paragraph — the
        authority backing a composition finding."""
        for g in self.store.coverage_groups(policy_id):
            if g["role"] == "required_secondary" and g["paragraph"]:
                return g["paragraph"][:220]
        return ""

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        # all diagnoses available on the claim (CMS-1500: any dx supports any line)
        claim_dx = [d.code for d in claim.diagnoses]

        # A claim that bills procedures/supplies but carries NO diagnosis at all
        # cannot establish medical necessity — every CMS-1500 service line must
        # point to a diagnosis. This also guards against upstream coding failures
        # that drop the diagnoses (e.g. an LLM pass that fails to parse).
        billable = [ln for ln in claim.lines if ln.code]
        non_em_billable = [ln for ln in billable if not _is_em(ln.code)]
        if non_em_billable and (claim.payer.kind == "unknown" or
                                not claim.payer.follows_medicare_coverage):
            return [self.finding(
                status=Status.UNKNOWN,
                codes=[ln.code for ln in non_em_billable],
                denial_risk=DenialRisk.HIGH,
                reason=("Payer-specific medical-necessity coverage authority is "
                        "unavailable for this claim; Medicare LCD/NCD data cannot "
                        "be treated as the payer's policy."),
                recommendation="Load the identified payer and plan's effective-dated "
                               "coverage policy or route for human review.",
                source_rule="payer-specific medical-necessity policy availability",
            )]
        if billable and not claim_dx:
            findings.append(self.finding(
                status=Status.FAIL, codes=[ln.code for ln in billable],
                denial_risk=DenialRisk.HIGH,
                reason="Claim bills procedures/supplies but has NO diagnosis code — medical "
                       "necessity cannot be established and every service line requires a "
                       "diagnosis pointer.",
                recommendation="Add the supporting ICD-10-CM diagnosis(es); if none exist, the "
                               "services are not billable.",
                source_rule="CMS-1500 diagnosis-pointer requirement / ICD↔CPT linkage",
                clause="no_diagnosis",
            ))

        for line in claim.lines:
            if _is_em(line.code):
                continue  # E/M necessity is an MDM question, not LCD code-pairing
            all_policies = self.store.coverage_policies_for_cpt(line.code)

            # LCDs/Articles are LOCAL policies — each governs only the states
            # its issuing MAC adjudicates. Without this filter, a CGS (KY/OH)
            # LCD gated Florida claims; verified live on note 031, where all
            # eight policies cited against CPT 29445 came from non-Florida
            # MACs. Unknown claim state or unknown contractor stays
            # conservative (policy kept).
            policies = [p for p in all_policies
                        if self.store.policy_applies_in_state(p, claim.state)]
            excluded = [p for p in all_policies if p not in policies]
            if not policies:
                if excluded:
                    # A previous run may have FAILed this exact line against
                    # these policies (pre-jurisdiction-scoping) — leave an
                    # explicit "checked and passed, here's why" trail so the
                    # disappearance of that FAIL reads as a rule decision,
                    # not a skipped or flaky check.
                    findings.append(self.finding(
                        status=Status.PASS, codes=[line.code], denial_risk=DenialRisk.NONE,
                        reason=f"{len(excluded)} coverage polic"
                               f"{'y' if len(excluded) == 1 else 'ies'} "
                               f"({', '.join(excluded)}) govern{'s' if len(excluded) == 1 else ''} "
                               f"{line.code} nationally, but none is issued by the MAC serving "
                               f"{claim.state} — out-of-jurisdiction policies do not gate this "
                               f"claim.",
                        recommendation="No action needed. If the servicing state is wrong, "
                                       "correct the note's location/insurance details.",
                        source_rule="MAC jurisdiction scoping (CMS 'Who are the MACs' / "
                                    "mac_jurisdictions.json)",
                        clause="jurisdiction_scope",
                    ))
                continue  # no coverage policy governs this code in this jurisdiction

            # Policies that publish no covered-ICD list (e.g. broad PT/OT
            # billing articles) impose documentation rules, not a diagnosis
            # gate — an empty list must read as "no ICD restriction", not
            # "no diagnosis can ever satisfy this". Only policies WITH a
            # published list can participate in the coverage decision.
            restrictive = [p for p in policies
                           if self.store.coverage_policy_has_dx_rules(p)]

            # A code can legitimately be governed by multiple LCDs/Articles for
            # different clinical indications (e.g. a debridement code covered
            # under both a routine-foot-care policy and a separate wound-care
            # policy). The claim only needs to satisfy the ONE policy relevant
            # to this encounter, not all of them simultaneously — so check
            # coverage across all governing policies together, and only FAIL
            # if none of them are satisfied. Each policy is evaluated by its
            # OWN group grammar (claim composition), not a flat covered test.
            linked_dx = line.linked_diagnoses or claim_dx
            verdicts = {policy_id: self._policy_coverage(policy_id, line.code,
                                                         linked_dx)
                        for policy_id in restrictive}
            satisfied = any(v["full"] for v in verdicts.values())
            # Policies where a primary-eligible diagnosis is present but its
            # required secondary is missing — the composition gap (repairable
            # with the note's own documented symptom, never by dropping care).
            composition_gaps = [(pid, v) for pid, v in verdicts.items()
                                if v["partial"]]
            # Group-N mirror list: a policy can also (or ONLY) publish
            # diagnoses that explicitly do NOT support medical necessity —
            # the one diagnosis signal available for policies without a
            # covered list. Only consulted when no policy is satisfied.
            noncov = None if satisfied else next(
                ((policy_id, dx) for policy_id in policies for dx in linked_dx
                 if self.store.coverage_icd_explicitly_noncovered(policy_id, dx)),
                None,
            )
            covered = satisfied or (not restrictive and not noncov)
            # CMS-1500 pointer scope: any claim diagnosis can support any
            # line — a covered dx that exists on the claim but wasn't POINTED
            # at this line is a fixable linkage problem, not absent medical
            # necessity. Without this, note 031's covered E11.42 (on the
            # claim, unlinked) couldn't rescue 29445 from a FAIL. Evaluated
            # through the same composition grammar: an unlinked dx repoints
            # only if ADDING it makes some policy fully satisfied (it can
            # also complete a composition, e.g. supply the missing required
            # secondary).
            repoint_dx = None
            if not covered:
                repoint_dx = next(
                    (dx for dx in claim_dx
                     if dx not in linked_dx
                     and any(self._policy_coverage(pid, line.code,
                                                   linked_dx + [dx])["full"]
                             for pid in restrictive)),
                    None,
                )
            if covered or repoint_dx:
                if covered and not restrictive:
                    # Same "checked and passed" trail as the jurisdiction
                    # exclusion above: the governing policies publish no
                    # covered-ICD list, so there is no diagnosis gate to fail.
                    findings.append(self.finding(
                        status=Status.PASS, codes=[line.code], denial_risk=DenialRisk.NONE,
                        reason=f"Coverage polic{'y' if len(policies) == 1 else 'ies'} "
                               f"{', '.join(policies)} govern{'s' if len(policies) == 1 else ''} "
                               f"{line.code} but publish{'es' if len(policies) == 1 else ''} no "
                               f"covered-diagnosis list — documentation polic"
                               f"{'y' if len(policies) == 1 else 'ies'}, not a diagnosis gate.",
                        recommendation="No diagnosis action needed; ensure the policy's "
                                       "documentation requirements are met.",
                        source_rule=f"Coverage polic{'y' if len(policies) == 1 else 'ies'} "
                                    f"{', '.join(policies)} (no published ICD list)",
                        clause="no_dx_gate",
                    ))
                if repoint_dx:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}'s linked diagnoses "
                               f"({', '.join(linked_dx) or 'none'}) do not satisfy the "
                               f"governing coverage polic{'y' if len(restrictive) == 1 else 'ies'} "
                               f"({', '.join(restrictive)}), but {repoint_dx} — already on this "
                               f"claim — does. Point {line.code} at {repoint_dx}.",
                        recommendation=f"Add {repoint_dx} to {line.code}'s diagnosis pointers "
                                       f"before submission.",
                        source_rule="LCD/Article ICD↔CPT list + CMS-1500 diagnosis-pointer scope",
                        auto_fixable=True,
                        suggested_fix={"action": "link_diagnosis", "code": line.code,
                                       "diagnosis": repoint_dx},
                        clause="diagnosis_pointer",
                    ))
                # Routine foot care (identified by the governing policy's own
                # CMS title, not a hardcoded CPT list) additionally requires a
                # class-findings modifier (Q7/Q8/Q9) — or ABN/noncovered
                # routing (GA/GX/GY/GZ) — on Medicare claims even when a
                # qualifying diagnosis is present. WARN, not FAIL: the
                # documentation may support the findings without the modifier
                # having been appended yet, and non-Medicare payers vary.
                rfc_policies = self.store.policies_titled(policies, _ROUTINE_FOOT_CARE_TITLE)
                mods = {m.strip().upper() for m in line.modifiers}
                if (
                    rfc_policies
                    # follows_medicare_coverage, not is_medicare: MA plans are
                    # bound to Original Medicare's coverage rules (42 CFR
                    # 422.101), so routine-foot-care class findings apply to
                    # both FFS and Medicare Advantage claims.
                    and claim.payer.follows_medicare_coverage
                    and not (mods & (_CLASS_FINDINGS_MODIFIERS | _ABN_ROUTING_MODIFIERS))
                ):
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.HIGH,
                        reason=f"{line.code} is governed by routine-foot-care polic"
                               f"{'y' if len(rfc_policies) == 1 else 'ies'} "
                               f"{', '.join(rfc_policies)} and billed to Medicare without a "
                               f"class-findings modifier (Q7/Q8/Q9) or ABN routing modifier — "
                               f"routine foot care denies without documented class findings.",
                        recommendation="Append Q7 (one class A), Q8 (two class B), or Q9 (one class B "
                                       "+ two class C) if the exam documents the findings; otherwise "
                                       "obtain an ABN (GA/GX) or do not bill as covered.",
                        source_rule=f"Routine foot care coverage policy "
                                    f"({', '.join(rfc_policies)}) — class-findings requirement",
                        clause="class_findings_modifier",
                    ))
                continue

            # ---- claim-composition gap (conjunction gate) --------------
            # A primary-eligible diagnosis is on the claim, but the same
            # policy's grammar demands a required-secondary code (the
            # patient's symptom) alongside it and none is present. This is
            # a claim-COMPOSITION edit, not a documentation test: the note
            # may document pain / difficulty walking perfectly, yet the
            # edit denies automatically unless a code carrying that finding
            # is ON the claim. Emitted as its own clause so no advisory
            # suppression scoped to a sibling clause can ever retire it.
            if composition_gaps:
                gap_pids = [pid for pid, _ in composition_gaps]
                primary_hit = composition_gaps[0][1]["hits"].get(
                    "primary_eligible", "")
                plural = "y" if len(gap_pids) == 1 else "ies"
                provenance = self._secondary_group_provenance(gap_pids[0])
                caveat = ""
                if claim.state is None:
                    caveat = (" The claim's servicing state could not be "
                              "determined, so jurisdiction was not verified "
                              "— confirm the governing MAC before repair.")
                reason = (
                    f"{line.code} reaches coverage polic{plural} "
                    f"{', '.join(gap_pids)} only through {primary_hit}, which "
                    f"the polic{plural} cover{'s' if len(gap_pids) == 1 else ''} "
                    f"as a PRIMARY-eligible diagnosis that must be accompanied "
                    f"by a secondary code for the patient's symptom (pain, "
                    f"difficulty in ambulation, or secondary infection) — no "
                    f"code from the required-secondary list is on this claim, "
                    f"and the coverage edit denies automatically without one."
                    f"{caveat}")
                if provenance:
                    reason += f' [policy group text: "{provenance}"]'
                fix = {"action": "add_required_secondary_dx",
                       "code": line.code, "policy_ids": gap_pids,
                       "primary_diagnosis": primary_hit,
                       "role": "required_secondary"}
                if claim.payer.follows_medicare_coverage:
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code],
                        denial_risk=DenialRisk.HIGH,
                        reason=reason,
                        recommendation=(
                            "Add the diagnosis code for the symptom the note "
                            "documents (from the policy's required-secondary "
                            "list) as a secondary diagnosis, or do not bill "
                            f"{line.code} as covered. Do not remove "
                            f"{primary_hit} — it is the required primary."),
                        source_rule=f"Coverage polic{plural} "
                                    f"{', '.join(gap_pids)} — claim-composition "
                                    f"requirement (primary + symptom secondary)",
                        suggested_fix=fix,
                        clause="coverage_composition",
                    ))
                else:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code],
                        denial_risk=DenialRisk.LOW,
                        reason=f"{reason} (Medicare coverage rule; this "
                               f"claim's payer ({claim.payer.name}) is not "
                               f"bound by Medicare LCDs — advisory only).",
                        recommendation=f"Check {claim.payer.name}'s own "
                                       f"medical policy for {line.code}'s "
                                       f"claim-composition requirements.",
                        source_rule=f"Medicare coverage polic{plural} "
                                    f"{', '.join(gap_pids)} (advisory for "
                                    f"non-Medicare payer)",
                        suggested_fix=fix,
                        clause="coverage_composition",
                    ))
                continue
            gating = restrictive or [noncov[0]]
            policy_list = ", ".join(gating)
            plural = "y" if len(gating) == 1 else "ies"
            if noncov and not restrictive:
                # no covered list anywhere, but a linked dx is explicitly on a
                # Group-N "does not support medical necessity" list
                mismatch = (f"{line.code}'s diagnosis {noncov[1]} is explicitly listed as NOT "
                            f"supporting medical necessity under polic{plural} {policy_list}")
            else:
                mismatch = (f"{line.code} is governed by coverage polic{plural} {policy_list}, "
                            f"but none of the claim's diagnoses satisfy any of them")
            # Jurisdiction-unverifiable denial predictions must not hard-FAIL:
            # when the claim's state is UNKNOWN, jurisdiction-scoped policies
            # are kept conservatively (see policy_applies_in_state) — but if
            # EVERY unsatisfied policy is scoped to a known, limited MAC
            # service area, the claim's actual state may well be outside all
            # of them, making "will deny" an unverifiable claim, not a fact.
            # (Observed live: 76942 FAILed against 15 policies from assorted
            # MACs on a claim whose note carried no state signal at all.)
            # A policy whose service area CAN'T be resolved (None) may be
            # national — those still FAIL.
            jurisdiction_unverifiable = claim.state is None and all(
                self.store.coverage_policy_states(p) is not None for p in gating
            )
            if jurisdiction_unverifiable and claim.payer.follows_medicare_coverage:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{mismatch} — however, every governing policy is a LOCAL coverage "
                           f"policy scoped to a specific MAC service area, and this claim's "
                           f"servicing state could not be determined from the note. If the "
                           f"practice is outside those jurisdictions, none of these policies "
                           f"apply — coverage cannot be verified either way without the state.",
                    recommendation="Determine the servicing state (letterhead/facility address) "
                                   "and re-verify; if a governing policy does apply, add a "
                                   "qualifying covered diagnosis or do not bill the line.",
                    source_rule=f"Coverage polic{plural} {policy_list} — jurisdiction "
                                f"unverifiable (claim state unknown)",
                    clause="diagnosis_coverage",
                ))
            elif claim.payer.follows_medicare_coverage:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{mismatch} — will deny as not medically necessary.",
                    recommendation=f"Add a qualifying diagnosis covered under one of {policy_list} "
                                   f"(and document it), or do not bill {line.code}.",
                    source_rule=f"Coverage polic{plural} {policy_list} "
                               f"(LCD/NCD/Article ICD↔CPT list)",
                    clause="diagnosis_coverage",
                ))
            else:
                # LCDs/Articles are MEDICARE coverage policies — they bind FFS
                # Medicare and MA plans (42 CFR 422.101), not Medicaid or
                # commercial payers, who publish their own coverage criteria.
                # For those payers a Medicare-policy mismatch is a heads-up to
                # verify the actual payer's policy, never a denial prediction.
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                    reason=f"{mismatch} (Medicare coverage rule). This claim's payer "
                           f"({claim.payer.name}) is not bound by Medicare LCDs — advisory only; "
                           f"verify against the payer's own medical policy.",
                    recommendation=f"Check {claim.payer.name}'s coverage criteria for {line.code}; "
                                   f"no action needed if its policy covers the documented indication.",
                    source_rule=f"Medicare coverage polic{plural} {policy_list} "
                               f"(advisory for non-Medicare payer)",
                    clause="diagnosis_coverage",
                ))
        return findings
