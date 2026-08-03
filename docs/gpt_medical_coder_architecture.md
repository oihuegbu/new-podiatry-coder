# GPT Medical Coder architecture

## Objective

This branch implements an autonomous professional-claim coding system that
turns a complete doctor note into a candidate claim and releases it only when
the record proves that the result is accurate enough to bill, valid for the
encounter date and payer context, and reproducibly defensible in an audit.

“Autonomous” means that routine, well-supported encounters can move from note
to a submission-ready claim without a human coder. It does not mean guessing
through ambiguity. Missing authority, conflicting facts, unsupported codes,
provider disagreement, or an audit dispute produces `REVIEW` or `REJECT`, never
a weaker claim labeled clean.

The design reuses useful repository assets—licensed/versioned code sources,
the deterministic validator templates, claim scrubber, payer registry,
submission adapter, and regression corpus—but replaces the former repeated
end-to-end voting process with the evidence-first architecture below.

## Online claim path

1. **Acquire the complete source.** Render every PDF page at high resolution,
   retain source/page hashes, require an explicit per-page transcription, and
   fail if any page is omitted. If the PDF has an independent embedded text
   layer, compare its token multiset with the vision transcript and block a
   material contradiction. Image-only scans are marked not applicable rather
   than treated as corroborated.
2. **Compile one immutable evidence packet.** Extract metadata and sections,
   preserve exact note text, normalize terminology, build evidence-bound
   clinical facts, retrieve candidates, and bind the applicable authoritative
   source versions. Canonical JSON and a cryptographic fingerprint make the
   nested packet deeply immutable across worker processes.
3. **Normalize terminology without replacing evidence.** Every abbreviation
   retains its raw phrase and source span. Candidate expansions come from the
   versioned terminology registry and are resolved using section, anatomy,
   laterality, surrounding terms, negation, and ambiguity rules. Retrieval sees
   both raw and expanded forms. Uniquely supported, high-confidence terms can
   proceed; unresolved terms capable of changing a billed line block release.
4. **Retrieve, do not memorize.** ICD-10-CM, CPT, HCPCS, modifiers, NCCI PTP,
   MUE, PFS indicators, coverage policy, place of service, payer requirements,
   and terminology are loaded from versioned authoritative data. Production
   decision code contains no medical-code literals, prefix families, or local
   approximations of source fields.
5. **Obtain provider-independent coding opinions.** One Anthropic profile and
   one OpenAI profile independently code the identical evidence packet. Their
   provider/model identities are persisted. Repeating one vendor cannot satisfy
   the independent-domain gate.
6. **Compare deterministic claim identities.** Diagnoses compare designation;
   services compare code, modifiers, and units. The comparison also proves that
   both runs used the same source document, extraction, patient/encounter
   context, terminology registry, clinical facts, and retrieval lexicon.
7. **Resolve only the disputed surface.** Agreement selects the canonical
   candidate. Disagreement goes to a bounded, authority-grounded adjudicator
   using independent provider profiles. It may touch only disputed lines or
   attributes, must cite note evidence and loaded authority, and must abstain
   when the sources do not decide. The mechanically realigned claim is replayed
   through validation; no majority vote releases a claim.
8. **Validate billability and claim composition.** Generic deterministic
   mechanics apply versioned rule configuration and date-of-service lookups for
   code activity, specificity, documentation, diagnosis linkage, modifiers,
   units, PTP edits, MUE, PFS indicators, payer/coverage policy, place of
   service, identifiers, and performed-service completeness.
9. **Record every mutation.** The pre-validation candidate claim is immutable.
   Every suppression or correction is diffed, attributed, and placed in the
   mutation ledger. Unaccounted changes block release.
10. **Run an independent whole-claim audit.** Two scored expert-audit passes
    inspect the complete note once, clinical facts, terminology decisions,
    final and candidate claims, model comparison, adjudication, mutations,
    scrub results, authority versions, and readiness controls. They must agree;
    unsupported or uncertain findings remain held.
11. **Issue an immutable readiness certificate.** Certification is rebuilt
    after cross-provider comparison and after any later correction. It binds
    the source PDF, evidence packet, final claim, source manifest, deterministic
    controls, model independence, and audit outcome. Any subsequent change
    invalidates the certificate.
12. **Submit only an authorized clean claim.** Submission requires a signed
    autonomy scope, valid identifiers and practice/payer configuration, a live
    readiness certificate, registry eligibility, and `CLEAN`. Missing data does
    not receive a placeholder or inferred value.

## Date-of-service authority

Current data is not automatically correct for a delayed or corrected claim.
Before retrieval, the system checks whether retained NCCI, MUE, and PFS
provenance covers the encounter quarter. Missing CMS snapshots are fetched by
target quarter and ingested additively; prior snapshots are never overwritten.
If CMS does not make the required historical file available and it is not
already retained, the claim stays blocked. Code-set edition and quarterly
release windows are independently enforced by the source-requirement pack.

## Online/offline separation

The production claim path is read-only with respect to decision logic. It may
refresh versioned authorities and append audit/claim events, but it cannot
synthesize, promote, merge, or rewrite rules. Rule proposals, regression replay,
template graduation, audit convergence, and pack consolidation run only in an
explicit offline-maintenance process. Verified exemplars default to shadow mode
and cannot silently change production prompts because a registry count crossed
a threshold.

## Defensibility bundle

Every released claim carries enough information to reconstruct why each line
was billed:

- source PDF, extracted-text, page, evidence, and clinical-fact fingerprints;
- exact supporting note spans and performed-event quotes;
- raw terminology, expansion, alternatives, source, and confidence;
- authoritative source record identifiers, checksums, release windows, and
  date-of-service applicability;
- candidate claim, final claim, and complete mutation ledger;
- independent model identities and item-level agreement/adjudication;
- deterministic validation and scrub outcomes;
- independent clinical-audit verdicts; and
- a final readiness certificate that becomes invalid if any bound input changes.

## Failure behavior

The system fails closed for incomplete pages, extraction/text-layer conflict,
unresolved billing-relevant terminology, absent date-of-service authority,
inactive or unsupported lines, incomplete performed-service accounting,
invalid identifiers, payer ambiguity, non-independent model execution,
unresolved model disagreement, unaccounted mutation, audit dispute, stale
source manifest, or changed certification inputs. Transient network/model
failures are retried within bounds and then surfaced; they never produce an
empty successful claim.

## External design patterns adopted

The architecture independently implements useful patterns publicly described
by autonomous-coding vendors: structured clinical narratives and exact audit
trails (Nym), direct-to-bill confidence gating and deficiency flags (Fathom),
source-text justification for recommendations (AKASA), and contextual coding
instead of isolated code prediction (CodaMetrix). Vendor performance claims are
marketing claims and are not treated as evidence of this system’s performance.

Official architecture/product pages:

- https://nym.health/autonomous-medical-coding/the-technology/
- https://www.fathomhealth.com/primary-care-coding-automation
- https://akasa.com/press/unveils-akasa-medical-coding
- https://www.codametrix.com/about

CMS update sources used by the refresh design:

- https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-medically-unlikely-edit-mue-archive
- https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-faq-library
- https://www.cms.gov/medicare/payment/fee-schedules/physician/pfs-relative-value-files

## Evidence still required for performance claims

Architecture, source authority, and fail-closed tests can prove process
integrity; they cannot prove a clinical accuracy percentage. Sensitivity,
specificity, false-positive rate, claim-level exact match, direct-to-bill rate,
and specialty/generalization performance require a sufficiently broad,
independently double-reviewed gold corpus. Until that evaluation exists, the
system must report measured operational outcomes and must not advertise an
unvalidated accuracy rate.
