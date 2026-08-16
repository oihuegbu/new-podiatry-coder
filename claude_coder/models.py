"""Core data model for claude-medical-coder.

Provenance is built in BY CONSTRUCTION: every code that reaches a claim carries
the evidence span it came from, the clinical fact it resolved, the authoritative
record that defines it, and how it was chosen. There is not a single medical
code literal in this file — codes only ever arrive as data pulled from the
authoritative source at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FactKind(str, Enum):
    PROCEDURE = "procedure"
    DIAGNOSIS = "diagnosis"
    SUPPLY = "supply"                 # DME / dressing / therapeutic shoe / insert
    DRUG = "drug"
    IMAGING = "imaging"
    EM = "evaluation_management"


# Which authoritative code system a fact of this kind resolves against. This is
# a structural mapping (fact category -> code system), NOT a code mapping.
SYSTEM_FOR_KIND: dict[FactKind, str] = {
    FactKind.PROCEDURE: "cpt",
    FactKind.DIAGNOSIS: "icd10",
    FactKind.SUPPLY: "hcpcs",
    FactKind.DRUG: "hcpcs",
    FactKind.IMAGING: "cpt",
    FactKind.EM: "cpt",
}


class Disposition(str, Enum):
    """Was the service actually rendered? Only PERFORMED/dispensed work is
    billable; everything else must be excluded or reviewed, never coded."""
    PERFORMED = "performed_today"
    ORDERED = "ordered"
    PLANNED = "planned"
    DISCUSSED = "discussed"
    HISTORICAL = "historical"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class EvidenceSpan:
    """Verbatim text copied from the note — the atom of defensibility. Phase-0 adds the
    VERIFIED anchor: the exact [start, end) offset of the quote in the source and its
    content hash. `anchored` is True only when note_text[start:end] == text was proven;
    an unanchorable quote keeps text (for audit) but anchored=False — never a guess."""
    text: str
    section: str | None = None
    start: int | None = None
    end: int | None = None
    text_sha256: str | None = None
    anchored: bool = False
    # Location-specific provenance. text_sha256 proves content but is not an
    # identity: the same quotation can occur more than once in a document. The
    # span id binds document/version + offsets + content (and optional page/section).
    document_sha256: str | None = None
    document_version: str | None = None
    span_id: str | None = None
    page: int | None = None
    # WHERE IN THE ORIGINAL DOCUMENT, and by WHOSE reading (issue #6 F6-R6-A).
    # `page`/`start`/`end` above locate the quote in the TRANSCRIPTION; these locate it
    # in the document the physician actually signed, and record which INDEPENDENT
    # channel proved it. `source_reconciliation` is a
    # `contracts.source_evidence.ReconciliationStatus` value; None means the question
    # was never asked (no source document accompanied the encounter), which is a
    # different — and separately routed — thing from "asked and unproven".
    page_image_sha256: str | None = None
    region: tuple[float, float, float, float] | None = None
    source_reconciliation: str | None = None
    verified_by_channel_id: str | None = None


@dataclass
class ClinicalFact:
    """A billable clinical event in plain clinical language — never a code.

    `attributes` carries the axes that DETERMINE a code (anatomy, laterality,
    count, depth, area, product, dose…) so a deterministic resolver can map the
    fact to a code from the authoritative data instead of the model guessing a
    code from memory.
    """
    kind: FactKind
    description: str
    attributes: dict[str, Any] = field(default_factory=dict)
    # Direct construction states a known intent (the callers here are trusted code
    # asserting a performed event). The fail-closed guard for UNTRUSTED input lives
    # at the trust boundary — extraction._coerce_disposition maps a missing/malformed
    # disposition from model output to UNCLEAR, so a real note never bills an event
    # whose disposition was not explicitly documented.
    disposition: Disposition = Disposition.PERFORMED
    # Assertion axes (ICD-10-CM outpatient rules): `certain` is False for a
    # suspected/probable/rule-out condition (never coded as confirmed); `experiencer`
    # is "family"/"other" when the condition belongs to someone other than the
    # patient (family history is not the patient's coded condition). Defaults assert
    # the common case; extraction sets them from the note.
    certain: bool = True
    experiencer: str = "patient"
    evidence: list[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0
    # Per-axis extraction confidence (e.g. occurrence, action, anatomy, laterality,
    # performer, relationship, measurement, temporal). Kept SEPARATE so a high overall
    # read cannot conceal a weak axis; gating uses the weakest, never an average. Empty
    # until extraction populates it -> min_confidence falls back to the scalar.
    axis_confidence: dict[str, float] = field(default_factory=dict)
    # Code-changing axes that TWO INDEPENDENT READINGS of the original document read
    # differently and the ORIGINAL PAGE could not settle (`claude_coder.graph_consensus`).
    # Each entry is the precise, self-contained question to send to the provider. A
    # non-empty list holds the event before retrieval and routes it to PROVIDER_QUERY --
    # never to a coder queue, which the product directive forbids for model disagreement.
    axis_conflicts: list[str] = field(default_factory=list)
    fact_id: str = ""

    @property
    def system(self) -> str:
        return SYSTEM_FOR_KIND[self.kind]

    @property
    def min_confidence(self) -> float:
        """The WEAKEST signal that should gate autonomy: the minimum over the recorded
        per-axis confidences AND the scalar (never an average). Falls back to the scalar
        when no per-axis values exist."""
        vals = list(self.axis_confidence.values())
        return min([self.confidence, *vals]) if vals else self.confidence

    @property
    def weakest_axis(self) -> str | None:
        """The name of the lowest-confidence axis, or None when none are recorded."""
        if not self.axis_confidence:
            return None
        return min(self.axis_confidence, key=self.axis_confidence.get)

    @property
    def billable(self) -> bool:
        # Never code an UNCERTAIN condition (suspected/probable/rule-out) as confirmed,
        # nor a condition belonging to someone other than the PATIENT (family history
        # / other) — ICD-10-CM outpatient coding rules. Then: a documented diagnosis
        # is codeable (it establishes necessity) unless purely historical; a
        # procedure/supply/drug/imaging is codeable only if actually PERFORMED today.
        if not self.certain or self.experiencer != "patient":
            return False
        if self.kind is FactKind.DIAGNOSIS:
            return self.disposition is not Disposition.HISTORICAL
        return self.disposition is Disposition.PERFORMED


@dataclass(frozen=True)
class CandidateCode:
    """A code offered by the authoritative source for a fact. The descriptor,
    activity window and any policy attributes are copied straight from the data
    — the coder never authors them."""
    code: str
    system: str
    descriptor: str
    score: float = 0.0                 # recall relevance (similarity), for ranking
    source: str = ""
    authority: dict[str, Any] = field(default_factory=dict)   # data provenance


class ResolutionMethod(str, Enum):
    DETERMINISTIC = "deterministic"   # one candidate whose descriptor entails the fact
    ARBITRATED = "llm_arbitrated"     # model picked among retrieved candidates -- including
                                      # an entailment whose corroborating second opinion did
                                      # not come from an INDEPENDENT origin (see below)
    VERIFIED = "verified_entailment"  # candidate whose authoritative descriptor the
                                      # documentation entails (propose-then-verify) AND
                                      # which an INDEPENDENT second model confirmed. Both
                                      # halves are enforced in `resolution._entailed_line`;
                                      # `autonomy` treats this as grounded, so agreement
                                      # between two calls to ONE provider must never mint it.
    ABSTAINED = "abstained"           # genuine ambiguity / no candidate -> review


@dataclass
class ResolvedLine:
    fact: ClinicalFact
    chosen: CandidateCode | None
    alternatives: list[CandidateCode] = field(default_factory=list)
    method: ResolutionMethod = ResolutionMethod.ABSTAINED
    rationale: str = ""
    modifiers: list[str] = field(default_factory=list)   # data-driven, e.g. RT/LT/50
    units: int = 1                                        # billing units (descriptor-driven)
    # set when a resolved code is NOT a separately reportable line (bundled /
    # non-covered per data): it is kept for the audit trail but not billed.
    excluded_reason: str | None = None
    # set when the line escalated because the best-matching code needs an element
    # the documentation does not state — carries the specific gap for a provider query.
    documentation_gap: str | None = None

    @property
    def resolved(self) -> bool:
        return self.chosen is not None


class Outcome(str, Enum):
    """Gate outcomes are POSITIVE assertions. Release requires PASS or a proven
    NOT_APPLICABLE; UNKNOWN / ERROR / BLOCKED all stop autonomy. There is no
    "clean = absence of failure" — a check that did not run is not a pass."""
    PASS = "PASS"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass
class GateResult:
    name: str
    outcome: Outcome
    detail: str = ""
    authority: str = ""               # what source/rule decided this
    # True when this gate could not clear because an AUTHORITY was UNAVAILABLE (data
    # not loaded / a lookup error) — an OPERATIONAL problem a retry can fix, NOT a
    # coding decision. The router sends these to SYSTEM_HOLD, never to a coder.
    retryable: bool = False

    @property
    def clears(self) -> bool:
        return self.outcome in (Outcome.PASS, Outcome.NOT_APPLICABLE)


class Verdict(str, Enum):
    AUTO_READY = "AUTO_READY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"


class Destination(str, Enum):
    """Where a non-auto-released encounter should ACTUALLY go — so operational
    failures, documentation gaps, and genuine coding judgement don't collapse into
    one human queue. A coder only ever sees REVIEW."""
    AUTO_READY = "AUTO_READY"          # release to billing, no human
    SYSTEM_HOLD = "SYSTEM_HOLD"        # operational/data failure -> retry + ops alert (not a coder)
    PROVIDER_QUERY = "PROVIDER_QUERY"  # documentation gap -> one structured question to the provider
    REVIEW = "REVIEW"                  # genuine coding/clinical judgement -> coder
    HOLD = "HOLD"                      # documentation cannot support a claim -> do not bill
    BLOCKED = "BLOCKED"                # a hard release gate failed


@dataclass
class CodingResult:
    encounter_id: str
    date_of_service: str | None
    lines: list[ResolvedLine] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    verdict: Verdict = Verdict.REVIEW_REQUIRED
    notes: list[str] = field(default_factory=list)
    certificate: dict[str, Any] | None = None   # tamper-evident evidence packet
    bypassed_ncci: list = field(default_factory=list)   # code pairs cleared by a modifier
    # NCCI PTP component codes DEMOTED during reconciliation (bundled into a payable
    # comprehensive code). Recorded as (component, payable) so the NCCI gate can report
    # that PTP was MATERIAL even when the RELEASED claim ends up with a single procedure
    # — instead of the misleading "fewer than two procedures / NOT_APPLICABLE".
    ncci_suppressed: list = field(default_factory=list)
    # actionable documentation recommendations (what to document/clarify to code it)
    recommendations: list[dict] = field(default_factory=list)
    # the actionable next-step destination (set by autonomy.decide) and the per-item
    # routing breakdown, so an encounter that isn't AUTO_READY is dispatched to the
    # RIGHT place (retry / provider / coder / hold) instead of one review queue.
    destination: "Destination | None" = None
    routing: list[dict] = field(default_factory=list)
    # Enforced pre-retrieval lineage. These are deliberately first-class rather
    # than hidden in a log so the certificate can bind the exact eligibility graph
    # that authorized every retrieval call.
    claim_line_intents: list[Any] = field(default_factory=list)
    relations: list["RelationAssertion"] = field(default_factory=list)
    audit_record_hashes: list[str] = field(default_factory=list)
    control_mode: str = "ENFORCED_FAIL_CLOSED"
    # WHY each released service was medically necessary, written by the necessity gate:
    # the claim-line diagnosis pointer plus the accepted relation's provenance (relation id,
    # reconciliation status, distinct assertion origins) and the coverage-policy disposition.
    # The certificate binds this, so the justification is auditable from the artifact rather
    # than only internally consistent. (Codex F6-R3.)
    necessity_support: list[dict] = field(default_factory=list)
    # Proof that every quotation this claim rests on says, in the ORIGINAL document,
    # what the transcription claims it says — a
    # `contracts.source_evidence.SourceReconciliation`. None means no source document
    # accompanied the encounter; `document_version` is what tells the gate whether
    # that absence is honest (text supplied directly) or a bypass (a document was
    # read, but nothing checked the reading). (Issue #6 F6-R6-A, directive §1.)
    source_reconciliation: Any = None
    document_version: str | None = None
    # WHERE `date_of_service` above came from and how it was proven -- an
    # `app.contracts.claim_bundle.ServiceDateBinding`, serialized. Bound into the
    # certificate so the claim's single most date-versioned value is answerable
    # after the fact instead of being an unattributed string. (Issue #6 F7-R4.)
    service_date_binding: dict | None = None
    # THE single clinical representation this encounter was decided from -- a
    # `claude_coder.graph.ClinicalGraph`. Extraction fills it, eligibility roles it,
    # retrieval is authorized by it, the certificate binds it and claim assembly reads
    # it to say which nodes/edges each released line rests on. None only on a
    # pre-retrieval system hold, which has no lines either.
    graph: Any = None
    # The two-reading axis comparison record (`graph_consensus.ConsensusReport`
    # serialized), or None when only one reading was taken.
    consensus: dict | None = None

    @property
    def billable_lines(self) -> list[ResolvedLine]:
        return [ln for ln in self.lines
                if ln.resolved and ln.fact.billable and not ln.excluded_reason]

    @property
    def procedure_lines(self) -> list[ResolvedLine]:
        return [ln for ln in self.billable_lines
                if ln.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING, FactKind.EM)]

    @property
    def diagnosis_lines(self) -> list[ResolvedLine]:
        return [ln for ln in self.billable_lines
                if ln.fact.kind is FactKind.DIAGNOSIS]


class RelationPredicate(str, Enum):
    """Generic CLINICAL relationships between events — never code logic. The eligibility
    engine (Phase 1) reads these; it does not let any single one decide billability."""
    PART_OF = "part_of"
    USED_IN = "used_in"
    REASON_FOR = "reason_for"
    SAME_EPISODE_AS = "same_episode_as"
    SEPARATE_FROM = "separate_from"
    PERFORMED_BY = "performed_by"
    ON_BEHALF_OF = "on_behalf_of"


class RelationState(str, Enum):
    ASSERTED = "asserted"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


@dataclass
class RelationAssertion:
    """A first-class, content-addressed edge between two clinical events (fact_ids).
    Records what the DOCUMENTATION asserts about their relationship, with the evidence
    and how sure the extractor was — separate from any eligibility/billability decision.
    Identity is (subject, predicate, object): the same edge asserted twice merges and
    accumulates support instead of creating a duplicate line."""
    subject_event_id: str
    predicate: RelationPredicate
    object_event_id: str
    state: RelationState = RelationState.ASSERTED
    evidence_span_ids: list[str] = field(default_factory=list)
    extraction_source: str = ""
    confidence: float = 0.0
    # GROUNDING: what the SOURCE DOCUMENT establishes about this edge, written only by
    # `provenance.reconcile_relations` (values: `provenance.RECONCILIATION_STATUSES`). A
    # claim-affecting control may accept only a member of
    # `provenance.GROUNDED_RECONCILIATION_STATUSES`.
    reconciliation_status: str = "unreconciled"
    # The verified span ids that ESTABLISHED `reconciliation_status` (the two endpoint
    # mentions for a directional proof, the shared passage for a co-location observation).
    # Written by the deterministic provenance layer so a certificate can show WHICH source
    # text proved the relationship, not merely that something did. A grounded status always
    # names at least one span; an empty list means nothing in the record was proved.
    reconciliation_evidence: list[str] = field(default_factory=list)
    # AGREEMENT: whether DISTINCT assertion origins asserted this same edge (values:
    # `provenance.CORROBORATION_STATUSES`; the literal default mirrors
    # `provenance.SINGLE_ORIGIN`, which this module cannot import without a cycle -- a
    # regression asserts they stay equal). Deliberately a SEPARATE field from
    # `reconciliation_status`: agreement between model runs is audit and confidence
    # information, and can never by itself make a relation claim-affecting, so no control
    # reads this value as justification. (Codex F6-R3, round 5.)
    corroboration_status: str = "single_origin"
    # RAW number of times this edge was asserted, across every origin. Observability only:
    # it says nothing about INDEPENDENCE, because one model can emit the same edge twice in
    # one response. Corroboration reads `independent_support`. (Codex F6-R3.)
    support: int = 1
    # The DISTINCT recorded assertion origins that asserted this edge -- one opaque id per
    # (run, provider/profile, prompt, schema) extraction call, stamped by
    # `extraction.extract_note` from the call's own recorded metadata. Two duplicate edges
    # from ONE response share one origin id and therefore corroborate nothing; the same edge
    # from two separate runs carries two ids and legitimately does.
    assertion_origins: list[str] = field(default_factory=list)

    @property
    def independent_support(self) -> int:
        """How many DISTINCT assertion origins asserted this edge. An edge with no recorded
        origin scores 0 -- unknown provenance can never be counted as independent support."""
        return len({str(o).strip() for o in (self.assertion_origins or []) if str(o).strip()})

    @property
    def relation_id(self) -> str:
        import hashlib
        pred = self.predicate.value if isinstance(self.predicate, RelationPredicate) \
            else str(self.predicate)
        raw = f"{self.subject_event_id}|{pred}|{self.object_event_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
