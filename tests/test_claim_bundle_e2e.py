"""The canonical claim path, end to end, through the DEPLOYED entrypoint.

================================================================================
WHAT THIS PROVES — issue #6 F6-R4-A1, product directive §5 acceptance test
================================================================================
    original document -> run.main -> ClaimBundle -> registry -> 837P dry-run

with the same codes, units, modifiers, diagnosis links, context fingerprint and
certificate at every hop — and with tampering or stale context/data stopping the
dry-run.

The finding this closes was not a crash. `run.py` emitted one result shape and
the retained claim path read another, so a perfectly good AUTO_READY encounter
arrived at the registry with zero diagnosis and zero service lines and was
refused as "pipeline did not succeed". Nothing logged an error. The only test
that can catch that class of defect is one that carries a real encounter across
every boundary and compares what came out with what went in, which is what this
module does.

WHAT IS REAL HERE
-----------------
`run.main()` — the exact Python target of `docker compose run app python run.py`
— `claude_coder.pipeline.code_encounter`, the real `AuthoritativeSource` (real
reference tables, real NCCI/MUE/coverage lookups, real capability manifest, real
data fingerprint), every release gate, the real certificate, the real
`app/contracts/claim_bundle.py` contract, the real `tools/claims_registry.py`
ingest and the real `tools/claim_submitter.py` 837P builder.

WHAT IS SUBSTITUTED, AND WHY IT CANNOT MANUFACTURE THE RESULT
-------------------------------------------------------------
  `run.extract_from_pdf`   the vision transcriber: an INPUT, not a decision
                           component. Stubbed so the test needs no API key.
  the extraction LLM       a well-formed fact/relation graph, so extraction
                           genuinely succeeds and the encounter is held (or
                           released) for the reason under test rather than for
                           a malformed-input reason.
  the retrieval index      a stub that returns ONE candidate per system, whose
                           code is chosen AT RUNTIME from the real authoritative
                           tables (active on the date of service, and — for the
                           procedure — ungoverned by any CMS coverage policy).
                           Building a real 60-90 minute Qdrant index inside a
                           test is not an option; the descriptor, activity
                           window, modifiers, units, MUE limit and NCCI status
                           all still come from the authoritative data.
  the verify/corroborate   answer "this descriptor is entailed". They stand in
  LLMs                     for two independent model judgements; the pipeline
                           still requires BOTH, from two declared providers,
                           before it will mint a VERIFIED line.

NO MEDICAL CODE, PAYER, POS VALUE OR NPI IS HARDCODED. Every one of them is
selected at runtime from the authoritative data the deployment loads, so this
test cannot rot into an assertion about a code set that has since changed — and
cannot silently pass because a literal happened to still exist.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run as entrypoint  # noqa: E402  — the exact module the deployment executes
from app.contracts.claim_bundle import (  # noqa: E402
    SCHEMA_ID, SCHEMA_VERSION, load_bundle)
from app.contracts.source_evidence import (  # noqa: E402
    ReconciliationStatus)
from app.ingestion.source_evidence import (  # noqa: E402
    EMBEDDED_TEXT_CHANNEL_ID)
from tests.source_pdf import build_pdf, digest_of, vision_extraction  # noqa: E402
from tools import claim_submitter as cs  # noqa: E402
from tools import claims_registry as reg  # noqa: E402

STEM = "NOTE_BUNDLE_E2E_001"
DOS_ISO = "2026-03-14"

#: Roster identifiers. Opaque strings — the point of the normalized context
#: source is that identities are declared once and referenced BY IDENTIFIER, so
#: what the identifiers say is irrelevant and what they LINK is everything.
PATIENT_ID = "PAT-E2E-1"
COVERAGE_ID = "COV-E2E-1"
PAYER_ID = "PAY-E2E-1"
ENTITY_ID = "ENT-E2E-1"
AFFILIATION_ID = "AFF-E2E-1"
FACILITY_ID = "FAC-E2E-1"
#: Every time-bound record starts well before the DOS and is open-ended, so a
#: test that changes a window is changing the ONLY thing under test.
ROSTER_START = "2020-01-01"
EXTRACTED_TEXT_SHA = "sha256:" + "c" * 64

#: Synthetic note. Disposition/negation/laterality logic turns on LINGUISTIC
#: markers, never on any clinical term, so nothing here needs to be real.
#: The DATE OF SERVICE IS WRITTEN ON THE PAGE, deliberately (issue #6 F7-R4).
#: The claim's date of service is no longer whatever the vision model reported in
#: its structured metadata: it binds only when the encounter-context source
#: declares one, or when the date the document reports is located on a page and
#: reconciled against an INDEPENDENT reading of that page. A fixture whose note
#: never states its own date could only ever hold, so the positive path here would
#: stop proving anything.
NOTE_TEXT = (
    f"Date of service: {DOS_ISO}. "
    "Procedure: excision of lesion alpha, right site two. "
    "Assessment: condition alpha, right side. "
    "Excision of lesion alpha was performed for condition alpha of the right side. "
    "Patient denies finding gamma."
)

#: The ORIGINAL DOCUMENT's identity, computed from the exact bytes this suite writes to
#: disk. The Source Evidence Compiler recomputes the digest of the file it compiles and
#: refuses a transcription that claims a different one (issue #6 F7-R5), so a fixture
#: cannot assert a document identity its document does not have.
DOCUMENT_VERSION = digest_of(build_pdf([[NOTE_TEXT]]))

#: The encounter's structured participant roster — what `claude_coder` resolves
#: claim OWNERSHIP against. Distinct from the encounter CONTEXT below: this says
#: who performed the service, that says who is billed and to whom.
BILLING_CONTEXT = {
    "billing_entity_id": "ORG_E2E",
    "participants": [
        {"id": "PERSON_E2E", "type": "person", "roles": ["performer"],
         "function": "performed the documented procedure",
         "affiliations": ["ORG_E2E"]},
        {"id": "ORG_E2E", "type": "organization"},
    ],
}

FACTS_JSON = json.dumps({
    "facts": [
        {"fact_id": "F1", "kind": "procedure",
         "description": "excision of lesion alpha",
         "attributes": {"laterality": "right", "anatomy": "site two",
                        "performer_id": "PERSON_E2E",
                        "organization_id": "ORG_E2E"},
         "disposition": "performed_today", "negated": False,
         "evidence": ["excision of lesion alpha, right site two",
                      "Excision of lesion alpha was performed"],
         "confidence": 0.97,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "performer": 0.99,
                             "relationship": 0.99}},
        {"fact_id": "F2", "kind": "diagnosis",
         "description": "condition alpha of the right side",
         "attributes": {"laterality": "right"},
         "disposition": "performed_today", "negated": False,
         "evidence": ["condition alpha, right side",
                      "condition alpha of the right side"],
         "confidence": 0.98,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "assertion": 0.99,
                             "experiencer": 0.99}},
    ],
    "relations": [
        {"subject_event_id": "F2", "object_event_id": "F1",
         "predicate": "reason_for", "state": "asserted",
         "evidence_fact_ids": ["F1", "F2"], "confidence": 0.99},
    ],
})


# --------------------------------------------------------------------------
# runtime-selected fixture data — nothing below is a literal from a code set
# --------------------------------------------------------------------------

def _valid_npi(nine_digits: str) -> str:
    """A 10-digit NPI with a correct CMS check digit (ISO 7812 Luhn, 80840).

    Computed, not copied: a literal NPI in a fixture is a number nobody can
    verify, and the submitter validates the check digit for real.
    """
    payload = "80840" + nine_digits
    total = 0
    for index, char in enumerate(reversed(payload)):
        digit = int(char) * (2 if index % 2 == 0 else 1)
        total += digit // 10 + digit % 10
    return nine_digits + str((10 - total % 10) % 10)


def _pick_codes(source) -> tuple[str, str]:
    """One diagnosis and one procedure code from the REAL loaded tables.

    The procedure must be active on the date of service AND governed by no CMS
    coverage policy, so medical necessity rests on the documented
    diagnosis->service linkage alone. Both properties are read from the
    authoritative data at runtime.
    """
    reference = source._reference()
    diagnosis = next(code for code in reference.icd10
                     if source.active_on(code, "icd10", DOS_ISO))
    procedure = next(code for code in reference.cpt
                     if source.active_on(code, "cpt", DOS_ISO)
                     and source.qualifying_dx_for(code, "cpt") is None)
    return diagnosis, procedure


def _pick_payer() -> tuple[str, str]:
    """(alias to write into the encounter context, expected trading-partner id).

    Taken from the declared payer registry so the test names a payer the
    deployment can actually route to, and stops naming it the moment the
    registry does.
    """
    from app.compliance.payer_registry import _load_payers
    payer = next(p for p in _load_payers()
                 if p.get("stedi_trading_partner_id") and p.get("aliases"))
    return max(payer["aliases"], key=len), payer["stedi_trading_partner_id"]


def _pick_place_of_service() -> str:
    raw = json.loads((REPO_ROOT / "data" / "codes" / "pos_codes.json").read_text())
    return sorted(raw.get("codes") or {})[0]


class _StubVectorStore:
    """One candidate per system — the code is real, the ranking is not.

    An index that offers exactly one authoritative candidate is the most
    conservative substitution available: it cannot make the coder choose
    something the descriptor does not support (both entailment judgements still
    have to agree) and it cannot smuggle in a code the authoritative tables do
    not carry.
    """

    def __init__(self, by_system: dict[str, str]):
        self.by_system = by_system

    def search(self, description, system, top_k=20):
        code = self.by_system.get(system)
        return [{"code": code, "similarity_score": 0.93}] if code else []


# --------------------------------------------------------------------------
# the deployment fixture
# --------------------------------------------------------------------------

class _Deployment:
    def __init__(self, paths: dict):
        self.__dict__.update(paths)

    def run(self, *, context_file: Path | None = None) -> dict:
        argv = ["--billing-context", str(self.billing_context_file),
                "--encounter-context", str(context_file or self.context_file)]
        assert entrypoint.main(argv) == 0, "the batch itself must not crash"
        return self.artifact()

    def artifact(self) -> dict:
        return json.loads(self.artifact_path.read_text())

    def write_artifact(self, payload: dict) -> None:
        self.artifact_path.write_text(json.dumps(payload, indent=2))

    def write_context(self, *, version: str,
                      mutate=None, path: Path | None = None) -> Path:
        """Write a NORMALIZED (`encounter_context/2`) source for this deployment.

        `mutate` receives a deep copy of the canonical roster and edits it in
        place, so each negative case changes exactly one thing and nothing else
        drifts between the positive and negative runs.
        """
        roster = copy.deepcopy(self.roster)
        roster["version"] = version
        if mutate is not None:
            mutate(roster)
        target = path or self.context_file
        target.write_text(json.dumps(roster, indent=2))
        return target

    def ingest(self) -> dict:
        return reg.ingest(self.output_dir, self.registry_path)

    def dry_run(self) -> dict:
        return cs.submit_all(self.output_dir, dry_run=True)


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    from app.core import config as app_config
    from claude_coder import arbitration
    from claude_coder import extraction
    from claude_coder import verify as verify_module
    from claude_coder.data_access import AuthoritativeSource

    notes_dir = tmp_path / "attachments"
    notes_dir.mkdir()
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    # A REAL document: its embedded text layer is the independent reading the
    # source-evidence gate reconciles the transcription against (F6-R6-A). A
    # placeholder byte string would hold every encounter as unverifiable, and the
    # AUTO_READY assertions below would then be proving the wrong thing.
    (notes_dir / f"{STEM}.pdf").write_bytes(build_pdf([[NOTE_TEXT]]))

    # Real source, real tables — used both to pick the fixture codes and (via
    # the patched vector store) by the run itself.
    source = AuthoritativeSource()
    diagnosis_code, procedure_code = _pick_codes(source)
    payer_alias, trading_partner = _pick_payer()
    place_of_service = _pick_place_of_service()
    rendering_npi = _valid_npi("188888888")
    billing_npi = _valid_npi("199999998")

    monkeypatch.setattr(entrypoint, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(entrypoint, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(app_config, "PROVENANCE_DB", tmp_path / "provenance.db")
    monkeypatch.setattr(entrypoint, "extract_from_pdf", lambda pdf_path:
                        vision_extraction(
                            [NOTE_TEXT], metadata={"date_of_service": DOS_ISO},
                            pdf_path=pdf_path,
                            extracted_text_sha256=EXTRACTED_TEXT_SHA))
    monkeypatch.setattr(extraction, "_default_llm", lambda system, user: FACTS_JSON)
    monkeypatch.setattr(
        AuthoritativeSource, "_vector_store",
        lambda self: _StubVectorStore({"icd10": diagnosis_code,
                                       "cpt": procedure_code}))
    monkeypatch.setattr(verify_module, "default_verify_llm",
                        verify_module.declare_model_profile(
                            lambda system, user:
                            '{"choice": 1, "reason": "stub"}', provider="openai"))
    monkeypatch.setattr(verify_module, "default_corroborate_llm",
                        verify_module.declare_model_profile(
                            lambda system, user:
                            '{"entailed": true, "missing_element": false, '
                            '"reason": "stub"}', provider="claude"))
    # The SECOND INDEPENDENT READING (product directive section 3; phase 4, commit
    # 97f748b). Substituted here for exactly the same reason as the two callables
    # above, and it MUST be: `run.main()` calls `code_encounter()` with no extractor,
    # so the pipeline treats the run as real mode and auto-enables
    # `extraction.default_second_extract_llm`, which calls a live vendor. Left
    # unsubstituted, every suite built on this fixture silently depended on a network
    # and a real second-provider API key; without them the second reading raised, the
    # pre-retrieval boundary held the encounter, and the DEPLOYED path produced zero
    # diagnosis and zero service lines.
    #
    # CROSS-PHASE FIXTURE CURRENCY -- read this before adding a control that calls a
    # model. This fixture is imported by `tests/test_encounter_context_e2e.py` and
    # `tests/test_deployment_entrypoint.py`, which drive the deployed entrypoint and
    # therefore cannot pass their own callables. A phase that adds a new real-mode
    # default LLM to the pipeline MUST substitute it here in the same commit.
    # `test_every_real_mode_default_llm_is_substituted` below derives that requirement
    # from the modules instead of restating a list, so the next one fails immediately
    # with a message naming the callable, rather than as an unexplained zero-line hold
    # discovered two phases later.
    #
    # Both channels return the same reading deliberately: two readings that agree on
    # every axis is the deterministic positive path, and it keeps the consensus control
    # ENABLED -- switching it off with GRAPH_CONSENSUS=0 would prove a configuration the
    # deployment does not run. The disagreement paths are covered end to end in
    # `tests/test_evidence_graph.py`.
    monkeypatch.setattr(extraction, "default_second_extract_llm",
                        verify_module.declare_model_profile(
                            lambda system, user: FACTS_JSON, provider="openai"))
    # `arbitration.arbitrate()` falls back to its OWN module-level default LLM whenever
    # the caller passes none -- and the deployed entrypoint passes none. It does not
    # fire under this fixture today only because the substituted index offers a single
    # candidate, so no line is left ambiguous; a fixture whose index offered two would
    # reach a live vendor from a test that believes it is hermetic. Substituted as a
    # TRIPWIRE rather than as a permissive answer: if arbitration ever does fire here,
    # the run must fail by name instead of quietly taking a different resolution path.
    def _arbitration_tripwire(system, user):
        raise AssertionError(
            "arbitration reached its real-mode default LLM from an e2e fixture; "
            "substitute it deliberately rather than calling a live vendor")

    monkeypatch.setattr(arbitration, "_default_llm", _arbitration_tripwire)

    # Durable audit anchor available, so a hold here would be attributable to
    # the claim path and not to the checkpoint (which round 6 already proves).
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ANCHOR", f"file:{tmp_path / 'anchor'}")

    billing_context_file = tmp_path / "billing_context.json"
    billing_context_file.write_text(json.dumps(BILLING_CONTEXT))

    # The normalized context source. Every identity is declared ONCE and
    # referenced by identifier; the encounter itself names only identifiers, so
    # nothing about who is billed can be inlined per encounter and drift.
    roster = {
        "schema": "encounter_context/2",
        "version": "context-edition-1",
        "patients": {PATIENT_ID: {
            "first_name": "Alexis", "last_name": "Quintero",
            "date_of_birth": "09/02/1982", "gender": "F",
            "record_number": "MRN-E2E-1"}},
        "payers": {PAYER_ID: {"name": payer_alias, "payer_id": "", "kind": "",
                              "plan": ""}},
        "coverages": [{
            "coverage_id": COVERAGE_ID, "patient_id": PATIENT_ID,
            "payer_id": PAYER_ID, "member_id": "MEM-E2E-1",
            "group_number": "GRP-E2E-1", "relationship_to_patient": "",
            "effective_start": ROSTER_START, "effective_end": ""}],
        "authorizations": [],
        "providers": {rendering_npi: {
            "first_name": "Robin", "last_name": "Vasquez",
            "display_name": "Robin Vasquez"}},
        "billing_entities": {ENTITY_ID: {
            "name": "Test Podiatry PLLC", "npi": billing_npi,
            "tax_id": "123456789", "taxonomy_code": "213E00000X"}},
        "affiliations": [{
            "affiliation_id": AFFILIATION_ID, "provider_npi": rendering_npi,
            "billing_entity_id": ENTITY_ID,
            "effective_start": ROSTER_START, "effective_end": ""}],
        "facilities": {FACILITY_ID: {
            "name": "Test Practice", "npi": billing_npi,
            "address1": "1 Test Way", "city": "Testville", "state": "FL",
            "postal_code": "33101", "place_of_service": place_of_service,
            "jurisdiction": "FL"}},
        "encounters": {STEM: {
            "patient_id": PATIENT_ID, "coverage_id": COVERAGE_ID,
            "rendering_provider_npi": rendering_npi,
            "facility_id": FACILITY_ID}},
    }

    practice_config_file = tmp_path / "practice_config.json"
    practice_config_file.write_text(json.dumps({
        "billing_provider": {
            "organization_name": "Test Podiatry PLLC", "npi": billing_npi,
            "tax_id": "123456789", "taxonomy_code": "213E00000X",
            "address": {"address1": "1 Test Way", "city": "Testville",
                        "state": "FL", "postal_code": "33101"},
            "contact_name": "Billing", "phone": "5555550100",
        },
        "rendering_providers": {"providers": [], "trust_note_npi": True,
                                "default": None},
        "submitter": {"organization_name": "Test Podiatry PLLC",
                      "contact_name": "Billing", "phone": "5555550100"},
        # Keyed by the runtime-selected procedure code — the fee schedule is
        # practice data, and the practice prices whatever it performs.
        "fee_schedule": {"charges": {procedure_code: 850.0},
                         "missing_code_policy": "block"},
        "claim_defaults": {
            "claim_frequency_code": "1", "signature_indicator": "Y",
            "plan_participation_code": "A", "release_information_code": "Y",
            "benefits_assignment_certification_indicator": "Y",
            "claim_filing_code": {"by_kind": {}, "default": "CI"},
            "payer_overrides": {},
        },
        "submission_policy": {"verification_tiers": ["auto"],
                              "require_clean_disposition": True},
    }))
    monkeypatch.setenv("PRACTICE_CONFIG_PATH", str(practice_config_file))

    registry_path = tmp_path / "claims_registry.jsonl"
    monkeypatch.setattr(cs, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(cs, "LEDGER_PATH", tmp_path / "submissions.jsonl")
    monkeypatch.setattr(cs, "DRYRUN_DIR", tmp_path / "submissions")

    deployment = _Deployment({
        "tmp_path": tmp_path,
        "output_dir": output_dir,
        "artifact_path": output_dir / f"{STEM}_results.json",
        "registry_path": registry_path,
        "dryrun_dir": tmp_path / "submissions",
        "context_file": tmp_path / "encounter_context.json",
        "billing_context_file": billing_context_file,
        "practice_config_file": practice_config_file,
        "roster": roster,
        "billing_npi": billing_npi,
        "diagnosis_code": diagnosis_code,
        "procedure_code": procedure_code,
        "trading_partner": trading_partner,
        "place_of_service": place_of_service,
        "rendering_npi": rendering_npi,
    })
    deployment.write_context(version="context-edition-1")
    return deployment


# --------------------------------------------------------------------------
# cross-phase fixture-currency guard
# --------------------------------------------------------------------------

def test_every_real_mode_default_llm_is_substituted(deployment):
    """No test driving the DEPLOYED entrypoint may reach a live model vendor.

    WHY THIS EXISTS. `run.main()` takes no LLM arguments, so a suite built on this
    fixture cannot opt out of real mode by passing its own callables the way a unit
    test does -- it can only substitute the module-level defaults the pipeline
    auto-enables. That makes the fixture the single point where a new model-calling
    control has to be declared, and nothing enforced it: phase 4 added
    `extraction.default_second_extract_llm`, this fixture was never extended, and the
    three suites that share it silently began requiring a network and a real
    second-provider key. Where those were present the suites passed while paying a
    vendor mid-test; where they were not, the pre-retrieval boundary held every
    encounter and 27 tests failed with zero diagnosis lines, zero service lines and no
    stated cause.

    WHAT IT CHECKS. The requirement is DERIVED, not restated: every module-level
    callable in the loaded `claude_coder` package whose name ends in `_llm` is a
    real-mode default by construction, and one still bound to its own defining module
    has not been substituted. A control added in a future phase is therefore covered
    the day it lands, and it fails HERE, naming the exact callable, instead of two
    phases later as an unexplained hold.
    """
    import sys

    package = [m for name, m in sorted(sys.modules.items()) if m is not None
               and (name == "claude_coder" or name.startswith("claude_coder."))]
    unsubstituted = []
    for module in package:
        for name, value in sorted(vars(module).items()):
            if not name.endswith("_llm") or not callable(value):
                continue
            # A substituted callable is defined by the TEST, so its `__module__` no
            # longer matches the module it is bound to. Comparing modules rather than
            # identities also survives a re-import of the package under test.
            if getattr(value, "__module__", None) == module.__name__:
                unsubstituted.append(f"{module.__name__}.{name}")

    assert not unsubstituted, (
        "these real-mode default LLM callables are still live under the deployment "
        "fixture, so this suite would call a model vendor (or hold every encounter "
        f"when it cannot): {sorted(set(unsubstituted))}. Substitute each of them in "
        "the `deployment` fixtures of tests/test_claim_bundle_e2e.py AND "
        "tests/test_deployment_entrypoint.py, in the same commit that adds it.")


# --------------------------------------------------------------------------
# the acceptance test
# --------------------------------------------------------------------------

def _submitted_payload(deployment, stats) -> dict:
    assert stats["submitted"] == 1, stats
    return json.loads((deployment.dryrun_dir / f"{STEM}_837p.json").read_text())


def test_document_to_bundle_to_registry_to_837p_preserves_the_claim(deployment):
    """The positive path, hop by hop, comparing what arrives with what left."""
    payload = deployment.run()

    # ---- hop 1: the entrypoint wrote ONE canonical artifact ----------------
    assert payload["schema_id"] == SCHEMA_ID
    # Read from the contract, not restated: a version bump is a deliberate change to
    # the artifact and belongs in one place, or this assertion becomes a second,
    # silently-drifting declaration of what the deployment writes.
    assert payload["schema_version"] == SCHEMA_VERSION
    bundle = load_bundle(payload)
    assert bundle.release_blockers() == (), bundle.release_blockers()
    assert bundle.release.destination.value == "AUTO_READY"
    assert bundle.encounter.document_id == STEM
    assert bundle.encounter.date_of_service == DOS_ISO
    assert bundle.encounter.source_document.document_version == DOCUMENT_VERSION

    # the finding, directly: encounter context obtained by `read_note` used to
    # be discarded before the artifact was written.
    assert bundle.context.resolution.value == "RESOLVED"
    assert bundle.context.provider_id == "versioned_roster/2"
    assert bundle.context.context_version == "context-edition-1"
    assert bundle.context.place_of_service == deployment.place_of_service
    assert bundle.context.rendering_provider.npi == deployment.rendering_npi
    # the identifier chain, not a per-encounter inline dump: the billing entity
    # is the one the AFFILIATION IN FORCE ON THE DOS names.
    assert bundle.context.billing_entity.entity_id == ENTITY_ID
    assert bundle.context.affiliation.affiliation_id == AFFILIATION_ID
    assert bundle.context.coverage.coverage_id == COVERAGE_ID

    (diagnosis,) = bundle.diagnoses
    (service,) = bundle.service_lines
    assert diagnosis.code == deployment.diagnosis_code
    assert diagnosis.sequence == 1 and diagnosis.primary is True
    assert service.code == deployment.procedure_code
    assert service.diagnosis_pointers == (1,), (
        "the service line lost its link to the diagnosis that justified it")
    assert service.units >= 1
    assert bundle.certificate is not None

    # ---- hop 1b: every released fact is anchored in the ORIGINAL DOCUMENT ----
    # Issue #6 F6-R6-A / directive §1, proven through the DEPLOYED ENTRYPOINT: the
    # note the coder read is one model's transcription of the PDF, and every
    # quotation behind these two lines was reconciled against the document's own
    # embedded text layer before this bundle could become releasable.
    references = [reference for line in (*bundle.diagnoses, *bundle.service_lines)
                  for reference in line.evidence]
    assert references, "the released claim carries no evidence at all"
    for reference in references:
        assert reference.source_reconciliation == ReconciliationStatus.AGREED.value, (
            f"a released fact was not confirmed against the original document: "
            f"{reference.text!r} -> {reference.source_reconciliation!r}")
        assert reference.verified_by_channel_id == EMBEDDED_TEXT_CHANNEL_ID
        assert reference.page == 1
        assert reference.page_image_sha256, "no source image identity"
        assert reference.region is not None, "no page region"
    source_evidence = bundle.certificate.certificate["source_evidence"]
    assert source_evidence["control_mode"] == "ENFORCED_FAIL_CLOSED"
    assert source_evidence["document_sha256"] == DOCUMENT_VERSION
    assert source_evidence["summary"]["DISAGREED"] == 0
    assert EMBEDDED_TEXT_CHANNEL_ID in source_evidence["independent_channel_ids"]
    assert bundle.certificate.certificate_sha256
    assert bundle.authority.data_fingerprint
    assert bundle.authority.source_manifest_fingerprint

    expected_modifiers = service.modifiers
    expected_units = service.units
    certificate_sha = bundle.certificate.certificate_sha256
    context_fingerprint = bundle.context.fingerprint

    # ---- hop 2: the registry accepted it and kept every line ---------------
    stats = deployment.ingest()
    assert stats["recorded"] == 1, stats["skip_reasons"]
    event = reg.current_view(reg.load_events(deployment.registry_path))[STEM]
    assert event["registry_version"] == reg.BUNDLE_REGISTRY_VERSION
    assert event["encounter_context_fingerprint"] == context_fingerprint
    assert event["certificate_sha256"] == certificate_sha
    recorded = reg.bundle_of_event(event)
    assert recorded.claim_content() == bundle.claim_content()
    assert [d.code for d in recorded.diagnoses] == [deployment.diagnosis_code]
    assert [s.code for s in recorded.service_lines] == [deployment.procedure_code]

    # ---- hop 3: the 837P dry-run carries the same claim --------------------
    claim = _submitted_payload(deployment, deployment.dry_run())["claimInformation"]
    assert claim["placeOfServiceCode"] == deployment.place_of_service
    (line,) = claim["serviceLines"]
    professional = line["professionalService"]
    assert professional["procedureCode"] == deployment.procedure_code
    assert professional["serviceUnitCount"] == str(expected_units)
    assert professional["compositeDiagnosisCodePointers"][
        "diagnosisCodePointers"] == [1]
    assert list(professional.get("procedureModifiers", [])) == \
        [m.upper() for m in expected_modifiers]
    (health_code,) = claim["healthCareCodeInformation"]
    assert health_code["diagnosisCode"] == \
        deployment.diagnosis_code.replace(".", "").upper()
    assert health_code["diagnosisTypeCode"] == "ABK", "first-listed diagnosis lost"


def test_recoding_the_same_note_re_verifies_and_stays_submittable(deployment):
    """Re-running a note must not deadlock it.

    The claim content is byte-identical across runs, but the certificate is not
    — it binds that run's durable audit records. The submitter requires the
    registry and the live artifact to carry the SAME certificate, so if ingest
    treated the re-run as "unchanged" (claim-only change detection) the note
    would become permanently unsubmittable with no event a re-ingest could ever
    add. The registry keys idempotence on the ATTESTATION, not just the claim.
    """
    first = load_bundle(deployment.run())
    assert deployment.ingest()["recorded"] == 1
    assert deployment.dry_run()["submitted"] == 1

    second = load_bundle(deployment.run())
    assert second.claim_content() == first.claim_content(), (
        "the same note, the same context and the same data must code the same")
    assert second.certificate.certificate_sha256 != \
        first.certificate.certificate_sha256, (
            "precondition: a re-run mints a fresh attestation over fresh audit rows")

    stats = deployment.ingest()
    assert stats["recorded"] == 1, stats
    event = reg.current_view(reg.load_events(deployment.registry_path))[STEM]
    assert event["certificate_sha256"] == second.certificate.certificate_sha256
    assert deployment.dry_run()["submitted"] == 1

    # ...and a genuine no-op re-ingest is still a no-op.
    assert deployment.ingest()["unchanged"] == 1


# --------------------------------------------------------------------------
# the negative cases — tampering and staleness must stop the dry-run
# --------------------------------------------------------------------------

def _verified_once(deployment) -> None:
    deployment.run()
    stats = deployment.ingest()
    assert stats["recorded"] == 1, stats["skip_reasons"]
    assert deployment.dry_run()["submitted"] == 1
    # Clear the dry-run output so a later assertion cannot pass by reading the
    # payload this baseline wrote.
    (deployment.dryrun_dir / f"{STEM}_837p.json").unlink()


def test_tampering_with_the_live_claim_stops_the_dry_run(deployment):
    """Edit the units on disk after verification: the claim no longer matches.

    Two independent controls must both notice — the artifact's own claim
    fingerprint stops reproducing, and the live artifact stops matching the one
    the registry verified. Either alone would be enough; the test asserts the
    dry-run stops and names the change.
    """
    _verified_once(deployment)

    payload = deployment.artifact()
    payload["service_lines"][0]["units"] += 1
    deployment.write_artifact(payload)

    tampered = load_bundle(payload)
    assert "claim fingerprint does not reproduce" in \
        " ".join(tampered.integrity_problems())

    stats = deployment.dry_run()
    assert stats["submitted"] == 0
    assert stats["blocked"] == 1
    assert "changed" in stats["docs"][STEM], stats["docs"][STEM]
    assert not (deployment.dryrun_dir / f"{STEM}_837p.json").exists()


def test_tampering_with_the_certificate_stops_the_dry_run(deployment):
    """Re-signing the claim by hand: the certificate stops self-addressing."""
    _verified_once(deployment)

    payload = deployment.artifact()
    payload["certificate"]["certificate"]["verdict"] = "AUTO_READY_TAMPERED"
    deployment.write_artifact(payload)

    tampered = load_bundle(payload)
    assert "certificate content address does not reproduce" in \
        " ".join(tampered.integrity_problems())

    stats = deployment.dry_run()
    assert stats["submitted"] == 0
    assert stats["blocked"] == 1


def test_a_changed_encounter_context_stops_the_dry_run(deployment):
    """A different payer/provider/POS is a different claim, and stale approval.

    Directive §2: "A different provider/facility/payer changes the context
    fingerprint and invalidates stale authorization." Re-running the SAME note
    against an edited context source must not be submittable under the earlier
    verification.
    """
    _verified_once(deployment)

    def _move_facility(roster):
        roster["facilities"][FACILITY_ID]["place_of_service"] = \
            _other_place_of_service(deployment.place_of_service)

    deployment.write_context(version="context-edition-2", mutate=_move_facility)
    fresh = deployment.run()
    assert load_bundle(fresh).context.fingerprint != \
        reg.bundle_of_event(
            reg.current_view(reg.load_events(deployment.registry_path))[STEM]
        ).context.fingerprint

    stats = deployment.dry_run()
    assert stats["submitted"] == 0
    assert stats["blocked"] == 1
    assert "context changed" in stats["docs"][STEM], stats["docs"][STEM]


def test_stale_authoritative_data_stops_the_dry_run(deployment):
    """The claim is unchanged but the data behind it is not the verified snapshot."""
    _verified_once(deployment)

    payload = deployment.artifact()
    payload["authority"]["data_fingerprint"] = "sha256:" + "0" * 64
    deployment.write_artifact(payload)

    stats = deployment.dry_run()
    assert stats["submitted"] == 0
    assert stats["blocked"] == 1
    assert "authoritative data" in stats["docs"][STEM], stats["docs"][STEM]


def test_an_unresolved_encounter_context_never_reaches_the_registry(deployment):
    """No context source configured -> the encounter holds, fail-safe and visibly.

    This is the deployment's current default. It must produce a bundle (a note
    that produced nothing on disk is invisible), that bundle must state its
    holds, and the registry must refuse it for the RIGHT reason — not because a
    field was empty, but because no authority resolved the context.
    """
    assert entrypoint.main(["--billing-context",
                            str(deployment.billing_context_file)]) == 0
    bundle = load_bundle(deployment.artifact())
    assert bundle.context.resolution.value == "UNRESOLVED"
    assert bundle.release_blockers()

    stats = deployment.ingest()
    assert stats["recorded"] == 0
    assert "context" in stats["skip_reasons"][STEM], stats["skip_reasons"]


def test_an_unknown_encounter_holds_only_that_encounter(deployment):
    """A context source that does not know this encounter resolves UNRESOLVED.

    Named, not silent: the hold says the encounter is absent from the context
    source, so an operator can tell "the roster has no entry" from "the roster
    could not be read" from "the roster is fine and the note is the problem".
    """
    def _rekey(roster):
        roster["encounters"] = {"SOME_OTHER_NOTE": roster["encounters"][STEM]}

    other = deployment.write_context(
        version="context-edition-1", mutate=_rekey,
        path=deployment.tmp_path / "other_context.json")
    bundle = load_bundle(deployment.run(context_file=other))
    assert bundle.context.resolution.value == "UNRESOLVED"
    assert any("not in the encounter context source" in hold
               for hold in bundle.release.holds), bundle.release.holds


def test_a_malformed_context_source_stops_the_batch(deployment):
    """An unreadable context source is a data-integrity incident, not "no roster".

    It must fail the invocation LOUDLY. Degrading to UNRESOLVED would make a
    corrupt file indistinguishable from an unconfigured deployment — both hold
    every note, so nobody would ever notice the corruption.
    """
    broken = deployment.tmp_path / "broken_context.json"
    broken.write_text("{ not json")
    assert entrypoint.main([
        "--billing-context", str(deployment.billing_context_file),
        "--encounter-context", str(broken)]) == 1
    assert not deployment.artifact_path.exists(), (
        "a batch that refused to start still wrote a result artifact")


# --------------------------------------------------------------------------
# the legacy adapter — readable, and structurally unable to release
# --------------------------------------------------------------------------

def test_the_round_six_artifact_the_finding_was_raised_against_still_reads():
    """The reviewer's own reproduction case, adapted rather than reinterpreted.

    A synthetic AUTO_READY `claude_coder.run/1` artifact — the shape whose
    diagnosis and service lines vanished in `extract_claim()` — must still be
    READABLE (its lines survive the adapter) and must be structurally incapable
    of releasing: it carries no encounter context and never did, so presenting
    it as a claim would invent the very facts it lacks.
    """
    from app.contracts.claim_bundle import BundleOrigin
    from app.contracts.legacy_adapter import bundle_from_legacy, is_legacy_artifact

    artifact = {
        "schema": "claude_coder.run/1",
        "document_id": "ROUND6_NOTE",
        "source_pdf": "ROUND6_NOTE.pdf",
        "date_of_service": DOS_ISO,
        "verdict": "AUTO_READY",
        "destination": "AUTO_READY",
        "releasable": True,
        "claim_lines": [
            {"system": "icd10", "code": "ZZ0.0", "descriptor": "synthetic dx",
             "modifiers": [], "units": 1},
            {"system": "cpt", "code": "00000", "descriptor": "synthetic service",
             "modifiers": ["ZZ"], "units": 3},
        ],
        "certificate": {"certificate_sha256": "not-verified-here"},
        "audit_record_hashes": ["h1", "h2"],
    }
    assert is_legacy_artifact(artifact)
    bundle = bundle_from_legacy(artifact)

    # the lines the registry used to lose
    assert [d.code for d in bundle.diagnoses] == ["ZZ0.0"]
    (service,) = bundle.service_lines
    assert (service.code, service.units, service.modifiers) == ("00000", 3, ("ZZ",))
    assert bundle.audit.audit_record_hashes == ("h1", "h2")

    # ...and it still cannot become a claim
    assert bundle.produced_by is BundleOrigin.LEGACY_CLAUDE_CODER_RUN_1
    assert bundle.release.producer_releasable is False
    assert bundle.release_blockers()

    from app.release.claim_readiness import verify_bundle_readiness
    ok, reason = verify_bundle_readiness(bundle)
    assert ok is False
    assert "adapted legacy artifact" in reason, reason


def test_a_legacy_artifact_can_never_be_read_as_a_canonical_one():
    """The dispatch is by DECLARED schema, so neither reader can capture the other."""
    from app.contracts.claim_bundle import (
        UnknownClaimBundleSchema, is_claim_bundle, load_bundle,
    )
    from app.contracts.legacy_adapter import is_legacy_artifact

    legacy = {"document_id": "n", "success": True, "icd_codes": [], "cpt_codes": []}
    assert is_legacy_artifact(legacy) and not is_claim_bundle(legacy)
    with pytest.raises(UnknownClaimBundleSchema):
        load_bundle(legacy)

    # A CORRUPT canonical artifact must not fall through to the legacy reader,
    # which would re-read it under weaker rules and find an empty claim.
    corrupt = {"schema_id": "claim_bundle", "schema_version": 1,
               "icd_codes": [{"code": "X"}]}
    assert is_claim_bundle(corrupt)
    assert not is_legacy_artifact(corrupt)


# --------------------------------------------------------------------------
# the same defect class, everywhere else it still exists
# --------------------------------------------------------------------------

#: Modules that read `*_results.json` AND index the retired `app.pipeline` claim
#: arrays off it. Given a canonical `ClaimBundle` artifact they will find no
#: `icd_codes`/`cpt_codes`/`hcpcs_codes` and see an EMPTY claim — silently. That
#: is the identical failure F6-R4-A1 named, and it is deliberately NOT fixed in
#: this phase: every one of them is retired growth-loop tooling that `run.py`'s
#: header already records as no longer invoked by the deployed entrypoint, and
#: none of them is on the path from a certified result to an 837P.
#:
#: The RETAINED claim path (`tools/claims_registry.py`,
#: `tools/claim_submitter.py`) is migrated and is excluded below; the registry
#: still contains legacy reads, but they are reached only through an explicit
#: schema dispatch.
#:
#: This is a ratchet, not an approval. It exists so the residual gap is counted
#: rather than rediscovered, and so a NEW module cannot join the list quietly.
_LEGACY_SHAPE_READERS = {
    "tools/audit_convergence_loop.py",
    "tools/audit_results.py",
    "tools/auto_actuate.py",
    "tools/calibration_dataset.py",
    "tools/coder_adjudicator.py",
    "tools/dryrun_layers.py",
    "tools/pack_consolidation.py",
    "tools/record_coherence.py",
    "tools/sweep_convergence_layers.py",
    "tools/sweep_new_layers.py",
    "tools/sweep_normalizations.py",
    "tools/unanimity_loop.py",
    "verify_notes.py",
}

_MIGRATED_TO_THE_CONTRACT = {
    "tools/claims_registry.py",
    "tools/claim_submitter.py",
}


def test_no_new_module_reads_a_result_in_the_retired_claim_shape():
    """Ratchet on the "two shapes, one boundary" defect class.

    A module that reaches into a result artifact for `icd_codes`/`cpt_codes`/
    `hcpcs_codes` is asserting a contract the deployed entrypoint no longer
    writes. The known set is frozen here: shrinking it is progress (update the
    set), growing it is the regression.
    """
    import re

    pattern = re.compile(
        r"""\[\s*["'](?:icd_codes|cpt_codes|hcpcs_codes)["']"""
        r"""|get\(\s*["'](?:icd_codes|cpt_codes|hcpcs_codes)["']""")
    candidates = [*(REPO_ROOT / "tools").rglob("*.py"),
                  *(REPO_ROOT / "app").rglob("*.py"),
                  *REPO_ROOT.glob("*.py")]
    found = set()
    for path in sorted(candidates):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith("app/contracts/") or path.name.startswith("test_"):
            continue
        text = path.read_text()
        if "_results.json" in text and pattern.search(text):
            found.add(relative)

    unmigrated = found - _MIGRATED_TO_THE_CONTRACT
    new = sorted(unmigrated - _LEGACY_SHAPE_READERS)
    assert not new, (
        f"these modules read a result artifact in the retired app.pipeline claim "
        f"shape and will silently see an empty claim for every ClaimBundle: {new}. "
        f"Read it through app/contracts/claim_bundle.py instead.")
    gone = sorted(_LEGACY_SHAPE_READERS - unmigrated)
    assert not gone, (
        f"these modules no longer read the retired shape — remove them from "
        f"_LEGACY_SHAPE_READERS so the ratchet keeps tightening: {gone}")
    assert _MIGRATED_TO_THE_CONTRACT <= found | _MIGRATED_TO_THE_CONTRACT


def _other_place_of_service(current: str) -> str:
    raw = json.loads((REPO_ROOT / "data" / "codes" / "pos_codes.json").read_text())
    return next(code for code in sorted(raw.get("codes") or {})
                if code != current)


def test_a_misread_page_cannot_reach_the_registry(deployment, monkeypatch):
    """The negative of the hop above, through the same entrypoint.

    The PDF on disk is unchanged; only the TRANSCRIPTION is perturbed — exactly what a
    vision misread looks like from every downstream component's point of view. Nothing
    after the transcription can tell it is wrong, which is why the check has to be
    against the document itself.
    """
    misread = NOTE_TEXT.replace("right site two", "left site two")
    assert misread != NOTE_TEXT
    monkeypatch.setattr(entrypoint, "extract_from_pdf", lambda pdf_path:
                        vision_extraction(
                            [misread], metadata={"date_of_service": DOS_ISO},
                            pdf_path=pdf_path,
                            extracted_text_sha256=EXTRACTED_TEXT_SHA))
    from claude_coder import extraction as extraction_module
    monkeypatch.setattr(
        extraction_module, "_default_llm",
        lambda system, user: FACTS_JSON.replace("right site two", "left site two"))

    bundle = load_bundle(deployment.run())
    assert not bundle.is_releasable
    assert bundle.release.destination.value == "BLOCKED"
    assert "source_evidence_reconciliation" in bundle.release.reason_codes
    outcome = next(o for o in bundle.outcomes
                   if o.name == "source_evidence_reconciliation")
    assert outcome.outcome == "BLOCKED"
    assert not outcome.retryable, (
        "a misread page is an integrity stop, not a retryable dependency failure")

    # and the retained claim path refuses it, which is the property that matters:
    # a bundle the registry will not record can never become an 837P.
    ingested = deployment.ingest()
    assert ingested.get("recorded", 0) == 0, (
        f"a claim resting on a misread page reached the registry: {ingested}")


# --------------------------------------------------------------------------
# F7-R1 — the certificate must be bound to THIS EXACT claim
# --------------------------------------------------------------------------
#
# Codex's reproduction, verbatim: take a valid AUTO_READY bundle, change the
# units 1 -> 9, add a modifier, change the patient identity, change the line
# authority and replace the graph relation/evidence ids, then RECOMPUTE the
# bundle's own context and claim fingerprints and leave the certificate alone.
# Every artifact-internal control reproduced, the sorted (system, code) multiset
# was unchanged, and `verify_bundle_readiness()` returned `(True, "")`.
#
# These tests drive that scenario through release authorization, the registry
# AND the 837P dry-run, because a unit-level check on one function is exactly
# the kind of evidence that let the summary comparison survive review.


def _revalidate(payload: dict):
    """Reload a hand-edited artifact and recompute its OWN fingerprints.

    This is the tamperer's move, not the producer's: `finalize()` is deliberately
    NOT used, because `finalize()` is the honest path. What is reproduced here is
    an attacker who knows exactly which two values the artifact checks against
    itself and updates both.
    """
    from app.contracts.claim_bundle import ClaimBundle
    tampered = ClaimBundle.model_validate(payload)
    context = tampered.context.model_copy(
        update={"fingerprint": tampered.context.compute_fingerprint()})
    tampered = tampered.model_copy(update={"context": context})
    return tampered.model_copy(
        update={"claim_fingerprint": tampered.compute_claim_fingerprint()})


def _codes(bundle) -> list:
    """The summary the old check compared — a sorted (system, code) multiset."""
    return sorted((line.system, line.code)
                  for line in (*bundle.diagnoses, *bundle.service_lines))


def test_a_recomputed_fingerprint_cannot_relaunder_a_tampered_claim(deployment):
    """Codex F7-R1, reproduced exactly, then proven caught at every consumer."""
    from app.release.claim_readiness import verify_bundle_readiness

    _verified_once(deployment)
    clean = load_bundle(deployment.artifact())
    ok, reason = verify_bundle_readiness(clean)
    assert ok, reason                      # the baseline really is releasable

    payload = deployment.artifact()
    payload["service_lines"][0]["units"] = 9
    payload["service_lines"][0]["modifiers"] = \
        list(payload["service_lines"][0]["modifiers"]) + ["ZZ"]
    payload["context"]["patient"]["last_name"] += "-TAMPERED"
    payload["service_lines"][0]["authority"]["edition"] = "not-the-edition-used"
    payload["service_lines"][0]["authority"]["detail"]["edition"] = \
        "not-the-edition-used"
    payload["graph"]["relation_ids"] = ["tampered-relation"]
    payload["graph"]["evidence_span_ids"] = ["tampered-span"]

    tampered = _revalidate(payload)

    # ---- the artifact is internally consistent, exactly as reported --------
    assert tampered.claim_fingerprint == tampered.compute_claim_fingerprint()
    assert tampered.context.fingerprint == tampered.context.compute_fingerprint()
    assert tampered.certificate.problems() == (), (
        "the certificate was not touched; it must still self-address")
    assert _codes(tampered) == _codes(clean), (
        "the tamper must be invisible to the summary the old check compared, "
        "or this test is not reproducing the finding")

    # ---- and it is refused, by the binding, naming what changed -----------
    problems = tampered.certificate_binding_problems()
    assert problems, "the certificate binding did not notice the tamper"
    joined = " | ".join(problems)
    assert "does not attest this exact claim" in joined, joined
    # and it says WHICH parts of the claim moved, not just that one did
    assert "the service lines" in joined, joined
    assert "the encounter context" in joined, joined
    assert "the clinical-graph binding" in joined, joined
    assert "units" in joined, joined
    assert "modifiers" in joined, joined
    assert "authoritative record" in joined, joined

    ok, reason = verify_bundle_readiness(tampered)
    assert ok is False
    assert "attest this exact claim" in reason, reason

    # ---- through the registry ---------------------------------------------
    # The RECOMPUTED artifact goes to disk, not the raw edit: writing the raw
    # edit would be stopped by the stored claim fingerprint, which is the
    # control the finding is explicitly about getting past.
    deployment.write_artifact(tampered.to_payload())
    stats = deployment.ingest()
    assert stats["recorded"] == 0, stats
    assert any("attest this exact claim" in str(r)
               for r in stats["skip_reasons"].values()), stats["skip_reasons"]

    # ---- and through the 837P dry-run --------------------------------------
    stats = deployment.dry_run()
    assert stats["submitted"] == 0
    assert stats["blocked"] == 1
    assert not (deployment.dryrun_dir / f"{STEM}_837p.json").exists()


def _tamper_cases():
    """One claim-affecting field per case, each applied ALONE.

    The acceptance criterion lists the fields the old comparison could not see.
    Each is asserted here on its own, so a fix that widened the comparison over
    some of them and not others cannot pass by aggregate.
    """
    def units(payload, deployment):
        payload["service_lines"][0]["units"] += 8

    def modifiers(payload, deployment):
        payload["service_lines"][0]["modifiers"] = \
            list(payload["service_lines"][0]["modifiers"]) + ["ZZ"]

    def modifier_order(payload, deployment):
        payload["service_lines"][0]["modifiers"] = ["ZZ", "YY"]

    def patient_identity(payload, deployment):
        payload["context"]["patient"]["last_name"] += "-OTHER"

    def line_authority(payload, deployment):
        payload["service_lines"][0]["authority"]["detail"]["edition"] = "other"

    def line_evidence(payload, deployment):
        payload["service_lines"][0]["evidence"][0]["text"] += " (edited)"

    def graph_relation_ids(payload, deployment):
        payload["graph"]["relation_ids"] = ["swapped"]

    def graph_evidence_ids(payload, deployment):
        payload["graph"]["evidence_span_ids"] = ["swapped"]

    def graph_digest(payload, deployment):
        payload["graph"]["graph_sha256"] = "0" * 64

    def diagnosis_primary(payload, deployment):
        payload["diagnoses"][0]["primary"] = False

    def diagnosis_pointers(payload, deployment):
        payload["service_lines"][0]["diagnosis_pointers"] = []

    def place_of_service(payload, deployment):
        payload["service_lines"][0]["place_of_service"] = \
            _other_place_of_service(deployment.place_of_service)

    def ndc(payload, deployment):
        payload["service_lines"][0]["ndc"] = "SYNTHETIC-NDC"

    def clinical_event_id(payload, deployment):
        payload["service_lines"][0]["clinical_event_id"] = "OTHER-EVENT"

    def source_document(payload, deployment):
        payload["encounter"]["source_document"]["document_version"] = "0" * 64

    def data_snapshot(payload, deployment):
        payload["authority"]["data_fingerprint"] = "sha256:" + "0" * 64

    def date_of_service(payload, deployment):
        payload["encounter"]["date_of_service"] = "2026-03-15"

    return [(f.__name__, f) for f in (
        units, modifiers, modifier_order, patient_identity, line_authority,
        line_evidence, graph_relation_ids, graph_evidence_ids, graph_digest,
        diagnosis_primary, diagnosis_pointers, place_of_service, ndc,
        clinical_event_id, source_document, data_snapshot, date_of_service)]


@pytest.mark.parametrize("name,mutate", _tamper_cases(),
                         ids=[n for n, _ in _tamper_cases()])
def test_every_claim_affecting_field_is_bound_to_the_certificate(
        deployment, name, mutate):
    """Change one field, recompute the artifact's own fingerprints, and the
    certificate binding must still refuse it."""
    from app.release.claim_readiness import verify_bundle_readiness

    deployment.run()
    payload = deployment.artifact()
    assert load_bundle(payload).release_blockers() == ()

    mutate(payload, deployment)
    tampered = _revalidate(payload)

    assert tampered.certificate_binding_problems(), (
        f"{name}: the certificate is not bound to this field, so it can be "
        f"changed after certification without detection")
    ok, _ = verify_bundle_readiness(tampered)
    assert ok is False, name


def test_rewriting_what_the_certificate_ATTESTS_is_caught_too(deployment):
    """The mirror image: leave the claim alone and rewrite the attestation.

    A maximally capable editor re-seals as it goes — it updates the producer
    certificate's own content address inside the binding and the certificate's
    outer address, so every digest in the artifact reproduces and the certified
    claim still matches the (untouched) bundle. Only the EXACT line-by-line
    comparison between what the certificate attests and what the claim bills
    can see it; a digest-only binding cannot.
    """
    from app.contracts.claim_bundle import (
        CERTIFIED_CLAIM_KEY, content_digest, load_bundle as _load)
    from app.release.claim_readiness import verify_bundle_readiness

    deployment.run()
    payload = deployment.artifact()
    service_code = _load(payload).service_lines[0].code

    certificate = payload["certificate"]["certificate"]
    attested = next(line for line in certificate["lines"]
                    if line["code"] == service_code)
    attested["units"] += 4
    body = {k: v for k, v in certificate.items()
            if k not in (CERTIFIED_CLAIM_KEY, "certificate_sha256")}
    certificate[CERTIFIED_CLAIM_KEY]["producer_certificate_sha256"] = \
        content_digest(body)
    resealed = content_digest({k: v for k, v in certificate.items()
                               if k != "certificate_sha256"})
    certificate["certificate_sha256"] = resealed
    payload["certificate"]["certificate_sha256"] = resealed

    tampered = _revalidate(payload)
    assert tampered.certificate.problems() == (), (
        "the re-sealed certificate must still self-address, or this test is "
        "catching a weaker forgery than the one it claims to")
    seal = tampered.certificate.certified_claim()
    assert seal["certified_claim_sha256"] == \
        tampered.compute_certified_claim_digest(), (
            "the claim itself was not touched, so the digest must still match")

    problems = " | ".join(tampered.certificate_binding_problems())
    assert "units" in problems, problems
    ok, _ = verify_bundle_readiness(tampered)
    assert ok is False


def test_a_certificate_with_no_claim_binding_can_never_release(deployment):
    """A native bundle whose certificate is not sealed to a claim is refused.

    This is what every artifact written before the binding existed looks like,
    and what a future producer path that forgot to seal would emit. Both are
    the same thing — a certificate whose subject cannot be established — and
    neither may release.
    """
    from app.contracts.claim_bundle import CERTIFIED_CLAIM_KEY, content_digest
    from app.release.claim_readiness import verify_bundle_readiness

    deployment.run()
    payload = deployment.artifact()
    certificate = payload["certificate"]["certificate"]
    certificate.pop(CERTIFIED_CLAIM_KEY)
    unsealed = content_digest({k: v for k, v in certificate.items()
                               if k != "certificate_sha256"})
    certificate["certificate_sha256"] = unsealed
    payload["certificate"]["certificate_sha256"] = unsealed

    stripped = _revalidate(payload)
    assert stripped.certificate.problems() == ()
    assert any("carries no certified-claim binding" in problem
               for problem in stripped.certificate_binding_problems())
    ok, reason = verify_bundle_readiness(stripped)
    assert ok is False
    assert "not bound to this claim" in reason, reason


def test_the_seal_preserves_the_producer_certificate_the_pipeline_recorded(
        deployment):
    """Sealing must not orphan the attestation the durable audit record bound.

    `claude_coder.pipeline` persists the producer certificate's content address
    in the terminal release decision BEFORE any claim exists. If sealing simply
    re-addressed the certificate, that record would point at an attestation no
    artifact carries any more. The producer address is therefore preserved
    inside the binding, and re-derivable from the artifact alone.
    """
    from app.contracts.claim_bundle import content_digest

    bundle = load_bundle(deployment.run())
    reference = bundle.certificate
    seal = reference.certified_claim()
    assert seal["producer_certificate_sha256"] == \
        content_digest(reference.producer_body())
    assert seal["certified_claim_sha256"] == \
        bundle.compute_certified_claim_digest()
    assert reference.certificate_sha256 != seal["producer_certificate_sha256"], (
        "the sealed certificate attests strictly more than the producer's did")
    assert bundle.graph.graph_sha256 == \
        bundle.certificate.certificate["clinical_graph"]["graph_sha256"]


def test_an_unusable_unit_count_is_refused_not_quietly_billed_as_one():
    """Zero or negative units must stop the claim, not become 1.

    The producer's certificate is built over the ORIGINAL unit count, so the
    old `max(1, ...)` coercion produced a claim that billed a quantity the
    certificate did not attest — and the codes-only comparison could not see
    the divergence. (Issue #6 F7-R1.)
    """
    from app.contracts.claim_bundle import (
        AuthorityBinding, InvalidClaimBundle, SourceDocument,
        bundle_from_coding_result)
    from app.contracts.encounter_context import EncounterContext
    from claude_coder.models import (
        CandidateCode, ClinicalFact, CodingResult, Disposition, EvidenceSpan,
        FactKind, ResolutionMethod, ResolvedLine)

    def _result(units) -> CodingResult:
        fact = ClinicalFact(
            FactKind.PROCEDURE, "synthetic service", fact_id="F1",
            disposition=Disposition.PERFORMED,
            evidence=[EvidenceSpan("synthetic service", anchored=True,
                                   span_id="s1")], confidence=0.9)
        line = ResolvedLine(fact, CandidateCode("SYNTHETIC", "cpt", "Synthetic"),
                            method=ResolutionMethod.DETERMINISTIC)
        line.units = units
        return CodingResult("enc", "2026-03-14", lines=[line])

    def _bundle(units):
        return bundle_from_coding_result(
            _result(units), source_document=SourceDocument(),
            context=EncounterContext(), authority=AuthorityBinding())

    assert _bundle(3).service_lines[0].units == 3
    for refused in (0, -1):
        with pytest.raises(InvalidClaimBundle) as caught:
            _bundle(refused)
        assert "at least 1" in str(caught.value)
    with pytest.raises(InvalidClaimBundle) as caught:
        _bundle(2.5)
    assert "whole number" in str(caught.value)
