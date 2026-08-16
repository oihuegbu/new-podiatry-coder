"""Acceptance tests for the Source Evidence Compiler — issue #6 F6-R6-A, directive §1.

The invariant under test, in the reviewer's words:

    An LLM transcription may be a candidate reading; it is never the authority
    against which its own correctness is proven.

So every test here works the same way: a REAL PDF carries the true text in its
embedded text layer, and the vision channel is fed a DIFFERENT (perturbed) reading of
the same page. The pipeline must notice, and must not release.

The directive's five acceptance tests map onto the classes below:

  1. `PerturbationTest`          laterality / ordinal / unit / decimal / measurement /
                                 negation disagreements are detected and cannot release
  2. `NonMaterialDifferenceTest` punctuation, case and spacing do not hold the encounter
  3. `SourceLocationTest`        every released fact resolves to an exact page region and
                                 source image/text hash
  4. `PageOutcomeTest`           blank / rotated / duplicated / missing / low-quality
                                 pages have explicit outcomes
  5. `AgnosticMechanismTest`     no medical term or code is in the mechanism

`BypassTest` and `IndependentChannelTest` cover the two questions the mechanism's own
failure paths raise: can an encounter reach a release WITHOUT being reconciled, and can
a second reading that merely repeats the first one's mistake be mistaken for proof.
"""
import json
import unittest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, Outcome, Verdict
from claude_coder.pipeline import code_encounter
from claude_coder.provenance import NullAuditRepository

from app.contracts.source_evidence import (
    BLOCKING_STATUSES, PAGE_SEPARATOR, PageStatus, ReconciliationStatus, SpanTarget,
    reconcile_service_date, reconcile_spans,
)
from app.ingestion.source_evidence import (
    EMBEDDED_TEXT_CHANNEL_ID, PRIMARY_CHANNEL_ID, SECONDARY_VISION_CHANNEL_ID,
    SourceEvidenceCompilationError, compile_source_evidence,
)
from tests.source_pdf import build_pdf, vision_extraction

# --------------------------------------------------------------------------
# a real, minimal PDF — the ORIGINAL document these tests reconcile against
# --------------------------------------------------------------------------


def _extraction(page_texts, **kwargs):
    return vision_extraction(page_texts, metadata={"date_of_service": "2026-03-14"},
                             page_separator=PAGE_SEPARATOR, **kwargs)


def _compile(tmpdir, pdf_pages, vision_pages, **kwargs):
    path = tmpdir / "note.pdf"
    path.write_bytes(build_pdf(pdf_pages))
    return compile_source_evidence(path, _extraction(vision_pages, **kwargs))


# --------------------------------------------------------------------------
# the encounter these tests code
# --------------------------------------------------------------------------

#: One page of a fully synthetic note. Every axis the directive names lives on the
#: first line, which is the procedure's own evidence quotation, so a perturbation of
#: any of them lands inside the span a BILLED line rests on.
TRUE_LINES = [
    "Procedure: excision of lesion alpha, right site two, measuring 3.5 cm, "
    "agent alpha 10 mg, without complication.",
    "Assessment: condition alpha, right side.",
    "Excision of lesion alpha was performed for condition alpha of the right side.",
    "Plan procedure beta correction next visit.",
    "Patient denies finding gamma.",
]

PROC = CandidateCode("PROC_ALPHA_EXC", "cpt",
                     "Excision, lesion alpha, single, each", 0.9, "retrieval")
DX = CandidateCode("DX_ALPHA_RIGHT", "icd10",
                   "condition alpha, right side", 0.9, "retrieval")

BILLING_CONTEXT = {"billing_entity_id": "actor-1",
                   "participants": [{"id": "actor-1", "type": "person",
                                     "roles": ["performer"]}]}


def _source():
    return MockSource(
        records={("PROC_ALPHA_EXC", "cpt"): {"active": True},
                 ("DX_ALPHA_RIGHT", "icd10"): {"active": True}},
        retrieval={("*", "cpt"): [PROC], ("*", "icd10"): [DX]})


def _facts_json(lines: list[str]) -> str:
    """The extractor response, quoting the lines it was actually given.

    Built FROM the transcription under test rather than frozen, because a perturbed
    transcription is one the extractor would have quoted verbatim — the whole point is
    that nothing downstream of the transcription can tell it is wrong.
    """
    return json.dumps({"facts": [
        {"kind": "procedure", "description": "excision of lesion alpha",
         "attributes": {"laterality": "right", "anatomy": "site two",
                        "performer_id": "actor-1", "billing_entity_id": "actor-1"},
         "disposition": "performed_today", "negated": False,
         "evidence": [lines[0], "Excision of lesion alpha"], "confidence": 0.97,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "performer": 0.99,
                             "relationship": 0.99}},
        {"kind": "diagnosis", "description": "condition alpha of the right side",
         "attributes": {"laterality": "right"}, "disposition": "performed_today",
         "negated": False,
         "evidence": ["condition alpha, right side",
                      "condition alpha of the right side"], "confidence": 0.98,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "assertion": 0.99,
                             "experiencer": 0.99}},
    ], "relations": [
        {"subject_event_id": "F2", "object_event_id": "F1", "predicate": "reason_for",
         "state": "asserted", "evidence_fact_ids": ["F1", "F2"], "confidence": 0.99},
    ]})


def _bundle(result):
    """The canonical claim artifact for a coding result — the shape the registry, the
    readiness check and the 837P builder read. Used here so a hold is asserted where a
    CONSUMER would see it, not only where the producer recorded it."""
    from app.contracts.claim_bundle import (
        AuthorityBinding, SourceDocument, bundle_from_coding_result)
    from app.contracts.encounter_context import EncounterContext
    return bundle_from_coding_result(
        result, source_document=SourceDocument(filename="note.pdf"),
        context=EncounterContext(), authority=AuthorityBinding())


class _Encounter:
    """One coding run over a document whose two channels may disagree."""

    def __init__(self, testcase, *, pdf_lines=None, vision_lines=None,
                 pdf_pages=None, vision_pages=None, reader=None, **kwargs):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        testcase.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        pdf_pages = pdf_pages or [list(pdf_lines or TRUE_LINES)]
        vision_pages = vision_pages or ["\n".join(vision_lines or TRUE_LINES)]
        self.document = _compile(root, pdf_pages, vision_pages, **kwargs)
        self.note_text = self.document.primary_text()
        self.result = code_encounter(
            "enc-1", self.note_text, "2026-03-14", source=_source(),
            extract_llm=lambda system, user: _facts_json(
                (vision_lines or TRUE_LINES)),
            arbitrate_llm=lambda system, user:
                '{"choice":0,"confidence":0.0,"reason":"unused"}',
            audit_repository=NullAuditRepository(),
            billing_context=BILLING_CONTEXT,
            document_version=self.document.document_sha256,
            source_evidence=self.document, source_reader=reader)

    @property
    def gate(self):
        return next(g for g in self.result.gates
                    if g.name == "source_evidence_reconciliation")


# --------------------------------------------------------------------------
# 1. a perturbed reading of a code-changing fact is detected and cannot release
# --------------------------------------------------------------------------

class PerturbationTest(unittest.TestCase):
    """Directive §1, acceptance test 1."""

    #: (name, what the DOCUMENT says, what the transcription says instead)
    PERTURBATIONS = [
        ("laterality", "right site two", "left site two"),
        ("ordinal", "right site two", "right site three"),
        ("unit", "agent alpha 10 mg", "agent alpha 10 mcg"),
        ("decimal", "measuring 3.5 cm", "measuring 35 cm"),
        ("measurement", "measuring 3.5 cm", "measuring 6.5 cm"),
        ("negation", "without complication", "with complication"),
    ]

    def test_a_clean_transcription_releases(self):
        """The control: identical readings must NOT be held, or the tests below prove
        only that the gate blocks everything."""
        encounter = _Encounter(self)
        self.assertEqual(encounter.gate.outcome, Outcome.PASS, encounter.gate.detail)
        self.assertEqual(encounter.result.verdict, Verdict.AUTO_READY,
                         encounter.result.notes)
        self.assertIsNotNone(encounter.result.certificate)

    def test_each_perturbed_axis_is_detected_and_cannot_auto_release(self):
        for name, truth, perturbed in self.PERTURBATIONS:
            with self.subTest(axis=name):
                lines = list(TRUE_LINES)
                self.assertIn(truth, lines[0], "the fixture must contain the axis")
                lines[0] = lines[0].replace(truth, perturbed)
                encounter = _Encounter(self, pdf_lines=TRUE_LINES,
                                       vision_lines=lines)
                gate = encounter.gate
                self.assertEqual(
                    gate.outcome, Outcome.BLOCKED,
                    f"{name}: a disagreement on the original page was not detected "
                    f"({gate.detail})")
                self.assertFalse(gate.retryable,
                                 f"{name}: a misread page is not a retryable "
                                 f"dependency failure")
                self.assertEqual(encounter.result.verdict, Verdict.BLOCKED)
                statuses = {s.status for s in
                            encounter.result.source_reconciliation.spans}
                self.assertIn(ReconciliationStatus.DISAGREED, statuses)
                # The certificate exists and RECORDS the block (it is the audit
                # artifact for the refusal, not a licence). What must be impossible is
                # a releasable claim, so that is what is asserted — through the same
                # consumer-side re-derivation the registry and the 837P builder use.
                self.assertEqual(encounter.result.certificate["verdict"], "BLOCKED")
                bundle = _bundle(encounter.result)
                self.assertFalse(bundle.is_releasable)
                self.assertIn("source_evidence_reconciliation",
                              bundle.release.reason_codes,
                              f"{name}: the bundle does not say WHY it cannot be "
                              f"released")

    def test_the_disagreeing_tokens_are_recorded_for_audit(self):
        lines = list(TRUE_LINES)
        lines[0] = lines[0].replace("right site two", "left site two")
        encounter = _Encounter(self, pdf_lines=TRUE_LINES, vision_lines=lines)
        differences = [d for span in encounter.result.source_reconciliation.spans
                       for d in span.differences]
        self.assertTrue(any(d.quoted == "left" and d.independent == "right"
                            for d in differences), differences)

    def test_a_perturbation_outside_a_billed_line_does_not_block(self):
        """Scoping, stated as a test: the mechanism reconciles what the CLAIM rests
        on. A transcription difference in prose no billed line quotes is recorded but
        must not hold an otherwise defensible encounter."""
        lines = list(TRUE_LINES)
        lines[3] = "Plan procedure beta correction next quarter."
        encounter = _Encounter(self, pdf_lines=TRUE_LINES, vision_lines=lines)
        self.assertEqual(encounter.gate.outcome, Outcome.PASS, encounter.gate.detail)
        self.assertEqual(encounter.result.verdict, Verdict.AUTO_READY)


# --------------------------------------------------------------------------
# 2. non-code-changing differences do not hold the encounter
# --------------------------------------------------------------------------

class NonMaterialDifferenceTest(unittest.TestCase):
    """Directive §1, acceptance test 2."""

    def test_punctuation_case_and_spacing_differences_release(self):
        lines = list(TRUE_LINES)
        lines[0] = ("PROCEDURE:  excision of lesion alpha;  right site two "
                    "-- measuring 3.5 cm,, agent alpha 10 mg (without complication)!")
        encounter = _Encounter(self, pdf_lines=TRUE_LINES, vision_lines=lines)
        self.assertEqual(encounter.gate.outcome, Outcome.PASS, encounter.gate.detail)
        self.assertEqual(encounter.result.verdict, Verdict.AUTO_READY,
                         encounter.result.notes)

    def test_a_word_broken_across_a_line_is_not_a_disagreement(self):
        """Typesetting, not reading: a text layer that recovers a hyphenated
        line-break as two tokens must not manufacture a disagreement."""
        document = _compile_two_readings(
            self, ["alpha well-", "healed margin beta"], "alpha well-healed margin beta")
        record = _reconcile_one(document, "alpha well-healed margin beta")
        self.assertEqual(record.status, ReconciliationStatus.AGREED, record.detail)

    def test_a_decimal_point_is_never_normalized_away(self):
        """The same tolerance must NOT extend to numbers: `3.5` and `35` are different
        readings of a measurement, and treating them as one would erase the exact
        class of error this module exists to catch."""
        document = _compile_two_readings(self, ["margin 3.5 cm"], "margin 35 cm")
        record = _reconcile_one(document, "margin 35 cm")
        self.assertEqual(record.status, ReconciliationStatus.DISAGREED, record.detail)


def _compile_two_readings(testcase, pdf_lines, vision_text):
    import tempfile
    from pathlib import Path
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    return _compile(Path(tmp.name), [pdf_lines], [vision_text])


def _reconcile_one(document, quotation):
    text = document.primary_text()
    start = text.find(quotation)
    assert start >= 0, "the quotation must be verbatim in the primary reading"
    record = reconcile_spans(document, [SpanTarget(
        span_id="s1", text=quotation, start=start, end=start + len(quotation),
        fact_id="F1")]).spans[0]
    return record


# --------------------------------------------------------------------------
# 3. every released fact resolves to an exact page region and source hashes
# --------------------------------------------------------------------------

class SourceLocationTest(unittest.TestCase):
    """Directive §1, acceptance test 3."""

    def test_every_released_span_resolves_to_a_page_region_and_hashes(self):
        encounter = _Encounter(self)
        self.assertEqual(encounter.result.verdict, Verdict.AUTO_READY)
        checked = 0
        for line in encounter.result.billable_lines:
            for span in line.fact.evidence:
                checked += 1
                self.assertEqual(span.source_reconciliation,
                                 ReconciliationStatus.AGREED.value)
                self.assertEqual(span.page, 1)
                self.assertTrue(span.page_image_sha256, "no source image identity")
                self.assertEqual(span.verified_by_channel_id, EMBEDDED_TEXT_CHANNEL_ID)
                self.assertIsNotNone(span.region, "no page region")
                x0, top, x1, bottom = span.region
                self.assertLess(x0, x1)
                self.assertLess(top, bottom)
        self.assertGreater(checked, 0)

    def test_the_certificate_binds_the_reading_and_every_proof(self):
        encounter = _Encounter(self)
        certificate = encounter.result.certificate
        evidence = certificate["source_evidence"]
        self.assertEqual(evidence["document_sha256"],
                         encounter.document.document_sha256)
        self.assertEqual(evidence["control_mode"], "ENFORCED_FAIL_CLOSED")
        self.assertIn(EMBEDDED_TEXT_CHANNEL_ID, evidence["independent_channel_ids"])
        self.assertTrue(evidence["spans"])
        self.assertTrue(all(page["image_sha256"] for page in evidence["page_outcomes"]))
        for line in certificate["lines"]:
            for span in line["evidence"]:
                self.assertEqual(span["source_reconciliation"],
                                 ReconciliationStatus.AGREED.value)
                self.assertTrue(span["page_image_sha256"])
                self.assertIsNotNone(span["region"])

    def test_the_claim_bundle_carries_the_region_and_the_proof(self):
        from app.contracts.claim_bundle import (
            SCHEMA_VERSION, AuthorityBinding, SourceDocument, bundle_from_coding_result,
            load_bundle)
        from app.contracts.encounter_context import EncounterContext
        encounter = _Encounter(self)
        bundle = bundle_from_coding_result(
            encounter.result,
            source_document=SourceDocument(
                filename="note.pdf",
                document_version=encounter.document.document_sha256),
            context=EncounterContext(), authority=AuthorityBinding())
        payload = bundle.to_payload()
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        reloaded = load_bundle(payload)
        references = [reference for line in
                      (*reloaded.diagnoses, *reloaded.service_lines)
                      for reference in line.evidence]
        self.assertTrue(references)
        for reference in references:
            self.assertEqual(reference.source_reconciliation,
                             ReconciliationStatus.AGREED.value)
            self.assertTrue(reference.page_image_sha256)
            self.assertIsNotNone(reference.region)
            self.assertEqual(reference.verified_by_channel_id,
                             EMBEDDED_TEXT_CHANNEL_ID)

    def test_a_v1_bundle_is_still_readable(self):
        """The version bump must not orphan artifacts already on disk."""
        from app.contracts.claim_bundle import load_bundle
        from app.contracts.encounter_context import EncounterContext
        from app.contracts.claim_bundle import (
            AuthorityBinding, SourceDocument, bundle_from_coding_result)
        encounter = _Encounter(self)
        payload = bundle_from_coding_result(
            encounter.result, source_document=SourceDocument(),
            context=EncounterContext(), authority=AuthorityBinding()).to_payload()
        payload["schema_version"] = 1
        for line in (*payload["diagnoses"], *payload["service_lines"]):
            for reference in line["evidence"]:
                for field in ("page_image_sha256", "region",
                              "source_reconciliation", "verified_by_channel_id"):
                    reference.pop(field)
        reloaded = load_bundle(payload)
        self.assertEqual(reloaded.schema_version, 1)


# --------------------------------------------------------------------------
# 4. blank / rotated / duplicated / missing / low-quality pages
# --------------------------------------------------------------------------

class PageOutcomeTest(unittest.TestCase):
    """Directive §1, acceptance test 4 — explicit outcomes, no silent skips, no crash."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_blank_page_is_blank_not_unreadable(self):
        document = _compile(self.root, [["alpha beta gamma"], []],
                            ["alpha beta gamma", ""],
                            statuses=["extracted", "blank"])
        page = document.page(2)
        self.assertEqual(page.status, PageStatus.BLANK)
        self.assertEqual(page.read_by(PRIMARY_CHANNEL_ID).status, PageStatus.BLANK)
        self.assertEqual(page.read_by(EMBEDDED_TEXT_CHANNEL_ID).status,
                         PageStatus.BLANK)

    def test_a_rotated_page_records_its_rotation(self):
        path = self.root / "note.pdf"
        path.write_bytes(build_pdf([["alpha beta"], ["gamma delta"]], rotate=[0, 90]))
        document = compile_source_evidence(
            path, _extraction(["alpha beta", "gamma delta"]))
        self.assertEqual(document.page(2).rotation, 90)
        self.assertIn("rotated:90", document.page(2).flags)
        # Rotation is an observation, not a failure: the page is still readable, so the
        # quotation on it must still be provable.
        self.assertTrue(document.page(2).read_by(EMBEDDED_TEXT_CHANNEL_ID).usable)

    def test_a_duplicated_page_is_flagged_on_both_pages(self):
        digest = "sha256:" + "bb" * 32
        document = _compile(self.root, [["alpha beta"], ["alpha beta"]],
                            ["alpha beta", "alpha beta"],
                            image_digests=[digest, digest])
        self.assertIn("duplicate_of_page:1", document.page(2).flags)
        self.assertTrue(any("renders identically" in a for a in document.anomalies))

    def test_a_page_the_text_layer_does_not_have_is_explicit(self):
        document = _compile(self.root, [["alpha beta"]],
                            ["alpha beta", "gamma delta"])
        page = document.page(2)
        self.assertEqual(page.read_by(EMBEDDED_TEXT_CHANNEL_ID).status,
                         PageStatus.MISSING)
        self.assertIn("text_layer_unavailable", page.flags)
        self.assertTrue(any("page(s) but the transcription returned" in a
                            for a in document.anomalies))

    def test_a_low_quality_page_holds_rather_than_agreeing(self):
        """A text layer that recovers a fraction of the page is not a reading of it.
        Declaring the difference a disagreement would block every scanned note; calling
        it agreement would certify an unchecked transcription. It holds."""
        text = " ".join(f"token{index}" for index in range(30))
        document = _compile(self.root, [["token0 token1"]], [text])
        page = document.page(1)
        self.assertIn("low_text_yield", page.flags)
        self.assertEqual(page.read_by(EMBEDDED_TEXT_CHANNEL_ID).status,
                         PageStatus.UNREADABLE)
        record = _reconcile_one(document, "token5 token6")
        self.assertEqual(record.status, ReconciliationStatus.UNVERIFIABLE)

    def test_an_image_only_page_holds_rather_than_agreeing(self):
        document = _compile(self.root, [[]], ["alpha beta gamma delta"])
        self.assertIn("no_embedded_text", document.page(1).flags)
        record = _reconcile_one(document, "alpha beta gamma")
        self.assertEqual(record.status, ReconciliationStatus.UNVERIFIABLE)

    def test_every_page_appears_in_the_reconciliation_record(self):
        document = _compile(self.root, [["alpha beta"], ["gamma delta"]],
                            ["alpha beta", "gamma delta"])
        record = reconcile_spans(document, [])
        self.assertEqual([page["page_number"] for page in record.page_outcomes], [1, 2])
        self.assertTrue(all(page["image_sha256"] for page in record.page_outcomes))

    def test_a_transcription_without_pages_is_refused_not_degraded(self):
        path = self.root / "note.pdf"
        path.write_bytes(build_pdf([["alpha beta"]]))
        payload = _extraction(["alpha beta"])
        payload["page_texts"] = []
        with self.assertRaises(SourceEvidenceCompilationError):
            compile_source_evidence(path, payload)

    def test_a_compiled_text_that_is_not_what_the_coder_reads_is_refused(self):
        path = self.root / "note.pdf"
        path.write_bytes(build_pdf([["alpha beta"]]))
        payload = _extraction(["alpha beta"])
        payload["sections"]["full_text"] = "alpha beta gamma"
        with self.assertRaises(SourceEvidenceCompilationError):
            compile_source_evidence(path, payload)

    def test_an_unopenable_document_yields_no_independent_channel(self):
        path = self.root / "note.pdf"
        path.write_bytes(b"not a pdf at all")
        document = compile_source_evidence(path, _extraction(["alpha beta"]))
        self.assertTrue(document.compiler_notes)
        record = _reconcile_one(document, "alpha beta")
        self.assertEqual(record.status, ReconciliationStatus.UNVERIFIABLE)


# --------------------------------------------------------------------------
# 5. the mechanism carries no medical vocabulary
# --------------------------------------------------------------------------

class AgnosticMechanismTest(unittest.TestCase):
    """Directive §1, acceptance test 5."""

    def test_no_code_or_domain_term_appears_in_the_new_modules(self):
        from pathlib import Path
        from tests.check_no_hardcoding import ROOT, scan_file, scan_terms
        for relative in ("app/contracts/source_evidence.py",
                         "app/ingestion/source_evidence.py",
                         "claude_coder/gates.py"):
            path = Path(ROOT) / relative
            with self.subTest(module=relative):
                self.assertEqual(scan_file(path), [])
                self.assertEqual(scan_terms(path), [])

    def test_the_mechanism_works_on_a_vocabulary_it_has_never_seen(self):
        """Proof by construction that no term list is involved: a disagreement in an
        invented vocabulary is detected exactly as a clinical one is."""
        document = _compile_two_readings(
            self, ["zzqq wibble 7.25 flurbs korvax"], "zzqq wobble 7.25 flurbs korvax")
        record = _reconcile_one(document, "zzqq wobble 7.25 flurbs korvax")
        self.assertEqual(record.status, ReconciliationStatus.DISAGREED)
        self.assertEqual([(d.quoted, d.independent) for d in record.differences],
                         [("wobble", "wibble")])

    def test_agreement_in_an_invented_vocabulary_is_agreement(self):
        document = _compile_two_readings(
            self, ["zzqq wibble 7.25 flurbs korvax"], "zzqq wibble 7.25 flurbs korvax")
        record = _reconcile_one(document, "zzqq wibble 7.25 flurbs")
        self.assertEqual(record.status, ReconciliationStatus.AGREED)


# --------------------------------------------------------------------------
# the mechanism's own failure paths
# --------------------------------------------------------------------------

class BypassTest(unittest.TestCase):
    """Can an encounter reach a release WITHOUT its reading being checked?"""

    def _run(self, **kwargs):
        return code_encounter(
            "enc-1", "\n".join(TRUE_LINES), "2026-03-14", source=_source(),
            extract_llm=lambda system, user: _facts_json(TRUE_LINES),
            arbitrate_llm=lambda system, user:
                '{"choice":0,"confidence":0.0,"reason":"unused"}',
            audit_repository=NullAuditRepository(),
            billing_context=BILLING_CONTEXT, **kwargs)

    def _gate(self, result):
        return next(g for g in result.gates
                    if g.name == "source_evidence_reconciliation")

    def test_a_document_backed_encounter_without_source_evidence_holds(self):
        """The bypass this phase exists to close: a caller that reads a PDF and hands
        only the transcription onward must not be able to release."""
        result = self._run(document_version="sha256:" + "cd" * 32)
        gate = self._gate(result)
        self.assertEqual(gate.outcome, Outcome.UNKNOWN, gate.detail)
        self.assertTrue(gate.retryable)
        self.assertNotEqual(result.verdict, Verdict.AUTO_READY)

    def test_text_supplied_directly_is_not_pretended_to_have_a_page(self):
        """The honest converse: with no source document there IS no original page, and
        claiming one would be a fabrication. The gate says so rather than holding."""
        result = self._run()
        self.assertEqual(self._gate(result).outcome, Outcome.NOT_APPLICABLE)


class IndependentChannelTest(unittest.TestCase):
    """Independence is a checked property, not a naming convention."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_second_model_read_can_cover_an_image_only_page(self):
        from app.contracts.source_evidence import ChannelKind, ReadChannel, build_page_read
        document = _compile(self.root, [[]], ["alpha beta gamma delta"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        covered = document.with_channel(
            channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1,
                                         "alpha beta gamma delta")})
        self.assertEqual(_reconcile_one(covered, "alpha beta gamma").status,
                         ReconciliationStatus.AGREED)

    def test_a_second_read_from_the_same_vendor_is_not_independent(self):
        from app.contracts.source_evidence import ChannelKind, ReadChannel, build_page_read
        from app.core.config import LLM_PROVIDER
        document = _compile(self.root, [[]], ["alpha beta gamma delta"])
        channel = ReadChannel(channel_id="same_vendor_second_pass",
                              kind=ChannelKind.VISION, provider=LLM_PROVIDER)
        covered = document.with_channel(
            channel, {1: build_page_read("same_vendor_second_pass", 1,
                                         "alpha beta gamma delta")})
        self.assertEqual(_reconcile_one(covered, "alpha beta gamma").status,
                         ReconciliationStatus.UNVERIFIABLE,
                         "one vendor agreeing with itself is repetition, not proof")

    def test_a_channel_that_repeats_the_misreading_cannot_overwrite_a_disagreement(self):
        from app.contracts.source_evidence import ChannelKind, ReadChannel, build_page_read
        document = _compile(self.root, [["alpha wibble gamma"]],
                            ["alpha wobble gamma"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        covered = document.with_channel(
            channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1,
                                         "alpha wobble gamma")})
        self.assertEqual(_reconcile_one(covered, "alpha wobble gamma").status,
                         ReconciliationStatus.DISAGREED)

    def test_a_channel_may_not_be_silently_replaced(self):
        from app.contracts.source_evidence import (
            ChannelKind, InvalidSourceEvidenceDocument, ReadChannel)
        document = _compile(self.root, [["alpha beta"]], ["alpha beta"])
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(
                ReadChannel(channel_id=EMBEDDED_TEXT_CHANNEL_ID,
                            kind=ChannelKind.EMBEDDED_TEXT, provider="pdf"), {})

    def test_the_paid_second_read_is_scoped_to_pages_that_change_the_answer(self):
        """The cost control, as a test: a page nobody quoted from, and a page already
        independently read, are never paid for again."""
        from app.contracts.source_evidence import pages_needing_independent_read
        document = _compile(self.root, [["alpha beta gamma"], []],
                            ["alpha beta gamma", "delta epsilon zeta"])
        text = document.primary_text()
        first = text.find("alpha beta")
        second = text.find("delta epsilon")
        record = reconcile_spans(document, [
            SpanTarget(span_id="s1", text="alpha beta", start=first,
                       end=first + len("alpha beta")),
            SpanTarget(span_id="s2", text="delta epsilon", start=second,
                       end=second + len("delta epsilon"))])
        self.assertEqual(
            pages_needing_independent_read(document, record, {"s1", "s2"}), (2,))
        self.assertEqual(
            pages_needing_independent_read(document, record, {"s1"}), ())

    def test_a_failed_second_read_holds_rather_than_releasing(self):
        class _Failing:
            def channel(self):
                raise RuntimeError("provider unavailable")

            def read_pages(self, page_numbers):
                raise RuntimeError("provider unavailable")

        encounter = _Encounter(self, pdf_pages=[[]],
                               vision_pages=["\n".join(TRUE_LINES)],
                               reader=_Failing())
        gate = encounter.gate
        self.assertEqual(gate.outcome, Outcome.UNKNOWN, gate.detail)
        self.assertTrue(gate.retryable)
        self.assertNotEqual(encounter.result.verdict, Verdict.AUTO_READY)

    def test_a_successful_second_read_releases_an_image_only_document(self):
        from app.contracts.source_evidence import ChannelKind, ReadChannel, build_page_read

        class _Reader:
            def channel(self):
                return ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                                   kind=ChannelKind.VISION, provider="openai")

            def read_pages(self, page_numbers):
                return {number: build_page_read(SECONDARY_VISION_CHANNEL_ID, number,
                                                "\n".join(TRUE_LINES))
                        for number in page_numbers}

        encounter = _Encounter(self, pdf_pages=[[]],
                               vision_pages=["\n".join(TRUE_LINES)],
                               reader=_Reader())
        self.assertEqual(encounter.gate.outcome, Outcome.PASS, encounter.gate.detail)
        self.assertEqual(encounter.result.verdict, Verdict.AUTO_READY)
        self.assertEqual(
            {span.verified_by_channel_id
             for line in encounter.result.billable_lines
             for span in line.fact.evidence},
            {SECONDARY_VISION_CHANNEL_ID})

    def test_a_second_read_that_disagrees_blocks_an_image_only_document(self):
        from app.contracts.source_evidence import ChannelKind, ReadChannel, build_page_read
        wrong = list(TRUE_LINES)
        wrong[0] = wrong[0].replace("right site two", "left site two")

        class _Reader:
            def channel(self):
                return ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                                   kind=ChannelKind.VISION, provider="openai")

            def read_pages(self, page_numbers):
                return {number: build_page_read(SECONDARY_VISION_CHANNEL_ID, number,
                                                "\n".join(wrong))
                        for number in page_numbers}

        encounter = _Encounter(self, pdf_pages=[[]],
                               vision_pages=["\n".join(TRUE_LINES)],
                               reader=_Reader())
        self.assertEqual(encounter.gate.outcome, Outcome.BLOCKED, encounter.gate.detail)
        self.assertEqual(encounter.result.verdict, Verdict.BLOCKED)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()


# --------------------------------------------------------------------------
# the DATE OF SERVICE, reconciled like every other code-changing fact (F7-R4)
# --------------------------------------------------------------------------

class ServiceDateReconciliationTest(unittest.TestCase):
    """Issue #6 F7-R4 at the compiler boundary.

    The DOS never reached `reconcile_spans` because it is a structured metadata
    FIELD of the transcription rather than a quotation supporting a fact — so the
    one value that selects the coverage, the affiliation, the authorization window
    and the effective code edition was the only code-changing value in the system
    nothing checked. These cases perturb it the three ways a transcription can get
    a date wrong.
    """

    PAGE = "Date of service: 2026-03-14. Procedure alpha performed on the left."

    def _document(self, vision_text=None):
        return _compile_two_readings(self, [self.PAGE], vision_text or self.PAGE)

    def test_a_date_written_on_the_page_is_proven_by_the_independent_channel(self):
        evidence = reconcile_service_date(self._document(), "2026-03-14")
        self.assertEqual(evidence.status, ReconciliationStatus.AGREED)
        self.assertTrue(evidence.reconciled)
        self.assertEqual(evidence.candidate, "2026-03-14")
        self.assertEqual(evidence.located_text, "2026-03-14")
        self.assertEqual(evidence.occurrences, 1)
        self.assertEqual(evidence.pages, (1,))
        self.assertEqual(evidence.verified_by_channel_id, EMBEDDED_TEXT_CHANNEL_ID)
        self.assertTrue(evidence.document_sha256)

    def test_a_metadata_only_misread_names_no_page_of_the_original(self):
        """The structured field says one date; the pages say another. There is
        nowhere in the document the claim's date could be pointed at."""
        for perturbed in ("2026-04-14", "2026-03-15", "2027-03-14"):
            with self.subTest(perturbed=perturbed):
                evidence = reconcile_service_date(self._document(), perturbed)
                self.assertEqual(evidence.status, ReconciliationStatus.NOT_LOCATED)
                self.assertFalse(evidence.reconciled)
                self.assertIn("written nowhere", evidence.detail)

    def test_a_transcription_wide_misread_is_caught_by_the_second_channel(self):
        """The whole transcription — metadata AND page text — reads the date wrong.
        Only a reading that is not the transcription's own can detect this.

        The status is BLOCKING either way; which blocking status it is depends on
        how the date is written, and that distinction is not a policy: an ISO date
        is ONE token, so a perturbed one shares no token with the independent
        reading and is reported as not appearing there (NOT_LOCATED), while a
        written form is several tokens and is reported as read differently
        (DISAGREED, below). Neither can bind, which is the only thing a claim
        depends on.
        """
        for perturbed in ("2026-04-14", "2026-03-15", "2027-03-14"):
            with self.subTest(perturbed=perturbed):
                misread = self.PAGE.replace("2026-03-14", perturbed)
                evidence = reconcile_service_date(
                    self._document(vision_text=misread), perturbed)
                self.assertIn(evidence.status, BLOCKING_STATUSES)
                self.assertFalse(evidence.reconciled)
                # It WAS anchored — a page of the original was named and read by
                # an independent channel — which is what distinguishes this from
                # the metadata-only misread above.
                self.assertEqual(evidence.pages, (1,))
                self.assertEqual(evidence.verified_by_channel_id,
                                 EMBEDDED_TEXT_CHANNEL_ID)
                self.assertTrue(evidence.span_id)

    def test_a_misread_day_inside_a_written_date_is_reported_as_a_difference(self):
        """The same perturbation on a multi-token written date: the independent
        channel reads the same place and reads it differently, token for token."""
        page = "Date of service: March 14, 2026. Procedure alpha performed."
        for perturbed, iso in (("March 14, 2027", "2027-03-14"),
                               ("April 14, 2026", "2026-04-14"),
                               ("March 15, 2026", "2026-03-15")):
            with self.subTest(perturbed=perturbed):
                document = _compile_two_readings(
                    self, [page], page.replace("March 14, 2026", perturbed))
                evidence = reconcile_service_date(document, iso)
                self.assertEqual(evidence.status, ReconciliationStatus.DISAGREED)
                self.assertFalse(evidence.reconciled)
                self.assertTrue(evidence.differences)

    def test_a_date_written_in_another_form_still_resolves_to_the_same_day(self):
        """The claim carries an ISO date; the page says what it says. Matching on
        the parsed CALENDAR DAY rather than on the string is what lets a document
        that writes 'March 14, 2026' prove an encounter dated 2026-03-14."""
        page = "Date of service: March 14, 2026. Procedure alpha performed."
        document = _compile_two_readings(self, [page], page)
        evidence = reconcile_service_date(document, "2026-03-14")
        self.assertEqual(evidence.status, ReconciliationStatus.AGREED)
        self.assertEqual(evidence.located_text, "March 14, 2026")

    def test_no_proposed_date_is_unverifiable_rather_than_absent(self):
        evidence = reconcile_service_date(self._document(), "")
        self.assertEqual(evidence.status, ReconciliationStatus.UNVERIFIABLE)
        self.assertEqual(evidence.candidate, "")
        self.assertFalse(evidence.reconciled)
