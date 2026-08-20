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


def _extraction(page_texts, path=None, **kwargs):
    return vision_extraction(page_texts, metadata={"date_of_service": "2026-03-14"},
                             pdf_path=path, page_separator=PAGE_SEPARATOR, **kwargs)


def _compile(tmpdir, pdf_pages, vision_pages, **kwargs):
    path = tmpdir / "note.pdf"
    path.write_bytes(build_pdf(pdf_pages))
    return compile_source_evidence(path, _extraction(vision_pages, path, **kwargs))


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
            path, _extraction(["alpha beta", "gamma delta"], path))
        self.assertEqual(document.page(2).rotation, 90)
        self.assertIn("rotated:90", document.page(2).flags)
        # Rotation is an observation, not a failure: the page is still readable, so the
        # quotation on it must still be provable.
        self.assertTrue(document.page(2).read_by(EMBEDDED_TEXT_CHANNEL_ID).usable)

    def test_a_duplicated_page_is_flagged_on_both_pages(self):
        # The SAME rendered bytes on both pages -- which is what a duplicated fax page
        # is, and now the only way to make two pages share a digest, because the
        # compiler recomputes each digest from the bytes it was given.
        same = b"identically-rendered-page"
        document = _compile(self.root, [["alpha beta"], ["alpha beta"]],
                            ["alpha beta", "alpha beta"],
                            image_payloads=[same, same])
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
        payload = _extraction(["alpha beta"], path)
        payload["page_texts"] = []
        with self.assertRaises(SourceEvidenceCompilationError):
            compile_source_evidence(path, payload)

    def test_a_compiled_text_that_is_not_what_the_coder_reads_is_refused(self):
        path = self.root / "note.pdf"
        path.write_bytes(build_pdf([["alpha beta"]]))
        payload = _extraction(["alpha beta"], path)
        payload["sections"]["full_text"] = "alpha beta gamma"
        with self.assertRaises(SourceEvidenceCompilationError):
            compile_source_evidence(path, payload)

    def test_an_unopenable_document_yields_no_independent_channel(self):
        path = self.root / "note.pdf"
        path.write_bytes(b"not a pdf at all")
        document = compile_source_evidence(path, _extraction(["alpha beta"], path))
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
        document = _compile(self.root, [[]], ["alpha beta gamma delta"])
        # The vendor that ACTUALLY produced the primary reading, as the compiler
        # recorded it from the client that answered -- not a configuration setting,
        # which may describe a call nobody made (issue #6 F7-R5).
        channel = ReadChannel(channel_id="same_vendor_second_pass",
                              kind=ChannelKind.VISION,
                              provider=document.primary_channel.provider)
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

    def test_the_same_channel_may_widen_to_pages_it_has_not_yet_read(self):
        """Issue #6 F7-R3: the independent vision reader's channel id is fixed (derived
        from the client, not chosen per call), and the SAME reader legitimately widens
        its own channel twice in one encounter — once proactively for pages no other
        channel covers, before recall extraction runs, and again later for whichever
        specific pages a disagreement or a candidate event turns out to need. A second
        `with_channel` call for the SAME channel identity must succeed when it only adds
        pages that channel has not read yet."""
        from app.contracts.source_evidence import ChannelKind, ReadChannel, build_page_read
        document = _compile(self.root, [[], []], ["alpha beta", "gamma delta"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        once = document.with_channel(
            channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1, "alpha beta")})
        twice = once.with_channel(
            channel, {2: build_page_read(SECONDARY_VISION_CHANNEL_ID, 2, "gamma delta")})
        self.assertEqual(_reconcile_one(twice, "alpha beta").status,
                         ReconciliationStatus.AGREED)
        self.assertEqual(_reconcile_one(twice, "gamma delta").status,
                         ReconciliationStatus.AGREED)
        # exactly one channel entry, not two -- widening is not a second channel
        self.assertEqual([c.channel_id for c in twice.channels].count(
            SECONDARY_VISION_CHANNEL_ID), 1)

    def test_widening_may_not_overwrite_a_page_the_channel_already_read(self):
        from app.contracts.source_evidence import (
            ChannelKind, InvalidSourceEvidenceDocument, ReadChannel, build_page_read)
        document = _compile(self.root, [[]], ["alpha beta"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        once = document.with_channel(
            channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1, "alpha beta")})
        with self.assertRaises(InvalidSourceEvidenceDocument):
            once.with_channel(
                channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1, "wobble")})

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


# --------------------------------------------------------------------------
# channel and source IDENTITY: derived from what ran, verified from the bytes
# (issue #6, Codex finding F7-R5)
# --------------------------------------------------------------------------

class _StubResponse:
    def __init__(self, payload):
        from types import SimpleNamespace
        self.stop_reason = "end_turn"
        self.content = [SimpleNamespace(type="text", text=payload)]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)


def _transcribe(testcase, pdf_path, page_text, client, *, provider_setting):
    """Drive the REAL transcriber against `client`, with the generic provider setting
    deliberately set to `provider_setting`.

    This is the whole point of the regression: the transcriber calls whatever client it
    was given, and the recorded channel identity must describe THAT call — not the
    setting, which a deployment is free to point at another vendor entirely.
    """
    import json
    from unittest import mock
    from PIL import Image
    from app.core import config as app_config
    from app.ingestion import pdf_parser

    parsed = {"patient_metadata": {"date_of_service": "2026-03-14"},
              "sections": {}, "note_category": "other",
              "procedures_performed_today": [], "imaging_performed_today": [],
              "supplies_dispensed_today": [], "prior_surgery_info": {},
              "physician_documented_codes": [],
              "page_texts": [{"page_number": 1, "status": "extracted",
                              "text": page_text}]}
    response = _StubResponse(json.dumps(parsed))
    client.response = response
    with mock.patch.object(pdf_parser, "convert_from_path",
                           return_value=[Image.new("RGB", (2, 2))]), \
            mock.patch.object(app_config, "LLM_PROVIDER", provider_setting), \
            mock.patch("app.core.llm_client.get_anthropic_client",
                       return_value=client), \
            mock.patch("app.core.llm_client._claude_message_via_batch",
                       return_value=response):
        return pdf_parser.extract_from_pdf(pdf_path)


class _StubClient:
    """Stands in for an SDK client object: answers `messages.create(...)`.

    What matters to the identity machinery is the PACKAGE that implements it, which is
    exactly what a real deployment's client carries and what a configuration setting
    cannot contradict.
    """

    def __init__(self, response=None):
        from types import SimpleNamespace
        self.response = response
        self.messages = SimpleNamespace(create=lambda **kwargs: self.response)


class _AnthropicLikeClient(_StubClient):
    """A client object whose implementing package is the Anthropic SDK's."""
    __module__ = "anthropic._client"


class _OpenAILikeClient(_StubClient):
    __module__ = "openai._client"


class _UnknownClient(_StubClient):
    __module__ = "some.vendor.sdk"


class MisdeclaredProviderTest(unittest.TestCase):
    """The recorded channel identity is the vendor that ANSWERED, not the configured one.

    Before this, the compiler read the primary channel's provider from the generic
    `LLM_PROVIDER` setting while the transcriber called one vendor unconditionally. Under
    a deployment configured for the other vendor the recorded identity was simply false,
    and every independence decision taken against it was a decision about a call that was
    never made — in BOTH directions.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "note.pdf"
        # An IMAGE-ONLY page: nothing deterministic in the document can prove a
        # quotation on it, so whether the paid second-vendor read counts as independent
        # is the ONLY thing standing between the quotation and a hold.
        self.path.write_bytes(build_pdf([[]]))

    def _compiled(self, client, provider_setting):
        extraction = _transcribe(self, self.path, "alpha beta gamma delta", client,
                                 provider_setting=provider_setting)
        return compile_source_evidence(self.path, extraction)

    def test_the_configured_vendor_does_not_overwrite_the_calling_one(self):
        document = self._compiled(_AnthropicLikeClient(), "openai")
        self.assertEqual(document.primary_channel.provider, "claude",
                         "the identity must name the client that actually answered")

    def test_a_second_vendor_read_is_not_rejected_because_config_named_it(self):
        """The costly half of the defect: a genuinely independent read held for nothing."""
        from app.contracts.source_evidence import (ChannelKind, ReadChannel,
                                                   build_page_read)
        document = self._compiled(_AnthropicLikeClient(), "openai")
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        covered = document.with_channel(
            channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1,
                                         "alpha beta gamma delta")},
            require_independent=True)
        self.assertEqual(_reconcile_one(covered, "alpha beta gamma").status,
                         ReconciliationStatus.AGREED)

    def test_a_same_vendor_read_is_not_accepted_because_config_named_another(self):
        """The dangerous half: the transcription read by the SAME vendor as the second
        channel, with the setting naming the other one. Independence must fail."""
        from app.contracts.source_evidence import (ChannelIndependenceError, ChannelKind,
                                                   ReadChannel, build_page_read)
        document = self._compiled(_OpenAILikeClient(), "claude")
        self.assertEqual(document.primary_channel.provider, "openai")
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        with self.assertRaises(ChannelIndependenceError):
            document.with_channel(
                channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1,
                                             "alpha beta gamma delta")},
                require_independent=True)

    def test_an_unidentifiable_client_establishes_no_independence(self):
        """Fail-closed: 'we cannot tell which vendor answered' is never 'independent'."""
        from app.contracts.source_evidence import (ChannelIndependenceError, ChannelKind,
                                                   ReadChannel, build_page_read)
        document = self._compiled(_UnknownClient(), "claude")
        self.assertEqual(document.primary_channel.provider, "")
        self.assertTrue(any("provider could not be established" in note
                            for note in document.compiler_notes))
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        with self.assertRaises(ChannelIndependenceError):
            document.with_channel(
                channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1,
                                             "alpha beta gamma delta")},
                require_independent=True)

    def test_a_transcription_that_declares_no_channel_identity_is_refused(self):
        payload = _extraction(["alpha beta gamma delta"], self.path)
        payload["note_integrity"].pop("vision_channel")
        with self.assertRaises(SourceEvidenceCompilationError):
            compile_source_evidence(self.path, payload)


class SameProviderIndependentReaderTest(unittest.TestCase):
    """A reader configured as the independent channel must FAIL CLOSED when it is not."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "note.pdf"
        self.path.write_bytes(build_pdf([[]]))
        self.document = compile_source_evidence(
            self.path, _extraction(["alpha beta gamma delta"], self.path))

    def _reader(self, client):
        from unittest import mock
        from app.ingestion.source_evidence import IndependentVisionReader
        reader = IndependentVisionReader(
            self.path, primary_channel=self.document.primary_channel)
        return reader, mock.patch("app.core.llm_client.get_openai_client",
                                  return_value=client)

    def test_a_reader_whose_client_is_the_primary_vendor_refuses_before_paying(self):
        from app.contracts.source_evidence import ChannelIndependenceError
        self.assertEqual(self.document.primary_channel.provider, "claude")
        # The reader's client resolves to the SAME vendor that produced the primary
        # reading — the case a fixed `provider="openai"` label could never notice.
        reader, patched = self._reader(_AnthropicLikeClient())
        with patched:
            with self.assertRaises(ChannelIndependenceError):
                reader.channel()
            # ...and nothing is rendered or sent: the refusal happens on identity.
            with self.assertRaises(ChannelIndependenceError):
                reader.read_pages((1,))

    def test_a_reader_bound_to_nothing_cannot_establish_independence(self):
        from app.contracts.source_evidence import ChannelIndependenceError
        from app.ingestion.source_evidence import IndependentVisionReader
        from unittest import mock
        reader = IndependentVisionReader(self.path)
        with mock.patch("app.core.llm_client.get_openai_client",
                        return_value=_OpenAILikeClient()):
            with self.assertRaises(ChannelIndependenceError):
                reader.channel()

    def test_a_genuinely_independent_reader_is_accepted(self):
        reader, patched = self._reader(_OpenAILikeClient())
        with patched:
            channel = reader.channel()
        self.assertEqual(channel.provider, "openai")
        self.assertEqual(channel.channel_id, SECONDARY_VISION_CHANNEL_ID)


class _ChatCompletionsClient:
    """A minimal OpenAI-chat-shaped client stub: answers
    `chat.completions.create(...)` with a fixed response TEXT, so the reader's own
    blank-inference logic can be exercised with a controlled model response."""
    __module__ = "openai._client"

    def __init__(self, content: str):
        from types import SimpleNamespace
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response))


class IndependentReaderBlankInferenceTest(unittest.TestCase):
    """Codex F7-R3-A, exact-SHA re-review, third pass: an EMPTY vision-model response
    must never be silently promoted to a positive BLANK finding -- an API error, a
    refusal, or a truncated response all look identical to silence, and only the
    model's own explicit assertion may certify the page as genuinely blank."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "note.pdf"
        self.path.write_bytes(build_pdf([[]]))          # an image-only page
        self.document = compile_source_evidence(
            self.path, _extraction(["alpha beta gamma delta"], self.path))

    def _reader(self, content: str):
        from unittest import mock
        from app.ingestion.source_evidence import IndependentVisionReader
        reader = IndependentVisionReader(
            self.path, primary_channel=self.document.primary_channel)
        return reader, mock.patch("app.core.llm_client.get_openai_client",
                                  return_value=_ChatCompletionsClient(content))

    def test_an_empty_response_is_unreadable_never_inferred_blank(self):
        from app.contracts.source_evidence import PageStatus
        reader, patched = self._reader("")
        with patched:
            reads = reader.read_pages((1,))
        self.assertEqual(reads[1].status, PageStatus.UNREADABLE)

    def test_an_explicit_blank_marker_is_certified_blank(self):
        from app.contracts.source_evidence import PageStatus
        from app.ingestion.source_evidence import BLANK_PAGE_MARKER
        reader, patched = self._reader(BLANK_PAGE_MARKER)
        with patched:
            reads = reader.read_pages((1,))
        self.assertEqual(reads[1].status, PageStatus.BLANK)

    def test_real_transcribed_text_still_reads_normally(self):
        from app.contracts.source_evidence import PageStatus
        reader, patched = self._reader("some real transcribed text")
        with patched:
            reads = reader.read_pages((1,))
        self.assertEqual(reads[1].status, PageStatus.READ)
        self.assertEqual(reads[1].text, "some real transcribed text")


class BuildPageReadValidationTest(unittest.TestCase):
    """Codex F7-R3-A, exact-SHA re-review, fourth pass: the shared `build_page_read`
    builder -- not just this project's own production reader -- is the ONE place a
    claim-affecting BLANK finding can be minted, so it validates rather than merely
    records. Any caller (including a third-party `source_reader`) is bound by these
    invariants, not only `IndependentVisionReader`."""

    def test_empty_text_with_no_explicit_status_is_unreadable_not_blank(self):
        from app.contracts.source_evidence import PageStatus, build_page_read
        read = build_page_read("chan", 1, "")
        self.assertEqual(read.status, PageStatus.UNREADABLE)

    def test_an_explicit_blank_claim_requires_provenance(self):
        from app.contracts.source_evidence import (InvalidPageReadError, PageStatus,
                                                    build_page_read)
        with self.assertRaises(InvalidPageReadError):
            build_page_read("chan", 1, "", status=PageStatus.BLANK)   # no detail
        # With provenance, the same claim is accepted.
        read = build_page_read("chan", 1, "", status=PageStatus.BLANK,
                               detail="a deterministic detector confirmed no marks")
        self.assertEqual(read.status, PageStatus.BLANK)

    def test_a_blank_claim_with_text_is_self_contradicting(self):
        from app.contracts.source_evidence import (InvalidPageReadError, PageStatus,
                                                    build_page_read)
        with self.assertRaises(InvalidPageReadError):
            build_page_read("chan", 1, "some text", status=PageStatus.BLANK,
                            detail="claims blank but carries text")

    def test_a_read_claim_with_no_tokens_is_self_contradicting(self):
        from app.contracts.source_evidence import (InvalidPageReadError, PageStatus,
                                                    build_page_read)
        with self.assertRaises(InvalidPageReadError):
            build_page_read("chan", 1, "", status=PageStatus.READ)

    def test_blank_missing_unreadable_and_read_remain_distinguishable(self):
        """Codex's exact durable-record complaint: BLANK, MISSING, and UNREADABLE
        must never look byte-equivalent in the record. Status AND detail differ for
        each, and a genuine blank finding always carries its own provenance."""
        from app.contracts.source_evidence import PageRead, PageStatus, build_page_read
        blank = build_page_read("chan", 1, "", status=PageStatus.BLANK,
                                detail="detector confirmed no marks on the page")
        unreadable = build_page_read("chan", 1, "", status=PageStatus.UNREADABLE,
                                     detail="empty model response, no assertion")
        missing = PageRead(channel_id="chan", page_number=1,
                           status=PageStatus.MISSING, detail="reader returned no page")
        read = build_page_read("chan", 1, "real content here")
        records = [blank, unreadable, missing, read]
        statuses = {r.status for r in records}
        details = {r.detail for r in records}
        self.assertEqual(len(statuses), 4, "all four statuses must be distinct")
        self.assertEqual(len(details), 4, "all four detail strings must be distinct")

    def test_direct_construction_is_held_to_the_same_invariants(self):
        """Codex F7-R3-A, exact-SHA re-review, fifth pass, exact reproduction: the
        prior fix closed the empty-response inference path in `build_page_read`
        alone, but `PageRead` itself stayed directly constructible with any
        status/content/detail combination -- so a caller-supplied `source_reader`
        that never goes through the shared builder at all could still manufacture
        `PageRead(status=BLANK)` with no provenance. The invariant is now on the
        MODEL, so direct construction is refused too, with no builder involved."""
        from pydantic import ValidationError

        from app.contracts.source_evidence import PageRead, PageStatus
        with self.assertRaises(ValidationError):
            PageRead(channel_id="chan", page_number=2, status=PageStatus.BLANK)
        with self.assertRaises(ValidationError):
            PageRead(channel_id="chan", page_number=2, status=PageStatus.BLANK,
                     text="some text", detail="claims blank but carries text")
        with self.assertRaises(ValidationError):
            PageRead(channel_id="chan", page_number=2, status=PageStatus.READ,
                     text="")
        # A genuinely valid direct construction still succeeds -- the fix rejects
        # the invalid COMBINATION, not the constructor itself.
        read = PageRead(channel_id="chan", page_number=2, status=PageStatus.BLANK,
                        detail="detector confirmed no marks on the page")
        self.assertEqual(read.status, PageStatus.BLANK)

    def test_missing_carries_no_content_but_unreadable_may_carry_a_partial_read(self):
        """MISSING means the channel did not return the page AT ALL, so any content
        contradicts the claim -- refused, just like BLANK. UNREADABLE is different: a
        text layer that recovers too few tokens to trust is legitimately marked
        UNREADABLE while still carrying those few real tokens (see
        app/ingestion/source_evidence.py's low_text_yield case) -- that must keep
        working, not be swept into the same constraint."""
        from pydantic import ValidationError

        from app.contracts.source_evidence import PageRead, PageStatus, SourceToken
        with self.assertRaises(ValidationError):
            PageRead(channel_id="chan", page_number=1, status=PageStatus.MISSING,
                     text="unexpected content", detail="reader returned no page")
        # UNREADABLE with a genuine partial read is still accepted.
        partial = PageRead(
            channel_id="chan", page_number=1, status=PageStatus.UNREADABLE,
            text="a few", tokens=(SourceToken(text="a", normalized="a"),
                                  SourceToken(text="few", normalized="few")),
            detail="text layer recovers 2 of 30 tokens; too partial to trust")
        self.assertEqual(partial.status, PageStatus.UNREADABLE)
        self.assertEqual(len(partial.tokens), 2)


class TrustBoundaryRevalidationTest(unittest.TestCase):
    """Codex F7-R3-A, exact-SHA re-review, sixth pass, exact reproductions: the model
    validator (fifth pass) is not the whole story. `model_copy(update=...)` never
    re-runs it, and it never checked that `text_sha256`/`SourceToken.normalized`
    actually describe the read's own `text`. `with_channel` -- the one trust boundary
    a caller-supplied reader's output crosses -- now reconstructs and re-verifies
    every filed read from its own raw fields before accepting it."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_model_copy_derived_blank_bypass_is_refused(self):
        """Codex's exact reproduction: build a valid READ, `model_copy` it to
        status=BLANK with empty text/tokens/detail (which never re-runs the model
        validator), return it from a caller-supplied reader. Before this fix,
        `with_channel` accepted it outright and the page was exempted as
        independently verified blank with no `recall_page_coverage` gate."""
        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    PageStatus, ReadChannel, build_page_read)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        valid_read = build_page_read(SECONDARY_VISION_CHANNEL_ID, 1, "alpha")
        forged_blank = valid_read.model_copy(
            update={"status": PageStatus.BLANK, "text": "", "text_sha256": "",
                   "tokens": (), "detail": ""})
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: forged_blank})

    def test_forged_tokens_and_empty_digest_are_refused(self):
        """Codex's exact reproduction: raw text 'different raw words', an empty
        digest, and a token whose normalized field claims unrelated content. Before
        this fix, nothing checked that `text_sha256`/`SourceToken.normalized`
        actually described this read's own `text`, so `reconcile_spans()` could
        return AGREED for a quotation this page never actually contained."""
        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    PageRead, PageStatus, ReadChannel,
                                                    SourceToken)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        forged = PageRead(
            channel_id=SECONDARY_VISION_CHANNEL_ID, page_number=1,
            status=PageStatus.READ, text="different raw words", text_sha256="",
            tokens=(SourceToken(text="different", normalized="documented phrase"),),
            detail="")
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: forged})

    def test_a_token_whose_text_does_not_appear_in_its_own_page_is_refused(self):
        """A token may not name text absent from the read it is filed under, even
        with a correctly-normalized form and a correct digest of a DIFFERENT text."""
        import hashlib

        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    PageRead, PageStatus, ReadChannel,
                                                    SourceToken)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        raw = "alpha"
        digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        forged = PageRead(
            channel_id=SECONDARY_VISION_CHANNEL_ID, page_number=1,
            status=PageStatus.READ, text=raw, text_sha256=digest,
            tokens=(SourceToken(text="nowhere", normalized="nowhere"),))
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: forged})

    def test_a_genuinely_valid_read_still_passes_the_boundary(self):
        """The fix rejects the invalid COMBINATIONS above, not every direct
        construction -- a self-consistent read (real digest, real matching tokens)
        still crosses the boundary normally."""
        import hashlib

        from app.contracts.source_evidence import (ChannelKind, PageRead, PageStatus,
                                                    ReadChannel, SourceToken)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        raw = "alpha beta"
        digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        genuine = PageRead(
            channel_id=SECONDARY_VISION_CHANNEL_ID, page_number=1,
            status=PageStatus.READ, text=raw, text_sha256=digest,
            tokens=(SourceToken(text="alpha", normalized="alpha"),
                   SourceToken(text="beta", normalized="beta")))
        covered = document.with_channel(channel, {1: genuine})
        self.assertTrue(covered.page(1).read_by(SECONDARY_VISION_CHANNEL_ID).usable)

    def test_reordered_tokens_fabricating_a_contiguous_quote_are_refused(self):
        """Codex F7-R3-A, exact-SHA re-review, seventh pass, exact reproduction: raw
        text 'phrase documented' filed with the ORDERED token stream
        ['documented', 'phrase'] -- both tokens genuinely occur in the text, so mere
        membership accepted this, and reconciliation could then return AGREED for
        the fabricated contiguous quote 'documented phrase', which the page never
        actually says."""
        import hashlib

        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    PageRead, PageStatus, ReadChannel,
                                                    SourceToken)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        raw = "phrase documented"
        digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        reordered = PageRead(
            channel_id=SECONDARY_VISION_CHANNEL_ID, page_number=1,
            status=PageStatus.READ, text=raw, text_sha256=digest,
            tokens=(SourceToken(text="documented", normalized="documented"),
                   SourceToken(text="phrase", normalized="phrase")))
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: reordered})

    def test_an_omitted_intervening_word_fabricating_a_contiguous_quote_is_refused(self):
        """Codex F7-R3-A, exact-SHA re-review, eighth pass, exact reproduction: raw
        text 'documented unrelated phrase' filed with the token stream
        ['documented', 'phrase'] -- SKIPPING the real intervening word 'unrelated'.
        Order and multiplicity alone do not catch this: the claimed tokens are in
        order and each occurs exactly once, but omitting a real word between them
        turns two separated words into an apparently CONTIGUOUS fabricated quote
        'documented phrase', which the page never actually states."""
        import hashlib

        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    PageRead, PageStatus, ReadChannel,
                                                    SourceToken)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        raw = "documented unrelated phrase"
        digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        omitted = PageRead(
            channel_id=SECONDARY_VISION_CHANNEL_ID, page_number=1,
            status=PageStatus.READ, text=raw, text_sha256=digest,
            tokens=(SourceToken(text="documented", normalized="documented"),
                   SourceToken(text="phrase", normalized="phrase")))
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: omitted})

    def test_a_token_claimed_more_times_than_it_actually_occurs_is_refused(self):
        """Codex's exact multiplicity reproduction: raw text contains one occurrence
        of a word, but the claimed token stream repeats it -- over-claiming
        occurrence count is a fabrication just like reordering."""
        import hashlib

        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    PageRead, PageStatus, ReadChannel,
                                                    SourceToken)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        raw = "alpha beta"
        digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        over_claimed = PageRead(
            channel_id=SECONDARY_VISION_CHANNEL_ID, page_number=1,
            status=PageStatus.READ, text=raw, text_sha256=digest,
            tokens=(SourceToken(text="alpha", normalized="alpha"),
                   SourceToken(text="alpha", normalized="alpha"),
                   SourceToken(text="beta", normalized="beta")))
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: over_claimed})

    def test_a_forged_read_baked_directly_into_an_incoming_document_is_refused(self):
        """Codex's exact reproduction: `source_evidence` is itself a directly
        caller-suppliable parameter to `code_encounter` -- a caller can hand in a
        fully-formed `SourceEvidenceDocument` whose reads were never filed through
        `with_channel` at all, bypassing that boundary entirely. `revalidated()`
        must catch a forged read wherever it already sits on an incoming document,
        not only ones added afterward."""
        import hashlib

        from app.contracts.source_evidence import (InvalidSourceEvidenceDocument, PageRead,
                                                    PageStatus, SourceToken)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        raw = "phrase documented"
        digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        forged_read = PageRead(
            channel_id=EMBEDDED_TEXT_CHANNEL_ID, page_number=1,
            status=PageStatus.READ, text=raw, text_sha256=digest,
            tokens=(SourceToken(text="documented", normalized="documented"),
                   SourceToken(text="phrase", normalized="phrase")))
        # Baked directly into the page's reads, bypassing `with_channel` entirely --
        # replacing the genuine embedded-text-layer read with a forged one carrying
        # the SAME channel id, exactly as a hand-built document could.
        forged_pages = tuple(
            page.model_copy(update={"reads": tuple(
                forged_read if r.channel_id == EMBEDDED_TEXT_CHANNEL_ID else r
                for r in page.reads)})
            if page.page_number == 1 else page
            for page in document.pages)
        forged_document = document.model_copy(update={"pages": forged_pages})
        with self.assertRaises(InvalidSourceEvidenceDocument):
            forged_document.revalidated()

    def test_a_genuine_document_revalidates_identically(self):
        """The fix must not disturb an ordinary, already-valid compiled document --
        `revalidated()` is safe to run unconditionally."""
        document = _compile(self.root, [["alpha beta"], ["gamma delta"]],
                            ["alpha beta", "gamma delta"])
        revalidated = document.revalidated()
        self.assertEqual(revalidated.document_sha256, document.document_sha256)
        self.assertEqual(
            [p.read_by(EMBEDDED_TEXT_CHANNEL_ID).text_sha256 for p in revalidated.pages],
            [p.read_by(EMBEDDED_TEXT_CHANNEL_ID).text_sha256 for p in document.pages])


class ChannelBindingTest(unittest.TestCase):
    """Codex F7-R3-A, exact-SHA re-review, fifth pass: `with_channel` attached
    whatever `PageRead` object a caller filed under a given page number, with no
    check that the object's OWN `channel_id`/`page_number` agreed with where it was
    being filed -- a read for one page or channel could be silently recorded as
    evidence for a different one."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_read_naming_a_different_page_than_it_is_filed_under_is_refused(self):
        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    ReadChannel, build_page_read)
        document = _compile(self.root, [["alpha"], ["beta"]], ["alpha", "beta"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        # Filed under page 1, but the read itself names page 2.
        mismatched = build_page_read(SECONDARY_VISION_CHANNEL_ID, 2, "beta")
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: mismatched})

    def test_a_read_naming_a_different_channel_than_it_is_filed_under_is_refused(self):
        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    ReadChannel, build_page_read)
        document = _compile(self.root, [["alpha"]], ["alpha"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        wrong_channel = build_page_read("some_other_channel", 1, "alpha")
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {1: wrong_channel})

    def test_a_read_for_a_page_this_document_does_not_have_is_refused(self):
        from app.contracts.source_evidence import (ChannelKind, InvalidSourceEvidenceDocument,
                                                    ReadChannel, build_page_read)
        document = _compile(self.root, [["alpha"]], ["alpha"])   # ONE page only
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        out_of_range = build_page_read(SECONDARY_VISION_CHANNEL_ID, 99, "alpha")
        with self.assertRaises(InvalidSourceEvidenceDocument):
            document.with_channel(channel, {99: out_of_range})


class CertificateChannelDistinguishabilityTest(unittest.TestCase):
    """Codex F7-R3-A, exact-SHA re-review, fifth pass, exact complaint: 'explicit
    MISSING and UNREADABLE runs produce identical consensus and identical
    human-readable SourceReconciliation after removing only the opaque fingerprint.
    The new test compares intermediate PageRead objects, not final certificate
    records.' This proves it through `certificate_record()` -- the ACTUAL bytes bound
    into the release certificate -- not through `PageRead` objects directly."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_missing_and_unreadable_are_distinct_in_the_final_certificate_record(self):
        from app.contracts.source_evidence import (ChannelKind, PageStatus, ReadChannel,
                                                    build_page_read, reconcile_spans)
        document = _compile(self.root, [["alpha"], ["beta"]], ["alpha", "beta"])
        channel = ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                              kind=ChannelKind.VISION, provider="openai")
        # Page 1: the independent channel tried and got nothing back -- UNREADABLE.
        # Page 2: the independent channel never covers it at all -- MISSING (no read
        # object is filed for it; `with_channel` only accepts pages it HAS read).
        covered = document.with_channel(
            channel, {1: build_page_read(SECONDARY_VISION_CHANNEL_ID, 1, "",
                                         status=PageStatus.UNREADABLE,
                                         detail="empty model response")})

        record = reconcile_spans(covered, [])
        cert = record.certificate_record()
        outcomes = {p["page_number"]: p for p in cert["page_outcomes"]}

        self.assertEqual(
            outcomes[1]["channel_statuses"][SECONDARY_VISION_CHANNEL_ID], "UNREADABLE")
        self.assertEqual(
            outcomes[2]["channel_statuses"][SECONDARY_VISION_CHANNEL_ID], "MISSING")
        # For THIS channel specifically, both pages look identical under the old
        # "not independently read" boolean (neither made the list) -- the PDF's own
        # embedded-text channel is independent too and covers both, which is exactly
        # why a coarser signal cannot be trusted to carry the distinction.
        self.assertNotIn(SECONDARY_VISION_CHANNEL_ID, outcomes[1]["independently_read_by"])
        self.assertNotIn(SECONDARY_VISION_CHANNEL_ID, outcomes[2]["independently_read_by"])
        self.assertNotEqual(outcomes[1]["channel_statuses"],
                            outcomes[2]["channel_statuses"])

        # Reader identity (not just a bare channel_id string) is bound in too.
        identities = {c["channel_id"]: c for c in cert["channel_identities"]}
        self.assertIn(SECONDARY_VISION_CHANNEL_ID, identities)
        self.assertEqual(identities[SECONDARY_VISION_CHANNEL_ID]["provider"], "openai")
        self.assertEqual(identities[SECONDARY_VISION_CHANNEL_ID]["kind"], "vision")


class SourceDigestVerificationTest(unittest.TestCase):
    """Source identity is a fact the compiler establishes, not an upstream assertion."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "note.pdf"
        self.path.write_bytes(build_pdf([["alpha beta gamma"]]))

    def test_a_document_digest_that_is_not_the_document_is_refused(self):
        payload = _extraction(["alpha beta gamma"], self.path)
        payload["note_integrity"]["source_pdf_sha256"] = "sha256:" + "ee" * 32
        with self.assertRaisesRegex(SourceEvidenceCompilationError,
                                    "not the same bytes"):
            compile_source_evidence(self.path, payload)

    def test_the_recorded_digest_is_the_one_the_compiler_computed(self):
        import hashlib
        document = compile_source_evidence(
            self.path, _extraction(["alpha beta gamma"], self.path))
        self.assertEqual(
            document.document_sha256,
            "sha256:" + hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_a_page_image_digest_that_the_bytes_do_not_produce_is_refused(self):
        payload = _extraction(["alpha beta gamma"], self.path)
        payload["note_integrity"]["page_images"][0]["sha256"] = "sha256:" + "ff" * 32
        with self.assertRaisesRegex(SourceEvidenceCompilationError, "digests to"):
            compile_source_evidence(self.path, payload)

    def test_a_page_image_digest_with_no_bytes_to_check_it_is_refused(self):
        payload = _extraction(["alpha beta gamma"], self.path)
        payload["page_image_bytes"] = {}
        with self.assertRaisesRegex(SourceEvidenceCompilationError, "cannot verify"):
            compile_source_evidence(self.path, payload)

    def test_a_swapped_document_is_caught_even_though_the_reading_is_consistent(self):
        """The end-to-end shape of the defect: the transcription is internally perfect
        and describes a DIFFERENT file from the one being compiled."""
        other = self.root / "other.pdf"
        other.write_bytes(build_pdf([["delta epsilon zeta"]]))
        payload = _extraction(["alpha beta gamma"], self.path)
        with self.assertRaises(SourceEvidenceCompilationError):
            compile_source_evidence(other, payload)
