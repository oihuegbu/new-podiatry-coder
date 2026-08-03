"""PDF extraction covers and fingerprints the complete source document."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from app.ingestion import pdf_parser


class PdfIntegrityTest(unittest.TestCase):
    def test_embedded_text_layer_corroborates_layout_independent_transcript(self):
        words = [f"token{index}" for index in range(30)]
        report = pdf_parser._text_layer_crosscheck(
            [" ".join(words)], ["\n".join(reversed(words))])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["pages"][0]["token_recall"], 1.0)
        self.assertNotIn(" ".join(words), json.dumps(report))

    def test_embedded_text_layer_material_mismatch_fails(self):
        expected = " ".join(f"expected{index}" for index in range(30))
        observed = " ".join(f"different{index}" for index in range(30))
        report = pdf_parser._text_layer_crosscheck([expected], [observed])
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["failed_pages"], [1])

    def test_image_only_pdf_has_explicit_not_applicable_status(self):
        report = pdf_parser._text_layer_crosscheck([""], ["vision text"])
        self.assertEqual(report["status"], "NOT_APPLICABLE")

    def test_all_pages_are_sent_and_bound_to_result(self):
        parsed = {
            "patient_metadata": {"date_of_service": "2026-01-05"},
            "sections": {"assessment_diagnoses": "Documented diagnosis",
                         "plan": "Performed service"},
            "note_category": "established_patient_visit",
            "procedures_performed_today": ["Performed service"],
            "imaging_performed_today": [], "supplies_dispensed_today": [],
            "prior_surgery_info": {}, "physician_documented_codes": [],
            "page_texts": [
                {"page_number": 1, "status": "extracted", "text": "Page one text"},
                {"page_number": 2, "status": "blank", "text": ""},
                {"page_number": 3, "status": "extracted", "text": "Page three text"},
            ],
        }
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(parsed))],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        images = [Image.new("RGB", (2, 2)) for _ in range(3)]
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "note.pdf"
            pdf.write_bytes(b"source-document")
            with mock.patch.object(pdf_parser, "convert_from_path",
                                   return_value=images) as convert, \
                    mock.patch("app.core.llm_client.get_anthropic_client",
                               return_value=object()), \
                    mock.patch("app.core.llm_client._claude_message_via_batch",
                               return_value=response), \
                    mock.patch.object(pdf_parser, "_embedded_text_crosscheck",
                                      return_value={"status": "NOT_APPLICABLE"}):
                result = pdf_parser.extract_from_pdf(pdf)
        kwargs = convert.call_args.kwargs
        self.assertEqual(kwargs, {"dpi": 300})
        self.assertEqual(result["note_integrity"]["page_count"], 3)
        self.assertEqual(result["note_integrity"]["extracted_page_count"], 3)
        self.assertTrue(result["note_integrity"]["complete"])
        self.assertTrue(result["note_integrity"]["source_pdf_sha256"].startswith(
            "sha256:"))
        self.assertEqual(
            [p["page_number"] for p in result["note_integrity"]["page_coverage"]],
            [1, 2, 3])

    def test_missing_page_proof_fails_closed(self):
        parsed = {
            "patient_metadata": {}, "sections": {}, "note_category": "other",
            "procedures_performed_today": [], "imaging_performed_today": [],
            "supplies_dispensed_today": [], "prior_surgery_info": {},
            "physician_documented_codes": [],
            "page_texts": [
                {"page_number": 1, "status": "extracted", "text": "Only first page"},
            ],
        }
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(parsed))],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        images = [Image.new("RGB", (2, 2)) for _ in range(2)]
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "note.pdf"
            pdf.write_bytes(b"source-document")
            with mock.patch.object(pdf_parser, "convert_from_path",
                                   return_value=images), \
                    mock.patch("app.core.llm_client.get_anthropic_client",
                               return_value=object()), \
                    mock.patch("app.core.llm_client._claude_message_via_batch",
                               return_value=response), \
                    mock.patch.object(pdf_parser, "_embedded_text_crosscheck",
                                      return_value={"status": "NOT_APPLICABLE"}):
                with self.assertRaisesRegex(RuntimeError, "coverage"):
                    pdf_parser.extract_from_pdf(pdf)


if __name__ == "__main__":
    unittest.main()
