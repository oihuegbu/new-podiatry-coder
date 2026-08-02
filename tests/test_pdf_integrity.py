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
    def test_all_pages_are_sent_and_bound_to_result(self):
        parsed = {
            "patient_metadata": {"date_of_service": "2026-01-05"},
            "sections": {"assessment_diagnoses": "Documented diagnosis",
                         "plan": "Performed service"},
            "note_category": "established_patient_visit",
            "procedures_performed_today": ["Performed service"],
            "imaging_performed_today": [], "supplies_dispensed_today": [],
            "prior_surgery_info": {}, "physician_documented_codes": [],
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
                               return_value=response):
                result = pdf_parser.extract_from_pdf(pdf)
        kwargs = convert.call_args.kwargs
        self.assertEqual(kwargs, {"dpi": 300})
        self.assertEqual(result["note_integrity"]["page_count"], 3)
        self.assertEqual(result["note_integrity"]["extracted_page_count"], 3)
        self.assertTrue(result["note_integrity"]["complete"])
        self.assertTrue(result["note_integrity"]["source_pdf_sha256"].startswith(
            "sha256:"))


if __name__ == "__main__":
    unittest.main()
