"""End-to-end tests for the claude-medical-coder pipeline.

Runs the WHOLE flow (extract -> resolve -> arbitrate -> gate -> autonomy ->
certificate) with a MockSource and stubbed LLMs, so it needs no API key, no RAG
index, and — deliberately — contains NO real medical code (the mock uses
synthetic identifiers). It asserts the safety properties, not just happy paths:
planned work is not billed, negated findings are dropped, unsupported evidence
blocks release, and autonomy is granted only when the chain closes.
"""
import unittest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, Outcome, ResolutionMethod, Verdict
from claude_coder.pipeline import code_encounter

# A note whose text contains, verbatim, every evidence span the extractor emits.
NOTE = (
    "Procedure: excision of the interdigital neuroma, right third interspace. "
    "Assessment: interdigital neuroma, right foot. "
    "Patient denies chest pain. "
    "Plan hammertoe correction next visit."
)

# What the (stubbed) CLU extractor returns: one performed procedure, one current
# diagnosis, one PLANNED procedure (must not bill), one NEGATED finding (drop).
FACTS_JSON = """{"facts":[
 {"kind":"procedure","description":"excision of interdigital neuroma",
  "attributes":{"laterality":"right","anatomy":"third interspace"},
  "disposition":"performed_today","negated":false,
  "evidence":["excision of the interdigital neuroma, right third interspace"],
  "confidence":0.97},
 {"kind":"diagnosis","description":"interdigital neuroma of the right foot",
  "attributes":{"laterality":"right"},"disposition":"performed_today","negated":false,
  "evidence":["interdigital neuroma, right foot"],"confidence":0.98},
 {"kind":"procedure","description":"hammertoe correction","attributes":{},
  "disposition":"planned","negated":false,
  "evidence":["Plan hammertoe correction next visit"],"confidence":0.9},
 {"kind":"diagnosis","description":"chest pain","attributes":{},
  "disposition":"performed_today","negated":true,
  "evidence":["denies chest pain"],"confidence":0.9}
]}"""

# Synthetic (non-code) identifiers — no real medical code anywhere in this test.
PROC = CandidateCode("PROC_NEUROMA_EXC", "cpt",
                     "Excision, interdigital neuroma, single, each", 0.9, "retrieval")
DX = CandidateCode("DX_NEUROMA_RIGHT", "icd10",
                   "interdigital neuroma, right foot", 0.9, "retrieval")


def _source():
    return MockSource(
        records={("PROC_NEUROMA_EXC", "cpt"): {"active": True},
                 ("DX_NEUROMA_RIGHT", "icd10"): {"active": True}},
        retrieval={("*", "cpt"): [PROC], ("*", "icd10"): [DX]},
    )


def _extract_stub(system, user):
    return FACTS_JSON


def _arbitrate_stub(system, user):
    return '{"choice":0,"confidence":0.0,"reason":"unused"}'


class AutonomousCoderTest(unittest.TestCase):

    def _run(self, note=NOTE, dos="2026-03-14"):
        return code_encounter("enc-1", note, dos, source=_source(),
                              extract_llm=_extract_stub, arbitrate_llm=_arbitrate_stub)

    def test_happy_path_auto_ready(self):
        r = self._run()
        codes = {ln.chosen.code for ln in r.billable_lines}
        self.assertEqual(codes, {"PROC_NEUROMA_EXC", "DX_NEUROMA_RIGHT"})
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_planned_work_not_billed(self):
        r = self._run()
        billed = {ln.chosen.code for ln in r.billable_lines}
        self.assertNotIn("hammertoe", " ".join(billed).lower())
        # the planned procedure produced no billable line at all
        self.assertEqual(len(r.billable_lines), 2)

    def test_negated_finding_dropped(self):
        r = self._run()
        descs = " ".join(ln.fact.description for ln in r.lines).lower()
        self.assertNotIn("chest pain", descs)

    def test_resolution_is_deterministic(self):
        r = self._run()
        for ln in r.billable_lines:
            self.assertEqual(ln.method, ResolutionMethod.DETERMINISTIC, ln.rationale)

    def test_missing_evidence_blocks_release(self):
        # a note that does NOT contain the procedure's evidence span
        r = self._run(note="Assessment: interdigital neuroma, right foot.")
        ev = next(g for g in r.gates if g.name == "verbatim_evidence")
        self.assertEqual(ev.outcome, Outcome.BLOCKED)
        self.assertEqual(r.verdict, Verdict.BLOCKED)

    def test_missing_dos_blocks_release(self):
        r = self._run(dos=None)
        dos = next(g for g in r.gates if g.name == "date_of_service")
        self.assertEqual(dos.outcome, Outcome.BLOCKED)
        self.assertEqual(r.verdict, Verdict.BLOCKED)

    def test_certificate_is_reproducible(self):
        a = self._run().certificate["certificate_sha256"]
        b = self._run().certificate["certificate_sha256"]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)


if __name__ == "__main__":
    unittest.main()
