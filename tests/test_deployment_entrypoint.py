"""The DEPLOYED entrypoint reaches the checkpoint-enforced coder — proved, not asserted.

Codex finding F6-R4-A1 (issue #6, P1): `docker-compose.yml` runs `python run.py` and
`terraform/templates/user_data.sh.tftpl` calls the same path on first boot and from the
note watcher, but `run.py` used to construct `app.pipeline.MedicalCodingPipeline` — a
separate implementation that never touches `claude_coder`'s provenance repository, source
gates, eligibility gates, certificate creation, or external terminal-head checkpoint.
Pinning `PROVENANCE_CHECKPOINT_REQUIRED=1` in Compose was therefore inert for the real note
processor, and the prior wiring regression only proved that the environment STRING existed.

This module closes that hole from the deployment side:

  * the structural tests pin the deployment command to this entrypoint, and pin this
    entrypoint's coding call to `claude_coder`, by parsing the checked-in source;

  * the end-to-end tests drive `run.main()` — the exact Python target of
    `docker compose run app python run.py` — over a note directory, with the checkpoint
    anchor REQUIRED and either unavailable or mismatched, and prove that no releasable
    claim and no certificate can come out of the deployed path.

What is substituted, and why each substitution cannot manufacture the result:

  `run.extract_from_pdf`   the Claude-Vision PDF transcriber. An input source, not a
                           decision component; stubbed so the test needs no API key. It
                           supplies note text, a DOS and a document version, i.e. MORE
                           input than the real thing has to produce.
  the extraction LLM       stubbed with a well-formed fact/relation graph so extraction
                           genuinely SUCCEEDS. This matters: if extraction failed, the
                           encounter would hold at the same `pre_retrieval_integrity`
                           boundary for a completely different reason and the test would
                           pass for the wrong one.
  `AuthoritativeSource._vector_store`
                           the RAG index, replaced by an index that returns no hits. A
                           test must not build a 60-90 minute Qdrant index. Everything
                           else on the source (reference tables, claim-assembly data,
                           gates, fingerprint) is the real `AuthoritativeSource`, and an
                           empty index can only ever make the outcome MORE conservative.
  the verify/corroborate   answer "nothing is entailed" — the conservative answer, and
  LLMs                     the only one an index with no hits could support. Stubbed so
                           the run needs no network at all; without this the encounter
                           would hold on a provider error, which is a DIFFERENT hold
                           than the one under test.

The control test is the discriminator: the SAME invocation with a working anchor commits
durable audit rows, so the two failing cases differ from it in exactly one thing — the
terminal-head checkpoint.

No real medical code appears anywhere in this file; the fixture note is synthetic.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run as entrypoint  # noqa: E402  — the exact module the deployment executes
from tests.source_pdf import build_pdf, digest_of, vision_extraction  # noqa: E402


# --------------------------------------------------------------- structural wiring
def test_the_compose_command_is_this_entrypoint():
    """The deployed container command must still be the file these tests drive."""
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    assert 'command: ["python", "run.py"]' in compose, (
        "docker-compose.yml no longer runs run.py; the end-to-end proof below no longer "
        "describes the deployed path")


def test_first_boot_and_the_note_watcher_invoke_the_same_entrypoint():
    user_data = (REPO_ROOT / "terraform" / "templates" / "user_data.sh.tftpl").read_text()
    assert "python run.py --setup-only" in user_data, (
        "first-boot dependency loading no longer goes through run.py --setup-only")
    assert "python run.py" in user_data.split("process-notes.sh")[-1] or \
           "run.py \"$@\"" in user_data, (
        "the process-notes helper (which the note-watcher service calls) no longer runs run.py")


def test_the_entrypoint_imports_the_claude_coder_pipeline_and_not_the_retired_one():
    """Parsed from the AST, not grepped: this file's own header discusses `app.pipeline`
    in prose, and a substring check would either miss the real thing or trip on the
    explanation of why it is gone."""
    tree = ast.parse((REPO_ROOT / "run.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "claude_coder.pipeline.code_encounter" in imported, (
        "the deployed entrypoint no longer imports the checkpoint-enforced coder")
    assert not [name for name in imported if name.startswith("app.pipeline")], (
        f"the deployed entrypoint imports the retired app.pipeline again: "
        f"{sorted(n for n in imported if n.startswith('app.pipeline'))}")


def test_retired_consistency_flags_are_refused_not_silently_downgraded():
    """`tools/unanimity_loop.py` still invokes this entrypoint with exactly these flags.
    Honouring them by running ONCE would hand a caller a fraction of the assurance it
    asked for, silently; the refusal makes that driver stop and say so. Returns before
    any dependency is loaded, so this is also the cheap proof that the refusal is
    unconditional rather than dependent on a healthy environment."""
    assert entrypoint.main(["--consistency", "3", "--consistency-workers", "12"]) == \
        entrypoint.EXIT_RETIRED_FLAG
    assert entrypoint.main(["--consistency", "2"]) == entrypoint.EXIT_RETIRED_FLAG
    assert entrypoint.EXIT_RETIRED_FLAG not in (0, 1, 2), (
        "the retired-flag exit code must be distinguishable from success, a batch "
        "failure, and argparse's usage error")


# ------------------------------------------------------------------- e2e fixture
#: Synthetic note. The pipeline's disposition/negation logic turns on LINGUISTIC
#: markers, never on any clinical term, so nothing here needs to be a real code.
#: The date of service is WRITTEN ON THE PAGE and reported identically in the
#: transcription's metadata, so the encounter's DOS binds from the reconciled
#: document (issue #6 F7-R4) and these tests keep failing for the ONE reason each
#: of them is about rather than for a missing date.
NOTE_TEXT = (
    "Date of service: 2026-03-14. "
    "Procedure: excision of lesion alpha, right site two. "
    "Assessment: condition alpha, right side. "
    "Excision of lesion alpha was performed for condition alpha of the right side. "
    "Patient denies finding gamma."
)

FACTS_JSON = json.dumps({
    "facts": [
        {"kind": "procedure", "description": "excision of lesion alpha",
         "attributes": {"laterality": "right", "anatomy": "site two"},
         "disposition": "performed_today", "negated": False,
         "evidence": ["excision of lesion alpha, right site two",
                      "Excision of lesion alpha was performed"],
         "confidence": 0.97,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "performer": 0.99, "relationship": 0.99}},
        {"kind": "diagnosis", "description": "condition alpha of the right side",
         "attributes": {"laterality": "right"}, "disposition": "performed_today",
         "negated": False,
         "evidence": ["condition alpha, right side",
                      "condition alpha of the right side"],
         "confidence": 0.98,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "assertion": 0.99, "experiencer": 0.99}},
    ],
    "relations": [
        {"subject_event_id": "F2", "object_event_id": "F1", "predicate": "reason_for",
         "state": "asserted", "evidence_fact_ids": ["F1", "F2"], "confidence": 0.99},
    ],
})

STEM = "NOTE_ENTRYPOINT_001"
#: Computed from the exact bytes written below: the compiler recomputes the digest of
#: the document it compiles and refuses a transcription claiming another (F7-R5).
DOCUMENT_VERSION = digest_of(build_pdf([[NOTE_TEXT]]))


class _StubVectorStore:
    """Stands in for the Qdrant hybrid index — an index that knows nothing.

    Retrieval IS reached here (a diagnosis has no performer, so it is not held by the
    unknown-ownership gate the way a procedure is), and building a real 60-90 minute
    Qdrant index inside a test is not an option. Returning no hits leaves every gate,
    every authoritative lookup and the whole release path real, and simply means the
    synthetic fixture concepts resolve to nothing — which is the honest answer for
    concepts that exist in no code set.
    """

    def search(self, description, system, top_k=20):
        return []


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A complete, isolated deployment: notes in, results out, own provenance store."""
    from app.core import config as app_config
    from claude_coder import arbitration
    from claude_coder import extraction
    from claude_coder import verify as verify_module
    from claude_coder.data_access import AuthoritativeSource

    notes_dir = tmp_path / "attachments"
    notes_dir.mkdir()
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    anchor_root = tmp_path / "anchor"
    # A REAL document, so the source-evidence reconciliation these tests now pass
    # through has an independent reading to work with (issue #6 F6-R6-A).
    (notes_dir / f"{STEM}.pdf").write_bytes(build_pdf([[NOTE_TEXT]]))

    monkeypatch.setattr(entrypoint, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(entrypoint, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(app_config, "PROVENANCE_DB", tmp_path / "provenance.db")
    monkeypatch.setattr(entrypoint, "extract_from_pdf", lambda pdf_path:
                        vision_extraction(
                            [NOTE_TEXT], metadata={"date_of_service": "2026-03-14"},
                            pdf_path=pdf_path))
    monkeypatch.setattr(extraction, "_default_llm", lambda system, user: FACTS_JSON)
    monkeypatch.setattr(AuthoritativeSource, "_vector_store",
                        lambda self: _StubVectorStore())
    # Declared providers mirror the deployment's (verifier OpenAI, corroborator
    # Anthropic) so the independence machinery sees a well-formed pair rather than an
    # undeclared one; both answer "not entailed", so nothing can be verified anyway.
    monkeypatch.setattr(verify_module, "default_verify_llm",
                        verify_module.declare_model_profile(
                            lambda system, user: '{"choice": 0, "reason": "stub"}',
                            provider="openai"))
    monkeypatch.setattr(verify_module, "default_corroborate_llm",
                        verify_module.declare_model_profile(
                            lambda system, user: '{"entailed": false, '
                                                 '"missing_element": false, '
                                                 '"reason": "stub"}',
                            provider="claude"))
    # The SECOND INDEPENDENT READING (product directive section 3; phase 4, commit
    # 97f748b). This file keeps its OWN copy of the deployment fixture, so it needs the
    # same substitution for the same reason as `tests/test_claim_bundle_e2e.py`: the
    # deployed entrypoint passes no extractor, the pipeline therefore auto-enables
    # `extraction.default_second_extract_llm`, and an unsubstituted one calls a live
    # vendor. Without a network that call raises, the pre-retrieval boundary holds the
    # encounter, and the anchor/source-manifest behaviour these tests exist to prove is
    # never reached -- the run holds for the wrong reason and the test says nothing.
    # A new real-mode default LLM must be substituted in BOTH fixtures in the same
    # commit; `test_every_real_mode_default_llm_is_substituted` in
    # `tests/test_claim_bundle_e2e.py` derives that requirement from the modules.
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

    class _Deployment:
        def __init__(self):
            self.output_dir = output_dir
            self.anchor_root = anchor_root

        def run(self) -> dict:
            assert entrypoint.main([]) == 0, "the batch itself must not crash"
            return json.loads((output_dir / f"{STEM}_results.json").read_text())

        def anchor_file(self) -> Path:
            (found,) = list(anchor_root.glob("*.checkpoint.json"))
            return found

    return _Deployment()


def _gates(payload) -> list[dict]:
    """The release-gate outcomes carried by the bundle."""
    return [o for o in payload["outcomes"] if o["stage"] == "release_gate"]


def _assert_nothing_releasable(payload):
    assert payload["processing_error"] == "", payload["processing_error"]
    assert payload["release"]["producer_releasable"] is False
    assert payload["release"]["holds"], (
        "a bundle with no recorded hold is a bundle that claims it may be billed")
    assert payload["certificate"] is None
    assert payload["service_lines"] == []
    assert payload["diagnoses"] == []
    assert payload["release"]["producer_verdict"] != "AUTO_READY"
    # SYSTEM_RETRY is the canonical destination for the producer's SYSTEM_HOLD:
    # a dependency failed, so this is system work and never a coder's queue.
    assert payload["release"]["destination"] == "SYSTEM_RETRY"
    assert payload["release"]["producer_destination"] == "SYSTEM_HOLD"
    assert payload["audit"]["audit_record_hashes"] == [], (
        "a durable audit row was committed even though the checkpoint refused")
    gates = _gates(payload)
    held = [g for g in gates if g["name"] == "pre_retrieval_integrity"]
    assert held, f"no enforced-boundary hold in {[g['name'] for g in gates]}"
    assert held[0]["outcome"] == "UNKNOWN"
    assert held[0]["retryable"] is True


# ------------------------------------------------------------------- the proof
def test_required_but_unavailable_checkpoint_releases_nothing(deployment, monkeypatch):
    """`PROVENANCE_CHECKPOINT_REQUIRED=1` with no anchor configured — the exact drift
    docker-compose.yml's pinned constant is designed to guarantee is the WORST case."""
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.delenv("PROVENANCE_CHECKPOINT_ANCHOR", raising=False)

    payload = deployment.run()
    _assert_nothing_releasable(payload)

    combined = json.loads((deployment.output_dir / "all_results.json").read_text())
    assert [p for p in combined if p["encounter"]["document_id"] == STEM]
    assert not [p for p in combined if not (p.get("release") or {}).get("holds")], (
        "the aggregate corpus file offers a releasable claim the per-note file refused")


def test_mismatched_checkpoint_releases_nothing(deployment, monkeypatch):
    """A configured anchor whose stored checkpoint cannot be verified. Unverifiable is
    not absent: the release must hold, never fall back to 'no anchor configured'."""
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ANCHOR",
                       f"file:{deployment.anchor_root}")

    first = deployment.run()
    assert first["audit"]["audit_record_hashes"], (
        "control precondition failed: the working anchor committed no audit rows")

    deployment.anchor_file().write_text("{ this is not a valid checkpoint record")

    payload = deployment.run()
    _assert_nothing_releasable(payload)


def test_control_a_working_anchor_reaches_the_durable_audit_chain(deployment,
                                                                  monkeypatch):
    """The discriminator. Same entrypoint, same note, same stubs — only the checkpoint
    differs — and here the encounter DOES reach the provenance repository and commit
    durable rows. So the two holds above are attributable to the checkpoint and to
    nothing else (a missing API key, a malformed note, an unbuilt index would all have
    held identically and silently)."""
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ANCHOR",
                       f"file:{deployment.anchor_root}")

    payload = deployment.run()

    assert payload["processing_error"] == "", payload["processing_error"]
    assert len(payload["audit"]["audit_record_hashes"]) >= 2, (
        "evidence anchoring and the relation graph must both be durably recorded")
    assert deployment.anchor_file().exists()
    # Still not releasable — no reviewed participant roster, no encounter-context
    # source, and an empty index — so nobody reads this control as 'the deployment
    # auto-releases'.
    assert payload["release"]["producer_releasable"] is False
    assert payload["release"]["holds"]
    assert payload["service_lines"] == []


# ------------------------------------------------------- round 6, Codex F6-R5-A
# The reviewer named two claim-affecting sources that were read out of filename
# literals in `app/**` and were therefore never source IDENTITIES: they reached the
# release manifest only through the incidental `data/codes/*.json` sweep, and a sweep
# cannot report a file that is not there. They are declared and REQUIRED now, so their
# absence has to be visible from the DEPLOYED entrypoint -- not just from a unit test
# against the registry.
#
# Being honest about which half of the finding this proves: post-cutover the deployed
# note→code path (`claude_coder.pipeline.code_encounter`) does not READ either file, so
# their corruption cannot reach a live claim; what binds here is the required-source
# disposition, which now holds the release when one of them goes missing. Their
# present-but-corrupt behavior is proved where they ARE read, in
# tests/test_release_atomicity.py.

@pytest.mark.parametrize("source_id", ["coding_semantics", "payer_registry"])
def test_a_missing_required_app_source_holds_the_deployed_entrypoint(deployment,
                                                                     monkeypatch,
                                                                     tmp_path,
                                                                     source_id):
    """Same invocation as the control test above, with a WORKING checkpoint anchor --
    so the only difference is the missing source, and the hold is attributable to it."""
    from app.release import source_manifest as sm
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ANCHOR", f"file:{deployment.anchor_root}")
    registry = dict(sm._AUTHORITATIVE)
    registry[source_id] = tmp_path / "absent-required-source.json"
    monkeypatch.setattr(sm, "_AUTHORITATIVE", registry)

    payload = deployment.run()

    assert payload["processing_error"] == "", payload["processing_error"]
    assert payload["release"]["producer_releasable"] is False
    assert payload["certificate"] is None
    assert payload["service_lines"] == []
    assert payload["release"]["producer_verdict"] != "AUTO_READY"
    gates = _gates(payload)
    blocked = [g for g in gates if g["name"] == "source_manifest"
               and g["outcome"] != "PASS"]
    assert blocked, (
        f"a missing REQUIRED source did not stop the deployed path: "
        f"{[(g['name'], g['outcome']) for g in gates]}")
    assert source_id in blocked[0]["detail"], blocked[0]["detail"]
