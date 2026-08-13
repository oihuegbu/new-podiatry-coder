"""Codex F6-R5: release attestation is order-safe and fail-closed.

The RETURNED verdict, the CERTIFICATE, the data fingerprint, and the LAST durable audit
decision can never disagree: a missing fingerprint or a failed certificate/audit write
downgrades the returned verdict AND is reflected in (or absent from) the durable record --
AUTO_READY is never persisted before it is real, and a certificate never exists without a
matching durable record.
"""
import hashlib
import json
import os
import pytest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode
from claude_coder.pipeline import code_encounter
import claude_coder.certificate as cert_mod
import claude_coder.provenance as provenance

_FACTS = ('{"facts":[{"kind":"procedure","description":"excision of lesion",'
          '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
          '"disposition":"performed_today","negated":false,'
          '"evidence":["excision of lesion performed"],"confidence":0.99}]}')
_NOTE = "excision of lesion performed today"
_CTX = {"billing_entity_id": "actor-1", "participants": [{"id": "actor-1", "type": "person", "roles": ["performer"]}]}


def _sel(system, user):
    sl = system.lower()
    if "propose" in sl:
        return '{"codes":[]}'
    if "independently" in sl:
        return '{"entailed":true,"missing_element":false,"reason":"x"}'
    return '{"choice":1,"reason":"x"}'


def _src():
    return MockSource(records={("PROC_X", "cpt"): {"active": True}},
                      retrieval={("*", "cpt"): [CandidateCode("PROC_X", "cpt",
                                                              "Excision, lesion, each", 0.9)]})


class _CapturingAudit:
    def __init__(self, fail_kind=None):
        self.records = []
        self.fail_kind = fail_kind

    def append(self, encounter_id, kind, record):
        if kind == self.fail_kind:
            raise RuntimeError("audit write failed")
        self.records.append((kind, record))
        return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()

    def last_release(self):
        rel = [r for k, r in self.records if k == "release_decision"]
        return rel[-1] if rel else None


def _run(monkeypatch=None, *, audit=None, fingerprint_fails=False, cert_fails=False):
    src = _src()
    if fingerprint_fails:
        def _boom():
            raise RuntimeError("no fingerprint")
        src.data_fingerprint = _boom
    if cert_fails:
        def _cboom(*a, **k):
            raise RuntimeError("cert build failed")
        monkeypatch.setattr(cert_mod, "build_certificate", _cboom)
    audit = audit or _CapturingAudit()
    r = code_encounter("e", _NOTE, "2026-03-14", source=src,
                       extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                       corroborate_llm=_sel, audit_repository=audit, billing_context=_CTX)
    return r, audit


def _has_gate(r, name):
    return any(g.name == name for g in r.gates)


# A valid fingerprint captured at IMPORT time, before any test monkeypatches the required-
# source declaration -- used to prove the validator resolves a broken declaration to
# "not certifiable" instead of raising out of `code_encounter`.
_UNSEALED_VALID = MockSource().data_fingerprint()


def test_normal_release_all_four_agree():
    r, audit = _run()
    rel = audit.last_release()
    assert r.certificate is not None
    assert rel is not None and rel["verdict"] == r.verdict.value           # audit == returned
    assert rel.get("certificate_sha256") == r.certificate["certificate_sha256"]  # binds cert


def test_fingerprint_failure_prevents_certification_and_agrees():
    r, audit = _run(fingerprint_fails=True)
    assert r.certificate is None                                           # not certifiable
    assert _has_gate(r, "data_fingerprint")
    assert audit.last_release()["verdict"] == r.verdict.value              # durable == returned


def test_certificate_failure_downgrades_and_agrees(monkeypatch):
    r, audit = _run(monkeypatch, cert_fails=True)
    assert r.certificate is None
    assert _has_gate(r, "release_evidence_persistence")
    rel = audit.last_release()
    assert rel["verdict"] == r.verdict.value                              # no AUTO_READY orphan
    assert "certificate_sha256" not in rel                               # no cert bound


def test_terminal_audit_write_failure_drops_certificate():
    from claude_coder.models import Verdict
    r, audit = _run(audit=_CapturingAudit(fail_kind="release_decision"))
    assert r.certificate is None                                          # no cert without record
    assert r.verdict is not Verdict.AUTO_READY                            # downgraded
    assert audit.last_release() is None                                  # nothing durably persisted
    assert _has_gate(r, "release_evidence_persistence")


def test_empty_or_partial_fingerprint_prevents_certification():
    """A source that swallows its own failure and returns {} / partial counts / a manifest
    that is not OK must NOT certify -- the outer 'is not None' check was insufficient."""
    import claude_coder.pipeline as pl
    assert pl._fingerprint_certifiable({}) is False
    assert pl._fingerprint_certifiable({"counts": {}}) is False
    assert pl._fingerprint_certifiable({"counts": {"icd10": 1, "cpt": 1, "hcpcs": 0}}) is False
    assert pl._fingerprint_certifiable(
        {"counts": {"icd10": 1, "cpt": 1, "hcpcs": 1}}) is False            # no manifest
    # F6-R5: a STATUS-ONLY manifest is no longer an identity and must NOT certify
    assert pl._fingerprint_certifiable(
        {"counts": {"icd10": 1, "cpt": 1, "hcpcs": 1},
         "source_manifest": {"status": "OK"}}) is False
    assert pl._fingerprint_certifiable(MockSource().data_fingerprint()) is True

    for bad in ({}, {"counts": {}}):
        src = _src()
        src.data_fingerprint = lambda b=bad: b
        r = code_encounter("e", _NOTE, "2026-03-14", source=src,
                           extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=_CapturingAudit(),
                           billing_context=_CTX)
        assert r.certificate is None                                       # not certified
        assert _has_gate(r, "data_fingerprint")


# ---- Codex F6-R5 (round 3): the fingerprint must identify the exact authoritative BYTES ----
def _valid_fp():
    import copy
    return copy.deepcopy(MockSource().data_fingerprint())


def _reseal(fp):
    """Recompute BOTH digests so a mutated fingerprint is fully SELF-CONSISTENT.

    Without this, every rejection below would only prove that the digest check fired --
    not that the required-source-set / role / release-metadata checks did. The reviewed
    counterexamples were all self-consistent objects. (Codex F6-R5.)"""
    import claude_coder.capability as cap
    man = fp["source_manifest"]
    man["manifest_sha256"] = cap.manifest_digest(man["sources"])
    fp["fingerprint_sha256"] = cap.fingerprint_digest(fp["counts"], man)
    return fp


def test_same_count_byte_change_invalidates_the_release_fingerprint(tmp_path):
    """The core defect: two materially different authoritative files with IDENTICAL row
    counts must not share a certifiable identity. Mutating the bytes of a required source
    while holding every count constant must change the fingerprint."""
    import json
    import claude_coder.pipeline as pl
    from app.release.source_manifest import sha256_file

    src_file = tmp_path / "authoritative.json"

    def _fingerprint():
        # the COMPLETE required-source set, with one required source content-addressed
        # against real bytes exactly as production records it; counts held CONSTANT
        fp = _valid_fp()
        stat = src_file.stat()
        target = fp["source_manifest"]["sources"][0]
        target["path"] = str(src_file)
        target["bytes"] = stat.st_size
        target["sha256"] = sha256_file(src_file)
        return _reseal(fp)

    src_file.write_text(json.dumps([{"code": "AAA", "description": "first edition"}]))
    first = _fingerprint()
    assert pl._fingerprint_certifiable(first) is True
    # SAME number of rows, DIFFERENT bytes
    src_file.write_text(json.dumps([{"code": "AAA", "description": "SECOND edition"}]))
    second = _fingerprint()
    assert pl._fingerprint_certifiable(second) is True
    assert first["counts"] == second["counts"]                             # cardinality equal
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]     # identity changed
    assert (first["source_manifest"]["manifest_sha256"]
            != second["source_manifest"]["manifest_sha256"])


def test_incomplete_or_tampered_manifest_is_not_certifiable():
    import claude_coder.pipeline as pl
    assert pl._fingerprint_certifiable(_valid_fp()) is True

    def _mutated(**changes):
        fp = _valid_fp()
        for key, value in changes.items():
            if key.startswith("manifest_"):
                fp["source_manifest"][key[len("manifest_"):]] = value
            else:
                fp[key] = value
        return fp

    # missing / malformed digests and required-source identity
    assert pl._fingerprint_certifiable(_mutated(fingerprint_sha256="")) is False
    assert pl._fingerprint_certifiable(_mutated(fingerprint_sha256="deadbeef")) is False
    assert pl._fingerprint_certifiable(_mutated(manifest_manifest_version="")) is False
    assert pl._fingerprint_certifiable(_mutated(manifest_missing_required=["cpt"])) is False
    assert pl._fingerprint_certifiable(
        _mutated(manifest_integrity_errors=["digest unavailable"])) is False
    assert pl._fingerprint_certifiable(_mutated(manifest_sources=[])) is False

    for mutation in ({"sha256": None}, {"sha256": "not-a-digest"}, {"present": False},
                     {"bytes": 0}, {"required": False}):
        fp = _valid_fp()
        fp["source_manifest"]["sources"][0].update(mutation)
        assert pl._fingerprint_certifiable(fp) is False, mutation

    # a manifest whose recorded digest no longer covers its sources (hand-edited)
    fp = _valid_fp()
    fp["source_manifest"]["sources"][0]["bytes"] = 999
    assert pl._fingerprint_certifiable(fp) is False


def test_real_manifest_is_content_addressed_and_version_aware():
    """The production manifest itself must carry per-source digests and release windows."""
    from claude_coder.capability import build_manifest, manifest_digest
    man = build_manifest()
    assert man["manifest_version"] and man["manifest_sha256"] == manifest_digest(man["sources"])
    for s in man["sources"]:
        assert "sha256" in s and "bytes" in s and "release" in s
        if s["present"]:
            assert str(s["sha256"]).startswith("sha256:") and len(str(s["sha256"])) == 71
            assert s["bytes"] > 0


def test_fingerprint_validator_is_total_and_fails_closed():
    """Post-fix review: the validator runs OUTSIDE the fingerprint try/except, so a raise
    would escape `code_encounter` and lose the structured hold. Every hostile shape must
    resolve to False instead."""
    import claude_coder.pipeline as pl
    for hostile in (None, [], "manifest", 7, {"counts": "not-a-dict"},
                    {"counts": {"icd10": 1, "cpt": 1, "hcpcs": 1},
                     "fingerprint_sha256": "sha256:" + "a" * 64, "source_manifest": []},
                    {"counts": {"icd10": 1, "cpt": 1, "hcpcs": 1},
                     "fingerprint_sha256": "sha256:" + "a" * 64,
                     "source_manifest": {"status": "OK", "manifest_version": "v",
                                         "missing_required": [], "integrity_errors": [],
                                         "sources": [{"required": True, "present": True,
                                                      "bytes": "not-an-int"}]}}):
        assert pl._fingerprint_certifiable(hostile) is False


def test_manifest_is_deterministic_so_certificates_stay_reproducible():
    """Post-fix review: the manifest must be a pure function of content. A per-call build
    timestamp inside the fingerprint would silently make release certificates irreproducible,
    because the certificate carries no other clock."""
    from claude_coder.capability import build_manifest
    first, second = build_manifest(), build_manifest()
    assert first == second
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert "generated_at" not in first


def test_source_drift_from_a_reviewed_lock_is_an_integrity_error(monkeypatch, tmp_path):
    """A present source whose bytes no longer match its reviewed lock must BLOCK, not pass."""
    import json
    import claude_coder.capability as cap
    from claude_coder.gates import source_manifest_gate
    from claude_coder.models import CodingResult, Outcome
    drifted = tmp_path / "locked_source.json"
    drifted.write_text('{"rows": 1}')
    monkeypatch.setattr(cap, "_source_locks", lambda: {
        str(drifted.resolve()): {"lock": "locked_source.lock.json",
                                 "output_sha256": "0" * 64, "effective_from": "2026-07-01",
                                 "source": "test"}})
    record, errors = cap._probe(drifted, True, "test", {}, cap._source_locks())
    assert record["present"] and errors and "reviewed lock" in errors[0]
    monkeypatch.setattr(cap, "build_manifest", lambda: {
        "missing_required": [], "integrity_errors": errors, "degraded_optional": []})
    g = source_manifest_gate(CodingResult("e", "2026-03-14", lines=[]))
    assert g.outcome is Outcome.BLOCKED


# ---- Codex F6-R5 (round 4): a PARTIAL required-source set and an unverified aggregate ----
# digest were both certifiable. The validator now compares the manifest against the COMPLETE
# declared required set (identities + roles + release-metadata expectation) and RECOMPUTES
# the aggregate fingerprint. Every case below is fully self-consistent (`_reseal`) so it
# isolates the check under test rather than tripping the digest comparison.


def test_required_release_source_declaration_agrees_with_the_authority():
    """The declaration is a statement ABOUT the registry, so it may not invent identities,
    and 'no release metadata' must be a REVIEWED exception with a stated reason -- never a
    blank field nobody looked at."""
    from app.release import source_manifest as sm
    spec = sm.required_release_sources()
    assert spec
    registered = sm.authoritative_paths()
    for source_id, entry in spec.items():
        assert source_id in registered, source_id            # registered, not invented
        assert str(entry["role"]).strip()
        provides = source_id in sm.RELEASE_METADATA_SOURCES
        assert entry["release_metadata_required"] is provides
        # silence is not an exemption: a source with no published window states why
        assert bool(str(entry["release_metadata_exemption"]).strip()) is not provides


def test_a_declaration_that_disagrees_with_the_authority_fails_loudly(monkeypatch):
    """Failure path of the declaration itself: it must raise, not silently return a
    partial set (which is exactly the defect this whole check exists to prevent)."""
    from app.release import source_manifest as sm
    metadata_bearing = sorted(sm.RELEASE_METADATA_SOURCES & set(sm.authoritative_paths()))
    assert metadata_bearing
    broken = [
        {"a_source_that_is_not_registered": {"role": "r"}},              # not registered
        {metadata_bearing[0]: {"role": "r",
                               "release_metadata_exemption": "stale"}},  # stale exemption
        {"validator_rules": {"role": "r"}},                              # no window, no reason
    ]
    for declaration in broken:
        monkeypatch.setattr(sm, "_REQUIRED_RELEASE_SOURCES", declaration)
        with pytest.raises(RuntimeError):
            sm.required_release_sources()


def test_a_broken_declaration_holds_the_release_instead_of_certifying(monkeypatch):
    """Boundary: the declaration raising must surface as a fail-closed HOLD everywhere it
    is consumed -- the capability gate, the fingerprint producer, and the pipeline -- never
    as an empty required set that certifies."""
    from app.release import source_manifest as sm
    from claude_coder.gates import source_manifest_gate
    from claude_coder.models import CodingResult, Outcome
    import claude_coder.pipeline as pl
    monkeypatch.setattr(sm, "_REQUIRED_RELEASE_SOURCES",
                        {"a_source_that_is_not_registered": {"role": "r"}})
    g = source_manifest_gate(CodingResult("e", "2026-03-14", lines=[]))
    assert g.outcome is Outcome.ERROR and g.retryable
    with pytest.raises(RuntimeError):
        MockSource().data_fingerprint()
    assert pl._fingerprint_certifiable(_UNSEALED_VALID) is False   # validator stays total
    r, _audit = _run()
    # The pipeline holds EARLIER than it used to, for the same broken declaration: the
    # modifier engine resolves its authority through that declaration, and (round 5, phase
    # 4) it no longer swallows an unresolvable one into an empty modifier table -- which is
    # the only reason the run previously got as far as the fingerprint check. Same
    # fail-closed conclusion, named at the boundary that actually detected it.
    assert r.certificate is None
    assert _has_gate(r, "authoritative_data_integrity")


def test_omitting_any_required_source_is_not_certifiable():
    """The reviewed counterexample: a self-consistent manifest with `missing_required=[]`,
    nonzero counts and a valid digest, but a real claim-affecting source simply absent."""
    import claude_coder.pipeline as pl
    from app.release.source_manifest import required_release_sources
    expected = required_release_sources()
    assert len(expected) > 1
    for source_id in expected:
        fp = _valid_fp()
        fp["source_manifest"]["sources"] = [
            s for s in fp["source_manifest"]["sources"] if s["source_id"] != source_id]
        assert fp["source_manifest"]["missing_required"] == []
        assert pl._fingerprint_certifiable(_reseal(fp)) is False, source_id
    # ...and the exact reported shape: ONE synthetic required source, every real
    # claim-affecting source omitted.
    fp = _valid_fp()
    fp["source_manifest"]["sources"] = [dict(fp["source_manifest"]["sources"][0],
                                             source="synthetic", source_id="synthetic",
                                             role="synthetic test source")]
    assert pl._fingerprint_certifiable(_reseal(fp)) is False


def test_duplicate_required_source_identity_is_not_certifiable():
    """Two records claiming one identity makes 'which bytes' ambiguous; last-one-wins would
    let a second record silently override the first."""
    import claude_coder.pipeline as pl
    fp = _valid_fp()
    sources = fp["source_manifest"]["sources"]
    sources.append(dict(sources[0], bytes=sources[0]["bytes"] + 1,
                        sha256="sha256:" + "b" * 64))
    assert pl._fingerprint_certifiable(_reseal(fp)) is False


def test_an_unregistered_required_source_is_not_certifiable():
    """An extra source asserting `required` is not in the declaration and must not pass;
    an extra OPTIONAL source is recorded detail and stays certifiable."""
    import claude_coder.pipeline as pl
    fp = _valid_fp()
    sources = fp["source_manifest"]["sources"]
    sources.append(dict(sources[0], source="rogue", source_id="rogue_source",
                        role="unreviewed source"))
    assert pl._fingerprint_certifiable(_reseal(fp)) is False

    ok = _valid_fp()
    ok_sources = ok["source_manifest"]["sources"]
    ok_sources.append(dict(ok_sources[0], source="aid", source_id="optional_aid",
                           required=False, role="recall aid"))
    assert pl._fingerprint_certifiable(_reseal(ok)) is True


def test_a_required_source_whose_role_disagrees_is_not_certifiable():
    """Identity alone is not enough: the manifest must agree about what each required
    source IS, so one source cannot be substituted for another's role."""
    import claude_coder.pipeline as pl
    from app.release.source_manifest import required_release_sources
    for source_id in required_release_sources():
        for bad_role in ("", "some other role"):
            fp = _valid_fp()
            for s in fp["source_manifest"]["sources"]:
                if s["source_id"] == source_id:
                    s["role"] = bad_role
            assert pl._fingerprint_certifiable(_reseal(fp)) is False, (source_id, bad_role)


def test_blank_release_metadata_where_the_authority_publishes_it_is_not_certifiable():
    """Where the authority publishes an effective/edition window it must be present. An
    ingest timestamp is not an edition, so `version` alone does not satisfy it. Where the
    authority publishes none, the exemption is explicit in the declaration."""
    import claude_coder.pipeline as pl
    from app.release.source_manifest import required_release_sources
    expected = required_release_sources()
    bearing = [sid for sid, s in expected.items() if s["release_metadata_required"]]
    exempt = [sid for sid, s in expected.items() if not s["release_metadata_required"]]
    assert bearing
    for source_id in bearing:
        for blank in ({}, None, "2026-01-01",
                      {"effective_from": "", "release_effective_from": "",
                       "version": "2026-08-05/2026-08-05T07:08:45"}):
            fp = _valid_fp()
            for s in fp["source_manifest"]["sources"]:
                if s["source_id"] == source_id:
                    s["release"] = blank
            assert pl._fingerprint_certifiable(_reseal(fp)) is False, (source_id, blank)
    for source_id in exempt:
        assert str(expected[source_id]["release_metadata_exemption"]).strip()
        fp = _valid_fp()
        for s in fp["source_manifest"]["sources"]:
            if s["source_id"] == source_id:
                s["release"] = {}
        assert pl._fingerprint_certifiable(_reseal(fp)) is True, source_id


def test_an_arbitrary_aggregate_fingerprint_is_not_certifiable():
    """The declared aggregate digest must BE the digest of the counts + manifest it claims
    to identify. Shape-matching accepted all-zero, all-`f`, and merely plausible hex."""
    import hashlib as _h
    import claude_coder.pipeline as pl
    good = _valid_fp()
    assert pl._fingerprint_certifiable(good) is True
    real_hex = good["fingerprint_sha256"].split(":", 1)[1]
    for bogus in ("sha256:" + "0" * 64,
                  "sha256:" + "f" * 64,
                  "sha256:" + _h.sha256(b"plausible but wrong").hexdigest(),
                  "sha256:" + real_hex[::-1]):
        fp = _valid_fp()
        fp["fingerprint_sha256"] = bogus
        assert pl._fingerprint_certifiable(fp) is False, bogus
    # the aggregate must actually COVER the counts: change them, keep the digest
    fp = _valid_fp()
    fp["counts"] = dict(fp["counts"], cpt=fp["counts"]["cpt"] + 1)
    assert pl._fingerprint_certifiable(fp) is False


def test_a_reordered_or_partially_copied_manifest_is_not_certifiable():
    import claude_coder.capability as cap
    import claude_coder.pipeline as pl
    # reordered, digests copied from the original -> the recorded digests no longer
    # describe the recorded manifest
    fp = _valid_fp()
    fp["source_manifest"]["sources"].reverse()
    assert pl._fingerprint_certifiable(fp) is False
    # reordered with the manifest digest recomputed but the AGGREGATE copied over
    fp = _valid_fp()
    original = fp["fingerprint_sha256"]
    fp["source_manifest"]["sources"].reverse()
    fp["source_manifest"]["manifest_sha256"] = cap.manifest_digest(
        fp["source_manifest"]["sources"])
    fp["fingerprint_sha256"] = original
    assert pl._fingerprint_certifiable(fp) is False
    # a fully recomputed reorder is the SAME content and stays certifiable -- order is
    # not the control; the digests and the complete required set are
    assert pl._fingerprint_certifiable(_reseal(fp)) is True
    # partially copied: a strict subset of the sources with both digests recomputed
    fp = _valid_fp()
    fp["source_manifest"]["sources"] = fp["source_manifest"]["sources"][:2]
    assert pl._fingerprint_certifiable(_reseal(fp)) is False


def test_a_manifest_built_against_another_required_definition_is_not_certifiable():
    """The manifest records WHICH required-source definition it was built against, so a
    certificate produced under a different (older/other) set is identifiable rather than
    silently comparable."""
    import claude_coder.pipeline as pl
    for value in (None, "", "release-required-sources-v0", "capability-manifest-v2"):
        fp = _valid_fp()
        if value is None:
            fp["source_manifest"].pop("required_sources_schema")
        else:
            fp["source_manifest"]["required_sources_schema"] = value
        assert pl._fingerprint_certifiable(_reseal(fp)) is False, value


def test_the_real_capability_manifest_declares_the_complete_required_set():
    """The producer side of the same invariant: the manifest production actually EMITS the
    declared required set with matching roles and a populated window where one is published
    -- otherwise the validator would reject every legitimate release."""
    from app.release.source_manifest import (
        REQUIRED_SOURCE_SCHEMA_VERSION, release_window_populated,
        required_release_sources)
    from claude_coder.capability import build_manifest
    man = build_manifest()
    assert man["required_sources_schema"] == REQUIRED_SOURCE_SCHEMA_VERSION
    expected = required_release_sources()
    declared = {s["source_id"]: s for s in man["sources"] if s["required"]}
    assert set(declared) == set(expected)
    for source_id, spec in expected.items():
        record = declared[source_id]
        assert record["role"] == spec["role"]
        if record["present"] and spec["release_metadata_required"]:
            assert release_window_populated(record["release"]), source_id


def test_the_release_record_declares_the_terminal_head_trust_boundary():
    """Codex F6-R4-A: the durable release record must state what chain-of-custody guarantee
    was ACTUALLY in force -- which terminal-head anchor backend, and whether it is a real
    external trust boundary. A repository that cannot answer is described as unanchored, never
    omitted: silence would read as 'fine' when the honest answer is 'not externally anchored'.
    """
    _r, audit = _run()
    anchor = audit.last_release()["terminal_head_anchor"]
    assert anchor["configured"] is False                     # this double has no anchor
    assert anchor["external_trust_boundary"] is False        # ...and never claims one


def test_the_release_record_reports_a_configured_anchor(tmp_path):
    """...and when an anchor IS configured, the record names it and its sequence, so an
    artifact can be checked against the anchor store after the fact."""
    from claude_coder.checkpoint import LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    repo = SqliteAuditRepository(tmp_path / "rel.db",
                                 checkpoint_anchor=LocalFileCheckpointAnchor(tmp_path / "a"))
    r, _audit = _run(audit=repo)
    assert r.encounter_id == "e"
    release = [rec for rec in repo.records("e") if rec["kind"] == "release_decision"]
    anchor = release[-1]["record"]["terminal_head_anchor"]
    assert anchor["backend"] == "local-file"
    assert anchor["configured"] is True
    assert anchor["external_trust_boundary"] is False        # reference backend, not a boundary
    assert anchor["anchored_seq"] == anchor["journal_seq"]
    assert repo.verify_chain("e") == []


# ---- Cross-finding integration: F6-R3 + F6-R4-A + F6-R5 compose on ONE release ------------
# The three remediations were implemented in separate commits but land on the SAME two
# artifacts -- the release certificate and the terminal `release_decision` audit record --
# and each was covered only by its own isolated suite. Passing three isolated suites does
# not establish that the three provenance facts coexist on one real release: a later change
# could bind the necessity support and quietly drop the fingerprint identity, or record an
# anchor while certifying against data it does not name, and every existing test would stay
# green. These two tests are the composition check.
def _auto_ready_release(tmp_path, *, encounter_id="enc-compose"):
    """One full AUTO_READY encounter through the REAL durable store (hash-chained SQLite +
    sealed witness journal) with a terminal-head checkpoint anchor configured."""
    from claude_coder.checkpoint import LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    from tests.test_claude_coder import NOTE, _arbitrate_stub, _extract_stub, _source
    repo = SqliteAuditRepository(
        tmp_path / "prov.db",
        checkpoint_anchor=LocalFileCheckpointAnchor(tmp_path / "anchor"))
    result = code_encounter(
        encounter_id, NOTE, "2026-03-14", source=_source(),
        extract_llm=_extract_stub, arbitrate_llm=_arbitrate_stub,
        audit_repository=repo,
        billing_context={"billing_entity_id": "actor-1",
                         "participants": [{"id": "actor-1", "type": "person",
                                           "roles": ["performer"]}]})
    return result, repo, NOTE


def test_necessity_binding_data_identity_and_terminal_anchor_coexist_on_one_release(tmp_path):
    """All THREE remediations are present, correct and mutually consistent on a single
    AUTO_READY release -- not merely green in three separate suites.

      * F6-R5  the certificate's bound data identity accounts for the COMPLETE required
               source set and its two digests are the RECOMPUTED ones, not digest-shaped;
      * F6-R3  the SAME certificate answers WHY each released service was necessary (the
               claim-line diagnosis pointer plus the accepted relation's provenance);
      * F6-R4-A the durable terminal release record states which terminal-head anchor was
               actually in force, and does not overstate it;
    and the returned verdict, the certificate and the last durable record all agree over an
    intact hash-chained, sealed, anchored store.
    """
    import claude_coder.capability as cap
    import claude_coder.pipeline as pl
    from app.release.source_manifest import (REQUIRED_SOURCE_SCHEMA_VERSION,
                                             required_release_sources)
    from claude_coder.models import Verdict

    r, repo, note = _auto_ready_release(tmp_path)
    assert r.verdict is Verdict.AUTO_READY, r.notes
    cert = r.certificate
    assert cert is not None

    # --- F6-R5: the identity of the data this claim was coded against -----------------
    fingerprint = cert["source_identity"]["data"]
    assert pl._fingerprint_certifiable(fingerprint) is True
    manifest = fingerprint["source_manifest"]
    assert manifest["required_sources_schema"] == REQUIRED_SOURCE_SCHEMA_VERSION
    assert ({s["source_id"] for s in manifest["sources"] if s["required"]}
            == set(required_release_sources()))
    # recomputed, not shape-matched: both digests must BE digests of what they describe
    assert manifest["manifest_sha256"] == cap.manifest_digest(manifest["sources"])
    assert fingerprint["fingerprint_sha256"] == cap.fingerprint_digest(
        fingerprint["counts"], manifest)

    # --- F6-R3: why each released service was medically necessary ---------------------
    procedures = {ln.fact.fact_id for ln in r.procedure_lines}
    assert procedures
    bound = {b["procedure_event_id"] for b in cert["necessity_support"]}
    assert bound == procedures                 # every released procedure is justified
    for binding in cert["necessity_support"]:
        assert binding["supports"]
        for support in binding["supports"]:
            assert support["diagnosis_event_id"] and support["diagnosis_code"]
            # the status was written by the deterministic provenance layer, never by the
            # extraction model, and the spans that proved it are the edge's own
            # ... and it GROUNDS the edge in the record: agreement between extraction runs is
            # recorded on its own axis and can never appear here. (Codex F6-R3, round 5.)
            assert support["reconciliation_status"] in provenance.GROUNDED_RECONCILIATION_STATUSES
            assert support["corroboration_status"] in provenance.CORROBORATION_STATUSES
            assert support["assertion_origins"] and all(support["assertion_origins"])
            assert set(support["reconciliation_evidence"]) <= set(support["evidence_span_ids"])

    # --- F6-R4-A: what the chain-of-custody guarantee ACTUALLY was --------------------
    releases = [rec for rec in repo.records(r.encounter_id)
                if rec["kind"] == "release_decision"]
    assert len(releases) == 1                  # exactly one terminal decision
    record = releases[-1]["record"]
    anchor = record["terminal_head_anchor"]
    assert anchor["configured"] is True and anchor["backend"] == "local-file"
    assert anchor["external_trust_boundary"] is False    # honest: reference backend only
    assert anchor["problems"] == []
    assert anchor["anchored_seq"] == anchor["journal_seq"]

    # --- the four artifacts agree, and the store verifies end to end ------------------
    assert record["verdict"] == r.verdict.value
    assert record["certificate_sha256"] == cert["certificate_sha256"]
    assert repo.verify_chain(r.encounter_id) == []

    # --- neither remediation displaced the other inside the ONE tamper-evident hash ----
    # Both the necessity binding (F6-R3) and the data identity (F6-R5) must be covered by
    # `certificate_sha256`; if either had been bound outside it, the other's regression
    # suite would still pass while the artifact silently stopped being tamper-evident.
    import copy

    def _rehash(payload):
        payload.pop("certificate_sha256")
        return cert_mod._sha(cert_mod._canonical(payload))

    # control: an UNCHANGED payload must reproduce the recorded hash, otherwise the
    # comparisons below would "pass" for the wrong reason
    assert _rehash(copy.deepcopy(cert)) == cert["certificate_sha256"]
    for mutate in (
            lambda p: p["necessity_support"][0]["supports"][0].update({"diagnosis_code": "X"}),
            lambda p: p["source_identity"]["data"].update(
                {"fingerprint_sha256": "sha256:" + "0" * 64}),
    ):
        payload = copy.deepcopy(cert)
        mutate(payload)
        assert _rehash(payload) != cert["certificate_sha256"]


def test_a_consistently_truncated_store_holds_the_whole_encounter(tmp_path):
    """The reviewer's exact attack, driven through `code_encounter` rather than through
    `append` alone: with an anchor configured, removing the terminal journal entry AND the
    terminal durable row together must stop the NEXT encounter at the pipeline boundary.

    The append-level regressions prove the repository raises; only this proves the raise is
    actually converted into a fail-closed pipeline outcome instead of being swallowed on the
    way out -- and that no certificate is produced for a claim written onto a rewritten
    history. (Codex F6-R4-A.)
    """
    from claude_coder.models import Destination, Verdict
    r, repo, _note = _auto_ready_release(tmp_path, encounter_id="enc-1")
    assert r.verdict is Verdict.AUTO_READY and repo.verify_chain() == []

    # drop the final witness line and the final audit row together
    import sqlite3
    witness = repo._witness_path()
    lines = [ln for ln in witness.read_text().splitlines() if ln.strip()]
    witness.write_text("\n".join(lines[:-1]) + "\n")
    conn = sqlite3.connect(str(repo.db_path))
    conn.execute("DROP TRIGGER audit_no_delete")
    conn.execute("DELETE FROM audit_log WHERE seq=(SELECT MAX(seq) FROM audit_log)")
    conn.commit()
    conn.close()
    assert any("SHORTER than its anchored checkpoint" in p for p in repo.verify_chain())

    held, _repo2, _note2 = _auto_ready_release(tmp_path, encounter_id="enc-2")
    assert held.verdict is not Verdict.AUTO_READY
    assert held.destination is Destination.SYSTEM_HOLD
    assert held.certificate is None
    assert not held.billable_lines


# ---- Codex F6-R5 (round 5): the required set must be DERIVED from the runtime graph ----
# Round 4 proved the manifest against a DECLARED set, but the declaration itself was
# under-derived: `coverage_policy` was read at decision time while marked optional, and the
# two claim-affecting control configs were not in the fingerprint at all. The cases below
# pin (a) the total disposition of every registered source, (b) that silence is now an
# error, (c) that each newly required identity's absence or byte change is provably
# certification-invalidating, and (d) that the remaining OPTIONAL sources really cannot
# change a released claim.

# The identities this round added after tracing what the claim path actually reads. Named
# explicitly so a future edit that quietly drops one fails here rather than in production.
_ROUND5_REQUIRED = (
    "coverage_policy",              # data_access._coverage_map -> governed / qualifying dx
    "necessity_relation_control",   # gates.load_necessity_control
    "relation_evidence_grammar",    # provenance.load_relation_grammar
    "pfs_indicators",               # data_access._pfs -> global period / bilateral
    "modifier_definitions",         # modifiers.load_modifier_defs
    "instructional_notes",          # data_access._excludes1_map -> Excludes1 gate
    "validator_rules",              # deterministic validation rule pack
    "snomed_root_concepts",         # validator confidence cap for root-level concepts
    "terminology_registry",         # governed interpretation source
)


def test_every_registered_source_is_required_or_a_reviewed_optional():
    """Totality: a registered source may not simply be unmentioned. That silence is what
    let `coverage_policy` be treated as optional while the necessity gate read it."""
    from app.release import source_manifest as sm
    required = sm.required_release_sources()
    optional = sm.optional_release_sources()
    assert set(sm._AUTHORITATIVE) <= (set(required) | set(optional))
    assert not (set(required) & set(optional))
    for source_id in _ROUND5_REQUIRED:
        assert source_id in required, source_id
    for source_id, spec in optional.items():
        # "optional" is a reviewed claim with a stated reason, never a blank
        assert str(spec["absence_justification"]).strip(), source_id
        assert str(spec["role"]).strip(), source_id


def test_a_registered_source_with_no_disposition_fails_loudly(monkeypatch):
    """Failure path of the totality check: adding a source without deciding what it is must
    raise in every consumer -- not default to optional, which is the original defect."""
    from app.release import source_manifest as sm
    from claude_coder.capability import build_manifest
    from claude_coder.gates import source_manifest_gate
    from claude_coder.models import CodingResult, Outcome
    import claude_coder.pipeline as pl
    registry = dict(sm._AUTHORITATIVE)
    registry["a_newly_read_source_nobody_dispositioned"] = sm.config.CPT_FILE
    monkeypatch.setattr(sm, "_AUTHORITATIVE", registry)
    for fn in (sm.required_release_sources, sm.optional_release_sources, build_manifest):
        with pytest.raises(RuntimeError):
            fn()
    with pytest.raises(RuntimeError):
        sm.declared_source_path("cpt_codes")
    g = source_manifest_gate(CodingResult("e", "2026-03-14", lines=[]))
    assert g.outcome is Outcome.ERROR and g.retryable
    assert pl._fingerprint_certifiable(_UNSEALED_VALID) is False


def test_declared_source_path_rejects_an_undeclared_identity():
    """The choke point decision-time readers resolve through: a file that is not a declared
    source cannot be read as if it were authoritative."""
    from app.release import source_manifest as sm
    with pytest.raises(RuntimeError):
        sm.declared_source_path("a_file_nobody_declared")
    for source_id in _ROUND5_REQUIRED:
        assert sm.declared_source_path(source_id).name, source_id
    # the control configs ship IN the repository, so they must resolve to real bytes on any
    # checkout -- including one without the generated authoritative data files
    for source_id in ("necessity_relation_control", "relation_evidence_grammar"):
        assert sm.declared_source_path(source_id).exists(), source_id


def test_removing_any_newly_required_source_blocks_the_real_capability_manifest(monkeypatch,
                                                                               tmp_path):
    """End of the chain, against the REAL manifest builder (not a hand-built dict): a
    required source that is absent produces BLOCKED / missing_required, which the
    fingerprint validator then refuses to certify."""
    from app.release import source_manifest as sm
    import claude_coder.capability as cap
    import claude_coder.pipeline as pl
    baseline = set(cap.build_manifest()["missing_required"])
    for source_id in _ROUND5_REQUIRED:
        registry = dict(sm._AUTHORITATIVE)
        registry[source_id] = tmp_path / "absent.json"
        monkeypatch.setattr(sm, "_AUTHORITATIVE", registry)
        manifest = cap.build_manifest()
        assert set(manifest["missing_required"]) - baseline, source_id
        assert manifest["status"] == "BLOCKED", source_id
        counts = {"icd10": 1, "cpt": 1, "hcpcs": 1}
        fp = {"counts": counts, "source_manifest": manifest,
              "fingerprint_sha256": cap.fingerprint_digest(counts, manifest)}
        assert pl._fingerprint_certifiable(fp) is False, source_id
        # ...and the same REAL manifest, carried through the whole pipeline, holds the
        # release: not certifiable at the unit level is worth nothing if `code_encounter`
        # still emits a certificate.
        src = _src()
        src.data_fingerprint = lambda f=fp: f
        r = code_encounter("e", _NOTE, "2026-03-14", source=src,
                           extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=_CapturingAudit(),
                           billing_context=_CTX)
        assert r.certificate is None, source_id
        # WHICH fail-closed boundary catches it is asserted exactly, not loosely:
        # `modifier_definitions` is read during claim ASSEMBLY, so (round 5, phase 4) the
        # pipeline's authoritative-data boundary detects it before the fingerprint check
        # ever runs; every other identity still surfaces as the fingerprint hold.
        assert _has_gate(r, "authoritative_data_integrity"
                         if source_id == "modifier_definitions"
                         else "data_fingerprint"), source_id
        monkeypatch.undo()


def test_control_config_bytes_are_bound_into_the_release_fingerprint(monkeypatch, tmp_path):
    """The specific round-5 hole: a control config's VERSION STRING appeared in audit data
    while its CONTENT was outside the certified identity. Changing the bytes -- without
    touching the declared version -- must change the fingerprint."""
    from app.release import source_manifest as sm
    import claude_coder.capability as cap
    for source_id in ("necessity_relation_control", "relation_evidence_grammar"):
        original = sm.declared_source_path(source_id).read_text()
        copy_path = tmp_path / f"{source_id}.json"
        copy_path.write_text(original)
        registry = dict(sm._AUTHORITATIVE)
        registry[source_id] = copy_path
        monkeypatch.setattr(sm, "_AUTHORITATIVE", registry)

        def _identity():
            manifest = cap.build_manifest()
            assert manifest["status"] == "OK"
            return cap.fingerprint_digest({"icd10": 1, "cpt": 1, "hcpcs": 1}, manifest)

        before = _identity()
        # same declared control version, different bytes
        payload = json.loads(original)
        payload["description"] = str(payload.get("description", "")) + " (edited)"
        copy_path.write_text(json.dumps(payload))
        after = _identity()
        assert json.loads(copy_path.read_text())["version"] == payload["version"]
        assert before != after, source_id
        monkeypatch.undo()


def test_a_manifest_omitting_a_newly_required_source_holds_through_code_encounter():
    """Not just the unit validator: an omitted required source must reach the caller as a
    fail-closed release outcome with no certificate."""
    for source_id in _ROUND5_REQUIRED:
        fp = _valid_fp()
        fp["source_manifest"]["sources"] = [
            s for s in fp["source_manifest"]["sources"] if s["source_id"] != source_id]
        src = _src()
        src.data_fingerprint = lambda f=_reseal(fp): f
        r = code_encounter("e", _NOTE, "2026-03-14", source=src,
                           extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=_CapturingAudit(),
                           billing_context=_CTX)
        assert r.certificate is None, source_id
        assert _has_gate(r, "data_fingerprint"), source_id


def test_unreadable_coverage_policy_holds_instead_of_reporting_ungoverned(monkeypatch,
                                                                          tmp_path):
    """The reviewed impact statement: missing/invalid coverage data silently became an empty
    map, and an empty map reads as 'governed by no policy' -- the LESS restrictive path. It
    must raise, and the necessity gate must convert that into a hold."""
    from app.release import source_manifest as sm
    from claude_coder.data_access import AuthoritativeSource, CoverageDataUnavailable
    corrupt = tmp_path / "coverage.json"
    for content in ("", "{not json", "{}", '{"lcd": [], "article": []}'):
        corrupt.write_text(content)
        for path in (corrupt, tmp_path / "absent.json"):
            registry = dict(sm._AUTHORITATIVE)
            registry["coverage_policy"] = path
            monkeypatch.setattr(sm, "_AUTHORITATIVE", registry)
            src = AuthoritativeSource()
            with pytest.raises(CoverageDataUnavailable):
                src.qualifying_dx_for("SYNTHETIC_PROC", "cpt")
            monkeypatch.undo()


def test_an_unavailable_coverage_authority_holds_the_necessity_gate():
    """Boundary: the raise above has to land as a HOLD in the gate that consumes it, not as
    an exception escaping the pipeline."""
    from claude_coder.data_access import CoverageDataUnavailable
    from claude_coder.gates import medical_necessity_gate
    from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, EvidenceSpan,
                                     FactKind, Outcome, ResolutionMethod, ResolvedLine)

    class _Unavailable:
        def qualifying_dx_for(self, code, system="cpt"):
            raise CoverageDataUnavailable("coverage authority unreadable")

    def _line(kind, code, system, fid):
        fact = ClinicalFact(kind=kind, description="x", fact_id=fid,
                            evidence=[EvidenceSpan("x", anchored=True, span_id="s1")])
        return ResolvedLine(fact=fact, chosen=CandidateCode(code, system, "d", 0.9),
                            method=ResolutionMethod.DETERMINISTIC)

    proc = _line(FactKind.PROCEDURE, "PROC_X", "cpt", "pf")
    dx = _line(FactKind.DIAGNOSIS, "DX_X", "icd10", "df")
    out = medical_necessity_gate(CodingResult("e", "2026-08-01", lines=[proc, dx]),
                                 _Unavailable())
    assert out.outcome is not Outcome.PASS


def test_an_absent_drug_dose_table_holds_rather_than_changing_billed_units():
    """The justification recorded for the OPTIONAL `hcpcs_drug_table` has to be true: with
    the table absent, a documented dose used to fall back to a COUNT (30 mg of a per-15-mg
    code billed as 1 unit instead of 2) -- an absent optional source changing the claim."""
    from claude_coder.gates import drug_units_gate
    from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, EvidenceSpan,
                                     FactKind, Outcome, ResolutionMethod, ResolvedLine)

    def _line(dose_text):
        fact = ClinicalFact(kind=FactKind.DRUG, description=dose_text, fact_id="df",
                            evidence=[EvidenceSpan(dose_text, anchored=True, span_id="s1")])
        return ResolvedLine(fact=fact, chosen=CandidateCode("DRUG_X", "hcpcs", "d", 0.9),
                            method=ResolutionMethod.DETERMINISTIC)

    dosed = CodingResult("e", "2026-08-01", lines=[_line("administered 30 mg")])
    undosed = CodingResult("e", "2026-08-01", lines=[_line("administered once")])
    absent = MockSource()                                   # no drug table loaded
    present = MockSource(drug_units={"DRUG_X": {"amount": 15, "unit": "mg"}})

    held = drug_units_gate(dosed, absent)
    assert held.outcome is Outcome.UNKNOWN and held.retryable   # holds, never a silent count
    assert drug_units_gate(dosed, present).outcome is Outcome.PASS
    # a drug line with NO documented dose is unaffected: units are the documented count
    assert drug_units_gate(undosed, absent).outcome is Outcome.NOT_APPLICABLE


#: The ONLY two modules allowed to name a source file: the declaration itself.  A path has
#: to be constructed somewhere; `app/core/config.py` is where, and `app/release/
#: source_manifest.py` is where each constructed path is bound to an identity and a
#: disposition.  Everything else RESOLVES an identity.  This list is deliberately tiny and
#: deliberately not a suppression list -- adding a module to it is a reviewable act.
_SOURCE_DECLARATION_MODULES = (("app", "core", "config.py"),
                               ("app", "release", "source_manifest.py"))


def test_no_decision_module_composes_an_authoritative_filename_literal():
    """Structural guard against the NEXT under-declaration: a claim-affecting module may not
    name an authoritative file itself. Composing `DATA_DIR / "x.json"` inline is exactly how
    `global_period.json`, `modifiers.json` and the two control configs came to be read at
    decision time while being certified by nobody. Paths come from the declaration.

    Round 6 (Codex F6-R5-A): this scanned only top-level `claude_coder/*.py`, while the
    DEPLOYED image runs `app/**` too -- the human-run 837P submission step resolves the
    payer through `app.compliance.payer_registry`, `app.compliance.datastore.store` owns
    the modifier-role and semantic-class vocabulary, and `app.release.scope_registry` owns
    what may be released without a human. Every one of those composed its own filename
    literal and so reached the manifest only through the incidental `data/codes/*.json`
    sweep, which cannot distinguish a source that is intentionally absent from one that
    silently disappeared. The scan now covers BOTH trees, recursively.
    """
    import ast
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parent.parent
    exempt = {repo.joinpath(*parts) for parts in _SOURCE_DECLARATION_MODULES}
    offenders = []
    for tree_root in (repo / "claude_coder", repo / "app"):
        for module in sorted(tree_root.rglob("*.py")):
            if module in exempt:
                continue
            tree = ast.parse(module.read_text())
            interpolated = {id(node) for parent in ast.walk(tree)
                            if isinstance(parent, ast.JoinedStr)
                            for node in ast.walk(parent)}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value.endswith(".json") and "*" not in node.value
                        and id(node) not in interpolated):
                    offenders.append(
                        f"{module.relative_to(repo)}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "authoritative filename literal(s) outside the release-source declaration: "
        + "; ".join(offenders))


def test_the_declaration_modules_are_the_only_exempted_ones():
    """Failure path of the guard above: its exemption list must stay two files that exist.
    A typo'd or stale entry would silently exempt nothing (harmless) -- but a THIRD entry
    added to quiet a failure is exactly the suppression this finding is about, so the count
    is pinned and the paths are proven real."""
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parent.parent
    assert len(_SOURCE_DECLARATION_MODULES) == 2
    for parts in _SOURCE_DECLARATION_MODULES:
        assert repo.joinpath(*parts).is_file(), parts


def test_every_declared_source_is_dispositioned_across_both_trees():
    """The registry half of the same guard: a filename literal is not the only way to read
    an undeclared file, so every identity the declaration carries must also be REQUIRED or
    reviewed-optional-with-a-justification -- including the app-side identities added in
    round 6, which the deployed image reads and the release must account for."""
    from app.release import source_manifest as sm
    required = sm.required_release_sources()
    optional = sm.optional_release_sources()
    assert not (set(required) & set(optional))
    assert set(sm._AUTHORITATIVE) <= set(required) | set(optional)
    # The two Codex named by file, present as first-class identities rather than as
    # incidental `codes/*.json` sweep entries.
    for source_id in ("coding_semantics", "payer_registry"):
        assert source_id in required, source_id
        assert str(required[source_id]["role"]).strip()
    manifest_ids = {record["source_id"] for record in sm.build_source_manifest()["records"]}
    for source_id in ("coding_semantics", "payer_registry"):
        assert source_id in manifest_ids
        assert f"codes/{sm._AUTHORITATIVE[source_id].name}" not in manifest_ids, (
            "the source is hashed twice, under its identity and under the sweep name")


# ---- Round 5, phase 4: PRESENT-BUT-CORRUPT required sources must hold, not degrade ------
# Phase 1 made the required set complete and made `coverage_policy` fail closed on a bad
# read. Its own report flagged the rest of the class: for every other required source,
# ABSENCE blocked (the capability manifest reports `missing_required`) but CORRUPTION did
# not -- a present file has bytes, the manifest hashes them, and the read path swallowed the
# parse failure into an empty table. For each of these sources an empty table is the
# PERMISSIVE answer, so corruption relaxed the claim rather than holding it.
#
# The corruption shapes below are one list, applied to every source, so a new required
# source cannot be signed off against a narrower set of failures than its siblings.
_CORRUPTIONS = (
    "",                       # truncated to nothing
    "{not json",              # malformed
    "[]",                     # valid JSON, wrong root type
    "{}",                     # parses, declares no table
    '{"codes": {}}',          # declares an EMPTY table -- the exact shape of the old {}
    '{"codes": []}',          # right key, wrong type
)


def _corrupted_source(monkeypatch, source_id, content, tmp_path):
    """Point a REQUIRED source's declared identity at bytes that are present but unusable.
    Present, not absent: absence is already blocked by the capability manifest, and the
    whole point of this class is that corruption never was."""
    from app.release import source_manifest as sm
    path = tmp_path / f"{source_id}.json"
    path.write_text(content)
    registry = dict(sm._AUTHORITATIVE)
    registry[source_id] = path
    monkeypatch.setattr(sm, "_AUTHORITATIVE", registry)
    return path


def test_corrupt_pfs_indicators_raise_instead_of_an_empty_indicator_table(monkeypatch,
                                                                         tmp_path):
    """`_pfs` swallowed every read failure into {}, which reports global period None and
    bilateral indicator None for EVERY code -- so `apply_global_package` bundles nothing and
    the laterality modifiers change. Both are the permissive direction."""
    from claude_coder.data_access import (AuthoritativeDataUnavailable, AuthoritativeSource,
                                          PfsIndicatorsUnavailable)
    for content in _CORRUPTIONS:
        _corrupted_source(monkeypatch, "pfs_indicators", content, tmp_path)
        src = AuthoritativeSource()
        for probe in (lambda: src.global_period("SYNTHETIC_PROC"),
                      lambda: src.bilat_indicator("SYNTHETIC_PROC"),
                      src.assert_claim_assembly_data_readable):
            with pytest.raises(PfsIndicatorsUnavailable):
                probe()
        assert issubclass(PfsIndicatorsUnavailable, AuthoritativeDataUnavailable)
        monkeypatch.undo()


def test_corrupt_modifier_definitions_raise_instead_of_a_bare_claim(monkeypatch, tmp_path):
    """The engine DISCOVERS every modifier it emits from these descriptions, so {} is
    indistinguishable from 'no modifier is warranted' and ships the claim bare."""
    from claude_coder.data_access import (AuthoritativeDataUnavailable,
                                          ModifierDefinitionsUnavailable)
    from claude_coder.modifiers import ModifierEngine, load_modifier_defs
    for content in _CORRUPTIONS + ('{"modifiers": {}}',):
        _corrupted_source(monkeypatch, "modifier_definitions", content, tmp_path)
        with pytest.raises(ModifierDefinitionsUnavailable):
            load_modifier_defs()
        with pytest.raises(ModifierDefinitionsUnavailable):
            ModifierEngine()                      # the default engine reads the authority
        # an engine handed REVIEWED definitions is unaffected -- it never reads the file
        assert ModifierEngine(defs={"MR": {"description": "Right side of the body"}})
        assert issubclass(ModifierDefinitionsUnavailable, AuthoritativeDataUnavailable)
        monkeypatch.undo()


def test_corrupt_instructional_notes_raise_instead_of_an_excludes1_clean_bill_of_health(
        monkeypatch, tmp_path):
    """The worst of the three: an empty note table does not merely lose a lookup, it makes
    every diagnosis pair look Excludes1-clean."""
    from claude_coder.data_access import (AuthoritativeDataUnavailable, AuthoritativeSource,
                                          InstructionalNotesUnavailable)
    shapes = _CORRUPTIONS + (
        # parses, has code entries, but not one declares an Excludes1 reference: schema
        # drift in the extract, indistinguishable at the gate from "no conflicts"
        '{"codes": {"SYNTHETIC_DX": {"includes": ["x"]}}}',
    )
    for content in shapes:
        _corrupted_source(monkeypatch, "instructional_notes", content, tmp_path)
        src = AuthoritativeSource()
        with pytest.raises(InstructionalNotesUnavailable):
            src.excludes1_refs("SYNTHETIC_DX", "icd10")
        assert issubclass(InstructionalNotesUnavailable, AuthoritativeDataUnavailable)
        monkeypatch.undo()


def _excludes1_result():
    from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, EvidenceSpan,
                                     FactKind, ResolutionMethod, ResolvedLine)

    def _dx(code, fid):
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="x", fact_id=fid,
                            evidence=[EvidenceSpan("x", anchored=True, span_id=fid)])
        return ResolvedLine(fact=fact, chosen=CandidateCode(code, "icd10", "d", 0.9),
                            method=ResolutionMethod.DETERMINISTIC)

    return CodingResult("e", "2026-08-01", lines=[_dx("DX_A", "a"), _dx("DX_B", "b")])


def test_unavailable_instructional_notes_hold_the_excludes1_gate_instead_of_passing():
    """Boundary: the raise has to land as a HOLD in the gate that consumes it. This is the
    one that is a COMPLIANCE GATE relaxing itself -- pre-fix the same input returned PASS,
    'no Excludes1 conflicts among diagnoses', on an authority nobody could read."""
    from claude_coder.data_access import InstructionalNotesUnavailable
    from claude_coder.gates import icd_excludes_gate
    from claude_coder.models import Outcome

    class _Unavailable:
        def excludes1_refs(self, code, system):
            raise InstructionalNotesUnavailable("instructional notes unreadable")

    class _Clean:
        def excludes1_refs(self, code, system):
            return set()

    held = icd_excludes_gate(_excludes1_result(), _Unavailable())
    assert held.outcome is Outcome.UNKNOWN and held.retryable
    assert "instructional notes unavailable" in held.detail.lower()
    # ...and a source that CAN read its authority and finds nothing still passes: the hold
    # is about unreadability, not about being conservative everywhere.
    assert icd_excludes_gate(_excludes1_result(), _Clean()).outcome is Outcome.PASS


def test_the_excludes1_gate_holds_on_a_corrupt_notes_file_end_to_end(monkeypatch, tmp_path):
    """The defect exactly as production would meet it, through the REAL source: the notes
    file is PRESENT -- so the capability manifest reports nothing missing and certification
    proceeds -- but its bytes are unusable. Pre-fix this returned PASS, "no Excludes1
    conflicts among diagnoses", on an authority nobody could read."""
    from claude_coder.data_access import AuthoritativeSource
    from claude_coder.gates import icd_excludes_gate
    from claude_coder.models import Outcome
    _corrupted_source(monkeypatch, "instructional_notes", "{not json", tmp_path)
    held = icd_excludes_gate(_excludes1_result(), AuthoritativeSource())
    assert held.outcome is Outcome.UNKNOWN and held.retryable


def test_a_single_diagnosis_never_consults_the_notes_at_all():
    """Failure path of the fix itself: the <2-diagnosis short-circuit must stay a genuine
    NOT_APPLICABLE and must not start holding claims on an unreadable file it never reads --
    one diagnosis cannot be in an Excludes1 relationship with anything."""
    from claude_coder.data_access import InstructionalNotesUnavailable
    from claude_coder.gates import icd_excludes_gate
    from claude_coder.models import Outcome

    class _Unavailable:
        def excludes1_refs(self, code, system):
            raise InstructionalNotesUnavailable("instructional notes unreadable")

    result = _excludes1_result()
    result.lines = result.lines[:1]
    assert icd_excludes_gate(result, _Unavailable()).outcome is Outcome.NOT_APPLICABLE


@pytest.mark.parametrize("source_id", ["pfs_indicators", "modifier_definitions"])
def test_corrupt_claim_assembly_data_holds_the_whole_pipeline(monkeypatch, tmp_path,
                                                              source_id):
    """The reads that happen during claim ASSEMBLY -- per-line modifiers, then the global
    surgical package -- run BEFORE the first gate, so no gate downstream could convert their
    unavailability into a hold. `code_encounter` must return a fail-closed system hold with
    no certificate rather than crashing or coding around the missing indicators."""
    from claude_coder.data_access import AuthoritativeSource
    from claude_coder.models import Destination, Outcome, Verdict
    _corrupted_source(monkeypatch, source_id, "{not json", tmp_path)
    r = code_encounter("e", _NOTE, "2026-03-14", source=AuthoritativeSource(),
                       extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                       corroborate_llm=_sel, audit_repository=_CapturingAudit(),
                       billing_context=_CTX)
    assert _has_gate(r, "authoritative_data_integrity")
    hold = [g for g in r.gates if g.name == "authoritative_data_integrity"][0]
    assert hold.outcome is Outcome.UNKNOWN and hold.retryable
    assert r.certificate is None
    assert r.verdict is not Verdict.AUTO_READY
    assert r.destination is Destination.SYSTEM_HOLD
    assert not r.billable_lines


def test_corrupt_snomed_root_concepts_raise_instead_of_disabling_the_confidence_cap(
        monkeypatch, tmp_path):
    """Post-fix review, adjacent instance in the SAME required set: the root table is a
    RESTRICTION (a root-level concept has its confidence CAPPED), and the loader warned and
    continued with an empty set -- so corruption silently lifted the cap. Absence of this
    required source blocks certification; presence with unreadable bytes must fail too."""
    import app.rag.code_reference as cr
    corrupt = tmp_path / "snomed_root_concepts.json"
    for content in ("", "{not json", "[]", "{}", '{"root_concepts": {}, "confidence_cap": 0.4}',
                    '{"root_concepts": {"1": "x"}}',            # no published cap
                    '{"root_concepts": {"1": "x"}, "confidence_cap": 0}'):
        corrupt.write_text(content)
        monkeypatch.setattr(cr, "SNOMED_ROOTS_FILE", corrupt)
        with pytest.raises(cr.SnomedRootsUnavailable):
            cr.CodeReferenceDB()._load_snomed_roots()
        monkeypatch.undo()
    # the real, intact table still loads and publishes both halves of the control
    db = cr.CodeReferenceDB()
    db._load_snomed_roots()
    assert db.snomed_roots and 0 < db.snomed_root_confidence_cap <= 1


def test_corrupt_validator_rule_pack_raises_instead_of_disabling_every_rule(tmp_path):
    """Same class again: every declarative rule is a RESTRICTION on what may be released, so
    a pack that degrades to `{"rules": []}` silently clears claims the reviewed pack flags."""
    from app.validation.rule_engine import RulePackUnavailable, load_rule_pack
    for content in ("", "{not json", "[]", "{}", '{"version": "x", "rules": []}',
                    '{"version": "x", "rules": {}}', '{"rules": [{"id": "r"}]}'):
        pack = tmp_path / "pack.json"
        pack.write_text(content)
        with pytest.raises(RulePackUnavailable):
            load_rule_pack(str(pack))
    with pytest.raises(RulePackUnavailable):
        load_rule_pack(str(tmp_path / "absent.json"))
    # The real, reviewed pack still loads -- addressed by its DECLARED identity, so this
    # asserts the required source itself rather than whatever the module global happens to
    # point at after some other test or tool swapped it.
    from app.release.source_manifest import declared_source_path
    real = load_rule_pack(str(declared_source_path("validator_rules")))
    assert real["rules"] and str(real["version"]).strip()


def test_the_rule_engines_default_pack_is_the_declared_release_source():
    """Guard for the leak this fix exposed: `RULES_FILE` is a module attribute that offline
    tools repoint to replay a candidate pack, and a swap that restores the WRONG value left
    the engine aimed at a deleted temp file -- which used to degrade silently to zero
    declarative rules. It must always be the declared `validator_rules` identity."""
    from pathlib import Path
    import app.validation.rule_engine as re_mod
    from app.release.source_manifest import declared_source_path
    assert Path(re_mod.RULES_FILE) == declared_source_path("validator_rules")


def test_every_required_source_the_coder_reads_fails_closed_on_corruption(monkeypatch,
                                                                          tmp_path):
    """The audit itself, executable: for EVERY required source the coder resolves through
    the declaration and reads at decision time, a present-but-corrupt file must raise a
    typed `AuthoritativeDataUnavailable` -- never return an empty table. A new required
    source wired into a decision path fails here until it fails closed."""
    from claude_coder.data_access import AuthoritativeDataUnavailable, AuthoritativeSource
    from claude_coder.modifiers import load_modifier_defs
    probes = {
        "coverage_policy": lambda s: s.qualifying_dx_for("SYNTHETIC_PROC", "cpt"),
        "pfs_indicators": lambda s: s.global_period("SYNTHETIC_PROC"),
        "instructional_notes": lambda s: s.excludes1_refs("SYNTHETIC_DX", "icd10"),
        "modifier_definitions": lambda s: load_modifier_defs(),
    }
    for source_id, probe in probes.items():
        for content in ("", "{not json", "[]", "{}"):
            _corrupted_source(monkeypatch, source_id, content, tmp_path)
            with pytest.raises(AuthoritativeDataUnavailable):
                probe(AuthoritativeSource())
            monkeypatch.undo()


def test_every_code_source_implements_the_claim_assembly_data_assertion():
    """Structural closure for the preflight: the pipeline asks the SOURCE to prove the
    tables claim assembly reads are readable, and a source that does not implement the
    method is skipped (a mock has no files to assert). That skip is only safe while every
    real implementation HAS it -- one that forgot would read its files mid-assembly, where
    no boundary converts the raise into a hold."""
    from claude_coder.data_access import AuthoritativeSource, MockSource
    for impl in (AuthoritativeSource, MockSource):
        assert callable(getattr(impl, "assert_claim_assembly_data_readable", None)), impl


def test_an_empty_mue_table_reads_as_unavailable_not_as_no_limits(monkeypatch):
    """Post-fix review, same class inside the same required set: `mue_available` asked
    `isinstance(table, dict)`, so a present-but-structurally-wrong `mue_limits` file
    produced an EMPTY dict that answered "no MUE is published for any code" -- and
    `mue_gate` reported NOT_APPLICABLE with every unit count unchecked."""
    from claude_coder.data_access import AuthoritativeSource
    from claude_coder.gates import mue_gate
    from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, EvidenceSpan,
                                     FactKind, Outcome, ResolutionMethod, ResolvedLine)

    class _Reference:
        mue: dict = {}

    src = AuthoritativeSource()
    monkeypatch.setattr(src, "_reference", lambda: _Reference())
    assert src.mue_available() is False
    _Reference.mue = {"SYNTHETIC_PROC": {"mue_value": 1}}
    assert src.mue_available() is True

    # ...and the gate holds rather than clearing the claim on the empty table
    _Reference.mue = {}
    fact = ClinicalFact(kind=FactKind.PROCEDURE, description="x", fact_id="pf",
                        evidence=[EvidenceSpan("x", anchored=True, span_id="s1")])
    line = ResolvedLine(fact=fact, chosen=CandidateCode("PROC_X", "cpt", "d", 0.9),
                        method=ResolutionMethod.DETERMINISTIC)
    held = mue_gate(CodingResult("e", "2026-08-01", lines=[line]), src)
    assert held.outcome is Outcome.UNKNOWN and held.retryable


# ---- Round 6, Codex F6-R5-A: the two app-side sources the reviewer named by file --------
# `coding_semantics.json` and `payers.json` were composed as filename literals inside their
# own readers, so they were never source IDENTITIES -- only incidental entries in the
# `data/codes/*.json` sweep, which a missing file simply drops out of. Both are now
# declared, required, and read through the same fail-closed mechanic every other required
# source uses. The tests below cover BOTH halves of the disposition: missing must hold, and
# present-but-corrupt must hold rather than degrading to `{}` / last-known-good.

def _repoint(monkeypatch, source_id, path):
    """Point a declared identity at `path`, through the registry, exactly as the release
    manifest and the reader both resolve it."""
    from app.release import source_manifest as sm
    registry = dict(sm._AUTHORITATIVE)
    registry[source_id] = path
    monkeypatch.setattr(sm, "_AUTHORITATIVE", registry)


def test_a_missing_required_app_source_is_reported_by_the_release_manifest(monkeypatch,
                                                                          tmp_path):
    """The absence half. Before this fix the file's absence was INVISIBLE: it was hashed
    only by the codes/ sweep, and a sweep cannot report what is not there. Now the identity
    is declared, so absence is an ERROR on the manifest -- which is what the source-manifest
    gate blocks the release on."""
    from app.release import source_manifest as sm
    for source_id in ("coding_semantics", "payer_registry"):
        _repoint(monkeypatch, source_id, tmp_path / "absent.json")
        errors = sm.build_source_manifest()["errors"]
        assert [e for e in errors if e.startswith(f"{source_id}:")], (source_id, errors)
        assert source_id in sm.required_release_sources()
        monkeypatch.undo()


def test_corrupt_coding_semantics_raise_instead_of_reclassifying_every_code(monkeypatch,
                                                                           tmp_path):
    """`_coding_semantics` logged and continued with `{}`. That is not a lost lookup: with
    `{}` every modifier-role query returns an empty set and every semantic-class question
    ("is this an E/M / a surgical procedure / an anaesthesia code / an external-cause
    diagnosis") answers False for EVERY code -- one unreadable file silently reclassifying
    the whole code set in the permissive direction."""
    from app.compliance.datastore.store import ComplianceDataStore, CodingSemanticsUnavailable
    from app.release.source_manifest import DeclaredSourceUnavailable
    assert issubclass(CodingSemanticsUnavailable, DeclaredSourceUnavailable)
    shapes = _CORRUPTIONS + (
        # parses, but publishes only part of the vocabulary -- schema drift in the extract,
        # indistinguishable downstream from "no modifier fills this role"
        '{"modifier_roles": {"r": {}}}',
        '{"modifier_roles": {"r": {}}, "code_classes": {}, "global_period_classes": {"a": 1}}',
    )
    for content in shapes:
        path = tmp_path / "coding_semantics.json"
        path.write_text(content)
        _repoint(monkeypatch, "coding_semantics", path)
        store = ComplianceDataStore.__new__(ComplianceDataStore)   # no DB build needed
        store._coding_semantics_cache = None
        with pytest.raises(CodingSemanticsUnavailable):
            store._coding_semantics()
        monkeypatch.undo()
    # ...and an ABSENT file fails identically: absence must never read as "no semantics
    # apply" in a process that already holds an open store.
    _repoint(monkeypatch, "coding_semantics", tmp_path / "absent.json")
    store = ComplianceDataStore.__new__(ComplianceDataStore)
    store._coding_semantics_cache = None
    with pytest.raises(CodingSemanticsUnavailable):
        store._coding_semantics()


def test_corrupt_payer_registry_raises_instead_of_silently_repricing_the_payer(monkeypatch,
                                                                              tmp_path):
    """`_load_payers` swallowed every failure and returned the last-known-good (initially
    EMPTY) registry. With an empty registry every note's insurance text matches nothing:
    payer_id None, is_medicare False, follows_medicare_coverage False -- i.e. every patient
    is silently reclassified as an unrecognized commercial payer, changing which coverage
    floor (LCD necessity, routine-foot-care findings, Medicare status-I validity) and which
    prior-authorization policy apply."""
    import app.compliance.payer_registry as pr
    from app.release.source_manifest import DeclaredSourceUnavailable
    assert issubclass(pr.PayerRegistryUnavailable, DeclaredSourceUnavailable)
    insurance = "Some Payer Plan, Member/Policy ID: X1, Group Number: G1"
    for content in ("", "{not json", "[]", "{}", '{"payers": []}', '{"payers": {}}'):
        path = tmp_path / "payers.json"
        path.write_text(content)
        _repoint(monkeypatch, "payer_registry", path)
        pr._payers_cache, pr._payers_mtime = [], -1
        with pytest.raises(pr.PayerRegistryUnavailable):
            pr.parse_insurance_text(insurance)
        monkeypatch.undo()
    _repoint(monkeypatch, "payer_registry", tmp_path / "absent.json")
    pr._payers_cache, pr._payers_mtime = [], -1
    with pytest.raises(pr.PayerRegistryUnavailable):
        pr.parse_insurance_text(insurance)
    monkeypatch.undo()
    # the real, declared registry still parses -- the hold is about unreadability, not
    # about refusing everything
    pr._payers_cache, pr._payers_mtime = [], -1
    assert pr.parse_insurance_text(insurance).raw_text == insurance


def test_a_corrupt_payer_registry_never_leaves_a_stale_one_serving_claims(monkeypatch,
                                                                         tmp_path):
    """The specific shape Codex named: 'silently retains empty/stale state'. A GOOD read
    followed by a corrupt rewrite must not keep answering from the superseded registry --
    the claim would then be composed against aliases no certified file backs."""
    import app.compliance.payer_registry as pr
    path = tmp_path / "payers.json"
    path.write_text(json.dumps({"payers": [
        {"payer_id": "synthetic_a", "canonical_name": "Synthetic A",
         "aliases": ["synthetic a"], "kind": "commercial"}]}))
    _repoint(monkeypatch, "payer_registry", path)
    pr._payers_cache, pr._payers_mtime = [], -1
    assert pr.parse_insurance_text("Synthetic A plan").payer_id == "synthetic_a"

    path.write_text("{ truncated")
    os.utime(path, ns=(0, 0))           # a genuinely different mtime, as a rewrite gives
    with pytest.raises(pr.PayerRegistryUnavailable):
        pr.parse_insurance_text("Synthetic A plan")
    # and it stays failed: the cache was never advanced past the last good read, so the
    # next call re-reads and re-raises rather than settling back into the stale registry
    with pytest.raises(pr.PayerRegistryUnavailable):
        pr.parse_insurance_text("Synthetic A plan")


def test_an_unreadable_payer_registry_stops_the_readiness_certificate(monkeypatch,
                                                                     tmp_path):
    """Boundary: the raise has to LAND somewhere that holds. `claim_readiness._context`
    caught `Exception` and passed, which recorded payer_kind/payer_id as "" and signed the
    certificate as though the note named no payer we recognize."""
    import app.release.claim_readiness as cr
    import app.compliance.payer_registry as pr
    path = tmp_path / "payers.json"
    path.write_text("{ truncated")
    _repoint(monkeypatch, "payer_registry", path)
    pr._payers_cache, pr._payers_mtime = [], -1
    with pytest.raises(pr.PayerRegistryUnavailable):
        cr._context({"patient_metadata": {"insurance": "Some Payer Plan"}})


def test_both_named_sources_survive_a_round_trip_through_the_declaration():
    """Certified bytes and read bytes are the SAME bytes: each reader resolves its file
    from the declaration, so the identity the manifest hashes is the path the reader opens.
    Asserted by identity, not by re-composing the filename here."""
    from pathlib import Path
    import app.compliance.payer_registry as pr
    from app.compliance.datastore import store as store_mod
    from app.release.source_manifest import declared_source_path
    assert Path(declared_source_path(pr._PAYERS_SOURCE_ID)).is_file()
    assert store_mod.CODING_SEMANTICS_SOURCE_ID == "coding_semantics"
    assert Path(declared_source_path(store_mod.CODING_SEMANTICS_SOURCE_ID)).is_file()


def test_unreadable_coding_semantics_holds_the_compliance_scrub(monkeypatch, tmp_path):
    """Boundary: the raise has to LAND as a hold in the path that consumes it. The
    compliance scrub wraps every agent in `except Exception` so one crash cannot sink the
    run -- the question is what it records. It must record a BLOCKING error ('this claim
    has not passed all required checks'), never a filter that quietly reported nothing and
    was counted as passed."""
    from app.compliance.agents.base import ComplianceAgent
    from app.compliance.datastore.store import ComplianceDataStore
    from app.compliance.engine import ClaimScrubber
    from app.compliance.models import Disposition, Status

    class _RoleConsumer(ComplianceAgent):
        filter_id = "SYNTHETIC_ROLE"
        filter_name = "synthetic modifier-role consumer"

        def check(self, claim):
            self.store.modifier_codes_for_role("synthetic_role")
            return []

    result = {"document_id": "SYNTHETIC", "patient_metadata": {},
              "rag_context": {}, "icd_codes": [], "cpt_codes": [], "hcpcs_codes": []}
    store = ComplianceDataStore.__new__(ComplianceDataStore)
    store._coding_semantics_cache = None

    corrupt = tmp_path / "coding_semantics.json"
    corrupt.write_text("{ truncated")
    _repoint(monkeypatch, "coding_semantics", corrupt)

    out = ClaimScrubber(store, [_RoleConsumer(store)]).scrub(result)
    assert out.clean is False
    assert out.disposition is Disposition.REVIEW
    assert [f for f in out.blocking_findings if f.status is Status.ERROR]
    assert [r for r in out.filter_results if r["status"] == Status.ERROR.value]
