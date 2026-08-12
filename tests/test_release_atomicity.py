"""Codex F6-R5: release attestation is order-safe and fail-closed.

The RETURNED verdict, the CERTIFICATE, the data fingerprint, and the LAST durable audit
decision can never disagree: a missing fingerprint or a failed certificate/audit write
downgrades the returned verdict AND is reflected in (or absent from) the durable record --
AUTO_READY is never persisted before it is real, and a certificate never exists without a
matching durable record.
"""
import hashlib
import json
import pytest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode
from claude_coder.pipeline import code_encounter
import claude_coder.certificate as cert_mod

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
    assert r.certificate is None and _has_gate(r, "data_fingerprint")


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
