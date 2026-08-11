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


def test_same_count_byte_change_invalidates_the_release_fingerprint(tmp_path):
    """The core defect: two materially different authoritative files with IDENTICAL row
    counts must not share a certifiable identity. Mutating the bytes of a required source
    while holding every count constant must change the fingerprint."""
    import json
    import claude_coder.capability as cap
    import claude_coder.pipeline as pl

    src_file = tmp_path / "authoritative.json"
    src_file.write_text(json.dumps([{"code": "AAA", "description": "first edition"}]))

    def _manifest():
        # one required source, content-addressed exactly as production records it
        from app.release.source_manifest import sha256_file
        stat = src_file.stat()
        sources = [{"source": "authoritative", "source_id": "authoritative", "required": True,
                    "present": True, "status": "loaded", "role": "test", "path": str(src_file),
                    "bytes": stat.st_size, "sha256": sha256_file(src_file),
                    "release": {"effective_from": "2026-01-01", "version": "v1"}}]
        man = {"manifest_version": cap.MANIFEST_VERSION, "generated_at": "x",
               "sources": sources, "missing_required": [], "degraded_optional": [],
               "integrity_errors": [], "status": "OK",
               "manifest_sha256": cap.manifest_digest(sources)}
        counts = {"icd10": 7, "cpt": 7, "hcpcs": 7}          # counts held CONSTANT
        return {"counts": counts, "source_manifest": man,
                "fingerprint_version": "release-data-fingerprint-v2",
                "fingerprint_sha256": cap.manifest_digest(
                    [{"counts": counts, "manifest": man["manifest_sha256"]}] + sources)}

    first = _manifest()
    assert pl._fingerprint_certifiable(first) is True
    # SAME number of rows, DIFFERENT bytes
    src_file.write_text(json.dumps([{"code": "AAA", "description": "SECOND edition"}]))
    second = _manifest()
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
