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
    assert pl._fingerprint_certifiable(
        {"counts": {"icd10": 1, "cpt": 1, "hcpcs": 1},
         "source_manifest": {"status": "OK"}}) is True

    for bad in ({}, {"counts": {}}):
        src = _src()
        src.data_fingerprint = lambda b=bad: b
        r = code_encounter("e", _NOTE, "2026-03-14", source=src,
                           extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=_CapturingAudit(),
                           billing_context=_CTX)
        assert r.certificate is None                                       # not certified
        assert _has_gate(r, "data_fingerprint")
