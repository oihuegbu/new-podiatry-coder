"""Codex F6-R8: the authoritative NCCI snapshot is built from PINNED, CHECKSUM-VERIFIED
release inputs — not from whichever quarter CMS happens to expose at build time.

Pinning only the OUTPUT detected drift but left the build irreproducible: CMS retains just
the current + prior quarter, so a clean rebuild of an already-reviewed commit could fail, or
resolve materially different files, with no source change at all. These tests assert the
input-provenance contract itself: identities and checksums are recorded, every fetch path
(network or controlled immutable copy) is verified against them, and a mismatch aborts
loudly instead of quietly redefining what the image contains.

No medical code appears here — this is build-input provenance, not code logic.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

builder = importlib.import_module("tools.build_ncci_ptp")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _lock() -> dict:
    return json.loads(builder.LOCK.read_text())


def test_reviewed_lock_pins_every_release_input():
    lock = _lock()
    assert _SHA256.fullmatch(str(lock["output_sha256"]).lower())
    assert int(lock["pairs"]) > 0
    assert lock["output"] and (ROOT / lock["output"]).parent.is_dir()
    inputs = lock["inputs"]
    assert isinstance(inputs, list) and inputs, "the build inputs must be pinned, not scraped"
    for spec in inputs:
        assert str(spec["url"]).startswith("http"), spec
        assert _SHA256.fullmatch(str(spec["sha256"]).lower()), spec
        assert int(spec["bytes"]) > 0 and spec["name"] and spec["effective_from"]
    # input ORDER is part of the pinned identity: the merge is order-dependent
    assert [s["effective_from"] for s in inputs] == sorted(s["effective_from"] for s in inputs)


def test_lock_without_pinned_inputs_aborts_the_build(tmp_path, monkeypatch):
    """A lock that names no inputs means the build would resolve mutable upstream listings —
    the exact irreproducibility being fixed. It must abort, not fall back to scraping."""
    empty = tmp_path / "ncci_ptp.lock.json"
    empty.write_text(json.dumps({"output_sha256": "0" * 64, "pairs": 1}))
    monkeypatch.setattr(builder, "LOCK", empty)
    with pytest.raises(SystemExit):
        builder._read_lock()
    monkeypatch.setattr(builder, "LOCK", tmp_path / "absent.lock.json")
    with pytest.raises(SystemExit):
        builder._read_lock()


def test_incomplete_input_spec_aborts_the_build(tmp_path, monkeypatch):
    partial = tmp_path / "ncci_ptp.lock.json"
    partial.write_text(json.dumps({"inputs": [{"name": "a.zip", "url": "https://x/a.zip"}]}))
    monkeypatch.setattr(builder, "LOCK", partial)
    with pytest.raises(SystemExit):
        builder._read_lock()


def _spec(raw: bytes, name="ptp.zip"):
    import hashlib
    return {"name": name, "url": f"https://cms.example/{name}",
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "effective_from": "2026-07-01"}


def test_downloaded_input_must_match_its_pinned_checksum(monkeypatch):
    raw = b"the reviewed release bytes"
    spec = _spec(raw)
    monkeypatch.delenv(builder.INPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(builder.runner, "download", lambda *a, **k: raw)
    assert builder._fetch_pinned(spec) == raw                      # verified -> accepted
    # CMS rotated/replaced the file behind the same URL: abort, never silently rebuild
    monkeypatch.setattr(builder.runner, "download", lambda *a, **k: b"a different release")
    with pytest.raises(SystemExit) as exc:
        builder._fetch_pinned(spec)
    assert "checksum" in str(exc.value)


def test_input_size_mismatch_aborts(monkeypatch):
    raw = b"bytes"
    spec = _spec(raw)
    spec["bytes"] = len(raw) + 1
    monkeypatch.delenv(builder.INPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(builder.runner, "download", lambda *a, **k: raw)
    with pytest.raises(SystemExit):
        builder._fetch_pinned(spec)


def test_controlled_immutable_copy_is_used_and_verified(tmp_path, monkeypatch):
    """The disaster-recovery path: a controlled copy of the same pinned files builds without
    CMS being reachable at all — and is checksum-verified identically, so the mirror cannot
    substitute different bytes either."""
    raw = b"the reviewed release bytes"
    spec = _spec(raw)
    (tmp_path / spec["name"]).write_bytes(raw)
    monkeypatch.setenv(builder.INPUT_DIR_ENV, str(tmp_path))

    def _no_network(*a, **k):
        raise AssertionError("the pinned build must not reach the network when a "
                             "controlled copy is provided")

    monkeypatch.setattr(builder.runner, "download", _no_network)
    assert builder._fetch_pinned(spec) == raw

    (tmp_path / spec["name"]).write_bytes(b"tampered mirror copy")
    with pytest.raises(SystemExit):
        builder._fetch_pinned(spec)


def test_refresh_is_a_separate_reviewed_workflow():
    """An intentional version upgrade writes a PROPOSED lock; it never redefines the lock the
    current image build uses."""
    assert builder.PROPOSED_LOCK != builder.LOCK
    assert builder.PROPOSED_LOCK.name.endswith(".proposed.json")
    src = builder.build.__doc__ or ""
    ap_source = importlib.import_module("tools.build_ncci_ptp").__doc__ or ""
    assert "--refresh" in ap_source
    # a default (non-refresh) build never resolves the mutable CMS listing
    import inspect
    payload_src = inspect.getsource(builder._payloads)
    assert "_all_quarter_files" in payload_src and "args.refresh" in payload_src


def test_prepare_refuses_a_lock_without_pinned_inputs(tmp_path, monkeypatch, capsys):
    prepare = importlib.import_module("tools.prepare_ncci")
    lock = tmp_path / "ncci_ptp.lock.json"
    lock.write_text(json.dumps({"output_sha256": "0" * 64, "pairs": 1}))
    monkeypatch.setattr(prepare, "LOCK", lock)
    assert prepare.main() == 1
    assert "pinned release inputs" in capsys.readouterr().err
