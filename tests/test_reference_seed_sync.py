import json
from pathlib import Path

import pytest

from tools.sync_reference_seed import ReferenceSeedError, synchronize


def _manifest(path: Path, *, roots=None, files=None, runtime_owned=None):
    path.write_text(json.dumps({
        "schema_version": 1,
        "managed_roots": roots or [],
        "managed_files": files or [],
        "runtime_owned_files": runtime_owned or [],
    }))


def test_reference_seed_reconciles_managed_files_and_preserves_runtime(tmp_path):
    seed = tmp_path / "seed"
    destination = tmp_path / "data"
    (seed / "terminology").mkdir(parents=True)
    (seed / "terminology" / "registry.json").write_text("new-registry")
    (seed / "source.json").write_text("new-source")
    _manifest(seed / "manifest.json", roots=["terminology"],
              files=["source.json"])
    (destination / "terminology").mkdir(parents=True)
    (destination / "terminology" / "registry.json").write_text("stale")
    (destination / "runtime.db").write_text("preserve-me")

    first = synchronize(seed, destination, seed / "manifest.json")
    assert first["changed"] is True
    assert (destination / "terminology" / "registry.json").read_text() == \
        "new-registry"
    assert (destination / "runtime.db").read_text() == "preserve-me"
    assert synchronize(seed, destination, seed / "manifest.json")["changed"] \
        is False

    (destination / "source.json").unlink()
    assert synchronize(seed, destination, seed / "manifest.json")["changed"] \
        is True
    assert (destination / "source.json").read_text() == "new-source"


def test_reference_seed_removes_only_retired_managed_files(tmp_path):
    seed = tmp_path / "seed"
    destination = tmp_path / "data"
    seed.mkdir()
    (seed / "keep.json").write_text("keep")
    (seed / "retired.json").write_text("retire")
    _manifest(seed / "manifest.json", files=["keep.json", "retired.json"])
    synchronize(seed, destination, seed / "manifest.json")
    (destination / "runtime.json").write_text("runtime")

    (seed / "retired.json").unlink()
    _manifest(seed / "manifest.json", files=["keep.json"])
    synchronize(seed, destination, seed / "manifest.json")
    assert not (destination / "retired.json").exists()
    assert (destination / "runtime.json").read_text() == "runtime"


def test_reference_seed_rejects_traversal_and_symlinks(tmp_path):
    seed = tmp_path / "seed"
    destination = tmp_path / "data"
    seed.mkdir()
    _manifest(seed / "manifest.json", files=["../outside.json"])
    with pytest.raises(ReferenceSeedError):
        synchronize(seed, destination, seed / "manifest.json")

    destination.mkdir(exist_ok=True)
    (destination / ".reference_seed.lock").unlink()
    (destination / ".reference_seed.lock").symlink_to(seed / "real.json")
    _manifest(seed / "manifest.json", files=["real.json"])
    with pytest.raises(ReferenceSeedError):
        synchronize(seed, destination, seed / "manifest.json")
    (destination / ".reference_seed.lock").unlink()

    (seed / "real.json").write_text("value")
    (seed / "link.json").symlink_to(seed / "real.json")
    _manifest(seed / "manifest.json", files=["link.json"])
    with pytest.raises(ReferenceSeedError):
        synchronize(seed, destination, seed / "manifest.json")


def test_reference_seed_preserves_runtime_refreshed_sources(tmp_path):
    seed = tmp_path / "seed"
    destination = tmp_path / "data"
    (seed / "codes").mkdir(parents=True)
    (seed / "codes" / "static.json").write_text("release")
    (seed / "codes" / "refreshed.json").write_text("old-release")
    (destination / "codes").mkdir(parents=True)
    (destination / "codes" / "refreshed.json").write_text("new-runtime")
    _manifest(seed / "manifest.json", roots=["codes"],
              runtime_owned=["codes/refreshed.json"])

    synchronize(seed, destination, seed / "manifest.json")
    assert (destination / "codes" / "static.json").read_text() == "release"
    assert (destination / "codes" / "refreshed.json").read_text() == \
        "new-runtime"
