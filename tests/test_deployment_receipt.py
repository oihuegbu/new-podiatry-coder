import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / \
    "write_deployment_receipt.sh"


def test_deployment_receipt_is_atomic_and_validated(tmp_path):
    commit = "a" * 40
    artifact = "app-source-test.zip"
    timestamp = "2026-08-03T00:00:00Z"
    completed = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path), commit, artifact, timestamp],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert (tmp_path / "DEPLOYMENT_RECEIPT").read_text() == (
        f"git_commit={commit}\nartifact={artifact}\ninstalled_at={timestamp}\n"
    )
    failed = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path), "short", artifact, timestamp],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert (tmp_path / "DEPLOYMENT_RECEIPT").read_text().startswith(
        f"git_commit={commit}\n"
    )
