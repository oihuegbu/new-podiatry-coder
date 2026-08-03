import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / \
    "refresh_runtime_env.sh"


def test_runtime_env_refresh_is_atomic_and_keeps_profile_json(tmp_path):
    secret = {
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "OPENAI_API_KEY": "openai-secret",
        "AUTHORIZED_MODEL_PROVIDERS": "claude,openai",
        "CODING_EXECUTION_PROFILES": json.dumps([
            {"profile_id": "primary", "provider": "claude",
             "model": "claude-model"},
            {"profile_id": "corroborator", "provider": "openai",
             "model": "openai-model"},
        ], separators=(",", ":")),
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_aws = bin_dir / "aws"
    fake_aws.write_text("#!/bin/sh\nprintf '%s' \"$FAKE_SECRET\"\n")
    fake_aws.chmod(0o700)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               FAKE_SECRET=json.dumps(secret, separators=(",", ":")))

    completed = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path), "secret-id", "us-east-1",
         "6335"], check=True, capture_output=True, text=True, env=env)

    assert completed.stdout == ""
    assert completed.stderr == ""
    values = dict(line.split("=", 1) for line in
                  (tmp_path / ".env").read_text().splitlines())
    assert values == {**secret, "QDRANT_HOST_PORT": "6335"}
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600


def test_runtime_env_refresh_rejects_invalid_secret_keys(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_aws = bin_dir / "aws"
    fake_aws.write_text("#!/bin/sh\nprintf '%s' '{\"bad-key\":\"value\"}'\n")
    fake_aws.chmod(0o700)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")

    failed = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path), "secret-id", "us-east-1"],
        capture_output=True, text=True, env=env)

    assert failed.returncode != 0
    assert not (tmp_path / ".env").exists()


def test_runtime_env_refresh_rejects_line_injection(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_aws = bin_dir / "aws"
    fake_aws.write_text(
        "#!/bin/sh\nprintf '%s' '{\"OPENAI_API_KEY\":\"safe\\nINJECTED=1\"}'\n")
    fake_aws.chmod(0o700)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")

    failed = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path), "secret-id", "us-east-1"],
        capture_output=True, text=True, env=env)

    assert failed.returncode != 0
    assert not (tmp_path / ".env").exists()
