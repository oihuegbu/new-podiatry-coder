"""Release-gate acceptance runner for issue #6's designated note.

issue #6, Codex's independent re-review, root cause 1: the acceptance run must
never omit an input the release contract already requires. A run with no
billing/encounter context makes every ownership/context hold inevitable --
that is not release evidence, it is a code-path smoke test wearing release
evidence's clothes, and it spends real model calls on a note predetermined to
fail. This script refuses to start when the versioned, non-PHI acceptance
context fixture is absent, prints its fingerprint, and then invokes the SAME
`run.py` entrypoint every real deployment uses -- no parallel context logic,
and production still resolves patient/payer/provider/billing-entity facts
through the existing `EncounterContextProvider` interface, never from note
prose.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGNATED_NOTE = "Right_Retrocalcaneal_Exostectomy_Operative_Note.pdf"
BILLING_CONTEXT = REPO_ROOT / "data" / "context" / "billing_context.json"
ENCOUNTER_CONTEXT = REPO_ROOT / "data" / "context" / "acceptance_encounter_context.json"


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    missing = [p for p in (BILLING_CONTEXT, ENCOUNTER_CONTEXT) if not p.exists()]
    if missing:
        print(
            "release-gate run refused: the versioned acceptance context fixture is "
            f"absent ({', '.join(str(p) for p in missing)}). An acceptance run must "
            "never spend model calls on a note whose ownership/encounter context is "
            "predetermined to hold on every line -- create the fixture first "
            "(see data/context/encounter_context.example.json for the schema).",
            file=sys.stderr)
        return 2

    raw = json.loads(ENCOUNTER_CONTEXT.read_text())
    if raw.get("schema") != "encounter_context/2":
        print(f"release-gate run refused: {ENCOUNTER_CONTEXT} does not declare "
             f"schema 'encounter_context/2'", file=sys.stderr)
        return 2
    version = str(raw.get("version") or "")
    if not version:
        print(f"release-gate run refused: {ENCOUNTER_CONTEXT} declares no version",
             file=sys.stderr)
        return 2

    print(f"context edition: {version!r}")
    print(f"encounter context fingerprint: sha256:{_fingerprint(ENCOUNTER_CONTEXT)}")
    print(f"billing context fingerprint:   sha256:{_fingerprint(BILLING_CONTEXT)}")

    command = [
        sys.executable, str(REPO_ROOT / "run.py"),
        "--note", DESIGNATED_NOTE,
        "--billing-context", str(BILLING_CONTEXT),
        "--encounter-context", str(ENCOUNTER_CONTEXT),
    ]
    print("exact command:", " ".join(command))
    return subprocess.call(command, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
