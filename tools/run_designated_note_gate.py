"""Release-gate acceptance runner for issue #6's designated note.

issue #6, Codex's independent re-review, root cause 1: the acceptance run must
never omit an input the release contract already requires. A run with no
billing/encounter context makes every ownership/context hold inevitable --
that is not release evidence, it is a code-path smoke test wearing release
evidence's clothes, and it spends real model calls on a note predetermined to
fail. This script refuses to start when the versioned, non-PHI acceptance
context fixture is absent OR malformed, prints both fixtures' fingerprints and
the exact command, then invokes the SAME run.py entrypoint every real
deployment uses -- no parallel context logic, and production still resolves
patient/payer/provider/billing-entity facts through the existing
EncounterContextProvider interface, never from note prose.

Round 2 (Codex's independent re-review): validating only that the fixture
FILES EXIST let a malformed billing context (e.g. `{}`) still start the paid
note run. Validation now goes through the SAME loaders/validators the real
pipeline uses -- `run.load_billing_context` + `claude_coder.extraction.
_participant_index` (the exact check `extraction.extract_note` itself runs
"BEFORE spending an extraction call"), and
`app.contracts.encounter_context.build_provider(...).preflight()` (the real
`VersionedRosterContextProvider`'s own schema/version/section validation) --
never a duplicated, weaker re-implementation of either schema.
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

    sys.path.insert(0, str(REPO_ROOT))

    # Billing context: the SAME strict validation `extraction.extract_note`
    # itself runs before spending an extraction call -- never a duplicate.
    try:
        from run import load_billing_context
        from claude_coder.extraction import _participant_index
        billing = load_billing_context(str(BILLING_CONTEXT))
        _participant_index(billing)
    except Exception as exc:
        print(f"release-gate run refused: billing context is invalid: {exc}",
             file=sys.stderr)
        return 2

    # Encounter context: the REAL provider's own preflight -- schema, version,
    # every required section -- never a hand-rolled re-check of that schema.
    try:
        from app.contracts.encounter_context import (
            EncounterContextUnavailable, build_provider)
        provider = build_provider(str(ENCOUNTER_CONTEXT))
        preflight = provider.preflight()
    except EncounterContextUnavailable as exc:
        print(f"release-gate run refused: encounter context is invalid: {exc}",
             file=sys.stderr)
        return 2

    designated_id = Path(DESIGNATED_NOTE).stem
    raw = json.loads(ENCOUNTER_CONTEXT.read_text())
    if designated_id not in (raw.get("encounters") or {}):
        print(
            f"release-gate run refused: {ENCOUNTER_CONTEXT} declares no entry for "
            f"the designated encounter {designated_id!r} -- this fixture cannot "
            f"resolve context for the note this gate is meant to verify.",
            file=sys.stderr)
        return 2

    version = preflight.get("version", "")
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
