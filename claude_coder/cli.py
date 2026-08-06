"""CLI: code a single note.

  python -m claude_coder.cli note.txt --dos 2026-03-14
  cat note.txt | python -m claude_coder.cli - --dos 2026-03-14 --json

Uses the real authoritative source (needs the RAG index + an LLM key). For a
key-free demonstration of the flow, see tests/test_claude_coder.py, which runs
the whole pipeline on a MockSource with stubbed LLMs.
"""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import code_encounter, render


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Autonomous medical coder")
    ap.add_argument("note", help="path to a note file, or '-' for stdin")
    ap.add_argument("--dos", help="date of service YYYY-MM-DD (required for release)")
    ap.add_argument("--id", default="enc-1", help="encounter id")
    ap.add_argument("--billing-context", help="JSON file containing billing_entity_id and known participants")
    ap.add_argument("--document-version", help="immutable source-document version/id")
    ap.add_argument("--json", action="store_true", help="print the release certificate as JSON")
    args = ap.parse_args(argv)

    note = sys.stdin.read() if args.note == "-" else open(args.note).read()
    billing_context = None
    if args.billing_context:
        with open(args.billing_context) as fh:
            billing_context = json.load(fh)
        if not isinstance(billing_context, dict):
            ap.error("--billing-context must contain a JSON object")
    result = code_encounter(args.id, note, args.dos,
                            billing_context=billing_context,
                            document_version=args.document_version)

    if args.json:
        print(json.dumps(result.certificate, indent=2))
    else:
        print(render(result))
        if result.certificate:
            print(f"\nCERTIFICATE sha256: {result.certificate['certificate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
