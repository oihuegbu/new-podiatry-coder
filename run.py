#!/usr/bin/env python3
"""
Podiatry Medical Coding System
NER → RAG (Qdrant hybrid) → LLM (Claude Opus 4.7) → Validation Pipeline

Usage:
    python run.py                     # Process all notes
    python run.py --note NOTE_01...   # Process single note
    python run.py --rebuild-index     # Force rebuild Qdrant collections
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import NOTES_DIR, OUTPUT_DIR
from app.core.logger import get_logger
from app.pipeline import MedicalCodingPipeline

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="Podiatry Medical Coding System")
    parser.add_argument("--note", type=str, help="Process a single PDF note (filename or path)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild Qdrant vector collections")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache lookup and force fresh processing")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PODIATRY MEDICAL CODING SYSTEM")
    logger.info(f"Pipeline: NER → RAG (Qdrant hybrid) → LLM (Claude Opus 4.7) → Validation")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # Initialize pipeline
    pipeline = MedicalCodingPipeline()
    pipeline.initialize(force_rebuild_index=args.rebuild_index)

    # Find notes to process
    if args.note:
        note_path = Path(args.note)
        if not note_path.exists():
            note_path = NOTES_DIR / args.note
        if not note_path.exists():
            logger.error(f"Note not found: {args.note}")
            sys.exit(1)
        note_files = [note_path]
    else:
        note_files = sorted(NOTES_DIR.glob("NOTE_*.pdf"))

    if not note_files:
        logger.error(f"No clinical notes found in {NOTES_DIR}")
        sys.exit(1)

    logger.info(f"\nProcessing {len(note_files)} clinical note(s)\n")

    # Process notes
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for pdf_path in note_files:
        try:
            result = pipeline.process_note(pdf_path, use_cache=not args.no_cache)
            results.append(result)

            output_file = OUTPUT_DIR / f"{pdf_path.stem}_results.json"
            with open(output_file, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            logger.info(f"  Saved → {output_file.name}")

        except Exception as e:
            logger.error(f"FAILED: {pdf_path.name} — {e}")
            import traceback
            traceback.print_exc()

    # Combined output
    if len(results) > 1:
        combined = OUTPUT_DIR / "all_results.json"
        with open(combined, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("BATCH COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total: {len(results)} | Success: {sum(1 for r in results if r.success)} | Failed: {sum(1 for r in results if not r.success)}")

    tiers = {"AUTO": 0, "REVIEW": 0, "REJECT": 0}
    for r in results:
        tiers[r.auto_coding_tier] = tiers.get(r.auto_coding_tier, 0) + 1

    logger.info(f"Auto: {tiers['AUTO']} | Review: {tiers['REVIEW']} | Reject: {tiers['REJECT']}")
    logger.info(f"Output → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
