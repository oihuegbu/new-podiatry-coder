"""Subprocess entrypoint — build ONE Qdrant collection, then exit.

Building all three collections (ICD-10 ~74K, CPT ~11K, HCPCS ~8K) back-to-back
in a single process lets the onnxruntime memory arena grow without returning
memory to the OS, which OOM-crashes lower-RAM / CPU-only machines. To avoid this,
`MedicalCodeVectorStore._rebuild_via_subprocesses()` invokes this module once per
collection:

    python -m app.rag.build_collection <icd10|cpt|hcpcs>

Because each invocation is its own process that fully exits, the OS reclaims ALL
of that memory (arena included) before the next collection is built.
"""
import sys

from app.rag.vector_store import MedicalCodeVectorStore, _CODE_SYSTEMS
from app.core.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in _CODE_SYSTEMS:
        logger.error(
            f"usage: python -m app.rag.build_collection <{'|'.join(_CODE_SYSTEMS)}>"
        )
        return 2

    name = sys.argv[1]
    logger.info(f"[subprocess] building '{name}' collection")
    store = MedicalCodeVectorStore()
    store.build_one(name)
    logger.info(f"[subprocess] '{name}' complete — exiting to release memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
