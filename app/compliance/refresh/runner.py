"""Refresh runner — pulls authoritative source files, parses them, and ingests
new snapshots into the ComplianceDataStore with full history retention.

Design:
  * additive ingestion — never deletes prior quarters (CMS purges them; we keep
    them as effective-dated rows so a claim is always scrubbed against the rules
    in force on its DOS);
  * idempotent — re-running the same quarter is a no-op (`ingest_snapshot`);
  * offline-capable — pass a local file to ingest without network (testing /
    air-gapped deploys);
  * graceful — a source that fails to download is logged and skipped, never
    crashes the run.
"""
from __future__ import annotations

import urllib.request
import urllib.error
from datetime import date

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.refresh.sources import SOURCES_BY_ID, due_sources
from app.compliance.refresh import parsers as P
from app.core.logger import get_logger

logger = get_logger(__name__)

_UA = "Mozilla/5.0 (compatible; ClaimScrubber/1.0; +compliance-refresh)"
# Tables that retain history (effective-dated, no primary key)
_HISTORY_TABLES = {"ncci_ptp", "mue", "global_period"}


def download(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _payload_text(source, raw: bytes) -> str:
    if source.fmt.startswith("zip"):
        return P.unzip_first(raw)
    return raw.decode("latin-1", errors="replace")


def refresh_source(store: ComplianceDataStore, source_id: str, *,
                   effective_from: str | None = None, local_bytes: bytes | None = None,
                   dry_run: bool = False) -> dict:
    """Refresh one source. Returns a summary dict."""
    src = SOURCES_BY_ID.get(source_id)
    if not src:
        return {"source": source_id, "ok": False, "error": "unknown source"}

    eff = effective_from or date.today().isoformat()
    try:
        raw = local_bytes if local_bytes is not None else download(src.url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning(f"refresh[{source_id}]: download failed ({e}) — skipped")
        return {"source": source_id, "ok": False, "error": str(e)}

    parser = P.PARSERS.get(src.parser)
    if not parser:
        return {"source": source_id, "ok": False, "error": f"no parser {src.parser}"}

    text = _payload_text(src, raw)

    # MCD articles go through the coverage loader (two tables)
    if source_id == "mcd_articles":
        articles = parser(text, eff)
        if not dry_run:
            store.load_coverage_articles(articles)
        return {"source": source_id, "ok": True, "articles": len(articles), "dry_run": dry_run}

    rows, cols = parser(text, eff)
    if dry_run:
        return {"source": source_id, "ok": True, "parsed_rows": len(rows), "dry_run": True}

    if src.target_table in _HISTORY_TABLES:
        n = store.ingest_snapshot(src.target_table, cols, rows, source_id, eff,
                                  file_name=src.url.rsplit("/", 1)[-1])
    else:  # reference tables (pos) — replace in place
        ph = ",".join("?" * len(cols))
        store.conn.executemany(
            f"INSERT OR REPLACE INTO {src.target_table} ({','.join(cols)}) VALUES ({ph})", rows
        )
        store.conn.commit()
        n = len(rows)
    return {"source": source_id, "ok": True, "ingested_rows": n}


def refresh_all(store: ComplianceDataStore, *, month: int | None = None,
                dry_run: bool = False) -> list[dict]:
    m = month or date.today().month
    results = []
    for src in due_sources(m):
        logger.info(f"refresh: {src.id} ({src.cadence}, due in month {m})")
        results.append(refresh_source(store, src.id, dry_run=dry_run))
    return results
