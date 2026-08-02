"""Automated freshness preflight for refreshable authoritative sources."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timezone

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.refresh.runner import refresh_source
from app.core.config import HCPCS_FILE, MCD_COVERAGE_CACHE_FILE


def _quarter_start(value: date) -> date:
    return date(value.year, ((value.month - 1) // 3) * 3 + 1, 1)


def _db_release(store: ComplianceDataStore, table: str) -> date | None:
    try:
        raw = store.conn.execute(
            f"SELECT MAX(effective_from) FROM {table}").fetchone()[0]
        return date.fromisoformat(str(raw))
    except (sqlite3.Error, TypeError, ValueError):
        return None


def _hcpcs_release() -> date | None:
    """Quarter identity from the CMS file name bound to every source row."""
    try:
        payload = json.loads(HCPCS_FILE.read_text())
        source_files = {
            str((row.get("metadata") or {}).get("source_file") or "")
            for row in payload if isinstance(row, dict)
        }
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if len(source_files) != 1:
        return None
    source_file = next(iter(source_files))
    match = re.search(
        r"HCPC(\d{4})_(JAN|APR|JUL|OCT)_ANWEB", source_file, re.I)
    if not match:
        return None
    month = {"JAN": 1, "APR": 4, "JUL": 7, "OCT": 10}[match.group(2).upper()]
    return date(int(match.group(1)), month, 1)


def stale_refreshable_sources(store: ComplianceDataStore, *,
                              today: date | None = None) -> list[str]:
    today = today or date.today()
    quarter = _quarter_start(today)
    stale = []
    for source_id, table in (("ncci_ptp", "ncci_ptp"),
                             ("mue", "mue"),
                             ("pfs_global", "global_period")):
        release = _db_release(store, table)
        if release is None or release < quarter:
            stale.append(source_id)
    hcpcs_release = _hcpcs_release()
    if hcpcs_release is None or hcpcs_release < quarter:
        stale.append("hcpcs")
    try:
        payload = json.loads(MCD_COVERAGE_CACHE_FILE.read_text())
        fetched = datetime.fromisoformat(
            str(payload.get("fetched_at") or "").replace("Z", "+00:00"))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        if (datetime.now(timezone.utc) - fetched.astimezone(timezone.utc)).days > 7:
            stale.append("mcd_articles")
    except (OSError, ValueError, TypeError):
        stale.append("mcd_articles")
    return stale


def refresh_stale_sources(*, require_current: bool = False) -> dict:
    store = ComplianceDataStore()
    try:
        store.build_or_load()
        before = stale_refreshable_sources(store)
        results = [refresh_source(store, source_id) for source_id in before]
        after = stale_refreshable_sources(store)
        errors = [str(row.get("error") or row.get("source")) for row in results
                  if not row.get("ok")]
        report = {"stale_before": before, "results": results,
                  "stale_after": after, "errors": errors,
                  "current": not after and not errors}
        if require_current and not report["current"]:
            raise RuntimeError(
                "authoritative refresh preflight failed: " +
                "; ".join(errors + after))
        return report
    finally:
        store.close()
