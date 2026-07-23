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

import re
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


# Preferred zip member per source, for archives bundling several datasets.
# PFS RVU zips ship GPCI + OPPSCAP + ANES + PPRRVU together; the global/
# status data lives in the PPRRVU file (non-QPP = the complete set).
_ZIP_MEMBER_PREFERENCE = {
    "pfs_global": (r"PPRRVU.*non.?QPP.*\.csv$", r"PPRRVU.*\.csv$", r"PPRRVU.*\.txt$"),
}


def _payload_text(source, raw: bytes) -> str:
    if source.fmt.startswith("zip"):
        return P.unzip_first(raw, prefer=_ZIP_MEMBER_PREFERENCE.get(source.id, ()))
    return raw.decode("latin-1", errors="replace")


# ------------------------------------------------------------- URL resolution
# CMS registers stable LANDING pages in sources.py but rotates the concrete
# file names every quarter (ccipra-v322r0-f1.zip, rvu26c-updated-...,
# Eff_07-01-2026, ...). These resolvers scrape the landing page for the
# current file link(s) at refresh time — previously the runner downloaded the
# landing page ITSELF and "ingested" 0 rows from its HTML.

def _abs(base: str, href: str) -> str:
    if href.startswith("http"):
        return href
    return "https://www.cms.gov" + href if href.startswith("/") else href


def _quarter_start(year: int, q: int) -> str:
    return f"{year}-{(q - 1) * 3 + 1:02d}-01"


def _resolve_ncci_ptp(url: str) -> tuple[list[str], str | None]:
    """Practitioner PTP edit files for the newest quarter on the page.
    CMS splits the full edit set across f1/f2/... members — ALL are needed,
    and they're served through an /license/ama? wrapper that still 200s the
    zip directly at /files/zip/ (verified live)."""
    html = download(url).decode("utf-8", errors="replace")
    hits = re.findall(
        r'href="([^"]*?(\d{4})q([1-4])-practitioner-ptp-edits[^"]*?-f\d[^"]*?\.zip)"', html, re.I)
    if not hits:
        return [], None
    year, q = max((int(y), int(qq)) for _, y, qq in hits)
    files = sorted({
        _abs(url, re.sub(r"^/license/ama\?file=", "", h))
        for h, y, qq in hits if (int(y), int(qq)) == (year, q)
    })
    return files, _quarter_start(year, q)


def _resolve_mue(url: str) -> tuple[list[str], str | None]:
    """Practitioner-services MUE table for the newest quarter on the page."""
    html = download(url).decode("utf-8", errors="replace")
    hits = re.findall(
        r'href="([^"]*?(\d{4})-?q([1-4])-practitioner-services-mue-table[^"]*?\.zip)"', html, re.I)
    if not hits:
        return [], None
    year, q = max((int(y), int(qq)) for _, y, qq in hits)
    files = [_abs(url, h) for h, y, qq in hits if (int(y), int(qq)) == (year, q)][:1]
    return files, _quarter_start(year, q)


def _resolve_pfs(url: str) -> tuple[list[str], str | None]:
    """Two-hop resolve: landing page → newest rvu<YY><a-d> page → its zip.
    The trailing letter is the quarterly revision (a=Jan ... d=Oct)."""
    html = download(url).decode("utf-8", errors="replace")
    hits = re.findall(r'href="([^"]*?/rvu(\d{2})([a-d])(?:-\d+)?)"', html, re.I)
    if not hits:
        return [], None
    yy, letter = max((int(y), l.lower()) for _, y, l in hits)
    page = next(_abs(url, h) for h, y, l in hits
                if int(y) == yy and l.lower() == letter)
    inner = download(page).decode("utf-8", errors="replace")
    zips = re.findall(r'href="([^"]+\.zip[^"]*)"', inner, re.I)
    if not zips:
        return [], None
    return [_abs(page, zips[0])], _quarter_start(2000 + yy, ord(letter) - ord("a") + 1)


# The MCD bulk export lives at a STABLE url (verified live) — the landing
# page in sources.py is javascript-rendered and exposes no scrapeable link.
_MCD_EXPORT_URL = ("https://downloads.cms.gov/medicare-coverage-database/"
                   "downloads/exports/current_article.zip")


def _resolve_mcd(url: str) -> tuple[list[str], str | None]:
    return [_MCD_EXPORT_URL], None


_RESOLVERS = {
    "ncci_ptp": _resolve_ncci_ptp,
    "mue": _resolve_mue,
    "pfs_global": _resolve_pfs,
    "mcd_articles": _resolve_mcd,
}


def _write_coverage_cache(articles: list[dict], effective: str | None) -> None:
    """Persist the parsed MCD export next to the other data/codes sources.

    compliance.db is REBUILT from data/codes/*.json whenever their
    fingerprint changes — a rebuild that read only the flat seed file
    (podiatry_lcd.json, no group data) would silently discard the covered-
    ICD group roles the weekly refresh ingested, reverting the claim-
    composition gate to the flat pre-group behavior until the next refresh
    fired. Caching the parsed export makes the rebuild self-sufficient:
    _ingest_lcd() overlays this cache (grouped, newer) over the seed
    (flat, older) per policy_id. Atomic write so a reader never sees a
    truncated file."""
    import json
    import os
    from datetime import datetime, timezone

    from app.core.config import CODES_DIR

    path = CODES_DIR / "mcd_coverage_cache.json"
    payload = {
        "source": _MCD_EXPORT_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "effective_from": effective,
        "note": ("Parsed CMS MCD bulk article export (parse_mcd_export) "
                 "including covered-ICD group roles; overlaid on the "
                 "podiatry_lcd.json seed at every compliance.db rebuild."),
        "articles": articles,
    }
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)
    logger.info(f"refresh[mcd_articles]: coverage cache written "
                f"({len(articles)} articles) → {path.name}")


def _resolve_urls(src) -> tuple[list[str], str | None]:
    """(concrete file URLs, effective date derived from the file's own
    quarter) — falls back to the registered URL when no resolver exists or
    resolution finds nothing (the 0-row guard downstream still catches a
    landing-page payload)."""
    resolver = _RESOLVERS.get(src.id)
    if not resolver:
        return [src.url], None
    try:
        urls, eff = resolver(src.url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning(f"refresh[{src.id}]: landing-page resolve failed ({e})")
        return [src.url], None
    if not urls:
        logger.warning(f"refresh[{src.id}]: no file links found on landing page — "
                       f"falling back to registered URL")
        return [src.url], None
    return urls, eff


def refresh_source(store: ComplianceDataStore, source_id: str, *,
                   effective_from: str | None = None, local_bytes: bytes | None = None,
                   dry_run: bool = False) -> dict:
    """Refresh one source. Returns a summary dict."""
    src = SOURCES_BY_ID.get(source_id)
    if not src:
        return {"source": source_id, "ok": False, "error": "unknown source"}
    if src.manual and local_bytes is None:
        # Manual sources have no automated fetch/parse path — say so plainly
        # instead of failing with a misleading download/parser error. They CAN
        # still be ingested by passing the file explicitly (local_bytes).
        logger.info(f"refresh[{source_id}]: manual source — update its JSON file by hand "
                    f"(compliance.db re-ingests changed files automatically); skipped")
        return {"source": source_id, "ok": True, "skipped": "manual source", "notes": src.notes}

    # Resolve the current concrete file URL(s) from the landing page; a
    # local file (offline/air-gapped ingest) bypasses resolution entirely.
    if local_bytes is not None:
        payloads, resolved_eff = [(local_bytes, "local-file")], None
    else:
        urls, resolved_eff = _resolve_urls(src)
        payloads = []
        for u in urls:
            try:
                payloads.append((download(u, timeout=300), u.rsplit("/", 1)[-1]))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                logger.warning(f"refresh[{source_id}]: download failed for {u} ({e})")
        if not payloads:
            return {"source": source_id, "ok": False, "error": "all downloads failed"}

    # Effective date precedence: explicit arg > the file's own quarter
    # (derived by the resolver from the filename) > today.
    eff = effective_from or resolved_eff or date.today().isoformat()

    parser = P.PARSERS.get(src.parser)
    if not parser:
        return {"source": source_id, "ok": False, "error": f"no parser {src.parser}"}

    # MCD articles go through the coverage loader (two tables). The live
    # bulk export is a nested relational zip (parse_mcd_export); a plain
    # CSV payload (offline ingest / tests) uses the flat-file parser.
    if source_id == "mcd_articles":
        raw, _name = payloads[0]
        if raw[:2] == b"PK":
            articles = P.parse_mcd_export(raw)
        else:
            articles = parser(raw.decode("latin-1", errors="replace"), eff)
        if not articles:
            # Zero parsed articles means a wrong/changed payload, not that
            # CMS published nothing — fail loudly instead of recording a
            # successful no-op refresh.
            logger.warning(f"refresh[{source_id}]: 0 articles parsed — payload format "
                           f"unrecognized or empty")
            return {"source": source_id, "ok": False,
                    "error": "0 articles parsed — payload format unrecognized; "
                             "check the export URL or ingest offline via --file"}
        if not dry_run:
            store.load_coverage_articles(articles)
            _write_coverage_cache(articles, eff)
        return {"source": source_id, "ok": True, "articles": len(articles), "dry_run": dry_run}

    # Tabular sources — a quarter's edit set may span multiple files
    # (NCCI PTP ships f1/f2/...); parse and combine them into ONE snapshot,
    # since ingest_snapshot is keyed by (source_id, effective_from) and a
    # second file for the same quarter would otherwise no-op as "already
    # present".
    rows, cols, file_names = [], None, []
    for raw, name in payloads:
        text = _payload_text(src, raw)
        r, cols = parser(text, eff)
        rows.extend(r)
        file_names.append(name)
    if not rows:
        # A 0-row parse previously ingested nothing but still wrote a
        # data_source_version provenance row, making the refresh look done.
        # This must surface as a failure the systemd journal shows red.
        logger.warning(f"refresh[{source_id}]: 0 rows parsed — payload is likely a landing "
                       f"page or an unrecognized format")
        return {"source": source_id, "ok": False,
                "error": "0 rows parsed — payload is likely a landing page or an "
                         "unrecognized format; check the source URL or ingest offline via --file"}
    if dry_run:
        return {"source": source_id, "ok": True, "parsed_rows": len(rows),
                "files": file_names, "effective_from": eff, "dry_run": True}

    if src.target_table in _HISTORY_TABLES:
        n = store.ingest_snapshot(src.target_table, cols, rows, source_id, eff,
                                  file_name=", ".join(file_names))
    else:  # reference tables (pos) — replace in place
        ph = ",".join("?" * len(cols))
        store.conn.executemany(
            f"INSERT OR REPLACE INTO {src.target_table} ({','.join(cols)}) VALUES ({ph})", rows
        )
        store.conn.commit()
        n = len(rows)
    return {"source": source_id, "ok": True, "ingested_rows": n, "effective_from": eff,
            "files": file_names}


def refresh_all(store: ComplianceDataStore, *, month: int | None = None,
                dry_run: bool = False) -> list[dict]:
    m = month or date.today().month
    results = []
    for src in due_sources(m):
        logger.info(f"refresh: {src.id} ({src.cadence}, due in month {m})")
        results.append(refresh_source(store, src.id, dry_run=dry_run))
    return results
