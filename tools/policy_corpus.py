#!/usr/bin/env python3
"""Policy corpus — the authoritative PROSE sources, as queryable artifacts.

The pipeline's code-shaped facts (MUE values, NCCI PTP pairs, global
periods, descriptors) are already authoritative DATA the system queries.
Its policy-shaped facts (coverage pathways, class-findings thresholds,
documentation principles) were only ever ATTESTED: the adjudicator cited
"Medicare Benefit Policy Manual Ch. 15 §290" and the system took the
citation on faith. A citation is not a lookup — a model can misname,
misremember, or invent a passage, and nothing would catch it.

This module closes that gap the same way data/codes/*.json closed the
code-data gap:

  fetch      download the real public documents (CMS manuals, ICD-10-CM
             Official Guidelines, NCCI Policy Manual) from the manifest,
             extract their text, and store it under data/policy/ with
             provenance (URL, sha256, fetch date, page count).
  verify     deterministically check that a verbatim quote occurs in a
             stored source (normalized containment, with a bounded
             token-overlap tolerance for PDF-extraction artifacts).
             tools/coder_adjudicator.py calls this on every verdict's
             authority_quote: corpus present + quote given + no match =
             the verdict pass is VOIDED (fail closed) — the model can
             still misread a real passage, but it can no longer invent
             one.

Attestation tiers derived here ride on recorded targets:
  document_quoted   the verdict's authority_quote was verified against a
                    stored source — strongest prose grounding available
                    without a human.
  attested_only     no quote was given (or no source matched a named
                    citation) while the corpus was available — the
                    weakest tier; actuation gates refuse to anchor rules
                    on such targets.
  unverified        the corpus was not available when the verdict was
                    recorded — grandfathered (nothing could have been
                    looked up), anchoring allowed, logged.

Upkeep is autonomous: ensure() fetches missing sources and re-checks
stored ones against upstream every POLICY_CORPUS_MAX_AGE_DAYS (default
30), and every adjudication-bearing entry point calls it before its
first verdict — no cron job, no manual fetch. The CLI remains for
inspection/debugging:
  python tools/policy_corpus.py fetch [source_id ...]
  python tools/policy_corpus.py ensure
  python tools/policy_corpus.py verify "<quote>"
  python tools/policy_corpus.py status
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from app.core.config import DATA_DIR  # noqa: E402

POLICY_DIR = DATA_DIR / "policy"
MANIFEST_PATH = POLICY_DIR / "manifest.json"

# The default manifest: public CMS/CDC documents the adjudicator already
# cites by name. Editable — adding an LCD or another manual chapter is a
# manifest entry plus `fetch`, never code. URLs are pinned to the
# documents' stable CMS locations; a moved document fails loudly at fetch
# time (never silently at verify time).
_DEFAULT_MANIFEST = [
    {
        "id": "mbpm_ch15",
        "title": "Medicare Benefit Policy Manual, Chapter 15 — Covered "
                 "Medical and Other Health Services (incl. §290 Foot Care)",
        "url": "https://www.cms.gov/Regulations-and-Guidance/Guidance/"
               "Manuals/Downloads/bp102c15.pdf",
        "type": "pdf",
    },
    {
        "id": "icd10cm_guidelines",
        "title": "ICD-10-CM Official Guidelines for Coding and Reporting",
        "url": "https://www.cms.gov/files/document/"
               "fy-2026-icd-10-cm-coding-guidelines.pdf",
        "type": "pdf",
    },
    {
        "id": "ncci_policy_manual",
        "title": "Medicare NCCI Policy Manual (all chapters, "
                 "effective Jan. 1, 2026)",
        "url": "https://www.cms.gov/files/document/"
               "2026-ncci-medicare-policy-manual-all-chapters.pdf",
        "type": "pdf",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        try:
            m = json.loads(MANIFEST_PATH.read_text())
            if isinstance(m, list) and m:
                return m
        except Exception as exc:
            logger.warning(f"policy manifest unreadable ({exc}) — "
                           f"using defaults")
    return list(_DEFAULT_MANIFEST)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _extract_pdf_text(data: bytes) -> tuple[str, int]:
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for pg in pdf.pages:
            pages.append(pg.extract_text() or "")
    return "\n".join(pages), len(pages)


def _extract_zip_text(data: bytes) -> tuple[str, int]:
    """Concatenated text of every PDF in the archive (the NCCI Policy
    Manual ships as one zip of chapter PDFs)."""
    texts, total_pages = [], 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".pdf"):
                continue
            try:
                t, n = _extract_pdf_text(zf.read(name))
                texts.append(f"\n\n===== {name} =====\n\n{t}")
                total_pages += n
            except Exception as exc:
                logger.warning(f"  {name}: extraction failed ({exc}) — "
                               f"skipped")
    return "".join(texts), total_pages


def fetch(source_ids: list[str] | None = None) -> dict:
    """Download and extract every manifest source (or the named ones).
    Idempotent by content hash: an unchanged document is re-verified, not
    re-extracted. Failures are per-source and loud — a missing document
    means quotes against it stay unverifiable, never silently verified."""
    import requests
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps(_DEFAULT_MANIFEST, indent=2))
    stats = {"fetched": [], "unchanged": [], "failed": {}}
    for src in manifest():
        sid = src["id"]
        if source_ids and sid not in source_ids:
            continue
        try:
            logger.info(f"Fetching {sid}: {src['url']}")
            resp = requests.get(src["url"], timeout=180, headers={
                "User-Agent": "podiatry-coder-policy-corpus/1.0"})
            resp.raise_for_status()
            digest = hashlib.sha256(resp.content).hexdigest()
            meta_path = POLICY_DIR / f"{sid}.meta.json"
            txt_path = POLICY_DIR / f"{sid}.txt"
            if meta_path.exists() and txt_path.exists():
                old = json.loads(meta_path.read_text())
                if old.get("sha256") == digest:
                    # touch last_checked so the staleness clock resets —
                    # without this an upstream-unchanged source stays
                    # "stale by age" forever and re-downloads on every
                    # ensure() pass
                    old["last_checked"] = _now()
                    meta_path.write_text(json.dumps(old, indent=2))
                    stats["unchanged"].append(sid)
                    logger.info(f"  {sid}: unchanged (sha256 match)")
                    continue
            if src.get("type") == "zip":
                text, pages = _extract_zip_text(resp.content)
            elif src.get("type") == "pdf":
                text, pages = _extract_pdf_text(resp.content)
            else:  # html/txt — strip tags crudely; policy prose survives
                text = re.sub(r"<[^>]+>", " ", resp.text)
                pages = 1
            if len(text) < 5000:
                raise ValueError(f"extracted only {len(text)} chars — "
                                 f"refusing to store a stub")
            # atomic writes: ensure() runs unattended from the pipeline,
            # possibly from concurrent workers — a reader must see the
            # old complete source or the new one, never a torn file
            tmp = txt_path.with_suffix(".txt.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(txt_path)
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "id": sid, "title": src.get("title", ""),
                "url": src["url"], "sha256": digest,
                "fetched_at": _now(), "last_checked": _now(),
                "pages": pages, "chars": len(text),
            }, indent=2))
            tmp.replace(meta_path)
            stats["fetched"].append(sid)
            logger.info(f"  {sid}: stored ({pages} pages, "
                        f"{len(text)} chars)")
        except Exception as exc:
            stats["failed"][sid] = str(exc)
            logger.warning(f"  {sid}: FETCH FAILED ({exc}) — quotes "
                           f"against this source stay unverifiable")
    return stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_norm_cache: dict[str, tuple[float, str, set]] = {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _sources() -> list[tuple[str, str, set]]:
    """[(source_id, normalized text, token set)] for every stored source,
    cached by file mtime."""
    out = []
    if not POLICY_DIR.exists():
        return out
    for f in sorted(POLICY_DIR.glob("*.txt")):
        sid = f.stem
        try:
            mtime = f.stat().st_mtime
            cached = _norm_cache.get(sid)
            if cached and cached[0] == mtime:
                out.append((sid, cached[1], cached[2]))
                continue
            text = _norm(f.read_text(encoding="utf-8"))
            toks = set(text.split())
            _norm_cache[sid] = (mtime, text, toks)
            out.append((sid, text, toks))
        except Exception as exc:
            logger.warning(f"policy source {f.name} unreadable: {exc}")
    return out


def corpus_available() -> bool:
    """True when at least one policy source is stored — the switch that
    turns quote verification (and attested_only demotion) on."""
    return bool(_sources())


def _frag_matches(nf: str, text: str, toks: set) -> bool:
    """One normalized quote fragment against one normalized source:
    containment, or a bounded extraction-artifact tolerance — >=85% of
    the fragment's tokens present AND >=60% of its ADJACENT WORD PAIRS
    occurring verbatim in the source. The pair requirement is what
    separates a real passage with a mangled word (loses only the pairs
    around the artifact) from an invented paraphrase assembled out of
    common policy words (tokens all present individually, adjacencies
    mostly absent)."""
    if nf in text:
        return True
    ftoks = nf.split()
    if not ftoks:
        return True
    if sum(1 for t in ftoks if t in toks) / len(ftoks) < 0.85:
        return False
    pairs = [f"{a} {b}" for a, b in zip(ftoks, ftoks[1:])]
    if not pairs:
        return False
    return sum(1 for p in pairs if p in text) / len(pairs) >= 0.6


def verify_quote(quote: str, source_hint: str = "") -> dict:
    """Deterministically locate a verbatim policy quote in the stored
    corpus. Ellipses split the quote into fragments; each fragment must
    occur in ONE source — normalized containment, or the bounded
    extraction-artifact tolerance of _frag_matches (token AND adjacency
    overlap; a real quote survives a mangled word, an invention fails).
    Returns {verified, source_id, why}. A short quote (< 5 content
    tokens) is too weak to verify anything and returns verified=False."""
    quote = str(quote or "").strip()
    srcs = _sources()
    if not srcs:
        return {"verified": False, "source_id": "",
                "why": "no policy corpus stored"}
    if not quote:
        return {"verified": False, "source_id": "", "why": "empty quote"}
    frags = [f for f in re.split(r"\.\.\.|\u2026", quote) if _norm(f)]
    if not frags:
        return {"verified": False, "source_id": "", "why": "empty quote"}
    all_toks = [t for f in frags for t in _norm(f).split()]
    if len(all_toks) < 5:
        return {"verified": False, "source_id": "",
                "why": "quote too short to verify (< 5 content tokens)"}

    hint = _norm(source_hint)
    ordered = sorted(srcs, key=lambda s: 0 if (hint and s[0] in hint)
                     else 1)
    for sid, text, toks in ordered:
        if all(_frag_matches(_norm(f), text, toks) for f in frags):
            return {"verified": True, "source_id": sid, "why": ""}
    return {"verified": False, "source_id": "",
            "why": "quote not found in any stored policy source"}


def attest(item: dict) -> str:
    """The attestation tier of one adjudication item:
      document_quoted   quote given and verified in a stored source —
                        the strongest prose grounding without a human
      data_backed       the item declares authority_basis
                        "reference_data": derivable from the case file's
                        own reference data (descriptors, MDM rows,
                        PTP/MUE edits), which the deterministic gates
                        re-verify against the data itself — no prose
                        quote applies. The declaration is the model's,
                        but its decisions still had to be unanimous
                        across independent passes, and the tier is
                        recorded so audits can sample it.
      attested_only     corpus available and the item rests on prose
                        policy with no verifiable quote (verdict-level
                        voiding for FABRICATED quotes happens upstream —
                        this tier is for the merely unquoted). Never
                        anchors actuation.
      unverified        no corpus stored — nothing could be looked up
    A verified quote outranks the basis declaration (checked first)."""
    quote = str(item.get("authority_quote") or "").strip()
    if not corpus_available():
        return "unverified"
    if quote:
        res = verify_quote(quote, str(item.get("authority") or ""))
        if res["verified"]:
            return "document_quoted"
    if str(item.get("authority_basis") or "").strip().lower() == \
            "reference_data" and not quote:
        return "data_backed"
    return "attested_only"


# Autonomous corpus upkeep: every adjudication-bearing entry point calls
# ensure() before its first verdict. 0 disables (air-gapped/test envs).
AUTOFETCH = os.getenv("POLICY_CORPUS_AUTOFETCH", "1")
# Re-check a stored source against its upstream URL after this many days
# (CMS revises manuals annually/ad hoc; the re-check is a download + sha
# compare, and an unchanged document just resets the clock).
MAX_AGE_DAYS = int(os.getenv("POLICY_CORPUS_MAX_AGE_DAYS", "30"))


def _stale_ids(max_age_days: int) -> list[str]:
    """Manifest sources that are missing from the store, or whose last
    successful upstream check is older than max_age_days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    out = []
    for src in manifest():
        sid = src["id"]
        meta_path = POLICY_DIR / f"{sid}.meta.json"
        txt_path = POLICY_DIR / f"{sid}.txt"
        if not meta_path.exists() or not txt_path.exists():
            out.append(sid)
            continue
        try:
            meta = json.loads(meta_path.read_text())
            checked = str(meta.get("last_checked")
                          or meta.get("fetched_at") or "")
            if datetime.fromisoformat(checked) < cutoff:
                out.append(sid)
        except Exception:
            out.append(sid)  # unreadable provenance = re-fetch
    return out


def ensure(max_age_days: int = MAX_AGE_DAYS) -> dict:
    """Bring the corpus up to date without human involvement: fetch every
    manifest source that is missing or overdue for an upstream re-check.
    Idempotent and cheap when the store is fresh (a few stat/JSON reads,
    no network). NEVER raises — quote verification degrades gracefully
    against whatever IS stored, and a source that cannot be fetched is a
    loud per-source log line plus 'quotes against it stay unverifiable',
    exactly the fail-closed posture verify_quote already has. Called by
    the adjudication-bearing entry points (unanimity loop, finalize
    scope, audit-convergence loop, run.py batch driver) so the ground-
    truth protection is active from the first verdict of every run."""
    if AUTOFETCH != "1":
        return {"skipped": "POLICY_CORPUS_AUTOFETCH=0"}
    try:
        stale = _stale_ids(max_age_days)
        if not stale:
            return {"fresh": True}
        logger.info(f"policy corpus: {len(stale)} source(s) missing/"
                    f"stale — fetching: {', '.join(stale)}")
        return fetch(stale)
    except Exception as exc:
        logger.warning(f"policy corpus ensure failed ({exc}) — "
                       f"verification continues against the stored "
                       f"sources only")
        return {"error": str(exc)}


def catalog() -> list[str]:
    """Titles of the STORED sources — what a prompt can truthfully tell
    the adjudicator is quotable. Order follows the manifest."""
    stored = {f.stem for f in POLICY_DIR.glob("*.txt")} \
        if POLICY_DIR.exists() else set()
    return [str(s.get("title") or s["id"]) for s in manifest()
            if s["id"] in stored]


def status() -> dict:
    out = {"corpus_available": corpus_available(), "sources": []}
    for src in manifest():
        meta_path = POLICY_DIR / f"{src['id']}.meta.json"
        entry = {"id": src["id"], "stored": meta_path.exists()}
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                entry.update({k: m.get(k) for k in
                              ("fetched_at", "pages", "chars", "sha256")})
            except Exception:
                pass
        out["sources"].append(entry)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",
                   choices=["fetch", "ensure", "verify", "status"])
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    if a.command == "fetch":
        print(json.dumps(fetch(a.args or None), indent=2))
    elif a.command == "ensure":
        print(json.dumps(ensure(), indent=2))
    elif a.command == "verify":
        print(json.dumps(verify_quote(" ".join(a.args)), indent=2))
    else:
        print(json.dumps(status(), indent=2))
