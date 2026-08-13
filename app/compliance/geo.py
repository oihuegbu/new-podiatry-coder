"""MAC jurisdiction resolution — which states a coverage policy's issuing
contractor actually adjudicates, and which state a claim belongs to.

LCDs and Billing & Coding Articles are LOCAL policies: each is issued by one
Medicare Administrative Contractor and only governs claims in the states that
contractor serves. The CMS Coverage API dataset identifies each policy's
contractor by NAME (e.g. "CGS Administrators, LLC (MAC - Part A, MAC - Part
B)"), so applicability is resolved name → jurisdiction states via
data/codes/mac_jurisdictions.json (sourced from CMS "Who are the MACs").

The same contractor can hold multiple jurisdictions (Noridian holds JE and
JF); the source data doesn't say which jurisdiction issued a given policy, so
the contractor's FULL service area is used — the finest scope the data
supports, and strictly better than the previous behavior of applying every
MAC's policies nationwide.

DME MAC policies are tagged "(DME MAC)" in the contractor string and resolve
through the DME jurisdiction map (different state groupings than A/B).
"""
from __future__ import annotations

import json
import re

from app.release.source_manifest import declared_source_path

# Declared identity, not a filename composed here -- the jurisdiction map decides which
# contractor's coverage policy applies to a claim. (Codex F6-R5-A, round 6.)
_MAC_FILE = declared_source_path("mac_jurisdictions")

_cache: dict = {}
_cache_mtime: int = -1


def _load() -> dict:
    global _cache, _cache_mtime
    try:
        mtime = _MAC_FILE.stat().st_mtime_ns
    except OSError:
        return _cache
    if mtime != _cache_mtime:
        try:
            with open(_MAC_FILE) as f:
                _cache = json.load(f)
            _cache_mtime = mtime
        except Exception:
            pass  # keep last-known-good map on a partial write/bad JSON
    return _cache


def _contractor_maps() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(A/B map, DME map): lowercase contractor alias → union of states
    across all jurisdictions that contractor holds. Keyed by every alias
    (spellings differ across CMS datasets: 'WPS Insurance Corporation' in
    the Coverage API vs the truncated 'Wisconsin Physicians Service
    Insurance C' in the MCD bulk export)."""
    data = _load()
    ab: dict[str, set[str]] = {}
    dme: dict[str, set[str]] = {}
    for entry in data.get("ab_mac_jurisdictions", []):
        for alias in entry.get("aliases") or [entry["contractor"]]:
            ab.setdefault(alias.lower(), set()).update(entry["states"])
    for entry in data.get("dme_mac_jurisdictions", []):
        for alias in entry.get("aliases") or [entry["contractor"]]:
            dme.setdefault(alias.lower(), set()).update(entry["states"])
    return ab, dme


def contractor_states(contractor: str) -> set[str] | None:
    """States where a policy's issuing contractor(s) adjudicate claims.

    Returns None when applicability can't be narrowed — unknown/absent
    contractor (e.g. refresh-loaded articles without contractor data) — in
    which case the policy must be treated as potentially applicable
    everywhere rather than silently dropped.

    A contractor string can name SEVERAL contractors (joint DME MAC policies
    list every DME MAC that adopted them); the union of all named
    contractors' states is returned. Whether each name resolves through the
    A/B or DME map is decided by the "(DME MAC)" tag in the text window
    between that name and the next one.
    """
    text = " ".join((contractor or "").split()).lower()
    if not text:
        return None
    ab, dme = _contractor_maps()
    # locate every known contractor alias in the string, in order; when
    # aliases overlap at the same position ("noridian" inside "noridian
    # healthcare solutions"), keep only the longest match there
    raw_hits: dict[int, str] = {}
    for name in set(ab) | set(dme):
        for m in re.finditer(re.escape(name), text):
            if len(name) > len(raw_hits.get(m.start(), "")):
                raw_hits[m.start()] = name
    if not raw_hits:
        return None
    # drop hits nested inside a longer hit that starts earlier
    hits: list[tuple[int, str]] = []
    for pos in sorted(raw_hits):
        name = raw_hits[pos]
        if hits and pos < hits[-1][0] + len(hits[-1][1]):
            continue
        hits.append((pos, name))
    states: set[str] = set()
    for i, (pos, name) in enumerate(hits):
        window_end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        window = text[pos:window_end]
        if "dme mac" in window:
            states.update(dme.get(name, set()))
        # "(HHH MAC, MAC - Part A...)" and plain A/B tags both resolve via
        # the A/B map; a name with ONLY a DME tag contributes only DME states.
        if "dme mac" not in window or "part a" in window or "part b" in window:
            states.update(ab.get(name, set()))
    return states or None


# "City, FL 33134" (letterhead) — a 2-letter token is only taken as a state
# when it's a real USPS abbreviation AND followed by a ZIP, so incidental
# all-caps words never match.
_ABBR_ZIP_RE = re.compile(r"\b([A-Z]{2})\s+\d{5}(?:-\d{4})?\b")


def is_state_abbr(value: str) -> bool:
    """Whether value is a real USPS state/territory abbreviation per
    mac_jurisdictions.json's own state-name map — used to validate a
    structured extracted field (e.g. the letterhead's service_facility
    state) before trusting it over free-text inference."""
    return str(value or "").strip().upper() in set(
        _load().get("state_names", {}).values())


def state_from_text(*texts: str) -> str | None:
    """Best-effort claim-state inference from free text (insurance line,
    facility name, note letterhead). Sources tried in order across all
    provided texts: 'ST 12345' abbreviation+ZIP, then full state names
    (longest match wins so 'West Virginia' never resolves as 'Virginia')."""
    names = _load().get("state_names", {})
    abbrs = set(names.values())
    for text in texts:
        for m in _ABBR_ZIP_RE.finditer(text or ""):
            if m.group(1) in abbrs:
                return m.group(1)
    for text in texts:
        low = (text or "").lower()
        best, best_len = None, 0
        for name, abbr in names.items():
            if len(name) > best_len and re.search(rf"\b{re.escape(name)}\b", low):
                best, best_len = abbr, len(name)
        if best:
            return best
    return None
