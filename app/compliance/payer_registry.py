"""Payer identification — maps a note's free-text insurance field to a
canonical payer_id, and parses out member/group identifiers.

The `insurance` field on patient_metadata is LLM-Vision-extracted prose with
no fixed format (e.g. "Blue Cross Blue Shield of Florida - Member/Policy ID:
BCB786579303, Group Number: GRP24592..." vs "Tricare Prime, Member/Policy ID
TRP853573823, Group Number MIL99593" — colon presence and separator style
both vary). This module is the single place that parses it, so every
downstream consumer (prior auth, eligibility/benefits) sees the same
normalized payer regardless of the source note's exact phrasing.

Payer aliases live in data/codes/payers.json — add a payer there, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.release.source_manifest import (DeclaredSourceUnavailable, declared_document,
                                         declared_source_path)

#: Declared identity of the payer registry.  The PATH is not composed here: it is
#: resolved from `app.release.source_manifest`, so the bytes the release certificate
#: attests and the bytes this module parses are provably the same file.
_PAYERS_SOURCE_ID = "payer_registry"

_MEMBER_ID_RE = re.compile(r"Member/Policy ID:?\s*([A-Za-z0-9\-]+)", re.IGNORECASE)
_GROUP_NUM_RE = re.compile(r"Group Number:?\s*([A-Za-z0-9\-]+)", re.IGNORECASE)


@dataclass
class ParsedInsurance:
    raw_text: str
    payer_name: str            # canonical name if matched, else the raw leading text
    payer_id: str | None       # None if no known payer matched — internal key (prior_auth_required)
    stedi_trading_partner_id: str | None  # Stedi's own payer ID namespace — for eligibility (270/271) calls only, NOT the same as payer_id
    is_medicare: bool          # ORIGINAL (FFS) Medicare only — never Medicare Advantage
    member_id: str | None
    group_number: str | None
    kind: str = "unknown"      # medicare_ffs | medicare_advantage | commercial | medicaid | tricare | workers_comp | unknown
    # True where the payer is bound to Original Medicare's coverage floor
    # (FFS itself, and MA plans per 42 CFR 422.101). Coverage-derived checks
    # (LCD medical necessity, routine-foot-care class findings, status-I
    # "not valid for Medicare") key off this — is_medicare stays FFS-only
    # for claims-routing/PA logic that genuinely differs under MA.
    follows_medicare_coverage: bool = False


class PayerRegistryUnavailable(DeclaredSourceUnavailable):
    """The authoritative payer registry could not be read.

    RAISED, never degraded to an empty or last-known-good registry, because "the payer
    registry is unreadable" and "this note names no payer we recognize" are opposite
    conclusions that used to be indistinguishable.  With an empty registry EVERY note's
    insurance text fails to match: `payer_id` None, `is_medicare` False,
    `follows_medicare_coverage` False — i.e. every patient is silently reclassified as an
    unrecognized commercial payer, which changes the coverage floor applied (LCD medical
    necessity, routine-foot-care class findings, Medicare status-I validity) and the
    prior-authorization policy consulted.  Keeping the LAST-KNOWN-GOOD registry across a
    corrupt write is the same defect one step removed: the claim is then composed against
    aliases nobody can point to a certified file for.  (Codex F6-R5-A, round 6.)
    """


_payers_cache: list[dict] = []
_payers_mtime: int = -1


def _load_payers() -> list[dict]:
    """mtime-cached read — editing payers.json (new payer, new alias) takes
    effect on the next parse without a process restart. Previously loaded
    once at import, so alias changes silently didn't apply until restart.

    FAIL-CLOSED: absent, unparseable, or publishing no payers all raise
    `PayerRegistryUnavailable` (see above). The cache is only ever advanced by a read
    that fully succeeded, so a corrupt write cannot leave the process serving a registry
    whose mtime says it is current."""
    global _payers_cache, _payers_mtime
    path = declared_source_path(_PAYERS_SOURCE_ID)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError as exc:
        raise PayerRegistryUnavailable(
            f"authoritative payer registry unavailable at {path}: {exc}") from exc
    if mtime != _payers_mtime:
        payers = declared_document(_PAYERS_SOURCE_ID, PayerRegistryUnavailable).get("payers")
        if not isinstance(payers, list) or not payers:
            raise PayerRegistryUnavailable(
                f"authoritative payer registry at {path} publishes no non-empty "
                f"'payers' list")
        _payers_cache, _payers_mtime = payers, mtime
    return _payers_cache


def parse_insurance_text(raw: str) -> ParsedInsurance:
    raw = (raw or "").strip()
    if not raw:
        return ParsedInsurance(raw_text=raw, payer_name="", payer_id=None,
                               stedi_trading_partner_id=None, is_medicare=False,
                               member_id=None, group_number=None)

    lower = raw.lower()
    # Longest matching alias wins across ALL payers — not first-file-order
    # match. With substring matching, order-based resolution misclassified
    # "Humana Medicare Advantage" as FFS Medicare because the bare
    # "medicare" alias matched before any more specific entry was consulted;
    # the longest-alias rule makes the most specific alias authoritative
    # regardless of where its payer sits in the file.
    matched, best_len = None, 0
    for payer in _load_payers():
        for alias in payer.get("aliases", []):
            if alias in lower and len(alias) > best_len:
                matched, best_len = payer, len(alias)

    m = _MEMBER_ID_RE.search(raw)
    member_id = m.group(1) if m else None
    g = _GROUP_NUM_RE.search(raw)
    group_number = g.group(1) if g else None

    if matched:
        return ParsedInsurance(
            raw_text=raw, payer_name=matched["canonical_name"], payer_id=matched["payer_id"],
            stedi_trading_partner_id=matched.get("stedi_trading_partner_id"),
            is_medicare=bool(matched.get("is_medicare", False)),
            member_id=member_id, group_number=group_number,
            kind=matched.get("kind", "unknown"),
            follows_medicare_coverage=bool(matched.get("follows_medicare_coverage", False)),
        )

    # No known payer matched — pass through the leading text (before the first
    # separator) rather than silently defaulting to Medicare, which would
    # apply the wrong payer's coverage/PA rules to an unrecognized insurer.
    leading = re.split(r"[,\-]", raw, maxsplit=1)[0].strip()
    return ParsedInsurance(
        raw_text=raw, payer_name=leading or raw, payer_id=None,
        stedi_trading_partner_id=None, is_medicare=False,
        member_id=member_id, group_number=group_number,
    )
