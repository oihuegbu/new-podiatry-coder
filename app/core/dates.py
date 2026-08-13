"""Date parsing for values that arrive as free prose from a clinical document.

Pure text processing — no medical-code, coverage, or claim semantics — which is
why it lives in `app.core` rather than inside a pipeline package: BOTH the
retired `app.pipeline` compliance engine and the deployed `claude_coder`
entrypoint (`run.py`) need to turn an extracted `date_of_service` string into a
real `date`, and neither should have to import the other to get it.

`app.compliance.engine` re-exports `_parse_date`/`_parse_dos` from here so its
existing callers are unaffected; this module is the single implementation.
"""
from __future__ import annotations

import re
from datetime import date, datetime

#: Formats seen in real extracted notes, most specific first. Two-digit years are
#: accepted LAST so an unambiguous four-digit year always wins.
_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y", "%m/%d/%y")


def parse_date(raw: str) -> date | None:
    """Parse a documented date string. Returns None when it cannot be parsed —
    never a guess, never today's date: callers treat None as "not documented"
    and fail closed on it."""
    raw = str(raw or "").strip()
    if not raw:
        return None
    for fmt in _FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    return None


def parse_date_of_service(meta: dict) -> date | None:
    """Date of service out of an extraction's patient-metadata mapping."""
    raw = (meta or {}).get("date_of_service") or (meta or {}).get("dos") or ""
    return parse_date(raw)
