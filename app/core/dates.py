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

import calendar
import re
from datetime import date, datetime

#: Formats seen in real extracted notes, most specific first. Two-digit years are
#: accepted LAST so an unambiguous four-digit year always wins.
_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y",
            "%m/%d/%y", "%m-%d-%y")


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


# --------------------------------------------------------------------------
# locating a date IN a document (issue #6 F7-R4)
# --------------------------------------------------------------------------
#
# `parse_date` above answers "what date is this string?". A claim also needs the
# other question -- "WHERE in the original document is this date written?" --
# because a date that cannot be pointed at on a page cannot be reconciled
# against an independent reading of that page, and an unreconciled date of
# service silently selects the wrong coverage, the wrong affiliation and the
# wrong effective code edition while every downstream field still populates.
#
# Month names come from the SAME table `datetime.strptime` reads for `%B`/`%b`,
# so the scanner and the parser can never disagree about what "March" means
# under whatever locale the process happens to run in.

_MONTH_NUMBERS: dict[str, int] = {
    name.lower().rstrip("."): index
    for index in range(1, 13)
    for name in (calendar.month_name[index], calendar.month_abbr[index])
    if name
}
_MONTH_ALTERNATION = "|".join(
    sorted((re.escape(name) for name in _MONTH_NUMBERS), key=len, reverse=True))

#: Every written form of a calendar date this project has to be able to point at.
#: Purely syntactic: a match is only a CANDIDATE, and it is kept only when the
#: groups it captured are a real calendar date ("2026-02-30" matches and is then
#: discarded), so this pattern can never assert a date the calendar does not have.
_DATE_LITERAL = re.compile(
    r"(?<![0-9A-Za-z])(?:"
    r"(?P<iso>\d{4}-\d{1,2}-\d{1,2})"
    r"|(?P<numeric>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    r"|(?:(?P<month>" + _MONTH_ALTERNATION + r")\.?\s+(?P<day>\d{1,2})"
    r"(?:st|nd|rd|th)?,?\s+(?P<year>\d{4}))"
    r"|(?:(?P<day2>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month2>" + _MONTH_ALTERNATION + r")"
    r"\.?,?\s+(?P<year2>\d{4}))"
    r")(?![0-9A-Za-z])",
    re.IGNORECASE)


def _date_from_match(match: "re.Match[str]") -> date | None:
    if match.group("iso") or match.group("numeric"):
        return parse_date(match.group(0))
    month = match.group("month") or match.group("month2") or ""
    day = match.group("day") or match.group("day2") or ""
    year = match.group("year") or match.group("year2") or ""
    number = _MONTH_NUMBERS.get(month.lower().rstrip("."))
    if number is None:                                # pragma: no cover - matched form
        return None
    try:
        return date(int(year), number, int(day))
    except ValueError:
        return None


def find_dates(text: str) -> tuple[tuple[int, int, date], ...]:
    """Every calendar date WRITTEN IN `text`, as `(start, end, date)` character
    offsets into that exact string.

    The offsets are the whole point: they are what lets a date be treated as an
    evidence span like any other quotation -- attributed to a page of the original
    document and proven against an independent reading of it. A written form this
    scanner does not recognise yields no offsets and therefore no proof, which
    holds the encounter; it never yields an approximate location.
    """
    out: list[tuple[int, int, date]] = []
    for match in _DATE_LITERAL.finditer(str(text or "")):
        parsed = _date_from_match(match)
        if parsed is not None:
            out.append((match.start(), match.end(), parsed))
    return tuple(out)
