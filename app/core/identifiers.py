"""Shared deterministic validation for claim-party identifiers."""

from __future__ import annotations

import re


_NPI_RE = re.compile(r"^\d{10}$")


def is_valid_npi(value) -> bool:
    """Validate the CMS NPI check digit (ISO 7812 Luhn with 80840 prefix)."""
    npi = str(value or "").strip()
    if not _NPI_RE.fullmatch(npi):
        return False
    payload = "80840" + npi[:-1]
    total = 0
    for index, char in enumerate(reversed(payload)):
        digit = int(char) * (2 if index % 2 == 0 else 1)
        total += digit // 10 + digit % 10
    return str((10 - total % 10) % 10) == npi[-1]
