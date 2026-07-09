"""Clearinghouse adapter — Stedi real-time eligibility (270/271).

The agents depend on this *interface*, never on Stedi directly, so the backend
is swappable (Stedi sandbox today → the client's clearinghouse or a FHIR payer
API later) with no change to the compliance logic.

Config: STEDI_API_KEY + STEDI_ELIGIBILITY_URL in .env. If the key is absent the
client reports `configured=False` and agents degrade gracefully (non-blocking).
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field

from app.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_URL = "https://healthcare.us.stedi.com/2024-04-01/change/medicalnetwork/eligibility/v3"


@dataclass
class EligibilityResult:
    configured: bool = False          # was the clearinghouse reachable/configured
    checked: bool = False             # did we actually perform a check
    active: bool | None = None        # coverage active? None = unknown
    errors: list[str] = field(default_factory=list)
    raw: dict | None = None


class ClearinghouseAdapter:
    """Interface every clearinghouse/eligibility backend implements."""
    def is_configured(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def check_eligibility(self, *, payer_id, member_id, first_name, last_name,
                          date_of_birth, npi, service_type_codes=None) -> EligibilityResult:
        raise NotImplementedError


class StediAdapter(ClearinghouseAdapter):
    def __init__(self, api_key: str | None = None, url: str | None = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("STEDI_API_KEY", "")
        self.url = url or os.getenv("STEDI_ELIGIBILITY_URL", _DEFAULT_URL)
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def check_eligibility(self, *, payer_id, member_id, first_name, last_name,
                          date_of_birth, npi, service_type_codes=None) -> EligibilityResult:
        if not self.is_configured():
            return EligibilityResult(configured=False)

        body = {
            "tradingPartnerServiceId": payer_id,
            "encounter": {"serviceTypeCodes": service_type_codes or ["30"]},
            "provider": {"organizationName": "PROVIDER", "npi": npi or "1999999984"},
            "subscriber": {
                "dateOfBirth": (date_of_birth or "").replace("-", ""),
                "firstName": first_name or "",
                "lastName": last_name or "",
                "memberId": member_id or "",
            },
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Authorization": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return EligibilityResult(configured=True, checked=True,
                                     errors=[f"HTTP {e.code}: {e.reason}"])
        except Exception as e:  # network/timeout
            return EligibilityResult(configured=True, checked=True, errors=[str(e)])

        errors = [str(x.get("description", x)) for x in (data.get("errors") or [])]
        # Active coverage is signalled by benefit info with an "active" indicator.
        benefits = data.get("benefitsInformation") or []
        active = None
        if benefits:
            active = any(b.get("code") in ("1", "A") or
                         "active" in str(b.get("name", "")).lower() for b in benefits)
        elif errors:
            active = None  # could not determine due to errors
        return EligibilityResult(configured=True, checked=True, active=active,
                                 errors=errors, raw=data)


def default_adapter() -> ClearinghouseAdapter:
    return StediAdapter()
