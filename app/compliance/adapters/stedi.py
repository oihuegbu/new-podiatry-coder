"""Clearinghouse adapter — Stedi real-time eligibility (270/271) and
professional claim submission (837P).

The agents depend on this *interface*, never on Stedi directly, so the backend
is swappable (Stedi sandbox today → the client's clearinghouse or a FHIR payer
API later) with no change to the compliance logic.

Config: STEDI_API_KEY + STEDI_ELIGIBILITY_URL / STEDI_CLAIMS_URL in .env. If
the key is absent the client reports `configured=False` and callers degrade
gracefully (eligibility agents stay non-blocking; claim submission refuses to
run rather than pretending it did).
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
_DEFAULT_CLAIMS_URL = "https://healthcare.us.stedi.com/2024-04-01/change/medicalnetwork/professionalclaims/v3/submission"


@dataclass
class EligibilityResult:
    configured: bool = False          # was the clearinghouse reachable/configured
    checked: bool = False             # did we actually perform a check
    active: bool | None = None        # coverage active? None = unknown
    errors: list[str] = field(default_factory=list)
    raw: dict | None = None


@dataclass
class SubmissionResult:
    configured: bool = False          # clearinghouse credentials present
    submitted: bool = False           # the claim was accepted for processing
    claim_reference: str | None = None  # clearinghouse/payer claim identifier
    errors: list[str] = field(default_factory=list)
    raw: dict | None = None


class ClearinghouseAdapter:
    """Interface every clearinghouse/eligibility backend implements."""
    def is_configured(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def check_eligibility(self, *, payer_id, member_id, first_name, last_name,
                          date_of_birth, npi, service_type_codes=None) -> EligibilityResult:
        raise NotImplementedError

    def submit_claim(self, claim_payload: dict) -> SubmissionResult:
        """Transmit a fully-built professional claim (clearinghouse-native
        JSON shape). Implementations must never mutate the payload."""
        raise NotImplementedError


class StediAdapter(ClearinghouseAdapter):
    def __init__(self, api_key: str | None = None, url: str | None = None,
                 claims_url: str | None = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("STEDI_API_KEY", "")
        self.url = url or os.getenv("STEDI_ELIGIBILITY_URL", _DEFAULT_URL)
        self.claims_url = claims_url or os.getenv("STEDI_CLAIMS_URL",
                                                  _DEFAULT_CLAIMS_URL)
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def submit_claim(self, claim_payload: dict) -> SubmissionResult:
        if not self.is_configured():
            return SubmissionResult(configured=False,
                                    errors=["STEDI_API_KEY not configured"])
        req = urllib.request.Request(
            self.claims_url, data=json.dumps(claim_payload).encode(),
            headers={"Authorization": self.api_key,
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                body = json.loads(e.read().decode())
                detail = f" — {body.get('message', body)}"
            except Exception:
                pass
            return SubmissionResult(
                configured=True,
                errors=[f"HTTP {e.code}: {e.reason}{detail}"])
        except Exception as e:  # network/timeout
            return SubmissionResult(configured=True, errors=[str(e)])

        errors = [str(x.get("description", x))
                  for x in (data.get("errors") or [])]
        # Stedi's professional-claims response carries the clearinghouse
        # trace/claim reference under claimReference; treat presence of a
        # reference with no errors as acceptance.
        ref_block = data.get("claimReference") or {}
        reference = (ref_block.get("correlationId")
                     or ref_block.get("patientControlNumber")
                     or data.get("controlNumber"))
        return SubmissionResult(
            configured=True, submitted=bool(reference) and not errors,
            claim_reference=str(reference) if reference else None,
            errors=errors, raw=data)

    def check_eligibility(self, *, payer_id, member_id, first_name, last_name,
                          date_of_birth, npi, service_type_codes=None) -> EligibilityResult:
        if not self.is_configured():
            return EligibilityResult(configured=False)

        body = {
            "tradingPartnerServiceId": payer_id,
            "encounter": {"serviceTypeCodes": service_type_codes or ["30"]},
            "provider": {"organizationName": "PROVIDER", "npi": npi or "1999999984"},
            "subscriber": {
                "dateOfBirth": date_of_birth or "",
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
            detail = ""
            try:
                body = json.loads(e.read().decode())
                detail = f" — {body.get('message', body)}"
            except Exception:
                pass
            return EligibilityResult(configured=True, checked=True,
                                     errors=[f"HTTP {e.code}: {e.reason}{detail}"])
        except Exception as e:  # network/timeout
            return EligibilityResult(configured=True, checked=True, errors=[str(e)])

        errors = [str(x.get("description", x)) for x in (data.get("errors") or [])]
        # Active coverage is signalled by benefit info with an "active"
        # indicator. A 271 response commonly carries multiple benefit lines
        # for DIFFERENT service types (medical, dental, vision, ...) in one
        # payload — scanning all of them with any() would report "active"
        # from an unrelated service type even when the one actually
        # requested (service_type_codes, default ["30"]) is inactive. Each
        # benefit line that names its own serviceTypeCodes (X12 EB03) is
        # filtered to the requested set first; a line with no
        # serviceTypeCodes of its own (general coverage info) is kept, since
        # there's nothing to filter it against.
        requested_types = set(service_type_codes or ["30"])
        benefits = data.get("benefitsInformation") or []
        relevant = [
            b for b in benefits
            if not b.get("serviceTypeCodes") or set(b.get("serviceTypeCodes")) & requested_types
        ]
        active = None
        if relevant:
            active = any(b.get("code") in ("1", "A") or
                         "active" in str(b.get("name", "")).lower() for b in relevant)
        elif errors:
            active = None  # could not determine due to errors
        return EligibilityResult(configured=True, checked=True, active=active,
                                 errors=errors, raw=data)


def default_adapter() -> ClearinghouseAdapter:
    return StediAdapter()
