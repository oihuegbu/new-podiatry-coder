"""Human-approved, HMAC-authenticated autonomous operating scopes.

The registry contains operational dimensions, never medical code families.
An unsigned or unverifiable scope is inert.  The signing key is supplied by
the deployment environment and is intentionally absent from the repository.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import date, datetime
from pathlib import Path

from app.core.config import DATA_DIR

DEFAULT_SCOPE_REGISTRY = DATA_DIR / "release" / "autonomous_scopes.json"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


def scope_fingerprint(scope: dict) -> str:
    unsigned = {k: v for k, v in scope.items() if k != "signature"}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def sign_scope(scope: dict, key: str) -> str:
    return "hmac-sha256:" + hmac.new(
        key.encode(), scope_fingerprint(scope).encode(), hashlib.sha256
    ).hexdigest()


def _registry_path() -> Path:
    return Path(os.getenv("AUTONOMOUS_SCOPE_REGISTRY",
                           str(DEFAULT_SCOPE_REGISTRY)))


def _matches(allowed, actual: str) -> bool:
    return bool(allowed) and str(actual or "") in {str(v) for v in allowed}


def approved_scope(context: dict, on_date: date | None = None
                   ) -> tuple[dict | None, str]:
    key = os.getenv("AUTONOMOUS_SCOPE_SIGNING_KEY", "")
    if not key:
        return None, "autonomous scope signing key is not configured"
    try:
        raw = json.loads(_registry_path().read_text())
    except Exception as exc:
        return None, f"autonomous scope registry is unavailable ({exc})"
    today = on_date or date.today()
    for scope in raw.get("scopes", []):
        if not scope.get("approved") or not scope.get("approved_by"):
            continue
        expected = sign_scope(scope, key)
        if not hmac.compare_digest(str(scope.get("signature") or ""),
                                   expected):
            continue
        try:
            start = datetime.fromisoformat(str(scope["effective_from"])).date()
            end = datetime.fromisoformat(str(scope["effective_to"])).date()
        except Exception:
            continue
        if not start <= today <= end:
            continue
        dimensions = scope.get("dimensions") or {}
        required = {
            "payer_kinds": context.get("payer_kind"),
            "provider_specialties": context.get("provider_specialty"),
            "places_of_service": context.get("place_of_service"),
            "note_categories": context.get("note_category"),
            "claim_families": context.get("claim_family"),
        }
        if all(_matches(dimensions.get(name), value)
               for name, value in required.items()):
            return scope, ""
    return None, "encounter is outside every authenticated autonomous scope"
