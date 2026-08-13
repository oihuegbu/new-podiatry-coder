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

from app.release.source_manifest import declared_source_path

# The registry of what may be released WITHOUT a human is itself a declared release
# source: its bytes are content-addressed by the manifest the certificate binds, so a
# substituted or truncated scope file is visible rather than silent. Resolved through
# the declaration rather than composed here. (Codex F6-R5-A, round 6.)
DEFAULT_SCOPE_REGISTRY = declared_source_path("autonomous_scopes")
_MIN_SIGNING_KEY_BYTES = 32


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
    values = {str(v) for v in (allowed or [])}
    actual = str(actual or "")
    return bool(actual) and (actual in values or "*" in values)


def approved_scope(context: dict, on_date: date | None = None
                   ) -> tuple[dict | None, str]:
    key = os.getenv("AUTONOMOUS_SCOPE_SIGNING_KEY", "")
    if len(key.encode()) < _MIN_SIGNING_KEY_BYTES:
        return None, "autonomous scope signing key is absent or shorter than 32 bytes"
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
        # Every release-relevant operational dimension is explicit. A scope
        # may deliberately use "*", but an absent encounter value never
        # matches a wildcard and can never silently broaden authorization.
        required = {
            "payer_kinds": context.get("payer_kind"),
            "payer_ids": context.get("payer_id"),
            "plans": context.get("plan"),
            "provider_specialties": context.get("provider_specialty"),
            "rendering_npis": context.get("rendering_npi"),
            "billing_npis": context.get("billing_npi"),
            "places_of_service": context.get("place_of_service"),
            "jurisdictions": context.get("jurisdiction"),
            "note_categories": context.get("note_category"),
            "claim_families": context.get("claim_family"),
        }
        if all(_matches(dimensions.get(name), value)
               for name, value in required.items()):
            return scope, ""
    return None, "encounter is outside every authenticated autonomous scope"
