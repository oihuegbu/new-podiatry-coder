"""Deterministically materialize an approved autonomous operating scope.

Authorization is a one-time deployment decision, not a per-claim review.  The
practice configuration supplies explicit allowed dimensions and an approver;
the deployment secret supplies authenticity.  No scope is inferred from
example data and no unsigned/wildcard fallback is possible.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from app.core.config import DATA_DIR
from app.core.identifiers import is_valid_npi
from app.release.scope_registry import DEFAULT_SCOPE_REGISTRY, sign_scope


_KEY_ENV = "AUTONOMOUS_SCOPE_SIGNING_KEY"
_MIN_KEY_BYTES = 32
_DIMENSIONS = (
    "payer_kinds", "payer_ids", "plans", "provider_specialties",
    "rendering_npis", "billing_npis", "places_of_service", "jurisdictions",
    "note_categories", "claim_families",
)


class ScopeBootstrapError(RuntimeError):
    pass


def _config_path() -> Path:
    return Path(os.getenv("PRACTICE_CONFIG_PATH",
                           str(DATA_DIR / "practice_config.json")))


def _placeholder(value: str) -> bool:
    text = str(value or "").strip().casefold()
    return not text or any(token in text for token in
                           ("example", "replace_me", "placeholder"))


def _scope_payload(config: dict) -> dict | None:
    autonomy = config.get("autonomy") or {}
    if not autonomy.get("enabled"):
        return None
    approved_by = str(autonomy.get("approved_by") or "").strip()
    if _placeholder(approved_by):
        raise ScopeBootstrapError(
            "autonomy.enabled requires a real approved_by identity")
    approval_reference = str(
        autonomy.get("approval_reference") or "").strip()
    if _placeholder(approval_reference):
        raise ScopeBootstrapError(
            "autonomy.enabled requires a real approval_reference")
    try:
        start = date.fromisoformat(str(autonomy["effective_from"]))
        end = date.fromisoformat(str(autonomy["effective_to"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ScopeBootstrapError(
            "autonomy requires valid effective_from/effective_to dates") from exc
    if end < start:
        raise ScopeBootstrapError("autonomy effective_to precedes effective_from")
    dimensions = autonomy.get("dimensions") or {}
    normalized = {}
    for name in _DIMENSIONS:
        values = dimensions.get(name)
        if not isinstance(values, list) or not values:
            raise ScopeBootstrapError(
                f"autonomy dimension {name} must be a non-empty explicit list")
        cleaned = sorted({str(value).strip() for value in values
                          if str(value).strip()})
        if not cleaned:
            raise ScopeBootstrapError(f"autonomy dimension {name} is empty")
        normalized[name] = cleaned

    for name in ("rendering_npis", "billing_npis"):
        invalid = [value for value in normalized[name]
                   if value != "*" and not is_valid_npi(value)]
        if invalid:
            raise ScopeBootstrapError(
                f"autonomy dimension {name} contains an invalid NPI")

    billing_npi = str((config.get("billing_provider") or {}).get("npi") or "")
    if _placeholder(billing_npi) or not is_valid_npi(billing_npi):
        raise ScopeBootstrapError(
            "autonomy cannot start with an absent/example/invalid billing NPI")
    if billing_npi not in normalized["billing_npis"]:
        raise ScopeBootstrapError(
            "autonomy billing_npis must include the configured billing provider NPI")
    return {
        "id": str(autonomy.get("scope_id") or "practice-autonomy-v1"),
        "approved": True,
        "approved_by": approved_by,
        "approval_reference": approval_reference,
        "effective_from": start.isoformat(),
        "effective_to": end.isoformat(),
        "dimensions": normalized,
    }


def bootstrap_scope() -> dict:
    """Write the exact signed scope if autonomy is configured.

    Returns a diagnostic object. An enabled-but-invalid configuration raises
    and stops batch startup; disabled autonomy remains an explicit safe state.
    """
    try:
        config = json.loads(_config_path().read_text())
    except Exception as exc:
        raise ScopeBootstrapError(f"practice configuration unavailable: {exc}") from exc
    scope = _scope_payload(config)
    if scope is None:
        return {"enabled": False, "changed": False,
                "reason": "practice autonomy is disabled"}
    key = os.getenv(_KEY_ENV, "")
    if len(key.encode()) < _MIN_KEY_BYTES:
        raise ScopeBootstrapError(
            f"{_KEY_ENV} is absent or shorter than {_MIN_KEY_BYTES} bytes")
    scope["signature"] = sign_scope(scope, key)
    payload = {"version": 1, "scopes": [scope]}
    path = Path(os.getenv("AUTONOMOUS_SCOPE_REGISTRY",
                           str(DEFAULT_SCOPE_REGISTRY)))
    existing = None
    try:
        existing = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        pass
    if existing == payload:
        return {"enabled": True, "changed": False, "scope_id": scope["id"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)
    return {"enabled": True, "changed": True, "scope_id": scope["id"]}
