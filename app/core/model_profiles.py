"""Execution profiles for independent medical-coding runs.

Self-consistency with repeated calls to one model is useful for instability
detection, but it is not independent corroboration.  This module makes the
provider/model identity an explicit, persisted input and supplies a profile
schedule that crosses provider boundaries whenever both configured providers
are available.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass

from app.core import config


@dataclass(frozen=True)
class CodingExecutionProfile:
    profile_id: str
    provider: str
    model: str
    independence_domain: str

    def model_dump(self) -> dict:
        return asdict(self)


_ACTIVE: ContextVar[CodingExecutionProfile | None] = ContextVar(
    "coding_execution_profile", default=None)


def _usable_secret(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and "REPLACE_ME" not in text and not text.startswith("test_")


def _validate(raw: dict, position: int) -> CodingExecutionProfile:
    if not isinstance(raw, dict):
        raise ValueError(f"coding profile {position} must be an object")
    provider = str(raw.get("provider") or "").strip().lower()
    if provider not in {"openai", "claude"}:
        raise ValueError(
            f"coding profile {position} has unsupported provider {provider!r}")
    model = str(raw.get("model") or "").strip()
    if not model:
        raise ValueError(f"coding profile {position} requires a model")
    profile_id = str(raw.get("profile_id") or
                     f"{provider}:{model}").strip()
    # Independence is intentionally provider-level. Two model names served by
    # one vendor are useful diversity, but share enough training, serving, and
    # policy infrastructure that they cannot satisfy the autonomous gate.
    domain = str(raw.get("independence_domain") or provider).strip().lower()
    if domain != provider:
        raise ValueError(
            f"coding profile {profile_id!r} must use provider as its "
            "independence_domain")
    return CodingExecutionProfile(profile_id, provider, model, domain)


def configured_profiles(*, require_credentials: bool = True
                        ) -> list[CodingExecutionProfile]:
    """Return validated, deduplicated profiles in deterministic order.

    ``CODING_EXECUTION_PROFILES`` may contain a JSON array. Without it, only
    the deployment's configured provider is used: possession of another API
    key is not authorization to disclose clinical-note data to that vendor.
    A missing second authorized profile does not crash ordinary/manual
    coding; the release gate records insufficient independence and withholds
    AUTO_READY.
    """
    authorized = {value.strip().lower() for value in os.getenv(
        "AUTHORIZED_MODEL_PROVIDERS", config.LLM_PROVIDER).split(",")
                  if value.strip()}
    if not authorized or not authorized <= {"openai", "claude"}:
        raise ValueError(
            "AUTHORIZED_MODEL_PROVIDERS must explicitly list supported providers")
    configured = os.getenv("CODING_EXECUTION_PROFILES", "").strip()
    if configured:
        try:
            raw_profiles = json.loads(configured)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"CODING_EXECUTION_PROFILES is invalid JSON: {exc}") from exc
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError("CODING_EXECUTION_PROFILES must be a non-empty array")
        profiles = [_validate(row, i) for i, row in enumerate(raw_profiles, 1)]
    else:
        defaults = [{
            "provider": config.LLM_PROVIDER,
            "model": (config.CLAUDE_MODEL if config.LLM_PROVIDER == "claude"
                      else config.OPENAI_MODEL),
        }]
        profiles = [_validate(row, i) for i, row in enumerate(defaults, 1)]

    unauthorized = sorted({p.provider for p in profiles} - authorized)
    if unauthorized:
        raise ValueError(
            "coding profiles contain providers not explicitly authorized for "
            "clinical-note processing: " + ", ".join(unauthorized))

    credentials = {
        "claude": _usable_secret(config.ANTHROPIC_API_KEY),
        "openai": _usable_secret(config.OPENAI_API_KEY),
    }
    if require_credentials:
        profiles = [p for p in profiles if credentials[p.provider]]
    seen, unique = set(), []
    for profile in profiles:
        identity = (profile.provider, profile.model)
        if identity not in seen:
            seen.add(identity)
            unique.append(profile)
    if not unique and require_credentials:
        # Keep ordinary startup diagnostics useful. The actual provider call
        # will produce the concrete missing-credential error; autonomy cannot
        # pass because the persisted profile set has zero independent domains.
        fallback_model = (config.CLAUDE_MODEL if config.LLM_PROVIDER == "claude"
                          else config.OPENAI_MODEL)
        unique = [_validate({"provider": config.LLM_PROVIDER,
                             "model": fallback_model}, 1)]
    return unique


def profiles_for_runs(run_count: int) -> list[CodingExecutionProfile]:
    if run_count < 1:
        raise ValueError("run_count must be positive")
    profiles = configured_profiles()
    return [profiles[i % len(profiles)] for i in range(run_count)]


def consistency_execution_plan(
        maximum_runs: int, mode: str = "adaptive"
        ) -> tuple[list[CodingExecutionProfile], int]:
    """Return ``(profile_schedule, initial_run_count)`` for consistency.

    Adaptive execution starts with the smallest provider-diverse set that can
    satisfy the autonomous independence gate. Remaining capacity is ordered
    primary-provider first and is consumed only after a disagreement. If the
    configured credentials cannot span the required domains, retain the fixed
    N-run behavior so manual/non-autonomous operation does not silently lose
    its same-provider instability checks; the autonomy preflight still blocks.
    """
    if maximum_runs < 1:
        raise ValueError("maximum_runs must be positive")
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"adaptive", "fixed"}:
        raise ValueError("consistency mode must be adaptive or fixed")
    if normalized_mode == "fixed" or maximum_runs == 1:
        schedule = profiles_for_runs(maximum_runs)
        return schedule, len(schedule)

    profiles = configured_profiles()
    required = config.MIN_INDEPENDENT_MODEL_DOMAINS
    initial: list[CodingExecutionProfile] = []
    seen_domains: set[str] = set()
    for profile in profiles:
        if profile.independence_domain in seen_domains:
            continue
        initial.append(profile)
        seen_domains.add(profile.independence_domain)
        if len(seen_domains) >= required:
            break

    if len(seen_domains) < required or len(initial) > maximum_runs:
        schedule = profiles_for_runs(maximum_runs)
        return schedule, len(schedule)

    primary_first = sorted(
        profiles,
        key=lambda profile: profile.provider != config.LLM_PROVIDER,
    )
    schedule = list(initial)
    index = 0
    while len(schedule) < maximum_runs:
        schedule.append(primary_first[index % len(primary_first)])
        index += 1
    return schedule, len(initial)


def default_profile() -> CodingExecutionProfile:
    configured = configured_profiles()
    for profile in configured:
        if profile.provider == config.LLM_PROVIDER:
            return profile
    return configured[0]


def active_profile() -> CodingExecutionProfile:
    return _ACTIVE.get() or default_profile()


def execution_record() -> dict:
    """Persist every coding model that can influence the run's claim."""
    profile = active_profile()
    models = [profile.model]
    if profile.provider == "claude":
        verify_model = str(config.CLAUDE_VERIFY_MODEL or "").strip()
        if verify_model:
            models.append(verify_model)
    return {**profile.model_dump(), "models_used": list(dict.fromkeys(models))}


def autonomous_execution_errors(
        scheduled: list[CodingExecutionProfile], run_count: int) -> list[str]:
    """Preflight whether a batch can satisfy model-side autonomy controls."""
    errors = []
    if run_count < config.MIN_INDEPENDENT_MODEL_DOMAINS:
        errors.append(
            "consistency run count is below the independent-domain requirement")
    domains = {
        profile.independence_domain for profile in scheduled[:run_count]
    }
    if len(domains) < config.MIN_INDEPENDENT_MODEL_DOMAINS:
        errors.append(
            "configured coding profiles do not span enough authorized providers")
    if os.getenv("CLINICAL_AUDIT", "1") != "1":
        errors.append("CLINICAL_AUDIT must be enabled for autonomous release")
    if os.getenv("AUTO_BUILD_TERMINOLOGY_PACK", "1") == "0":
        errors.append(
            "AUTO_BUILD_TERMINOLOGY_PACK must be enabled for autonomous release")
    if os.getenv("AUTO_REFRESH_AUTHORITIES", "1") == "0":
        errors.append(
            "AUTO_REFRESH_AUTHORITIES must be enabled for autonomous release")
    if os.getenv("CODER_ADJUDICATION", "1") != "1":
        errors.append("CODER_ADJUDICATION must be enabled for autonomous release")
    if os.getenv("AUDIT_CONVERGENCE", "1") != "1":
        errors.append("AUDIT_CONVERGENCE must be enabled for autonomous release")
    try:
        audit_passes = int(os.getenv("CLINICAL_AUDIT_PASSES", "1"))
    except ValueError:
        audit_passes = 0
    if audit_passes < 2:
        errors.append(
            "CLINICAL_AUDIT_PASSES must be at least 2 for autonomous release")

    coding_identities = set()
    for profile in scheduled:
        coding_identities.add((profile.provider, profile.model))
        if profile.provider == "claude" and config.CLAUDE_VERIFY_MODEL:
            coding_identities.add((profile.provider, config.CLAUDE_VERIFY_MODEL))
    adjudicator_provider = config.LLM_PROVIDER
    adjudicator_model = os.getenv(
        "CODER_ADJUDICATOR_MODEL", "claude-fable-5").strip()
    adjudicator_alt = os.getenv("CODER_ADJUDICATOR_ALT_MODEL", "").strip()
    try:
        adjudication_passes = max(
            2, int(os.getenv("CODER_ADJUDICATION_PASSES", "2")))
    except ValueError:
        adjudication_passes = 0
    adjudication_identities = set()
    adjudication_domains = set()
    adjudication_schedule = (
        [scheduled[index % len(scheduled)]
         for index in range(adjudication_passes)]
        if scheduled and adjudication_passes else [])
    for index, profile in enumerate(adjudication_schedule):
        model = profile.model
        if profile.provider == adjudicator_provider:
            if profile.provider == "claude":
                model = adjudicator_model or profile.model
            if index > 0 and adjudicator_alt:
                model = adjudicator_alt
        adjudication_identities.add((profile.provider, model))
        adjudication_domains.add(profile.independence_domain)
    if len(adjudication_domains) < config.MIN_INDEPENDENT_MODEL_DOMAINS:
        errors.append(
            "autonomous adjudication does not span enough authorized providers")
    auditor_provider = config.LLM_PROVIDER
    auditor_model = (os.getenv("CLINICAL_AUDITOR_MODEL", "claude-fable-5")
                     if auditor_provider == "claude" else
                     next((profile.model for profile in scheduled
                           if profile.provider == auditor_provider), ""))
    if (not auditor_model
            or (auditor_provider, auditor_model) in coding_identities
            or (auditor_provider, auditor_model) in adjudication_identities):
        errors.append(
            "clinical auditor must use a model not used by any coding or "
            "adjudication pass")
    return errors


@contextmanager
def use_execution_profile(profile: CodingExecutionProfile | dict):
    if isinstance(profile, dict):
        profile = _validate(profile, 1)
    token = _ACTIVE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE.reset(token)
