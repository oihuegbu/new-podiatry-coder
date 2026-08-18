"""Where real-world evidence attaches to a claim — validation-ladder rungs 5-7.

The product directive (issue #6 §9) builds accuracy evidence as a ladder. Rungs 1-4
(schema/property, metamorphic, plausible-alternative, authoritative-descriptor-derived)
are provable today from synthetic and data-generated fixtures, and live in
`tests/test_validation_ladder.py`. Rungs 5-7 cannot be:

  5. de-identified historical notes with blinded duplicate runs;
  6. shadow comparison against the current/previous pipeline;
  7. denial / remittance / corrected-claim feedback.

All three need something this deployment does not have yet — encounters that were
really submitted and really adjudicated. This module is the STRUCTURE those rungs
plug into, deliberately NOT a stand-in for their data:

  * it defines the exact identity an observation must carry before it counts as
    evidence — the ClaimBundle it describes, the authoritative data snapshot that
    answered it, and the model profiles that read it (rung 7: "linked to the exact
    ClaimBundle and data/model versions");
  * it REFUSES an observation whose identity is incomplete, or that contradicts the
    bundle it names, instead of storing an unattributable row that would silently
    average into a metric later;
  * it reports every rung as POPULATED or AWAITING_DEPLOYMENT_DATA with a count, so
    "not measured yet" is a visible state rather than a confident-looking zero.

An empty ledger is the honest answer for this deployment, and `ladder_status()` says
so in as many words. Nothing here invents a historical corpus.

Control mode: OBSERVATIONAL. Recording an outcome never changes a claim, a release
decision or a routing destination; it accumulates the evidence a later, reviewed
change would need. (`COLLABORATION.md`, "Control mode".)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from app.contracts.claim_bundle import (
    ClaimBundle,
    LineMethod,
    ReleaseDestination,
    content_digest,
    is_claim_bundle,
    load_bundle,
)
from app.release.attempt_ledger import atomic_write_json

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LEDGER_DIR = ROOT / "data" / "feedback" / "outcomes"

#: Schema of one stored observation. Versioned for the same reason the bundle is:
#: a reader that cannot recognise the shape must refuse it, not guess.
OBSERVATION_SCHEMA = "ladder_observation/1"

CONTROL_MODE = "OBSERVATIONAL"


class LedgerError(RuntimeError):
    """Base class — every refusal here is typed and loud."""


class UnattributableObservation(LedgerError):
    """The observation cannot be tied to an exact claim/data/model version.

    Raised rather than stored. An observation whose subject is unknown is not weak
    evidence, it is no evidence: it would be indistinguishable from a row about a
    different claim, a different data snapshot or a different model once aggregated.
    """


class Rung(int, Enum):
    """The ladder rungs this module carries. 1-4 are tests, not stored observations."""

    BLINDED_DUPLICATE = 5     # same note, blinded duplicate runs, disagreement analysis
    SHADOW_COMPARISON = 6     # this pipeline vs. the previous one on the same input
    OUTCOME_FEEDBACK = 7      # denial / remittance / corrected claim from the payer

    @property
    def slug(self) -> str:
        return self.name.lower()


class RungStatus(str, Enum):
    POPULATED = "POPULATED"
    #: Structurally ready; waiting on real submitted/adjudicated encounters. This is
    #: NOT a failure state and must never be reported as a measured result.
    AWAITING_DEPLOYMENT_DATA = "AWAITING_DEPLOYMENT_DATA"


#: The identity fields an observation MUST carry. Each one can change the answer, so
#: an observation missing any of them cannot be compared with another observation.
REQUIRED_IDENTITY = (
    "encounter_id",            # which encounter
    "document_version",        # which bytes of which source document
    "claim_fingerprint",       # which exact claim (codes, units, modifiers, pointers)
    "data_fingerprint",        # which authoritative data set
    "database_snapshot_digest",  # which compiled database actually answered the edits
)

#: Recorded when present, never required: a held encounter has no certificate, and an
#: index or model profile may legitimately be absent from a deterministic-only run.
OPTIONAL_IDENTITY = (
    "certificate_sha256",
    "index_build_id",
    "model_profiles_digest",
    "schema_version",
    "destination",
)


@dataclass(frozen=True)
class ClaimIdentity:
    """The exact (claim, data, model) triple an observation is about."""

    encounter_id: str
    document_version: str
    claim_fingerprint: str
    data_fingerprint: str
    database_snapshot_digest: str
    certificate_sha256: str = ""
    index_build_id: str = ""
    model_profiles_digest: str = ""
    schema_version: int = 0
    destination: str = ""

    def missing(self) -> tuple[str, ...]:
        return tuple(f for f in REQUIRED_IDENTITY if not str(getattr(self, f) or "").strip())

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in REQUIRED_IDENTITY + OPTIONAL_IDENTITY}

    @property
    def key(self) -> str:
        """Content address of the identity — the join key an observation is filed under."""
        return content_digest({f: getattr(self, f) for f in REQUIRED_IDENTITY})


def claim_identity(bundle: ClaimBundle | dict) -> ClaimIdentity:
    """Derive the identity triple from a ClaimBundle (model or payload).

    The claim fingerprint is RECOMPUTED from the bundle's own content rather than read
    from its `claim_fingerprint` field: an observation keyed on a stored value would
    still join cleanly to a bundle whose claim had been altered underneath it, which is
    exactly the binding failure F7-R1 was about.
    """
    if isinstance(bundle, dict):
        if not is_claim_bundle(bundle):
            raise UnattributableObservation(
                "payload is not a ClaimBundle; an observation cannot be attached to a "
                "shape whose claim identity is undefined")
        bundle = load_bundle(bundle)
    profiles = dict(bundle.authority.model_profiles or {})
    return ClaimIdentity(
        encounter_id=str(bundle.encounter.encounter_id or ""),
        document_version=str(bundle.encounter.source_document.document_version or ""),
        claim_fingerprint=bundle.compute_claim_fingerprint(),
        data_fingerprint=str(bundle.authority.data_fingerprint or ""),
        database_snapshot_digest=str(bundle.authority.database_snapshot_digest or ""),
        certificate_sha256=(bundle.certificate.certificate_sha256
                            if bundle.certificate else ""),
        index_build_id=str(bundle.authority.index_build_id or ""),
        model_profiles_digest=(content_digest(profiles) if profiles else ""),
        schema_version=int(bundle.schema_version),
        destination=str(bundle.release.destination.value),
    )


@dataclass
class Observation:
    """One recorded real-world fact about one exact claim."""

    rung: Rung
    identity: ClaimIdentity
    observed_at: str
    #: Free-form, rung-specific body (CARC list, shadow diff, duplicate-run diff).
    #: Deliberately opaque: this module owns ATTRIBUTION, not the semantics of each
    #: feedback source, which belong to the tool that parses it.
    body: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "rung": int(self.rung),
            "rung_name": self.rung.slug,
            "identity": self.identity.as_dict(),
            "identity_key": self.identity.key,
            "observed_at": self.observed_at,
            "source": self.source,
            "body": self.body,
        }


class OutcomeLedger:
    """Append-only, one file per rung. Durable via the attempt ledger's atomic write."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory or DEFAULT_LEDGER_DIR)

    def path(self, rung: Rung) -> Path:
        return self.directory / f"rung_{int(rung)}_{rung.slug}.json"

    def observations(self, rung: Rung) -> list[dict[str, Any]]:
        path = self.path(rung)
        if not path.exists():
            return []
        try:
            rows = json.loads(path.read_text())
        except Exception as exc:                       # corrupt file is loud, not empty
            raise LedgerError(
                f"outcome ledger for rung {int(rung)} is unreadable at {path}: {exc}; "
                f"refusing to report it as an empty rung") from exc
        if not isinstance(rows, list):
            raise LedgerError(
                f"outcome ledger for rung {int(rung)} at {path} is not a list of "
                f"observations")
        return rows

    def record(self, rung: Rung, bundle: ClaimBundle | dict, body: dict[str, Any], *,
               observed_at: str, source: str = "") -> Observation:
        """Attach one observation to one exact claim, or refuse.

        Refuses when the bundle cannot supply every field in `REQUIRED_IDENTITY`. A
        held or system-retried encounter legitimately has no certificate, but it always
        has an encounter, a document version, a claim content and the data identity that
        answered it — if any of those is missing, the artifact itself is incomplete and
        an outcome filed against it could never be compared to anything.
        """
        identity = claim_identity(bundle)
        missing = identity.missing()
        if missing:
            raise UnattributableObservation(
                f"cannot record a rung-{int(rung)} observation: the bundle does not "
                f"bind {', '.join(missing)}. An outcome that cannot name the exact "
                f"claim, data snapshot and document version it describes is not "
                f"evidence and is not stored.")
        obs = Observation(rung=rung, identity=identity, observed_at=str(observed_at),
                          body=dict(body or {}), source=str(source or ""))
        rows = self.observations(rung)
        record = obs.as_dict()
        # Idempotent on (identity, source, body): re-ingesting the same remittance file
        # must not inflate the denial count.
        addr = content_digest({k: record[k] for k in ("identity_key", "source", "body")})
        for existing in rows:
            if content_digest({k: existing.get(k) for k in
                               ("identity_key", "source", "body")}) == addr:
                return obs
        rows.append(record)
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path(rung), rows, indent=2)
        return obs

    def status(self, rung: Rung) -> dict[str, Any]:
        rows = self.observations(rung)
        return {
            "rung": int(rung),
            "name": rung.slug,
            "status": (RungStatus.POPULATED.value if rows
                       else RungStatus.AWAITING_DEPLOYMENT_DATA.value),
            "observations": len(rows),
            "distinct_claims": len({r.get("identity_key") for r in rows}),
        }

    def ladder_status(self) -> dict[str, Any]:
        """What is actually measured, and what is only structurally ready.

        Rungs 1-4 are asserted by the test suite on every run, so they are reported as
        the tests they are, not as a count of stored rows. Rungs 5-7 report their real
        population — which for a deployment with no submitted claims is zero, said out
        loud rather than smoothed into a metric.
        """
        return {
            "control_mode": CONTROL_MODE,
            "test_rungs": {
                "1": "schema/property invariants — tests/test_validation_ladder.py",
                "2": "metamorphic, one axis at a time — tests/test_metamorphic.py",
                "3": "plausible alternatives rejected on the correct axis — "
                     "tests/test_tie_policy.py, tests/test_validation_ladder.py",
                "4": "authoritative-descriptor/index-derived cases — "
                     "tests/test_validation_ladder.py",
            },
            "observation_rungs": [self.status(r) for r in Rung],
        }


# --------------------------------------------------------------------------
# rung 8 — calibration inputs, and the directive's separately-tracked metrics
# --------------------------------------------------------------------------

#: The directive's metric list (§9, "Track separately"). Every entry is either a real
#: measurement or `None` — never a zero standing in for "we have no outcomes yet",
#: which is the single most misleading thing this reporting could do.
METRIC_NAMES = (
    "document_field_extraction_agreement",
    "eligible_service_precision_recall_proxy",
    "candidate_recall",
    "unique_deterministic_resolution_rate",
    "complete_claim_bundle_rate",
    "auto_ready_rate",
    "provider_query_rate",
    "escape_rate_invalid_inactive_ncci_mue_coverage",
    "repeat_run_claim_stability",
    "denial_correction_rate",
)


def _rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def metrics(bundles: Iterable[ClaimBundle | dict],
            ledger: OutcomeLedger | None = None) -> dict[str, Any]:
    """Compute the directive's separately-tracked metrics over real artifacts.

    Anything that genuinely requires data this deployment does not have (repeat-run
    stability needs duplicate runs; denial rate needs adjudicated claims) returns
    `None` with its reason recorded in `unavailable`, so a dashboard cannot render an
    unmeasured axis as a good score.
    """
    loaded: list[ClaimBundle] = []
    for b in bundles:
        loaded.append(load_bundle(b) if isinstance(b, dict) else b)
    total = len(loaded)

    by_destination: dict[str, int] = {}
    complete = auto_ready = queries = deterministic = escapes = 0
    for bundle in loaded:
        dest = bundle.release.destination
        by_destination[dest.value] = by_destination.get(dest.value, 0) + 1
        if dest is ReleaseDestination.AUTO_READY:
            auto_ready += 1
        if dest is ReleaseDestination.AUTO_QUERY:
            queries += 1
        if not bundle.integrity_problems():
            complete += 1
        lines = tuple(bundle.service_lines) + tuple(bundle.diagnoses)
        if lines and all(ln.method is LineMethod.DETERMINISTIC for ln in lines):
            deterministic += 1
        # An ESCAPE is a released claim carrying an unmet hard authority decision.
        # Target zero: a non-clean edit/coverage/validity outcome must never reach
        # AUTO_READY, so anything counted here is a control failure, not a rate to tune.
        if dest is ReleaseDestination.AUTO_READY and any(
                o.outcome in ("BLOCKED", "ERROR", "UNKNOWN") for o in bundle.outcomes):
            escapes += 1

    ledger = ledger or OutcomeLedger()
    duplicates = ledger.observations(Rung.BLINDED_DUPLICATE)
    outcomes = ledger.observations(Rung.OUTCOME_FEEDBACK)

    unavailable: dict[str, str] = {}
    values: dict[str, Any] = {
        "complete_claim_bundle_rate": _rate(complete, total),
        "auto_ready_rate": _rate(auto_ready, total),
        "provider_query_rate": _rate(queries, total),
        "unique_deterministic_resolution_rate": _rate(deterministic, total),
        "escape_rate_invalid_inactive_ncci_mue_coverage": _rate(escapes, total),
        "repeat_run_claim_stability": (
            _rate(sum(1 for d in duplicates if not d.get("body", {}).get("differs")),
                  len(duplicates)) if duplicates else None),
        "denial_correction_rate": (
            _rate(sum(1 for o in outcomes if o.get("body", {}).get("denied")),
                  len(outcomes)) if outcomes else None),
    }
    if not duplicates:
        unavailable["repeat_run_claim_stability"] = (
            "no blinded duplicate runs recorded (ladder rung 5)")
    if not outcomes:
        unavailable["denial_correction_rate"] = (
            "no payer outcomes recorded (ladder rung 7)")
    # Measured against a labelled corpus, which by design does not exist yet. Named
    # here so the gap is part of the report instead of being quietly absent from it.
    for name, why in (
            ("document_field_extraction_agreement",
             "needs per-field agreement capture across the two read channels"),
            ("eligible_service_precision_recall_proxy",
             "needs reviewed eligible-service labels (ladder rung 5)"),
            ("candidate_recall",
             "needs reviewed correct-code labels (ladder rung 5)")):
        values[name] = None
        unavailable[name] = why

    return {
        "control_mode": CONTROL_MODE,
        "claims": total,
        "by_destination": by_destination,
        "metrics": {name: values.get(name) for name in METRIC_NAMES},
        "unavailable": unavailable,
    }
