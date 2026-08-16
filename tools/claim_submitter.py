#!/usr/bin/env python3
"""Claim submission: registry-verified claims -> clearinghouse 837P (Stedi).

The pipeline's output stops being a JSON artifact here and becomes a real
professional claim. Four principles govern this module:

  1. ONLY VERIFIED CLAIMS TRANSMIT. The claims registry
     (data/registry/claims_registry.jsonl) is the sole source of billable
     content — a claim submits only when its registry verification tier is
     allowed by policy (auto / adjudicated / human) AND its disposition is
     CLEAN. Nothing is ever built from a raw result file's arrays.

  2. EVERY ENVELOPE VARIABLE IS DYNAMIC. Charge amounts (practice fee
     schedule), billing/rendering provider NPIs and TIN, taxonomy codes,
     service-facility details, submitter contact, claim-level indicators,
     and filing codes all come from data/practice_config.json — re-read
     (mtime-cached) on every run, so an edit takes effect immediately with
     no code change. Payer trading-partner IDs come from
     data/codes/payers.json via the existing payer registry (same
     hot-reload behavior). Patient demographics, subscriber identifiers,
     rendering-provider identity and place of service come from the
     ClaimBundle's ENCOUNTER CONTEXT — resolved by an authoritative
     `EncounterContextProvider`, not read off the note by a model. A
     missing variable BLOCKS that one claim with a precise reason (fail
     closed); it never crashes the batch and the system never invents a
     value.

  3. SUBMISSION IS IDEMPOTENT AND AUDITED. Every attempt is appended to
     data/registry/submissions.jsonl. A claim (document + exact claim
     fingerprint) transmits at most once; if the verified claim CHANGES
     after a successful submission, the new version is blocked with a
     "requires replacement claim" reason — corrected/void resubmission
     (frequency codes 7/8) is a deliberate human decision, not an
     automatic one.

  4. ONE CLAIM CONTRACT. Every payload is assembled from a `ClaimBundle`
     (app/contracts/claim_bundle.py). A canonical registry event carries
     the bundle itself; a legacy event is VIEWED as one through
     app/contracts/legacy_adapter.py. There is exactly one 837P builder,
     so a field the producer carries can no longer be lost because this
     module was written against a different result shape (issue #6,
     F6-R4-A1).

CLI (inside the app container):
  python tools/claim_submitter.py [--docs stem1,stem2] [--dry-run]
      [--results-dir DIR]

--dry-run builds and validates every payload and writes it to
output/submissions/ for inspection without transmitting anything.

Env:
  PRACTICE_CONFIG_PATH   override config location (default
                         data/practice_config.json)
  STEDI_API_KEY          clearinghouse credentials (absent -> submission
                         refuses to transmit; dry-run still works)
  STEDI_USAGE_INDICATOR  "T" (test, default) or "P" (production)
  AUTO_SUBMIT_CLAIMS     "1" lets run.py invoke this automatically after
                         registry ingest (default off — transmission is
                         irreversible and stays opt-in)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from tools.claims_registry import (REGISTRY_PATH, _claim_key,  # noqa: E402
                                   current_view, load_events)

DEFAULT_RESULTS = ROOT / "output" / "results"
DEFAULT_CONFIG = ROOT / "data" / "practice_config.json"
POS_CODES_FILE = ROOT / "data" / "codes" / "pos_codes.json"
LEDGER_PATH = ROOT / "data" / "registry" / "submissions.jsonl"
DRYRUN_DIR = ROOT / "output" / "submissions"

_NPI_RE = re.compile(r"^\d{10}$")
_pos_cache: tuple[int, frozenset[str]] | None = None

#: X12 837P transaction cardinalities: at most four diagnosis pointers on a
#: service line, at most twelve diagnoses on a claim. Envelope structure from
#: the professional-claim implementation guide — not medical-code facts.
_MAX_DX_POINTERS = 4
_MAX_CLAIM_DIAGNOSES = 12


def _valid_npi(value) -> bool:
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


def _valid_pos(value) -> bool:
    """Validate against the checked-in CMS Place of Service source."""
    global _pos_cache
    try:
        mtime = POS_CODES_FILE.stat().st_mtime_ns
        if _pos_cache is None or _pos_cache[0] != mtime:
            raw = json.loads(POS_CODES_FILE.read_text())
            _pos_cache = (mtime, frozenset(str(code) for code in
                                           (raw.get("codes") or {})))
    except (OSError, ValueError):
        return False
    return str(value or "").strip() in _pos_cache[1]


# --------------------------------------------------------------------------
# practice config — hot-reloaded, validated, never hardcoded
# --------------------------------------------------------------------------

_cfg_cache: dict = {}
_cfg_mtime: int = -1
_cfg_path_cached: str = ""


def config_path() -> Path:
    return Path(os.getenv("PRACTICE_CONFIG_PATH", str(DEFAULT_CONFIG)))


def load_practice_config() -> dict:
    """mtime-cached read — editing practice_config.json (new provider, fee
    change, payer indicator) takes effect on the next submission run without
    a restart. Missing/corrupt file returns {} and every claim blocks with
    a config-level reason rather than crashing."""
    global _cfg_cache, _cfg_mtime, _cfg_path_cached
    path = config_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return {}
    if mtime != _cfg_mtime or str(path) != _cfg_path_cached:
        try:
            _cfg_cache = json.loads(path.read_text())
            _cfg_mtime = mtime
            _cfg_path_cached = str(path)
        except Exception as exc:
            logger.warning(f"practice config unreadable ({exc}) — keeping "
                           f"last-known-good")
    return _cfg_cache


def validate_config(cfg: dict) -> list[str]:
    """Envelope-level problems that block EVERY claim. Field-level gaps
    (a code missing from the fee schedule, an unmatched provider) are
    reported per claim instead."""
    problems = []
    bp = cfg.get("billing_provider") or {}
    for field in ("organization_name", "npi", "tax_id"):
        if not str(bp.get(field) or "").strip():
            problems.append(f"billing_provider.{field} missing")
    if bp.get("npi") and not _valid_npi(bp["npi"]):
        problems.append("billing_provider.npi fails the CMS NPI check digit")
    tax_id = str(bp.get("tax_id") or "").strip()
    if tax_id and (not re.fullmatch(r"\d{9}", tax_id) or len(set(tax_id)) == 1):
        problems.append("billing_provider.tax_id must be a non-placeholder 9-digit TIN")
    addr = bp.get("address") or {}
    for field in ("address1", "city", "state", "postal_code"):
        if not str(addr.get(field) or "").strip():
            problems.append(f"billing_provider.address.{field} missing")
    sub = cfg.get("submitter") or {}
    if not str(sub.get("organization_name") or "").strip():
        problems.append("submitter.organization_name missing")
    if not isinstance((cfg.get("fee_schedule") or {}).get("charges"), dict):
        problems.append("fee_schedule.charges missing")
    defaults = cfg.get("claim_defaults") or {}
    for field in (
        "claim_frequency_code", "signature_indicator",
        "plan_participation_code", "release_information_code",
        "benefits_assignment_certification_indicator",
    ):
        if not str(defaults.get(field) or "").strip():
            problems.append(f"claim_defaults.{field} missing")
    filing = defaults.get("claim_filing_code") or {}
    if not isinstance(filing, dict) or not (
            filing.get("default") or filing.get("by_kind")):
        problems.append("claim_defaults.claim_filing_code missing")
    return problems


# --------------------------------------------------------------------------
# dynamic field resolution
# --------------------------------------------------------------------------

_DATE_FORMATS = ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%B %d, %Y",
                 "%b %d, %Y", "%d %B %Y", "%m/%d/%y")


def _to_ccyymmdd(raw: str) -> str | None:
    raw = str(raw or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return None


def _split_name(full: str) -> tuple[str, str] | None:
    """'First [Middle] Last' -> (first, last). Handles 'Last, First' too."""
    full = str(full or "").strip()
    if not full:
        return None
    if "," in full:
        last, _, first = full.partition(",")
        first = first.strip().split()[0] if first.strip() else ""
        if first and last.strip():
            return first, last.strip()
        return None
    parts = full.split()
    if len(parts) < 2:
        return None
    return parts[0], parts[-1]


_CREDENTIALS = {"dpm", "md", "do", "pa", "pa-c", "np", "aprn", "dnp",
                "rn", "cpc", "facfas", "faafp"}


def _split_provider_name(full: str) -> tuple[str, str] | None:
    """Provider names carry honorifics and credentials ('Dr. Sandra Kim,
    DPM') — the generic splitter would read the credential comma as a
    'Last, First' format. Strip both, then split First/Last."""
    full = re.sub(r"^\s*dr\.?\s+", "", str(full or "").strip(),
                  flags=re.IGNORECASE)
    parts = [p for p in re.split(r"[\s,]+", full)
             if p and p.strip(".").lower() not in _CREDENTIALS]
    if len(parts) < 2:
        return None
    return parts[0], parts[-1]


def _rendering_from_identity(provider, npi: str, rp_cfg: dict) -> dict:
    """The claim's rendering provider, built from the RESOLVED identity itself.

    The practice config contributes only a fallback taxonomy code — an
    attribute of how the practice bills, not an answer to who rendered the
    service.
    """
    first = str(getattr(provider, "first_name", "") or "")
    last = str(getattr(provider, "last_name", "") or "")
    if not (first and last):
        split = _split_provider_name(
            str(getattr(provider, "display_name", "") or ""))
        first, last = split if split else ("", "")
    return {"first_name": first, "last_name": last, "npi": npi,
            "taxonomy_code": (str(getattr(provider, "taxonomy_code", "") or "")
                              or rp_cfg.get("default_taxonomy_code"))}


def resolve_rendering_provider(cfg: dict, provider, *,
                               authoritative: bool) -> tuple[dict | None, str]:
    """Rendering provider for the claim, from the bundle's provider identity.

    ISSUE #6, DIRECTIVE §2 — "resolve by stable identifiers, not model
    inference", and "must not invent or select a provider identity from a broad
    roster". This function used to do both, on the canonical path, AFTER the
    encounter context had already resolved the provider BY NPI:

      * a practice-config roster entry whose `match` pattern was a SUBSTRING of
        the resolved provider's display name replaced the resolved NPI with the
        config's own — so a config entry matching "lee" would sign this claim
        as a different person the moment a "Leeson" was resolved, silently and
        with every downstream check still passing;
      * and when nothing matched, a configured DEFAULT provider's NPI was put
        on the claim instead.

    Both put a different human's NPI on an autonomously released claim. So the
    behaviour is now split by whether the identity is AUTHORITATIVE:

      authoritative  the resolved participant IS the rendering provider. The
                     config may supply a taxonomy code it did not carry, and
                     nothing else. An invalid resolved NPI BLOCKS — substituting
                     another provider for a bad identifier is how a wrong claim
                     becomes a clean one.
      legacy         (an adapted `app.pipeline`/`claude_coder.run/1` artifact,
                     whose "context" is note-extracted text and which can never
                     auto-release) keeps the original fidelity order, because
                     for those artifacts the practice config genuinely is the
                     better authority.
    """
    rp_cfg = cfg.get("rendering_providers") or {}
    npi = str(getattr(provider, "npi", "") or "").strip()

    if authoritative:
        if not _valid_npi(npi):
            return None, (
                f"the authoritatively resolved rendering provider NPI {npi!r} "
                f"is not a valid CMS NPI; the practice config may not "
                f"substitute another provider for a resolved one")
        return _rendering_from_identity(provider, npi, rp_cfg), ""

    display_name = str(getattr(provider, "display_name", "") or "").lower()
    for entry in rp_cfg.get("providers") or []:
        for pattern in entry.get("match") or []:
            if pattern and str(pattern).lower() in display_name:
                if _valid_npi(entry.get("npi")):
                    return entry, ""
                return None, (f"roster entry for '{pattern}' has an invalid "
                              f"NPI — fix rendering_providers in the "
                              f"practice config")
    if rp_cfg.get("trust_note_npi") and _valid_npi(npi):
        return _rendering_from_identity(provider, npi, rp_cfg), ""
    default = rp_cfg.get("default")
    if default and _valid_npi(default.get("npi")):
        return default, ""
    return None, ("rendering provider unresolvable: no roster match for "
                  f"'{getattr(provider, 'display_name', '')}', no valid "
                  f"encounter NPI, no default")


def line_charge(cfg: dict, code: str, units: int) -> tuple[float | None, str]:
    """Charge for a claim line from the practice fee schedule — never
    invented. Missing code follows fee_schedule.missing_code_policy
    ('block' is the only safe value and the default)."""
    fees = (cfg.get("fee_schedule") or {}).get("charges") or {}
    per_unit = fees.get(str(code).upper())
    if per_unit is None:
        return None, (f"no fee schedule entry for {code} — add it to "
                      f"fee_schedule.charges in the practice config")
    try:
        return round(float(per_unit) * max(1, int(units or 1)), 2), ""
    except (TypeError, ValueError):
        return None, f"fee schedule charge for {code} is not numeric"


def claim_filing_code(cfg: dict, parsed) -> str:
    d = (cfg.get("claim_defaults") or {}).get("claim_filing_code") or {}
    overrides = (cfg.get("claim_defaults") or {}).get("payer_overrides") or {}
    payer_override = (overrides.get(parsed.payer_id or "") or {})
    return (payer_override.get("claim_filing_code")
            or (d.get("by_kind") or {}).get(parsed.kind)
            or d.get("default") or "")


# --------------------------------------------------------------------------
# 837P builder
# --------------------------------------------------------------------------

def _control_number(doc: str) -> str:
    """Deterministic 9-digit control number derived from the document id —
    stable across retries of the same claim, distinct across notes."""
    import hashlib
    h = hashlib.sha256(doc.encode()).hexdigest()
    return str(int(h[:12], 16) % 900000000 + 100000000)


def bundle_for(reg_event: dict, result: dict):
    """The `ClaimBundle` this registry event's claim is built from.

    ONE 837P builder, two sources — never two builders:

      * a canonical event carries the whole bundle (`claim_bundle`), which IS
        the verified claim;
      * a legacy event carries the retired code arrays, which
        `app/contracts/legacy_adapter.py` VIEWS as a bundle, using the exact
        arrays the registry verified plus the result file's demographics.

    The legacy view keeps `tools/claim_submitter`'s founding principle intact:
    billable content still comes from the registry's verified claim, never from
    whatever the result file happens to say now.
    """
    from tools.claims_registry import bundle_of_event, is_bundle_artifact
    bundle = bundle_of_event(reg_event)
    if bundle is not None:
        return bundle
    if is_bundle_artifact(result):
        # Legacy event, canonical artifact. The legacy reader would find none of
        # the keys it needs on a bundle and return a bundle with no demographics
        # and no claim — an empty claim that LOOKS like a complete one. Refuse.
        # `_policy_gate` already blocks this pairing; raising here means the
        # refusal does not depend on being called in the right order.
        from app.contracts.claim_bundle import InvalidClaimBundle
        raise InvalidClaimBundle(
            "registry event carries a legacy claim but the result artifact is a "
            "ClaimBundle; re-ingest this note before building a claim from it")
    from app.contracts.legacy_adapter import bundle_from_legacy
    return bundle_from_legacy(result, reg_event.get("claim") or {})


def build_claim(doc: str, reg_event: dict, result: dict,
                cfg: dict) -> tuple[dict | None, list[str]]:
    """Assemble the clearinghouse professional-claim JSON from the verified
    `ClaimBundle` + the practice config. Returns (payload, blocks); any block
    -> no payload.

    The bundle is the authority for ENCOUNTER-level content (who the patient,
    subscriber, payer, rendering provider and place of service are; which codes,
    units, modifiers and diagnosis pointers). The practice config remains the
    authority for PRACTICE-level content (billing provider, submitter, fee
    schedule, claim defaults). Neither invents the other's fields, and a missing
    value on either side blocks this one claim with a precise reason.
    """
    from app.compliance.payer_registry import (PayerRegistryUnavailable,
                                                parse_insurance_text)

    blocks: list[str] = []
    bundle = bundle_for(reg_event, result)
    context = bundle.context
    defaults = cfg.get("claim_defaults") or {}

    # -- payer ------------------------------------------------------------
    # This is the one production step that still transmits a claim, and the payer
    # registry is what decides WHO it is transmitted to and under whose coverage
    # rules. An unreadable registry is a BLOCK (no payload), never an exception out
    # of a submission run and never a claim built against a payer nobody can name.
    # (Codex F6-R5-A, round 6.)
    try:
        parsed = parse_insurance_text(context.payer.name)
    except PayerRegistryUnavailable as exc:
        return None, [f"payer registry unavailable: {exc}"]
    if not parsed.stedi_trading_partner_id:
        blocks.append(f"payer '{parsed.payer_name or 'unknown'}' has no "
                      f"stedi_trading_partner_id in the declared payer registry")
    structured_member = context.subscriber.member_id.strip()
    parsed_member = str(parsed.member_id or "").strip()
    if structured_member and parsed_member and structured_member != parsed_member:
        blocks.append("structured and insurance-text Member/Policy IDs disagree")
    member_id = structured_member or parsed_member
    structured_group = context.subscriber.group_number.strip()
    parsed_group = str(parsed.group_number or "").strip()
    if structured_group and parsed_group and structured_group != parsed_group:
        blocks.append("structured and insurance-text group identifiers disagree")
    group_number = structured_group or parsed_group
    if not member_id:
        blocks.append("no Member/Policy ID present in structured encounter "
                      "metadata or the insurance text")

    # -- patient / subscriber ----------------------------------------------
    name = ((context.patient.first_name, context.patient.last_name)
            if context.patient.first_name and context.patient.last_name
            else None)
    if not name:
        blocks.append("patient name missing or not splittable into "
                      "first/last")
    dob = _to_ccyymmdd(context.patient.date_of_birth)
    if not dob:
        blocks.append(f"patient DOB unparseable: "
                      f"{context.patient.date_of_birth!r}")
    dos = _to_ccyymmdd(bundle.encounter.date_of_service or "")
    if not dos:
        blocks.append(f"date of service unparseable: "
                      f"{bundle.encounter.date_of_service!r}")
    gender = context.patient.gender.strip()[:1].upper()
    if gender not in {"F", "M", "U"}:
        blocks.append("patient gender/sex is missing or not valid for the claim "
                      "transaction")
    pos = context.place_of_service.strip()
    if not _valid_pos(pos):
        blocks.append("place of service is missing or absent from the authoritative "
                      "CMS POS set; autonomous submission cannot infer it")
    filing_code = claim_filing_code(cfg, parsed)
    if not filing_code:
        blocks.append("claim filing code is unresolved for this payer; configure "
                      "claim_defaults.claim_filing_code")
    for field in (
        "claim_frequency_code", "signature_indicator",
        "plan_participation_code", "release_information_code",
        "benefits_assignment_certification_indicator",
    ):
        if not str(defaults.get(field) or "").strip():
            blocks.append(f"claim_defaults.{field} missing")

    # -- providers ----------------------------------------------------------
    # A RESOLVED context resolved this provider BY NPI. Passing that fact down
    # is what stops the practice config from substituting a different provider
    # for one the authoritative source already named. (Directive §2.)
    from app.contracts.claim_bundle import ContextResolution
    rendering, why = resolve_rendering_provider(
        cfg, context.rendering_provider,
        authoritative=context.resolution is ContextResolution.RESOLVED)
    if rendering is None:
        blocks.append(why)

    # -- diagnoses ------------------------------------------------------------
    # Already ordered by the contract (`sequence`, primary first) and verified
    # in-sequence by `ClaimBundle.integrity_problems()`. Re-sorting here would
    # be a second, competing opinion about the claim's diagnosis order — the
    # order the pointers below are relative to.
    ordered_dx = list(bundle.diagnoses)
    if not ordered_dx:
        blocks.append("verified claim has no diagnoses")

    # -- service lines --------------------------------------------------------
    if not bundle.service_lines:
        blocks.append("verified claim has no billable service lines")
    service_lines, total = [], 0.0
    for line in bundle.service_lines:
        code = line.code.upper()
        units = line.units
        # Linkage is checked BEFORE price, deliberately: a line the record never
        # justified is a claim-integrity failure whether or not the practice has
        # a fee for it, and reporting the fee gap first would bury it.
        pointers = [p for p in line.diagnosis_pointers if 1 <= p <= len(ordered_dx)]
        if not pointers:
            # No fallback to "every documented diagnosis". A service line whose
            # diagnosis linkage the record never established must not acquire
            # one at the moment of submission; that is a fabricated medical
            # necessity assertion on a transmitted claim.
            blocks.append(
                f"service line {code} has no diagnosis pointer into this "
                f"claim's diagnoses; the record established no linkage")
            continue
        charge, why = line_charge(cfg, code, units)
        if charge is None:
            blocks.append(why)
            continue
        total += charge
        mods = [m.upper() for m in line.modifiers][:4]
        svc = {
            "serviceDate": dos or "",
            "professionalService": {
                "procedureIdentifier": "HC",
                "procedureCode": code,
                "lineItemChargeAmount": f"{charge:.2f}",
                "measurementUnit": "UN",
                "serviceUnitCount": str(units),
                "compositeDiagnosisCodePointers": {
                    "diagnosisCodePointers": pointers[:_MAX_DX_POINTERS],
                },
            },
        }
        if mods:
            svc["professionalService"]["procedureModifiers"] = mods
        service_lines.append(svc)

    if blocks:
        return None, blocks

    bp = cfg["billing_provider"]
    sub = cfg["submitter"]
    health_codes = []
    for i, diagnosis in enumerate(ordered_dx[:_MAX_CLAIM_DIAGNOSES]):
        health_codes.append({
            "diagnosisTypeCode": "ABK" if i == 0 else "ABF",
            "diagnosisCode": diagnosis.code.replace(".", "").upper(),
        })

    payload = {
        "controlNumber": _control_number(doc),
        "tradingPartnerServiceId": parsed.stedi_trading_partner_id,
        "usageIndicator": os.getenv("STEDI_USAGE_INDICATOR", "T"),
        "submitter": {
            "organizationName": sub["organization_name"],
            "contactInformation": {
                "name": sub.get("contact_name") or sub["organization_name"],
                "phoneNumber": str(sub.get("phone") or ""),
                **({"email": sub["email"]} if sub.get("email") else {}),
            },
        },
        "receiver": {"organizationName": parsed.payer_name},
        "subscriber": {
            "memberId": member_id,
            "firstName": name[0],
            "lastName": name[1],
            "dateOfBirth": dob,
            "gender": gender,
            **({"groupNumber": group_number} if group_number else {}),
            "paymentResponsibilityLevelCode": "P",
        },
        "billing": {
            "providerType": "BillingProvider",
            "organizationName": bp["organization_name"],
            "npi": str(bp["npi"]),
            "employerId": str(bp["tax_id"]),
            **({"taxonomyCode": bp["taxonomy_code"]}
               if bp.get("taxonomy_code") else {}),
            "address": {
                "address1": bp["address"]["address1"],
                "city": bp["address"]["city"],
                "state": bp["address"]["state"],
                "postalCode": str(bp["address"]["postal_code"]),
            },
            "contactInformation": {
                "name": bp.get("contact_name") or bp["organization_name"],
                "phoneNumber": str(bp.get("phone") or ""),
            },
        },
        "rendering": {
            "providerType": "RenderingProvider",
            "npi": str(rendering["npi"]),
            **({"firstName": rendering["first_name"],
                "lastName": rendering["last_name"]}
               if rendering.get("last_name") else {}),
            **({"taxonomyCode": rendering["taxonomy_code"]}
               if rendering.get("taxonomy_code") else {}),
        },
        "claimInformation": {
            "claimFilingCode": filing_code,
            "patientControlNumber": str(context.patient.record_number or doc)[:20],
            "claimChargeAmount": f"{total:.2f}",
            "placeOfServiceCode": pos,
            "claimFrequencyCode": str(defaults["claim_frequency_code"]),
            "signatureIndicator": str(defaults["signature_indicator"]),
            "planParticipationCode": str(defaults["plan_participation_code"]),
            "releaseInformationCode": str(defaults["release_information_code"]),
            "benefitsAssignmentCertificationIndicator": str(
                defaults["benefits_assignment_certification_indicator"]),
            "healthCareCodeInformation": health_codes,
            "serviceLines": service_lines,
        },
    }

    fac = cfg.get("service_facility") or {}
    if fac.get("organization_name") and fac.get("npi") and fac.get("address"):
        if not _valid_npi(fac.get("npi")):
            return None, ["service_facility.npi fails the CMS NPI check digit"]
        payload["claimInformation"]["serviceFacilityLocation"] = {
            "organizationName": fac["organization_name"],
            "npi": str(fac["npi"]),
            "address": {
                "address1": fac["address"].get("address1", ""),
                "city": fac["address"].get("city", ""),
                "state": fac["address"].get("state", ""),
                "postalCode": str(fac["address"].get("postal_code", "")),
            },
        }
    return payload, []


# --------------------------------------------------------------------------
# submission ledger — append-only, idempotent
# --------------------------------------------------------------------------

def load_ledger(path: Path | None = None) -> list[dict]:
    path = path or LEDGER_PATH
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping corrupt submissions ledger line")
    return out


def append_ledger(event: dict, path: Path | None = None) -> None:
    path = path or LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def submitted_keys(events: list[dict]) -> dict[str, str]:
    """document_id -> complete payload fingerprint of the last submission."""
    out: dict[str, str] = {}
    for e in events:
        if e.get("event") == "submitted":
            out[str(e.get("document_id"))] = str(
                e.get("submission_key") or e.get("claim_key") or "")
    return out


def _last_block(events: list[dict]) -> dict[str, tuple[str, str]]:
    """document_id -> (claim_key, reason) of the latest blocked/rejected
    event — so an unchanged block isn't re-appended on every run (the
    ledger records state CHANGES, not every scan)."""
    out: dict[str, tuple[str, str]] = {}
    for e in events:
        doc = str(e.get("document_id"))
        if e.get("event") in ("blocked", "rejected"):
            out[doc] = (str(e.get("claim_key") or ""),
                        str(e.get("reason") or ""))
        elif e.get("event") == "submitted":
            out.pop(doc, None)
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def _policy_gate(cfg: dict, reg_event: dict, result: dict) -> str | None:
    """May this verified claim transmit? The FIRST reason it may not, or None.

    Dispatches on the contract the registry event recorded. Both branches
    answer the same three questions — is the verification tier allowed, is the
    claim still the one that was verified, and does its release authorization
    still hold — but they answer them with the controls their own shape has.
    """
    policy = cfg.get("submission_policy") or {}
    tiers = [str(t).lower() for t in
             (policy.get("verification_tiers")
              or ["auto", "adjudicated", "human"])]
    tier = str(reg_event.get("verification") or "").lower()
    if tier not in tiers:
        return (f"verification tier '{tier}' not in submission policy "
                f"{tiers}")

    from tools.claims_registry import bundle_of_event, is_bundle_artifact
    verified_bundle = bundle_of_event(reg_event)
    if verified_bundle is not None:
        return _bundle_policy_gate(verified_bundle, result, tier)
    if is_bundle_artifact(result):
        # A canonical artifact on disk under a LEGACY registry event: the two
        # do not describe the same contract, and the legacy battery below would
        # read `success`/`final_disposition`/`patient_metadata` off a bundle,
        # find none of them, and produce a confident answer about controls that
        # never ran. Refuse instead; re-ingest records the note canonically.
        return ("registry event predates the canonical ClaimBundle contract "
                "but the result artifact is a ClaimBundle — re-ingest this "
                "note before submitting it")

    if policy.get("require_clean_disposition", True):
        disp = str((reg_event.get("claim") or {})
                   .get("final_disposition") or "").upper()
        if disp != "CLEAN":
            return f"disposition {disp or 'unknown'} != CLEAN"
    from app.release.claim_readiness import (
        encounter_context_fingerprint, verify_readiness_certificate,
    )
    expected_context = str(reg_event.get("encounter_context_fingerprint") or "")
    if not expected_context:
        return "registry event predates immutable encounter binding"
    if expected_context != encounter_context_fingerprint(result):
        return "encounter context changed after registry verification"
    if tier in {"auto", "adjudicated"}:
        cert = reg_event.get("claim_readiness_certificate") or {}
        if not cert or cert != (result.get("claim_readiness_certificate") or {}):
            return "registry and result do not carry the same readiness certificate"
        ok, reason = verify_readiness_certificate(result, cert)
        if not ok:
            return f"claim readiness authorization failed: {reason}"
    return None


def _bundle_policy_gate(verified, result: dict, tier: str) -> str | None:
    """The canonical branch: the verified bundle must still be the live one.

    Three distinct failures are checked separately, because they mean different
    things and a single combined comparison would report the wrong one:

      TAMPERED   the verified bundle no longer verifies on its own terms — its
                 claim fingerprint or its certificate's content address does
                 not reproduce, its context is unresolved, its pointers dangle.
      STALE      the note has been re-coded since it was verified: the live
                 artifact's claim, context or certificate differs from the one
                 the registry recorded.
      UNREADABLE the live artifact cannot be parsed as the contract it declares
                 — never treated as "no live artifact to compare against".

    A missing live artifact is NOT fatal here: `submit_all` already blocks a
    document whose result file is absent, and the registry's recorded bundle is
    the verified claim. What must never happen is a DIFFERENT live artifact
    passing unnoticed.

    The `human` tier skips the AUTOMATED release authorization — a coder
    recorded that claim, which is what the tier means, and the legacy branch
    has always worked this way. It is not an unguarded path: the integrity and
    staleness checks below still run, and `build_claim` still refuses to
    assemble a claim whose encounter context is missing any field the
    professional transaction requires.
    """
    from app.contracts.claim_bundle import ClaimBundleError, load_bundle
    from app.release.claim_readiness import verify_bundle_readiness

    if tier in {"auto", "adjudicated"}:
        ok, reason = verify_bundle_readiness(verified)
        if not ok:
            return f"claim readiness authorization failed: {reason}"
    try:
        live = load_bundle(result)
    except ClaimBundleError as exc:
        return f"live result artifact is not a readable ClaimBundle: {exc}"
    # The live artifact's OWN coherence, re-derived — not just compared. A hand
    # edit to a code, a unit, a modifier or the certificate body leaves the
    # STORED fingerprints untouched, so every comparison below would still
    # match while the artifact no longer describes the claim it was verified
    # for. Re-deriving is the only check that sees it. (Found by the
    # tampering case in tests/test_claim_bundle_e2e.py, not by review.)
    live_problems = live.integrity_problems()
    if live_problems:
        return f"live result artifact is not internally coherent: {live_problems[0]}"
    # The SAME re-derivation on the artifact the 837P is actually built FROM.
    # `build_claim` reads the REGISTRY's bundle, not this file, and for the
    # `human` tier nothing above re-derives it: the release authorization that
    # would have is deliberately skipped for a coder-verified claim. Coherence
    # is not a release policy, though — a recorded bundle whose certificate no
    # longer attests its own claim must not become a claim under any tier.
    # (Adjacent instance of the same class as the live-artifact check above;
    # issue #6 F7-R1.)
    verified_problems = verified.integrity_problems()
    if verified_problems:
        return (f"registry-verified claim is not internally coherent: "
                f"{verified_problems[0]}")
    if live.context.compute_fingerprint() != verified.context.compute_fingerprint():
        return "encounter context changed after registry verification"
    # RE-DERIVED and COMPLETE, not the stored billable-payload fingerprint: the
    # certified-claim digest additionally covers line order, per-line evidence
    # and authoritative record, the clinical-graph digest and the authoritative
    # snapshot, and it is recomputed from each artifact's own content so a
    # stored value edited to match cannot pass. The stored fingerprints are
    # still compared below — two independent controls, neither the only one
    # that would notice. (Issue #6 F7-R1.)
    if live.compute_certified_claim_digest() != \
            verified.compute_certified_claim_digest():
        return "claim changed after registry verification"
    if live.claim_fingerprint != verified.claim_fingerprint:
        return "claim changed after registry verification"
    live_certificate = (live.certificate.certificate_sha256
                        if live.certificate else "")
    verified_certificate = (verified.certificate.certificate_sha256
                            if verified.certificate else "")
    if live_certificate != verified_certificate:
        return "registry and result do not carry the same release certificate"
    if live.authority.data_fingerprint != verified.authority.data_fingerprint:
        return ("the authoritative data behind this claim changed after "
                "registry verification")
    return None


def _submission_key(payload: dict) -> str:
    """Fingerprint the exact clearinghouse payload, not only its code arrays."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def submit_all(results_dir: Path = DEFAULT_RESULTS,
               docs: list[str] | None = None, dry_run: bool = False,
               adapter=None) -> dict:
    """Submit every eligible registry-verified claim. Per-claim fail-closed:
    each block is recorded with its reason; nothing partial transmits."""
    cfg = load_practice_config()
    stats = {"submitted": 0, "blocked": 0, "already_submitted": 0,
             "dry_run": dry_run, "docs": {}}

    cfg_problems = validate_config(cfg)
    if cfg_problems:
        stats["config_problems"] = cfg_problems
        logger.error(f"practice config invalid — nothing submits: "
                     f"{'; '.join(cfg_problems)}")
        return stats

    view = current_view(load_events(REGISTRY_PATH))
    ledger_events = load_ledger()
    prior = submitted_keys(ledger_events)
    prior_blocks = _last_block(ledger_events)

    def _record_block(doc: str, key: str, reason: str) -> None:
        if prior_blocks.get(doc) != (key, reason):
            append_ledger({"event": "blocked", "document_id": doc,
                           "at": _now(), "claim_key": key,
                           "reason": reason})

    if adapter is None and not dry_run:
        from app.compliance.adapters.stedi import StediAdapter
        adapter = StediAdapter()
        if not adapter.is_configured():
            stats["config_problems"] = ["STEDI_API_KEY not configured — "
                                        "use --dry-run to build payloads"]
            logger.error(stats["config_problems"][0])
            return stats

    for doc in sorted(view):
        if docs is not None and doc not in docs:
            continue
        reg_event = view[doc]
        result_file = results_dir / f"{doc}_results.json"
        try:
            result = json.loads(result_file.read_text())
        except OSError:
            stats["blocked"] += 1
            stats["docs"][doc] = ("blocked: result file with the encounter "
                                  "context not found "
                                  f"({result_file.name})")
            continue
        except ValueError as exc:
            # Corrupt JSON is ONE document's problem. Letting it raise would
            # abort the whole submission run at whichever note happened to sort
            # first, leaving every later claim unexamined and unexplained.
            stats["blocked"] += 1
            stats["docs"][doc] = (f"blocked: result artifact is not readable "
                                  f"JSON ({exc})")
            continue

        registry_key = _claim_key({
            "claim": reg_event.get("claim") or {},
            "encounter_context_fingerprint": reg_event.get(
                "encounter_context_fingerprint") or "",
        })
        try:
            why = _policy_gate(cfg, reg_event, result)
        except Exception as exc:
            # Same rule for a malformed REGISTRY event: fail this claim closed,
            # with its reason, and keep examining the rest.
            why = (f"submission policy could not be evaluated "
                   f"({type(exc).__name__}: {exc})")
        if why:
            stats["blocked"] += 1
            stats["docs"][doc] = f"blocked: {why}"
            if not dry_run:
                _record_block(doc, registry_key, why)
            continue

        try:
            payload, blocks = build_claim(doc, reg_event, result, cfg)
        except Exception as exc:
            payload, blocks = None, [f"claim could not be assembled "
                                     f"({type(exc).__name__}: {exc})"]
        if blocks:
            stats["blocked"] += 1
            stats["docs"][doc] = "blocked: " + "; ".join(blocks)
            if not dry_run:
                _record_block(doc, registry_key, "; ".join(blocks))
            continue

        key = _submission_key(payload)
        if doc in prior:
            if prior[doc] == key:
                stats["already_submitted"] += 1
                stats["docs"][doc] = "already submitted (unchanged payload)"
            else:
                stats["blocked"] += 1
                stats["docs"][doc] = (
                    "blocked: verified submission payload changed after a "
                    "successful submission — requires a replacement claim "
                    "(frequency code 7), which is a manual decision")
                _record_block(doc, key, "submission payload changed after submission")
            continue

        if dry_run:
            DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
            out = DRYRUN_DIR / f"{doc}_837p.json"
            out.write_text(json.dumps(payload, indent=2))
            stats["submitted"] += 1
            stats["docs"][doc] = f"DRY RUN: payload built -> {out.name}"
            continue

        res = adapter.submit_claim(payload)
        if res.submitted:
            stats["submitted"] += 1
            stats["docs"][doc] = f"submitted (ref {res.claim_reference})"
            append_ledger({
                "event": "submitted", "document_id": doc, "at": _now(),
                "claim_key": key, "submission_key": key,
                "verification": reg_event.get("verification"),
                "trading_partner": payload["tradingPartnerServiceId"],
                "usage_indicator": payload["usageIndicator"],
                "claim_charge_amount":
                    payload["claimInformation"]["claimChargeAmount"],
                "claim_reference": res.claim_reference,
            })
        else:
            stats["blocked"] += 1
            reason = "; ".join(res.errors) or "clearinghouse rejected"
            stats["docs"][doc] = f"rejected: {reason}"
            append_ledger({"event": "rejected", "document_id": doc,
                           "at": _now(), "claim_key": key,
                           "reason": reason})
    return stats


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    p.add_argument("--docs", default="",
                   help="comma-separated note stems to restrict to")
    p.add_argument("--dry-run", action="store_true",
                   help="build + validate payloads, write to "
                        "output/submissions/, transmit nothing")
    args = p.parse_args()
    docs = [s.strip() for s in args.docs.split(",") if s.strip()] or None
    stats = submit_all(Path(args.results_dir), docs=docs,
                       dry_run=args.dry_run)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
