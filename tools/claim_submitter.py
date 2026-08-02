#!/usr/bin/env python3
"""Claim submission: registry-verified claims -> clearinghouse 837P (Stedi).

The pipeline's output stops being a JSON artifact here and becomes a real
professional claim. Three principles govern this module:

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
     hot-reload behavior). Patient demographics and subscriber identifiers
     come from the note's own extracted metadata. A missing variable BLOCKS
     that one claim with a precise reason (fail closed); it never crashes
     the batch and the system never invents a value.

  3. SUBMISSION IS IDEMPOTENT AND AUDITED. Every attempt is appended to
     data/registry/submissions.jsonl. A claim (document + exact claim
     fingerprint) transmits at most once; if the verified claim CHANGES
     after a successful submission, the new version is blocked with a
     "requires replacement claim" reason — corrected/void resubmission
     (frequency codes 7/8) is a deliberate human decision, not an
     automatic one.

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
from app.core.identifiers import is_valid_npi as _valid_npi  # noqa: E402

from tools.claims_registry import (REGISTRY_PATH, _claim_key,  # noqa: E402
                                   current_view, load_events)

DEFAULT_RESULTS = ROOT / "output" / "results"
DEFAULT_CONFIG = ROOT / "data" / "practice_config.json"
POS_CODES_FILE = ROOT / "data" / "codes" / "pos_codes.json"
LEDGER_PATH = ROOT / "data" / "registry" / "submissions.jsonl"
DRYRUN_DIR = ROOT / "output" / "submissions"

_pos_cache: tuple[int, frozenset[str]] | None = None


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


def resolve_rendering_provider(cfg: dict, meta: dict) -> tuple[dict | None, str]:
    """Rendering provider, resolved dynamically: config roster match on the
    note's provider name first, then the note's own extracted NPI (when
    trusted), then the configured default. Returns (provider, reason)."""
    rp_cfg = cfg.get("rendering_providers") or {}
    note_provider = str(meta.get("provider") or "").lower()
    for entry in rp_cfg.get("providers") or []:
        for pattern in entry.get("match") or []:
            if pattern and str(pattern).lower() in note_provider:
                if _valid_npi(entry.get("npi")):
                    return entry, ""
                return None, (f"roster entry for '{pattern}' has an invalid "
                              f"NPI — fix rendering_providers in the "
                              f"practice config")
    note_npi = str(meta.get("provider_npi") or meta.get("npi") or "").strip()
    if rp_cfg.get("trust_note_npi") and _valid_npi(note_npi):
        name = _split_provider_name(meta.get("provider") or "")
        return {"first_name": name[0] if name else "",
                "last_name": name[1] if name else "",
                "npi": note_npi,
                "taxonomy_code": rp_cfg.get("default_taxonomy_code")}, ""
    default = rp_cfg.get("default")
    if default and _valid_npi(default.get("npi")):
        return default, ""
    return None, ("rendering provider unresolvable: no roster match for "
                  f"'{meta.get('provider')}', no valid note NPI, no default")


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

def _dx_pointers(entry: dict, n_dx: int,
                 dx_codes: list[str] | None = None) -> list[int]:
    """Diagnosis pointers for a service line, in fidelity order: numeric
    pointers the pipeline produced; else the line's own linked_diagnoses
    (code strings) translated to 1-based positions in the claim's diagnosis
    order; else every documented diagnosis (primary first). Capped at the
    837P maximum of 4."""
    raw = entry.get("dx_pointers") or entry.get("diagnosis_pointers")
    if isinstance(raw, list) and raw:
        ptrs = []
        for p in raw:
            try:
                p = int(p)
            except (TypeError, ValueError):
                continue
            if 1 <= p <= min(n_dx, 12) and p not in ptrs:
                ptrs.append(p)
        if ptrs:
            return ptrs[:4]
    linked = entry.get("linked_diagnoses")
    if isinstance(linked, list) and linked and dx_codes:
        norm_order = [str(c or "").replace(".", "").upper()
                      for c in dx_codes]
        ptrs = []
        for code in linked:
            n = str(code or "").replace(".", "").upper()
            if n in norm_order:
                p = norm_order.index(n) + 1
                if p not in ptrs:
                    ptrs.append(p)
        if ptrs:
            return ptrs[:4]
    return list(range(1, min(n_dx, 4) + 1))


def _control_number(doc: str) -> str:
    """Deterministic 9-digit control number derived from the document id —
    stable across retries of the same claim, distinct across notes."""
    import hashlib
    h = hashlib.sha256(doc.encode()).hexdigest()
    return str(int(h[:12], 16) % 900000000 + 100000000)


def build_claim(doc: str, reg_event: dict, result: dict,
                cfg: dict) -> tuple[dict | None, list[str]]:
    """Assemble the clearinghouse professional-claim JSON from the registry's
    verified claim + the note's demographics + the practice config. Returns
    (payload, blocks); any block -> no payload."""
    from app.compliance.payer_registry import parse_insurance_text

    blocks: list[str] = []
    claim = reg_event.get("claim") or {}
    meta = result.get("patient_metadata") or {}
    defaults = cfg.get("claim_defaults") or {}

    # -- payer ------------------------------------------------------------
    parsed = parse_insurance_text(str(meta.get("insurance") or ""))
    if not parsed.stedi_trading_partner_id:
        blocks.append(f"payer '{parsed.payer_name or 'unknown'}' has no "
                      f"stedi_trading_partner_id in data/codes/payers.json")
    structured_member = str(meta.get("member_id") or
                            meta.get("insurance_id") or "").strip()
    parsed_member = str(parsed.member_id or "").strip()
    if structured_member and parsed_member and structured_member != parsed_member:
        blocks.append("structured and insurance-text Member/Policy IDs disagree")
    member_id = structured_member or parsed_member
    structured_group = str(meta.get("group_number") or "").strip()
    parsed_group = str(parsed.group_number or "").strip()
    if structured_group and parsed_group and structured_group != parsed_group:
        blocks.append("structured and insurance-text group identifiers disagree")
    group_number = structured_group or parsed_group
    if not member_id:
        blocks.append("no Member/Policy ID present in structured encounter "
                      "metadata or the insurance text")

    # -- patient / subscriber ----------------------------------------------
    name = _split_name(meta.get("patient_name") or "")
    if not name:
        blocks.append("patient name missing or not splittable into "
                      "first/last")
    dob = _to_ccyymmdd(meta.get("date_of_birth") or "")
    if not dob:
        blocks.append(f"patient DOB unparseable: "
                      f"{meta.get('date_of_birth')!r}")
    dos = _to_ccyymmdd(meta.get("date_of_service") or "")
    if not dos:
        blocks.append(f"date of service unparseable: "
                      f"{meta.get('date_of_service')!r}")
    gender = str(meta.get("gender") or meta.get("sex") or "").strip()[:1].upper()
    if gender not in {"F", "M", "U"}:
        blocks.append("patient gender/sex is missing or not valid for the claim "
                      "transaction")
    pos = str(meta.get("place_of_service") or "").strip()
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
    rendering, why = resolve_rendering_provider(cfg, meta)
    if rendering is None:
        blocks.append(why)

    # -- diagnoses ------------------------------------------------------------
    icds = claim.get("icd_codes") or []
    if not icds:
        blocks.append("verified claim has no diagnoses")
    primaries = [e for e in icds if e.get("type") == "primary"]
    ordered_dx = primaries + [e for e in icds if e.get("type") != "primary"]
    if not primaries and icds:
        ordered_dx = list(icds)  # first-listed stands in for principal

    # -- service lines --------------------------------------------------------
    lines = (claim.get("cpt_codes") or []) + (claim.get("hcpcs_codes") or [])
    if not lines:
        blocks.append("verified claim has no billable service lines")
    service_lines, total = [], 0.0
    for e in lines:
        code = str(e.get("code") or "").upper()
        try:
            units = max(1, int(e.get("units") or 1))
        except (TypeError, ValueError):
            blocks.append(f"units for {code} not numeric: "
                          f"{e.get('units')!r}")
            continue
        charge, why = line_charge(cfg, code, units)
        if charge is None:
            blocks.append(why)
            continue
        total += charge
        mods = [str(m).upper() for m in (e.get("modifiers") or [])][:4]
        svc = {
            "serviceDate": dos or "",
            "professionalService": {
                "procedureIdentifier": "HC",
                "procedureCode": code,
                "lineItemChargeAmount": f"{charge:.2f}",
                "measurementUnit": "UN",
                "serviceUnitCount": str(units),
                "compositeDiagnosisCodePointers": {
                    "diagnosisCodePointers": _dx_pointers(
                        e, len(ordered_dx),
                        [d.get("code", "") for d in ordered_dx]),
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
    for i, e in enumerate(ordered_dx[:12]):
        health_codes.append({
            "diagnosisTypeCode": "ABK" if i == 0 else "ABF",
            "diagnosisCode": str(e["code"]).replace(".", "").upper(),
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
            "patientControlNumber": str(meta.get("mrn") or doc)[:20],
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
    policy = cfg.get("submission_policy") or {}
    tiers = [str(t).lower() for t in
             (policy.get("verification_tiers")
              or ["auto", "adjudicated", "human"])]
    tier = str(reg_event.get("verification") or "").lower()
    if tier not in tiers:
        return (f"verification tier '{tier}' not in submission policy "
                f"{tiers}")
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
            stats["docs"][doc] = ("blocked: result file with patient "
                                  "demographics not found "
                                  f"({result_file.name})")
            continue

        registry_key = _claim_key({
            "claim": reg_event.get("claim") or {},
            "encounter_context_fingerprint": reg_event.get(
                "encounter_context_fingerprint") or "",
        })
        why = _policy_gate(cfg, reg_event, result)
        if why:
            stats["blocked"] += 1
            stats["docs"][doc] = f"blocked: {why}"
            if not dry_run:
                _record_block(doc, registry_key, why)
            continue

        payload, blocks = build_claim(doc, reg_event, result, cfg)
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
