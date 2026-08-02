"""Authoritative compliance data-source registry — mirrors
`Claimly_Compliance_Data_Sources.xlsx`. Each source declares its publisher,
update cadence, target table, parser, and license so the refresh runner knows
what to pull, how often, and how to ingest it.

URLs point at the CMS landing pages; the concrete quarterly file URL is resolved
at refresh time (or overridden via the `url` field) because CMS rotates file
names each quarter (RVU+YY+quarter, Eff_MM-DD-YYYY, …).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    id: str
    layer: str
    publisher: str
    cadence: str                  # weekly | quarterly | annual
    fmt: str                      # zip-csv | csv | txt | html | json
    target_table: str
    parser: str                   # name of the parser fn in parsers.py
    url: str
    license: str = "Free"
    notes: str = ""
    # True = updated by hand (no automated fetch/parse path exists yet).
    # Manual sources are excluded from due_sources()/cron scheduling and the
    # runner reports them as skipped-manual instead of failing with
    # "no parser" — previously these were SCHEDULED in the crontab and
    # silently error-logged every cycle, which read as automated coverage
    # that didn't actually exist.
    manual: bool = False


# Cadence → months it fires on (quarterly = Jan/Apr/Jul/Oct).
QUARTER_MONTHS = {1, 4, 7, 10}

SOURCES: list[Source] = [
    Source(
        id="ncci_ptp", layer="NCCI", publisher="CMS", cadence="quarterly",
        fmt="zip-csv", target_table="ncci_ptp", parser="parse_ncci",
        url="https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-procedure-procedure-ptp-edits",
        notes="Column1/Column2 + modifier indicator 0/1/9. CMS keeps only current+prior quarter — retain history.",
    ),
    Source(
        id="mue", layer="NCCI", publisher="CMS", cadence="quarterly",
        fmt="zip-csv", target_table="mue", parser="parse_mue",
        url="https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-medically-unlikely-edits-mues",
        notes="Carries MAI 1/2/3. Practitioner/Outpatient/DME contexts.",
    ),
    Source(
        id="pfs_global", layer="PFS", publisher="CMS", cadence="quarterly",
        fmt="zip-csv", target_table="global_period", parser="parse_pfs",
        url="https://www.cms.gov/medicare/payment/fee-schedules/physician/pfs-relative-value-files",
        notes="GLOB DAYS 000/010/090/MMM/XXX/YYY/ZZZ + status indicator.",
    ),
    Source(
        id="pos", layer="POS", publisher="CMS", cadence="annual",
        fmt="html", target_table="pos", parser="parse_pos",
        url="https://www.cms.gov/medicare/coding-billing/place-of-service-codes/code-sets",
        notes="Refreshes CMS names only. Facility vs non-facility status is "
              "preserved from the authoritative installed reference; a new code "
              "without that second authority fails closed.",
    ),
    Source(
        id="mcd_articles", layer="MEDICAL_NECESSITY", publisher="CMS", cadence="weekly",
        fmt="zip-csv", target_table="coverage_cpt/coverage_icd", parser="parse_mcd_articles",
        url="https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx",
        notes="ICD<->CPT covered-code lists live in Billing & Coding ARTICLES, not LCDs.",
    ),
    Source(
        id="hcpcs", layer="CODE_SET", publisher="CMS", cadence="quarterly",
        fmt="zip-fixed", target_table="code_set", parser="parse_hcpcs",
        url="https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/quarterly-update",
        notes="Official quarterly alpha-numeric contractor file. The fixed-width parser "
              "uses the CMS-published record layout, persists the complete versioned JSON "
              "source atomically, and triggers compliance.db fingerprint re-ingestion.",
    ),
    Source(
        id="prior_auth_medicare", layer="PRIOR_AUTH", publisher="CMS", cadence="quarterly",
        fmt="html", target_table="prior_auth_required", parser="parse_prior_auth_medicare",
        url="https://www.cms.gov/research-statistics-data-and-systems/monitoring-programs/"
            "medicare-ffs-compliance-programs/dmepos/downloads/dmepos_pa_required-prior-authorization-list.pdf",
        manual=True,
        notes="DMEPOS Required Prior Authorization List — exact HCPCS codes. Source PDF blocks "
              "automated fetching (403); data/codes/prior_auth_medicare.json is currently a "
              "manually-verified partial list (7 of ~74 items), not yet wired to this refresh "
              "cadence. parse_prior_auth_medicare doesn't exist yet — add it when the fetch-block "
              "is resolved (e.g. a browser-rendered fetch) rather than scraping around it.",
    ),
    # Commercial/other-federal payers (Tricare, BCBS Florida, Florida Medicaid) are
    # NOT registered here — unlike CMS's sources, they don't publish at a stable
    # URL/cadence: Tricare's list is a TriWest-issued PDF with no versioned API,
    # BCBS Florida's list is provider-portal-gated (no public URL at all), and
    # Florida Medicaid fragments across per-MCO plans (Sunshine Health, Humana
    # Healthy Horizons, etc.) with no single authoritative list. Their
    # data/codes/prior_auth_<payer>.json files are manually sourced/updated —
    # same trust tier as modifiers.json/podiatry_lcd.json, not an automated feed.
    Source(
        id="icd10cm", layer="CODE_SET", publisher="CMS+CDC", cadence="annual",
        fmt="zip-txt", target_table="code_set", parser="parse_icd10",
        url="https://www.cms.gov/medicare/coding-billing/icd-10-codes",
        manual=True,
        notes="Annual (Oct) + possible April update. parse_icd10 not implemented yet — "
              "data/codes/icd10cm_codes.json is refreshed manually from the CMS/CDC annual "
              "release; compliance.db re-ingests it automatically on file change.",
    ),
    Source(
        id="cpt", layer="CODE_SET", publisher="AMA", cadence="annual",
        fmt="csv", target_table="code_set", parser="parse_cpt",
        url="https://www.ama-assn.org/practice-management/cpt",
        license="LICENSED (AMA data agreement)",
        manual=True,
        notes="Descriptors require the AMA license — client holds it. parse_cpt not "
              "implemented yet — data/codes/cpt_codes.json is refreshed manually from the "
              "licensed file; compliance.db re-ingests it automatically on file change.",
    ),
]

SOURCES_BY_ID = {s.id: s for s in SOURCES}


def due_sources(month: int) -> list[Source]:
    """Sources whose cadence fires for the given calendar month.

    Manual sources are excluded — they have no fetch/parse path, so
    scheduling them only produces guaranteed failures in the refresh log."""
    out = []
    for s in SOURCES:
        if s.manual:
            continue
        if s.cadence == "weekly":
            out.append(s)
        elif s.cadence == "quarterly" and month in QUARTER_MONTHS:
            out.append(s)
        elif s.cadence == "annual" and month in (1, 10):  # CPT Jan, ICD Oct
            out.append(s)
    return out
