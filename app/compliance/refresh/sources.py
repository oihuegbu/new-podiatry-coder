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
        notes="Facility vs non-facility differential.",
    ),
    Source(
        id="mcd_articles", layer="MEDICAL_NECESSITY", publisher="CMS", cadence="weekly",
        fmt="zip-csv", target_table="coverage_cpt/coverage_icd", parser="parse_mcd_articles",
        url="https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx",
        notes="ICD<->CPT covered-code lists live in Billing & Coding ARTICLES, not LCDs.",
    ),
    Source(
        id="hcpcs", layer="CODE_SET", publisher="CMS", cadence="quarterly",
        fmt="zip-csv", target_table="code_set", parser="parse_hcpcs",
        url="https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system",
    ),
    Source(
        id="icd10cm", layer="CODE_SET", publisher="CMS+CDC", cadence="annual",
        fmt="zip-txt", target_table="code_set", parser="parse_icd10",
        url="https://www.cms.gov/medicare/coding-billing/icd-10-codes",
        notes="Annual (Oct) + possible April update.",
    ),
    Source(
        id="cpt", layer="CODE_SET", publisher="AMA", cadence="annual",
        fmt="csv", target_table="code_set", parser="parse_cpt",
        url="https://www.ama-assn.org/practice-management/cpt",
        license="LICENSED (AMA data agreement)",
        notes="Descriptors require the AMA license — client holds it.",
    ),
]

SOURCES_BY_ID = {s.id: s for s in SOURCES}


def due_sources(month: int) -> list[Source]:
    """Sources whose cadence fires for the given calendar month."""
    out = []
    for s in SOURCES:
        if s.cadence == "weekly":
            out.append(s)
        elif s.cadence == "quarterly" and month in QUARTER_MONTHS:
            out.append(s)
        elif s.cadence == "annual" and month in (1, 10):  # CPT Jan, ICD Oct
            out.append(s)
    return out
