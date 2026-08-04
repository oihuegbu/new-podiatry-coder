# Authoritative source audit

## Imported from claude-medical-coder

The new branch reuses source data, not Claude decision logic. Imported assets
include licensed CPT descriptors, CMS HCPCS, CDC/NCHS ICD-10-CM order and
index/instruction files, CMS MUE seed data, add-on relationships, global
surgery attributes, modifier references, MCE data, place-of-service data,
Medicare coverage exports, and official policy text.

Generated synonym files, model prompts, specialty heuristics, benchmark labels,
claims registries, and Claude validation code were intentionally excluded.

## Live refresh verification

On 2026-08-04 the EC2 refresh tool resolved and downloaded the CMS 2026 Q3
practitioner PTP release as four archive members, parsed 2,633,388 effective-
dated relationships, and recorded each publisher URL, response timestamp, and
SHA-256 digest. It also downloaded and parsed the 2026 Q3 practitioner MUE
release into 15,162 rows.

The normalized PTP and MUE payloads are generated artifacts and are not stored
in Git. A clean environment regenerates them from the allowlisted CMS sources
before compiling the active snapshot.

## Data-quality findings

- The inherited HCPCS JSON has 1,326 records whose old termination date
  precedes a later activation date. The source-pack adapter explicitly treats
  that earlier termination as belonging to a prior activation; no generic
  silent date repair occurs.
- The inherited ncci_aoc_edits file is an add-on relationship table, not a PTP
  table. The new pack labels it accordingly.
- The inherited MUE effective date was one day earlier than the release named
  by its source filename. The live refresh now derives the quarter from CMS's
  release link and writes canonical effective dates.

## Refresh coverage

Automated and exercised on EC2:

- CMS professional PTP
- CMS practitioner MUE

Implemented authenticated connector:

- AMA licensed CPT JSON

Other imported seed assets remain provenance-checked but need additional
publisher-format connectors before they may be represented as continuously
refreshed capabilities. Their absence must not be treated as permission to
release a claim.

