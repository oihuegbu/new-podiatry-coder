# claude-medical-coder

A **facts-first, deterministic, autonomous** medical coder. It turns a clinician's
note into accurate, billable, defensible CPT / ICD-10-CM / HCPCS codes — and
releases to billing with *no human* only when it can prove the claim, escalating
everything else with a precise reason.

It is built on one design decision: **invert the spine** of a conventional
LLM-coder. The model does the genuinely language-shaped job (understanding messy
prose); *authoritative data* does the code assignment. There is **not a single
medical-code literal, condition, drug, or eponym anywhere in the Python** — codes
and clinical vocabulary exist only as data pulled at runtime, so next quarter's
NCCI/MUE/CPT/ICD update changes the answers with zero code change.

---

## Table of contents

1. [The inversion — why this architecture](#the-inversion)
2. [Design invariants](#design-invariants)
3. [Pipeline architecture](#pipeline-architecture)
4. [The resolution ladder](#the-resolution-ladder)
5. [Diagnosis specificity & Excludes1 (the entailment-is-not-enough fixes)](#diagnosis-specificity--excludes1)
6. [Claim-shaping mechanics](#claim-shaping-mechanics)
7. [Release gates](#release-gates)
8. [The autonomy controller](#the-autonomy-controller)
9. [The learned verified-resolution index](#the-learned-verified-resolution-index)
10. [Documentation recommendations](#documentation-recommendations)
11. [The release certificate](#the-release-certificate)
12. [Data grounding — the `CodeSource` port](#data-grounding)
13. [Module reference](#module-reference)
14. [Workflow — a note end to end](#workflow)
15. [Running it](#running-it)
16. [Data files & data-prep tooling](#data-files--tooling)
17. [Testing & the hardcoding guards](#testing--guards)
18. [Engineering conventions](#engineering-conventions)
19. [Honest boundaries](#honest-boundaries)

---

## The inversion

A conventional pipeline uses the **LLM as the coder**: it reads the note *and*
picks the codes, and deterministic validators try to catch its mistakes
afterward. That forces medical-code knowledge *into the prompts* (which go stale
every quarter) and forces expensive consistency/convergence loops to paper over
the model's nondeterministic code choices.

Medical coding is mostly **deterministic given the facts**. The genuinely
LLM-shaped job is understanding the note. So the spine is inverted:

```
note ─► 1. EXTRACT    Clinical Language Understanding. The LLM emits evidence-
        │             linked clinical FACTS (act, anatomy, laterality, count,
        │             depth, product, dose, performed-vs-planned, negation).
        │             It is never asked for, and never outputs, a code.
        ▼
        2. RESOLVE    Deterministic ontological linking. Retrieval is demoted to
        │             RECALL (narrow 10^5 codes to a pool); the DECISION is made
        │             by structured rules over features PARSED FROM authoritative
        │             descriptors — laterality, measurement intervals, cardinality,
        │             concept — eliminating contradictions and ranking specificity.
        │             What the deterministic path can't ground goes to →
        ▼
        2b. VERIFY    Propose-then-verify (license-clean CPT-Index substitute):
        │             the LLM PROPOSES candidate code numbers (validated against
        │             the registry — it can't invent one), a code is accepted only
        │             when its AUTHORITATIVE DESCRIPTOR is ENTAILED by the facts,
        │             and an INDEPENDENT second model must corroborate. Nothing
        │             bills on recall or on one model's say-so.
        ▼
        3. SHAPE      Claim-level mechanics from the data: dedup, section
        │             applicability, modifiers, NCCI PTP bundling, integral
        │             bundling, global surgical package.
        ▼
        4. GATE       Positive, fail-closed release gates (DOS, verbatim evidence,
        │             code-active-on-DOS, medical necessity, NCCI, MUE, ICD
        │             Excludes1) — each a PASS / NOT_APPLICABLE / UNKNOWN / BLOCKED
        │             assertion. Silence is never consent.
        ▼
        5. AUTONOMY   Release to billing only when the chain CLOSES (all gates
                      clear, every performed fact resolved, confidence ≥ floor);
                      otherwise escalate with a precise reason + a documentation
                      recommendation. Emit a SHA-256 release certificate.
```

This mirrors the architecture the autonomous-coding leaders use: extract clinical
meaning, apply rules-based ontologies that encode AMA/CMS/WHO guidance **as data**,
produce an audit trail for every decision, and **route to billing above a
confidence threshold, human review below**.

---

## Design invariants

These five properties are enforced, not aspirational — the first two by a CI guard
(`tests/check_no_hardcoding.py`).

1. **No hardcoded medical codes.** No CPT/ICD/HCPCS/modifier literal, no
   `code.startswith("...")` code-family classification, no prefix ranges. Every
   code-dependent decision is resolved by querying authoritative data.
2. **No scenario-specific vocabulary.** No real condition, eponym, drug, or region
   is named in the logic *or the prompts* — prompts state general principles, never
   worked medical examples. The coder is specialty-agnostic; a term-denylist guard
   enforces it across the package.
3. **Fail-closed.** Every uncertainty escalates to review rather than guessing. A
   check that cannot run is `UNKNOWN` (stops autonomy), never a silent pass.
4. **Provenance by construction.** Every code that reaches a claim carries the
   evidence span it came from, the fact it resolved, the authoritative record that
   defines it, and how it was chosen — captured in a tamper-evident certificate.
5. **New deterministic rules are config, not code.** Rule behaviour lives as
   versioned data (descriptor grammar, lexicons, authority citations); Python
   implements only generic mechanics. Every rule cites its authority
   (CPT/ICD guideline, CMS manual, NCCI policy).

---

## Pipeline architecture

`pipeline.code_encounter()` is the orchestrator. Every step is pluggable — pass a
`MockSource` and stub LLMs to run the whole thing deterministically in a test, or
the real `AuthoritativeSource` in production.

```
extract_facts ─► for each fact:
                   EM   → em.resolve_em            (MDM-grid leveling)
                   else → resolution.resolve       (deterministic → propose-verify)
                          ├─ arbitration.arbitrate (only residual ambiguity, non-PV kinds)
                          ├─ learned.observe        (feed VERIFIED procedures to the learned index)
                          └─ resolution.refine_diagnosis_specificity  (dx specificity upgrade)
                 ─► dedup_lines
                 ─► apply_section_applicability     (anesthesia-section suppression)
                 ─► modifier_engine.assign_claim    (E/M-25, distinct-service 59/X)
                 ─► apply_ncci_bundling             (demote the PTP component)
                 ─► apply_integral_bundling         (escalated ancillary → integral)
                 ─► apply_global_package            (E/M into the global surgical package)
                 ─► gates.run_gates
                 ─► autonomy.decide                 (verdict + audit notes)
                 ─► recommendations.build           (actionable documentation guidance)
                 ─► certificate.build               (SHA-256 defensibility packet)
```

---

## The resolution ladder

`resolution.resolve()` tries the cheapest, most authoritative path first and only
escalates to a more expensive, less certain one when needed. **Every rung is
grounded in authoritative data; nothing bills on vector rank or model memory.**

| Rung | Mechanism | When it fires |
|---|---|---|
| **Authoritative index** | ICD-10-CM Alphabetic Index (`index_codes`) → SNOMED→ICD map (`snomed_codes`) for diagnoses; CMS Table of Drugs (`drug_index_codes`) → AMA CPT Index (`cpt_index_codes`) → learned index (`learned_index_codes`) → CPT/HCPCS descriptor index (`procedure_index_codes`) for procedures/supplies/drugs | A single, unambiguous authoritative term→code hit. Taken deterministically. |
| **Structured decision** | `_decide` / `_evaluate`: eliminate candidates that *contradict* documented attributes (wrong laterality, measurement outside the descriptor's interval); rank survivors by recall, then specificity, then concept support. | The index has no clean hit; a retrieval pool exists. |
| **Propose-then-verify** | `verify.propose_codes` (LLM proposes, registry validates) → `verify.select_entailed` (descriptor entailment) → `verify.corroborate` (independent second model). Bounded re-selection on a wrong-concept rejection; a `missing_element` rejection becomes a provider query. | Procedures/imaging, and diagnoses that reach the embedding fallback. The license-clean substitute for the AMA CPT Index. |
| **Arbitration** | `arbitration.arbitrate`: a single bounded LLM pick over the *retrieved* candidate descriptors — it can never recall or invent a code. | Residual ambiguity for kinds that did **not** go through propose-then-verify. |

Retrieval is the repo's hybrid dense(bge)+sparse(BM25) RRF store, reused **as
recall only**: it supplies candidate code identities and a relevance score; the
authoritative record always supplies the descriptor and the truth.

---

## Diagnosis specificity & Excludes1

Two root-cause fixes for a class the original design missed: **entailment is
necessary but not sufficient.**

### Specificity upgrade — `resolution.refine_diagnosis_specificity`
An *unspecified* / NOS descriptor is entailed by **every** case in its concept, so
it passes an entailment check trivially and can be billed when a more specific code
is supported. Two steps, most conservative first:

1. **Structural laterality upgrade** (no LLM) — an unspecified-*laterality* code →
   the documented-side sibling in the same descriptor family.
2. **Verified specificity upgrade** (real mode) — for a broader unspecified/NOS
   code whose specific counterpart lives in a *different* descriptor family, gather
   the code's more-specific, on-concept, documented-side relatives from its **own
   authoritative category leaves**, offer `{chosen + relatives}` to the same
   entailment verifier, and adopt a strictly-more-specific relative it selects *and*
   an independent model confirms. If a specific relative is proposed but the
   independent check splits, **escalate** rather than silently bill the unspecified
   code (fail-closed). If no specific relative exists, keep the unspecified code.

*Effect (real data): `M77.9 "Enthesopathy, unspecified"` → `M77.51 "Other
enthesopathy of right foot and ankle"`.*

### ICD-10-CM Excludes1 gate — `gates.icd_excludes_gate`
The gate suite modeled *procedure*-side edits (NCCI/MUE) but never *diagnosis–
diagnosis* Tabular constraints. Excludes1 means two conditions are mutually
exclusive and must not be reported together — **unless genuinely unrelated**, an
FY-guideline exception that is a human judgement. The gate reads `excludes1` refs
from `icd10cm_instructional_notes.json` (data that was loaded but had no consumer),
walks each diagnosis's category ancestors, and returns **`UNKNOWN`** (stops
autonomy → review) on a co-occurrence — never auto-releasing, never silently
dropping a code.

*Effect (real data): billed `M71.571` + `M77.x` → `UNKNOWN — confirm the conditions
are unrelated before reporting together`.*

---

## Claim-shaping mechanics

All directionality comes from the data; no code is named in any of these.

- **`dedup_lines`** — two phrases that resolve to the same code are one line
  (evidence merged), preventing accidental double-billing.
- **`apply_section_applicability`** — an **anesthesia-section** service (detected
  from descriptor grammar, not a code range) is not separately reportable by the
  operating provider unless a separate anesthesia provider is documented. Also
  decides an *escalated* line deterministically when its leading, dominant
  candidate is an anesthesia code.
- **`ModifierEngine`** (`modifiers.py`) — laterality (RT/LT), bilateral (50, gated
  by the CMS bilateral indicator), E/M-25 (only when a separately-identifiable E/M
  is documented), and distinct-service (X{ESU}/59, only with a documented basis —
  never appended just to clear an edit). Modifier *values* are discovered from
  `modifiers.json` by matching each modifier's own description.
- **`apply_ncci_bundling`** — an NCCI PTP edit is a *resolution*, not a block: demote
  the authoritative column-2 component; honour the CPT "(separate procedure)"
  designation.
- **`apply_integral_bundling`** — an *escalated* ancillary procedure that is an NCCI
  indicator-0 (always-bundled) component of a billed primary is decided as integral
  (bundled), not sent to review. Safe by construction: only ever turns an escalation
  into a non-billed exclusion.
- **`apply_global_package`** — a same-day E/M is bundled into a procedure's global
  surgical package unless significant, separately-identifiable work is documented.

---

## Release gates

Every gate is a **positive assertion**; release requires PASS or a proven
NOT_APPLICABLE. `UNKNOWN`/`ERROR`/`BLOCKED` all stop autonomy.

| Gate | Asserts | Source |
|---|---|---|
| `date_of_service` | a DOS is present (every date check depends on it) | input contract |
| `verbatim_evidence` | every billed line is supported by verbatim note text | the note |
| `code_active_on_dos` | every code is active on the DOS | authoritative source |
| `medical_necessity` | every procedure has a supporting diagnosis (structural floor) | necessity |
| `ncci_ptp` | no unresolved PTP conflicts among billed procedures | NCCI PTP data |
| `mue` | units within the MUE limit | MUE data |
| `icd_excludes1` | no Excludes1 conflict among billed diagnoses | ICD-10-CM Tabular data |

---

## The autonomy controller

`autonomy.decide` grants hands-off release **only when the chain closes**: no
gate BLOCKED/ERROR, no gate UNKNOWN, every *performed* fact resolved, and every
released line's confidence ≥ the floor (`AUTONOMY_CONFIDENCE = 0.95`, a policy
dial). `DETERMINISTIC` and `VERIFIED` (cross-model-confirmed) lines are gated only
by how well the underlying fact is documented; a single-model `ARBITRATED`
tie-break is discounted. Verdicts: `AUTO_READY`, `REVIEW_REQUIRED`, `BLOCKED` —
each with an audit note naming exactly why.

---

## The learned verified-resolution index

Turns propose-then-verify from probabilistic into deterministic-on-repeat, without
a licensed index and without human sign-off:

- **Observe** — every accepted (entailed) resolution appends `(normalized phrase →
  code)` + evidence to an append-only log.
- **Promote** — an offline step (`tools/build_learned_index.py`) promotes a mapping
  to deterministic trust only when it is confirmed across ≥ `PROMOTE_AT` (default 3)
  **distinct** encounters and dominates any competitor. No single note can promote
  itself.
- **Resolve** — `learned_index_codes` serves the promoted crosswalk, but
  **self-invalidates**: an entry is honoured only while the code still exists *and*
  its current authoritative descriptor still matches the one that was verified.

---

## Documentation recommendations

`recommendations.build_recommendations` turns every escalation into actionable,
provider-facing guidance: `documentation_gap` (a code fits but an element is
undocumented → a targeted provider query), `unresolved_service` (documentation too
thin to pick any code), and `gate_<name>` (what to fix to clear a blocked gate).
Deterministic and agnostic — it reads fact kinds, methods, and gate outcomes, never
a code.

---

## The release certificate

`certificate.build_certificate` binds the released claim to everything it depends
on — the note (by SHA-256), the DOS, each billed line with its verbatim evidence
and authoritative record, every gate outcome, and the verdict — then
content-addresses the whole packet with a SHA-256. Re-running the same inputs
reproduces the same hash; changing the note, a code, an evidence span, a gate
outcome, or the source edition invalidates it. This is what makes an autonomous
claim answerable after the fact and byte-for-byte replayable.

---

## Data grounding

`data_access.py` defines the `CodeSource` **Protocol** (port) with two
implementations: `AuthoritativeSource` (production, over the repo's real data) and
`MockSource` (tests, synthetic identifiers only). Everything the coder knows about
codes it asks this layer:

- **Descriptors** — all authoritative tiers (long / medium / consumer / short),
  richest first, from `data/codes/<system>_codes.json`.
- **Recall** — the hybrid RAG vector store (`retrieve`).
- **Term indexes** — ICD-10-CM Alphabetic Index, SNOMED→ICD map, AMA CPT Index,
  CMS Table of Drugs, CPT/HCPCS descriptor index, learned index.
- **Policy** — CMS PFS global period & bilateral indicator, NCCI PTP directional
  edits (`check_ncci`), MUE limits, code activity windows, separately-billable
  signals, and ICD Excludes1 refs.

Swap the data files and the answers change with no code change.

---

## Module reference

| Module | Responsibility |
|---|---|
| `pipeline.py` | Orchestration (`code_encounter`) + all claim-shaping mechanics + `render`. |
| `extraction.py` | Stage 1 CLU — note → evidence-linked `ClinicalFact`s (code-free prompt). |
| `resolution.py` | Stage 2 — deterministic resolution ladder, propose-then-verify driver, laterality + specificity upgrades. |
| `verify.py` | Propose / select-entailed / corroborate — the license-clean CPT-Index substitute. |
| `em.py` | E/M leveling from the MDM 2-of-3 grid + descriptor setting/new-vs-established. |
| `arbitration.py` | Bounded single-LLM pick over *retrieved* descriptors (residual ambiguity only). |
| `ontology.py` | Descriptor grammar: `parse_descriptor`, measurement intervals, `code_section`, `is_separate_procedure`, `support_score`, dose/drug-unit parsing, `billing_units`. |
| `terminology.py` | `TerminologyIndex` — inverts authoritative term→code maps into robust (exact / compound / order-&-plural-independent) lookups. |
| `modifiers.py` | Data-driven modifier discovery + per-line and claim-level assignment. |
| `gates.py` | Positive, fail-closed release gates. |
| `autonomy.py` | Calibrated abstention → `AUTO_READY` / `REVIEW_REQUIRED` / `BLOCKED`. |
| `learned.py` | Observe → promote → self-invalidating learned index. |
| `recommendations.py` | Escalations → actionable documentation guidance. |
| `certificate.py` | SHA-256 tamper-evident release certificate. |
| `data_access.py` | The `CodeSource` port + `AuthoritativeSource` / `MockSource`. |
| `models.py` | Core dataclasses/enums — provenance built in; zero code literals. |
| `cli.py` | `python -m claude_coder.cli note.txt --dos YYYY-MM-DD [--json]`. |

---

## Workflow

Taking a right retrocalcaneal exostectomy operative note as the running example:

1. **Extract** → facts: exostectomy (procedure, right, calcaneus), Achilles
   debridement/reattachment (procedures), retrocalcaneal bursectomy (procedure),
   suture anchors (supply), intraoperative fluoroscopy (imaging), Haglund deformity
   / retrocalcaneal bursitis / insertional Achilles degeneration (diagnoses),
   anesthesia (procedure), plus historical PMH & prior imaging (marked historical).
2. **Resolve** → the exostectomy grounds to a calcaneal ostectomy via
   propose-then-verify (descriptor entailed, independently confirmed, no plantar-spur
   variant); the anchors resolve deterministically; diagnoses resolve and the
   enthesopathy is **sharpened from the unspecified code to the right foot/ankle
   code** by the specificity upgrade.
3. **Shape** → anesthesia suppressed (operating-provider claim); historical items
   not billed; units/modifiers applied to CPT/HCPCS only.
4. **Gate** → DOS, evidence, activity, necessity, NCCI, MUE pass; **Excludes1** flags
   the bursitis/enthesopathy pair for the unrelated-conditions judgement.
5. **Autonomy** → genuine ambiguities (is the reattachment a *secondary* repair? is
   the debridement separately reportable?) become precise **provider queries**;
   verdict `REVIEW_REQUIRED`; certificate emitted. The system codes what it can
   defend and steps back — with reasons — from the rest.

---

## Running it

```bash
# Real note (needs the RAG index + an LLM key; real mode auto-enables
# propose-then-verify + cross-model corroboration):
python -m claude_coder.cli path/to/note.txt --dos 2026-01-05
python -m claude_coder.cli - --dos 2026-01-05 --json   # stdin, JSON certificate

# Key-free, no index — the whole pipeline on a MockSource with stubbed LLMs:
python -m pytest tests/test_claude_coder.py -q
```

Verify env: `CLAUDE_VERIFY_MODEL` / `CLAUDE_VERIFY_EFFORT` select the independent
corroboration model (typically the Opus verification tier); `LEARNED_PROMOTE_AT`
tunes the learned-index promotion threshold.

---

## Data files & tooling

Consumed from `data/codes/` (all sourced from real AMA/CMS/NCHS files with
provenance): `<system>_codes.json`, `icd10cm_index_terms.json`,
`icd10cm_instructional_notes.json`, `snomed_icd10_map.json`, `cpt_index_terms.json`,
`hcpcs_drug_table.json`, `learned_cpt_index.json`, `global_period.json`,
`modifiers.json`, `em_mdm_grid.json`; plus `compliance.db` (NCCI/MUE) and the RAG
index.

Data-prep tools (automated source ingestion → preparation → integration; each a
drop-in that the coder degrades gracefully without): `tools/parse_icd10cm_index.py`,
`tools/parse_cpt_index.py`, `tools/build_snomed_icd10_map.py`,
`tools/build_hcpcs_drug_table.py`, `tools/build_global_period.py`,
`tools/build_learned_index.py`, `tools/refresh_authoritative_data.py`.

---

## Testing & guards

- **`tests/test_claude_coder.py`** — the whole pipeline over a `MockSource` with
  stubbed LLMs; synthetic identifiers only (the suite embeds no real code).
- **`tests/check_no_hardcoding.py`** — the CI guard. AST-based detection of code
  literals, `range()` code families, `code.startswith/endswith` classification, and
  ≥2-modifier collections across `app/`, `claude_coder/`, and `tools/` (test scripts
  excluded per contract). Plus a **domain-term denylist** over `claude_coder/` + its
  data-prep tools that flags any real condition/eponym/drug/region in code,
  comments, or docstrings — so the coder stays specialty-agnostic.

---

## Engineering conventions

From the repo's `CLAUDE.md`, enforced here:

- **Fixes are architectural, not patches.** Fix the source of truth and prove it
  from a clean build — never a hand-patch on a live system.
- **No hardcoded medical codes in fixes.** Query authoritative data; never
  `if code in {...}` or a prefix range.
- **New deterministic rules are config, not code.** Generic mechanics in Python;
  rules as versioned data that cites its authority.
- **Every fix gets a deliberate post-fix review** — re-read the full path, check the
  failure paths, look for adjacent instances of the same bug class, check boundary
  interactions.

---

## Honest boundaries

Each boundary **fails closed** (escalates to review), so the system is safe to run
today and improves by adding gates/axes, never by hardcoding codes.

- **Resolution** decides on laterality, measurement-interval containment,
  cardinality, and concept entailment; more axes (wound-depth families,
  multi-measurement area arithmetic, richer descriptor grammar) are additive.
- **The AMA CPT Alphabetic Index** is licensed; until ingested, propose-then-verify
  + the learned index are the license-clean substitute.
- **SNOMED→ICD** long-tail synonym resolution needs a (free) UMLS license to build;
  empty until then, and the coder degrades to the ICD Index + retrieval.
- **Medical necessity** enforces the structural floor; full LCD/NCD dx→procedure
  coverage linkage would query policy data.
- **Facility vs professional claim context** (e.g. HCPCS C-codes are facility/OPPS
  device codes, often packaged) and **CPT-policy determinations** (integral vs
  separately reportable when no NCCI edit exists) are surfaced for review rather
  than auto-decided.
- **The certificate** provides an integrity hash; an HMAC with a private key would
  add non-repudiation.
- **Two independent models can be jointly wrong** — cross-model corroboration lowers
  that risk but does not eliminate it; the autonomy floor and human review are the
  backstop.
