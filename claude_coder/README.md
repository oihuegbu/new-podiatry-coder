# claude-medical-coder

A facts-first, deterministic, **autonomous** medical coder. Built as an answer to
one question: *if you were building this from scratch, would you build it the way
the current pipeline is built?* No — you'd invert the spine.

## The inversion

The existing pipeline uses the **LLM as the coder**: it reads the note *and*
picks the CPT/ICD/HCPCS codes, and deterministic validators try to catch its
mistakes afterward. That forces medical-code knowledge *into the prompts* (which
go stale every quarter) and forces expensive consistency/convergence loops to
paper over the model's nondeterministic code choices.

Medical coding is mostly **deterministic given the facts**. The genuinely
LLM-shaped job is understanding the messy note. So here:

```
note ─► 1. EXTRACT   Clinical Language Understanding: the LLM emits evidence-
        │            linked clinical FACTS (what was done, anatomy, laterality,
        │            count, depth, product, dose, performed-vs-planned, negation).
        │            It is never asked for, and never outputs, a code.
        ▼
        2. RESOLVE   Deterministic ontological linking. The DECISION is made by
        │            structured rules over features PARSED FROM the authoritative
        │            descriptors (laterality, measurement intervals, cardinality,
        │            core concept) — not by vector rank. Retrieval is demoted to
        │            RECALL: it only narrows ~10^5 codes to a candidate pool; the
        │            rules then eliminate contradictions (wrong side, size out of
        │            the descriptor's range) and pick the most specific survivor.
        ▼
        3. ARBITRATE Only on residual ambiguity: the LLM picks among the
        │            RETRIEVED candidate descriptors — it can never recall or
        │            invent a code, so a wrong pick costs a review, not a bill.
        ▼
        4. GATE      Positive, fail-closed release gates (DOS, verbatim evidence,
        │            code-active-on-DOS, medical necessity, NCCI, MUE) — each a
        │            PASS / NOT_APPLICABLE / UNKNOWN / BLOCKED assertion.
        ▼
        5. AUTONOMY  Release to billing only when the chain CLOSES (all gates
                     clear, every performed fact resolved, confidence ≥ floor);
                     otherwise escalate to a human with a precise reason.
```

This is the same shape the category leaders use. **Nym Health** (whose CLU also
powers **Fathom**) extracts clinical meaning, then applies **rules-based
ontologies that encode AMA/CMS/WHO guidelines as data**, produces an audit trail
for every decision, and **routes to billing above a confidence threshold, human
review below**. This package follows that architecture deliberately.

## How it delivers the four requirements

- **Accurate** — codes come from descriptor *entailment* against authoritative
  data (site/laterality/count must match), not the model's memory. Ambiguity is
  arbitrated over a retrieved shortlist, never guessed.
- **Billable** — positive gates check code activity on the DOS, medical necessity
  (a procedure needs a diagnosis), NCCI PTP edits and MUE limits — all read from
  the data, so they track quarterly updates automatically.
- **Defensible** — provenance is built in by construction. Every billed line
  carries its verbatim evidence span, the authoritative descriptor/edition, and
  how it was chosen; a **tamper-evident release certificate** (SHA-256 over note
  + codes + evidence + gate outcomes + verdict) proves the submitted claim is the
  one that passed every control and makes it replayable.
- **Autonomous** — the autonomy controller grants hands-off release only when the
  evidence chain closes, and abstains precisely otherwise. It codes every note it
  can defend and steps back from the rest, rather than coding everything.

## No hardcoded medical codes — anywhere

There is not a single CPT/ICD-10-CM/HCPCS literal in this package. Codes exist
only as **data pulled at runtime** from the authoritative source (`data_access`).
Swapping in next quarter's data files changes the answers with zero code change.
Even the test suite uses synthetic identifiers so it embeds no real code.

## What it reuses from this repo

The authoritative data (`data/codes/*`, `compliance.db`), `CodeReferenceDB`, the
hybrid RAG index, and the LLM client — wrapped behind the `CodeSource` port in
`data_access.py`. The *logic* (the inverted spine) is new; the *data foundation*
is reused, because it's the genuinely valuable part of the existing system.

## Run it

```bash
# key-free, no index: the full pipeline on a mock (see the tests)
PYTHONPATH=. python3 tests/test_claude_coder.py -v

# a real note (needs the RAG index + an LLM key)
python -m claude_coder.cli note.txt --dos 2026-03-14
```

## Honest boundaries (scaffolded, not yet complete)

- **Resolution** decides on laterality, measurement-interval containment,
  cardinality and concept entailment parsed from descriptors; a production build
  would add more axes (wound-depth families, multi-measurement area arithmetic,
  unit conversion) and a richer descriptor grammar.
- **Medical necessity** enforces the structural floor (a procedure needs a
  diagnosis); full LCD/NCD dx→procedure coverage linkage would query policy data.
- **Modifiers** are not yet assigned; when NCCI reports a pair as bypassable-with-
  a-modifier the line surfaces as UNKNOWN (fail-closed) rather than auto-appending
  one — a modifier engine would resolve it from the documented facts.
- **Certificate** provides an integrity hash; an HMAC with a private key would add
  non-repudiation.

Each boundary fails *closed* — it escalates to review rather than guessing — so
the system is safe to run today and improves by adding gates and attribute axes,
never by hardcoding codes.
