# GPT Medical Coder

This repository implements a source-compiled, evidence-to-claim coding kernel.
Language models may extract documented facts and rank candidates that the
terminology service supplied; they cannot create a billable identifier or
override a deterministic gate.

The software kernel contains no medical identifiers, descriptor dictionaries,
specialty terms, or code-family shortcuts. Those values live in versioned
source packs compiled from licensed or official releases.

## Decision path

1. Preserve immutable source spans for every extracted fact.
2. Normalize facts independently of coding.
3. Resolve the date-effective source snapshot.
4. Retrieve candidates only from that snapshot.
5. Validate identity, effective date, billability, descriptor requirements,
   specificity, constraints, units, claim context, coverage, and provenance.
6. Release only when every mandatory gate passes; model confidence is never a
   release gate.

The first imported pack comes from the claude-medical-coder branch, but only
its authoritative code/policy assets are reused. Its model prompts, coding
logic, generated synonyms, benchmarks, and specialty heuristics are excluded.

## Build a snapshot

    python -m medical_coder compile-sources \
      --pack source-packs/authoritative/pack.json \
      --output build/snapshots

The command emits an immutable directory named by the manifest hash. Reusing
the same source bytes is idempotent; changing any source or pack definition
creates a different snapshot.

## Safety boundary

This foundation can autonomously release only encounters for which imported
source semantics fully describe every material requirement. A descriptor-only
source can supply candidates but cannot, by itself, prove descriptor
entailment. Missing source semantics route to a structured technical state,
not to fabricated logic or a provider query.

