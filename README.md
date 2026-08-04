# GPT Medical Coder

GPT Medical Coder is an autonomous, evidence-to-claim coding service for medium-sized private surgical practices. It converts a doctor’s note into proposed professional claim lines only when two independent evidence extractions and every deterministic source-backed gate agree.

The service is designed to produce billable and defensible output, not to let a language model invent billing identifiers. OpenAI and Anthropic each read the note once and return clinical facts with exact source spans. They cannot return medical codes. All identifiers come from an immutable SQLite snapshot compiled from licensed AMA or official CMS/CDC source files.

## Processing contract

1. Ingest PDF, DOCX, text, or image input; hash the original bytes and preserve page/character boundaries.
2. Extract facts independently with OpenAI and Anthropic using strict structured output.
3. Normalize each provider’s raw clinical phrases separately against the date-effective SNOMED CT US terminology, preserving the raw phrase, source span, candidate expansion, confidence, and alternatives.
4. Reconcile facts, attributes, negation, status, anatomy, relationships, and SNOMED identities deterministically. Material disagreement cannot reach billing normalization or claim construction.
5. Resolve the agreed clinical facts against licensed billing descriptors and retrieve candidate identifiers from licensed descriptors and official indexes. Index, SNOMED, and UMLS synonym matches are candidates only; they never satisfy descriptor confirmation by themselves.
6. Construct procedure-diagnosis, unit, and modifier relationships only from agreed evidence.
7. Evaluate effective date, active/billable status, descriptor entailment, specificity, ICD instructions, NCCI PTP/add-on edits, MUE units, modifier evidence, place of service, payer/jurisdiction coverage, medical necessity, and provenance.
8. Release claim lines only when every mandatory gate passes. Persist a content-addressed decision certificate and an append-only audit hash chain.

Completed encounters are idempotent: retrying the same encounter, note hash, context, and source snapshot returns the stored result without paying for another pair of model calls.

## CPT retrieval without CPT Link

CPT Link is optional. When it is unavailable, the compiler builds its search corpus from every licensed CPT descriptor field in `cpt_codes.json`: long, medium, short, and consumer descriptions. The long descriptor remains the confirmation target.

This is deliberately not a generated imitation of the AMA Alphabetic Index. Generated medical-code synonym maps from earlier branches are excluded. Rare eponyms or abbreviations that the licensed descriptors cannot resolve uniquely remain a source-data gap rather than being guessed.

If CPT Link later becomes available, `licensed-cpt-link-index` accepts the licensed delimited file through a configured AMA endpoint. Its parser is header-driven, preserves index terms, expands ranges only to members of the licensed CPT release, and imports entries as retrieval candidates. Coding still requires independent descriptor confirmation.

SNOMED CT improves clinical phrase normalization but is not a substitute for CPT Link and is never treated as a direct CPT map. The optional UMLS adapter uses only non-suppressed English MRCONSO rows, limits the target side to identifiers present in the installed licensed CPT release, and associates SNOMED descriptions by shared UMLS concept solely for candidate recall. A UMLS association cannot pass the descriptor-entailment gate.

## Authoritative source plane

`source-packs/authoritative/pack.json` compiles:

- licensed CPT descriptors and concept identifiers;
- the NLM SNOMED CT US Edition active concept and English-description snapshots;
- CMS HCPCS and place-of-service data;
- CDC/NCHS ICD-10-CM tabular codes, Alphabetic Index terms, and instructional notes;
- CMS NCCI practitioner PTP, add-on, and practitioner MUE constraints;
- CMS PFS global/bilateral attributes;
- CMS Medicare Coverage Database articles, diagnosis-group composition, procedure scope, effective periods, and MAC jurisdictions;
- authoritative modifier definitions and policy documents.

`source-packs/refresh/sources.json` defines allowlisted, schema-checked refresh adapters. The deployed refresh timer downloads PTP, MUE, PFS attributes, and MCD articles; when `UMLS_API_KEY` is configured it also downloads the current NLM SNOMED CT US RF2 and UMLS MRCONSO releases. It parses all sources in a staging tree, compiles and tests a shadow snapshot, then installs the entire batch and atomically activates the new snapshot. Any failure rolls the source files back. Credentials are not written into download provenance. Licensed CPT and CPT Link adapters require explicitly configured AMA credentials/endpoints.

An already licensed SNOMED CT US archive can be prepared without extracting archive paths:

```bash
python -m medical_coder prepare-snomed-us \
  --archive data/sources/licensed/SnomedCT_US.zip \
  --output data/codes/snomed_us_terminology.json
```

The importer validates ZIP CRCs, required RF2 snapshot members and headers, active relationships, minimum content counts, and records the release identifier plus source SHA-256. RF2 language-reference-set acceptability is not inferred: the generated `display_term` is a deterministic label, while all active English descriptions remain searchable.

No code-dependent decision is encoded as a medical identifier, prefix, or family in Python.

## Build and verify

```bash
python -m medical_coder compile-sources \
  --repository-root . \
  --pack source-packs/authoritative/pack.json \
  --output build/snapshots
python -m unittest discover -s tests
python tools/check_no_hardcoding.py
```

Each snapshot directory is named by a hash of the compiler, pack, and source bytes. It is read-only and can be activated atomically:

```bash
python -m medical_coder activate-snapshot \
  --snapshot build/snapshots/<snapshot-id> \
  --link build/snapshots/current
```

## Service and deployment

The service exposes authenticated `POST /v1/code`, `GET /healthz`, and `GET /readyz` endpoints. It listens on loopback by default and is intended to sit behind the practice’s private TLS ingress.

Deployment is source-driven:

```bash
sudo deploy/install.sh
```

The installer refuses placeholder credentials, requires explicit PHI-transmission authorization, runs the full test and no-hardcoding suites, compiles and activates a snapshot, and installs hardened systemd service/refresh units.

## Honest validation boundary

The architecture makes unsupported claims mechanically difficult and records why every released line passed. It does not establish a measured claim-accuracy rate. Without a sufficiently broad, independently double-reviewed gold corpus, sensitivity, specificity, false-positive rate, and claim-level accuracy improvement cannot yet be quantified.
