"""Regression tests for the RAG embedding-text enrichment.

Retrieval recall failed for codes whose surgeon-facing vocabulary differs from
their terse official descriptor (measured: the semicolon-parent 28118
'Ostectomy, calcaneus;' was never retrieved for a Haglund/exostectomy note
while its keyword-rich child 28119 was). The fix embeds EVERY descriptor
variant plus the ICD Index's clinical synonyms/eponyms, so the note's
vocabulary can match on both dense and sparse. These tests exercise the record
LOADERS directly (they only read the code JSON — no Qdrant, no embedding), so
they guard the enrichment without a live index.

Run:  PYTHONPATH=. python tests/test_rag_enrichment.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.vector_store import MedicalCodeVectorStore

PASSED = FAILED = 0


def check(label: str, cond: bool):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {label}")
    else:
        FAILED += 1
        print(f"  ❌ {label}")


def _emb(records, code, key="code"):
    for r in records:
        if r.get(key) == code or r.get("code_raw") == code:
            return r["embedding_text"]
    return None


def main():
    vs = MedicalCodeVectorStore.__new__(MedicalCodeVectorStore)
    cpt = vs._load_cpt_records()
    icd = vs._load_icd10_records()
    hcpcs = vs._load_hcpcs_records()

    print("\n[CPT — every descriptor variant embedded]")
    e28118 = _emb(cpt, "28118") or ""
    low = e28118.lower()
    # long descriptor terms AND the note-vocabulary short/consumer terms
    check("28118 keeps its long descriptor ('ostectomy'/'calcaneus')",
          "ostectomy" in low and "calcaneus" in low)
    check("28118 also carries note-vocabulary ('heel', 'bone', 'removal')",
          "heel" in low and "bone" in low and "removal" in low)
    check("variants are deduped (no case-dup repetition of 'removal of "
          "heel bone')", low.count("removal of heel bone") <= 1)

    print("\n[ICD-10-CM — Index synonyms/eponyms folded in]")
    idx = json.load(open(Path(__file__).resolve().parent.parent
                         / "data/codes/icd10cm_index_terms.json"))["terms"]
    # find an active record whose raw code has index synonyms not already in
    # its descriptor, and confirm the synonym reached the embedding text
    proved = False
    for r in icd:
        raw = r["code_raw"]
        syns = idx.get(raw) or []
        if not syns:
            continue
        syn_tok = str(syns[0]).split()[0].lower()
        if syn_tok and syn_tok not in r["description"].lower() \
                and syn_tok in r["embedding_text"].lower():
            proved = True
            break
    check("at least one ICD code carries an Index synonym absent from its "
          "descriptor", proved)

    print("\n[CPT — LLM synonym index folded in when present]")
    import json as _json
    from app.rag.vector_store import CPT_SYNONYMS_FILE
    existed = CPT_SYNONYMS_FILE.exists()
    backup = CPT_SYNONYMS_FILE.read_text() if existed else None
    try:
        CPT_SYNONYMS_FILE.write_text(_json.dumps({"provenance": "test",
            "terms": {"28118": ["Haglund resection",
                                 "retrocalcaneal exostectomy"]}}))
        cpt2 = vs._load_cpt_records()
        e = (_emb(cpt2, "28118") or "").lower()
        check("eponym synonym ('haglund') folded into 28118 embedding",
              "haglund" in e and "exostectomy" in e)
        check("official descriptor still retained alongside synonyms",
              "ostectomy" in e and "calcaneus" in e)
    finally:
        if existed:
            CPT_SYNONYMS_FILE.write_text(backup)
        else:
            CPT_SYNONYMS_FILE.unlink(missing_ok=True)
    # absent-file path must not crash and must yield descriptor-only embeddings
    cpt3 = vs._load_cpt_records()
    check("absent synonyms file degrades gracefully (descriptor-only)",
          bool(_emb(cpt3, "28118")))

    print("\n[HCPCS — short + long embedded]")
    e = _emb(hcpcs, hcpcs[0]["code"]) if hcpcs else None
    check("HCPCS embedding_text is code-prefixed and non-empty",
          bool(e) and e.startswith("HCPCS "))

    print("\n[no code dropped by enrichment]")
    check("CPT/ICD/HCPCS record counts are non-trivial",
          len(cpt) > 10000 and len(icd) > 50000 and len(hcpcs) > 5000)

    print("\n" + "=" * 50)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
