"""issue #6 F9-R12-A: ICD-10-CM Alphabetic Index cross-reference resolution
and its DOWNSTREAM consumers.

`tools/parse_icd10cm_index.py`'s `subtree_codes` recursively follows a
<see>/<seeAlso> redirect it meets WHILE walking a subtree (not just the
top-level one-hop alias), matching the real CMS Index's own multi-hop
redirect shape (e.g. "Paronychia" -> seeAlso -> "Cellulitis, digit" ->
its own codeless "finger"/"toe" children -> see -> the direct "Cellulitis,
finger"/"toe" entries that carry the real codes). The synthetic XML below
mirrors that STRUCTURAL pattern only -- not the actual CMS content.

A second, independent regression here: the parser now writes redirect
aliases to their OWN `cross_reference_terms` map, kept separate from
`terms` (direct Index entries) specifically so `app/rag/vector_store.py`
-- which only ever reads `terms` -- can never flatten a broad redirect
alias into a per-code embedding vector (a redirect alias can span
thousands of descendant codes; embedding it into every one pollutes
retrieval for the whole family, displacing that code's own precise,
Index-authored terms out of the capped per-code slot).

Previously untested: this whole module had no dedicated test file (fixed
directly alongside the two-hop parser bug itself, discovered via CI's
pytest-collection blind spot for a DIFFERENT file -- see
tests/test_validator_checks.py's own history).
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.parse_icd10cm_index import navigate, plain_title, subtree_codes


def _main_terms(xml: str) -> dict:
    root = ET.fromstring(xml)
    out: dict = {}
    for letter in root.findall("letter"):
        for mt in letter.findall("mainTerm"):
            title = plain_title(mt).lower()
            if title:
                out.setdefault(title, mt)
    return out


# "Paronychia" -seeAlso-> "Cellulitis, digit" (codeless) -> its own codeless
# "finger"/"toe" children, each -see-> the DIRECT "Cellulitis, finger"/"toe"
# entries (siblings of "digit" under the SAME "Cellulitis" mainTerm) that
# carry the real codes -- the exact two-hop shape that silently resolved to
# zero codes before this fix.
_TWO_HOP_XML = """
<index>
  <letter>
    <mainTerm>
      <title>Cellulitis</title>
      <term>
        <title>digit</title>
        <term><title>finger</title><see>Cellulitis, finger</see></term>
        <term><title>toe</title><see>Cellulitis, toe</see></term>
      </term>
      <term><title>finger</title><code>L03.011</code></term>
      <term><title>toe</title><code>L03.031</code></term>
    </mainTerm>
    <mainTerm>
      <title>Paronychia</title>
      <seeAlso>Cellulitis, digit</seeAlso>
    </mainTerm>
  </letter>
</index>
"""

_THREE_HOP_XML = """
<index>
  <letter>
    <mainTerm>
      <title>Alpha</title>
      <term><title>beta</title><see>Alpha, gamma</see></term>
      <term>
        <title>gamma</title>
        <term><title>delta</title><see>Alpha, epsilon</see></term>
      </term>
      <term><title>epsilon</title><code>Z99.1</code></term>
    </mainTerm>
  </letter>
</index>
"""

_CYCLE_XML = """
<index>
  <letter>
    <mainTerm>
      <title>Loopa</title>
      <see>Loopb</see>
    </mainTerm>
    <mainTerm>
      <title>Loopb</title>
      <see>Loopa</see>
    </mainTerm>
  </letter>
</index>
"""


def test_two_hop_seealso_redirect_resolves_to_the_real_codes():
    main_terms = _main_terms(_TWO_HOP_XML)
    target = navigate("Cellulitis, digit", main_terms)
    assert target is not None
    assert subtree_codes(target, main_terms) == {"L03011", "L03031"}


def test_one_hop_only_walk_reproduces_the_original_bug():
    """Omitting `main_terms` (the ORIGINAL, one-hop-only signature every other
    caller still uses) must reproduce the exact silent-zero-codes bug this
    fix closes -- proving the two-hop chase above is what actually changed."""
    main_terms = _main_terms(_TWO_HOP_XML)
    target = navigate("Cellulitis, digit", main_terms)
    assert subtree_codes(target) == set()


def test_three_hop_redirect_chain_resolves():
    main_terms = _main_terms(_THREE_HOP_XML)
    beta = main_terms["alpha"].find("term")
    assert plain_title(beta) == "beta"
    # beta -see-> gamma (codeless) -> gamma's own child delta -see-> epsilon (coded):
    # two chained <see> hops, proving the recursion isn't hardcoded to depth 2.
    assert subtree_codes(beta, main_terms) == {"Z991"}


def test_a_cyclical_reference_returns_no_codes_and_terminates():
    main_terms = _main_terms(_CYCLE_XML)
    # Must terminate (no infinite recursion via the visited-set guard) and
    # find nothing -- there is no real code anywhere in the cycle.
    assert subtree_codes(main_terms["loopa"], main_terms) == set()


def test_cross_reference_aliases_are_kept_out_of_the_direct_terms_map(tmp_path, monkeypatch):
    """The parser's own `main()`, end to end: a redirect alias must land in
    `cross_reference_terms`, never merged into `terms` (issue #6 F9-R12-A --
    this is what previously let a broad alias reach the embedding loader,
    which only ever reads `terms`)."""
    import tools.parse_icd10cm_index as mod
    src = tmp_path / "index.xml"
    src.write_text(_TWO_HOP_XML)
    dst = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["parse_icd10cm_index.py", str(src), str(dst)])
    mod.main()
    data = json.loads(dst.read_text())
    assert "paronychia" not in data["terms"].get("L03011", [])
    assert "paronychia" not in data["terms"].get("L03031", [])
    assert "paronychia" in data["cross_reference_terms"].get("L03011", [])
    assert "paronychia" in data["cross_reference_terms"].get("L03031", [])


def test_a_broad_cross_reference_alias_never_enters_per_code_embedding_text(
        tmp_path, monkeypatch):
    """issue #6 F9-R12-A: `app/rag/vector_store._load_icd10_records` only ever
    reads the `terms` key -- a broad redirect alias belongs in
    `cross_reference_terms` and must never reach a per-code embedding vector,
    and must never displace that code's own direct terms out of the capped
    per-code slot."""
    from app.rag import vector_store as vs_mod

    code = "L03011"
    direct_terms = [f"direct term {i}" for i in range(vs_mod._INDEX_TERMS_PER_CODE)]
    icd10_records = [{"code": "L03.011", "long_description": "cellulitis of right finger",
                      "status": "active"}]
    index_terms = {"version": "test", "source": "synthetic",
                  "terms": {code: direct_terms},
                  # A broad alias real Index redirects can carry (e.g.
                  # 'paronychia' spanning the whole family) -- must stay OUT
                  # of the embedding text entirely.
                  "cross_reference_terms": {code: ["paronychia"]}}

    icd10_file = tmp_path / "icd10cm_codes.json"
    icd10_file.write_text(json.dumps(icd10_records))
    index_terms_file = tmp_path / "icd10cm_index_terms.json"
    index_terms_file.write_text(json.dumps(index_terms))
    missing_synonyms_file = tmp_path / "icd10_synonyms_absent.json"

    monkeypatch.setattr(vs_mod, "ICD10_FILE", icd10_file)
    monkeypatch.setattr(vs_mod, "ICD10_INDEX_TERMS_FILE", index_terms_file)
    monkeypatch.setattr(vs_mod, "ICD10_SYNONYMS_FILE", missing_synonyms_file)

    store = vs_mod.MedicalCodeVectorStore.__new__(vs_mod.MedicalCodeVectorStore)
    records = store._load_icd10_records()
    assert len(records) == 1
    embedding_text = records[0]["embedding_text"].lower()

    assert "paronychia" not in embedding_text
    for term in direct_terms:
        assert term in embedding_text, (term, embedding_text)
