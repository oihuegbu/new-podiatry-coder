"""Parse the official ICD-10-CM Alphabetic Index XML (CDC/NCHS,
icd10cm-index-*.xml) into data/codes/icd10cm_index_terms.json:

    {"version": "...",
     "terms": {"<dotless code>": ["phrase", ...]},
     "cross_reference_terms": {"<dotless code>": ["alias phrase", ...]},
     "reference_directives": {"total": N, "resolved": N,
                              "external_table_reference": N, "unresolved": N,
                              "unresolved_directives": ["<tag>: <raw ref text>", ...]}}

Two DISTINCT trust tiers, kept in separate maps (issue #6 F9-R12-A) rather
than merged, because they mean different things and downstream consumers
must not treat them alike:

1. `terms` -- direct entries: every term node carrying a <code>, phrase =
   the chain of plain title texts from the mainTerm down (nonessential
   modifiers in <nemod> are dropped). Trailing '-' on codes (incomplete
   stems like L03.03-) is stripped; consumers prefix-match. This is a
   precise clinician-term -> code mapping, safe to embed per-code.
2. `cross_reference_terms` -- redirect aliases: any Index node (mainTerm OR
   a nested <term> at ANY depth) whose <see>/<seeAlso> points elsewhere
   contributes its own FULL navigational path as an alias phrase on every
   code under the referenced target's subtree ("Paronychia — see also
   Cellulitis, digit" maps 'paronychia' to the L03.0x cellulitis family,
   not the L03.04x lymphangitis family). A redirect means "keep navigating
   under this term", not "this bare word is an equivalent synonym for
   every descendant code" -- a broad redirect can fan out to hundreds of
   codes, so `cross_reference_terms` must never be flattened into a
   per-code embedding vector (that pollutes retrieval for every code in
   the family). It is only for exact Index lookup, which already defers a
   multi-code hit to candidate narrowing rather than trusting it blindly
   (see `claude_coder.resolution`).

issue #6 F9-R12-B (Codex's independent structural check against the real
FY2026 CDC/NCHS source archive): an earlier version of this compiler only
scanned TOP-LEVEL <mainTerm> nodes for their OWN <see>/<seeAlso>, skipped
any node that already carried a direct <code>, and read only ONE of
<see>/<seeAlso> when a node had both. Measured against the real source:
5,400 reference directives -- 104,823 source code associations -- were
silently dropped this way. Fixed by walking EVERY node at every depth
(`iter_index_nodes`), reading ALL <see>/<seeAlso> elements on each node
(`reference_texts`), and letting a node contribute both its own direct
code AND every one of its redirects (`subtree_codes` no longer treats
"has a code" and "has a redirect" as mutually exclusive). Every directive
is now classified (resolved / a known "Table of ..." external-reference
convention this compiler has no table to navigate into / genuinely
unresolved) and the counts persist in the artifact's own
`reference_directives` block -- so a directive this compiler cannot
resolve is a visible, counted gap, never a silent drop.

issue #6 F9-R12-C (Codex's re-review of the F9-R12-B fix): two further
accuracy gaps in that same measurement. (1) "resolved" only meant
`navigate()` found A node, not that the node's own subtree actually
reaches any code -- 55 such directives counted as resolved while
producing zero candidate edges; fixed by requiring a NONEMPTY code set.
(2) A main term whose own title is a composite of several official
comma-joined heading forms was indexed under one full-title key, but a
directive elsewhere routinely cites just one heading component, which
never matched; 6,392 of the "unresolved" directives had a first segment
identifying exactly one such component. Fixed by also indexing each
component, but ONLY when it names a single main term across the whole
source (an ambiguous component -- 5 in the real source -- is left
unresolved rather than guessed).

issue #6 F9-R12-C, second re-review: the single flat `_components` map
above was itself structurally wrong -- an unrelated main term's own
SECONDARY (non-first) comma qualifier could collide with, and block, a
totally different main term's CANONICAL first heading, leaving 102
distinct reference texts (449 directives) unresolved even though they
identified a unique canonical heading. Fixed with three PRIORITY tiers
(exact full title, then each main term's OWN position-0 heading, then
its remaining position-1+ qualifiers), consulted in order -- a tier
claims a token (resolving it if unique, leaving it unresolved but BLOCKED
if ambiguous) before the next, lower-priority tier is ever consulted for
that same token. `unresolved_directives` in the artifact is now the
COMPLETE list, never a capped sample.

Usage: python tools/parse_icd10cm_index.py <icd10cm-index-*.xml> [out.json]
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

CODE_RE = re.compile(r"^[A-Z][0-9][0-9A-Z](?:\.[0-9A-Za-z]{1,4})?-?$")

#: The Index's OWN structural convention for a redirect that leaves this
#: document entirely, into a separate table this compiler does not model
#: (e.g. "see Table of Drugs and Chemicals", "see Table of Neoplasms") --
#: a formatting/section-heading convention of the source, not a diagnosis
#: name, matched the same way `_rvu_release_effective_from` elsewhere in
#: this codebase matches a source's own declared convention rather than
#: guessing.
_EXTERNAL_TABLE_PREFIX = "table of"

#: The Index's OWN universal INSTRUCTIONAL convention -- "see condition"
#: on a bare adjective/qualifier mainTerm ('Accidental', 'Acute', ...)
#: tells the CODER to look up their actual diagnosis term as the main
#: entry; it is not a literal pointer anywhere in this document. A real
#: mainTerm happens to ALSO be titled "Condition" in the source, so
#: navigating this instruction literally resolves it to that unrelated
#: entry's own subtree -- discovered via this round's own verification
#: (606 directives use this convention; broadened component matching
#: made "condition" newly, and wrongly, resolvable). Matched by exact
#: normalized text, the same structural-marker discipline as
#: `_EXTERNAL_TABLE_PREFIX` above -- not a clinical phrase.
_SEE_CONDITION_INSTRUCTION = "condition"


def plain_title(node) -> str:
    t = node.find("title")
    return (t.text or "").strip() if t is not None else ""


def norm_code(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not CODE_RE.match(raw):
        return None
    return raw.replace(".", "").rstrip("-").upper()


def walk(node, path, out):
    title = plain_title(node)
    p = path + [title] if title else path
    code = norm_code(node.findtext("code") or "")
    if code and p:
        out[code].add(" ".join(p).lower())       # full navigational path
        if title:
            out[code].add(title.lower())           # the leaf clinician term alone,
        # ...and every suffix sub-path in between, so a note's phrasing matches at
        # any depth. Generic suffixes that collide across codes are handled by the
        # resolver's single-code trust rule (a multi-code hit defers to retrieval).
        for i in range(1, len(p)):
            out[code].add(" ".join(p[i:]).lower())
    for child in node.findall("term"):
        walk(child, p, out)


def iter_index_nodes(node, path=()):
    """Every node in `node`'s own subtree (itself included), each paired with
    its full navigational title path -- issue #6 F9-R12-B: the ONLY way to
    find a redirect attached to a nested <term> rather than a top-level
    <mainTerm>, which the alias-emission loop below previously never
    visited at all."""
    title = plain_title(node)
    current = (*path, title) if title else path
    yield node, current
    for child in node.findall("term"):
        yield from iter_index_nodes(child, current)


def reference_texts(node):
    """Every <see>/<seeAlso> directive on `node`, as (tag, ref text) pairs --
    issue #6 F9-R12-B: a node reading only `findtext("see") or
    findtext("seeAlso")` silently drops a second directive when both are
    present (5 such nodes in the real FY2026 source); `findall` reads every
    element of each tag."""
    for tag in ("see", "seeAlso"):
        for element in node.findall(tag):
            value = (element.text or "").strip()
            if value:
                yield tag, value


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def navigate(ref, main_terms):
    """Follow a full 'see' reference path ('Entity, part, qualifier') into the
    subtree, matching each comma part to a nested term title. Returns the precise
    target node — NOT the whole main term — so an alias lands only on the referenced
    codes. Conservative: if any step can't be matched, returns None (no alias).
    Whitespace-normalized on both sides (issue #6 F9-R12-B): the source XML's
    <see>/<seeAlso> text occasionally carries doubled internal spacing
    ('Abnormal,  diagnostic imaging') that a bare .strip() does not collapse,
    which otherwise fails an exact title match that is genuinely present.

    issue #6 F9-R12-C, second re-review: a bare 'see condition' (no further
    qualifier) is the Index's own universal INSTRUCTION to look up the
    actual diagnosis term -- never a literal pointer, even though a real
    mainTerm happens to ALSO be titled 'Condition'. Broadening component
    matching made that coincidental title newly, wrongly reachable (606
    directives use this exact convention); refused here, at the one place
    every caller -- top-level classification AND subtree_codes' own
    internal recursive resolution -- funnels through."""
    parts = [_norm_ws(p).lower() for p in ref.split(",") if _norm_ws(p)]
    if not parts or parts == [_SEE_CONDITION_INSTRUCTION]:
        return None
    node = main_terms.get(parts[0])
    for part in parts[1:]:
        if node is None:
            return None
        node = next((t for t in node.findall("term")
                     if _norm_ws(plain_title(t)).lower() == part), None)
    return node


def classify_reference(ref: str, codes: set[str]) -> str:
    """"resolved" (issue #6 F9-R12-C: means the directive actually reaches at
    least one code -- a navigable-but-codeless target, or a target whose own
    subtree resolves to nothing, is NOT "resolved"; it produces no candidate
    edge at all, so counting it as resolved hid a real gap), "external_table_
    reference" (the Index's own "Table of ..." convention -- a real, known
    kind of redirect this compiler has no external table to follow), or
    "unresolved" (neither -- a genuine, counted gap; every directive must be
    classified, never silently dropped)."""
    if codes:
        return "resolved"
    if ref.strip().lower().startswith(_EXTERNAL_TABLE_PREFIX):
        return "external_table_reference"
    return "unresolved"


def subtree_codes(node, main_terms, visited=None) -> set[str]:
    """Every code reachable under `node`'s subtree: `node`'s own direct code
    (if any) UNION every code its own <see>/<seeAlso> directives resolve to
    (a coded node may ALSO redirect -- issue #6 F9-R12-B: 1,008 such nodes
    in the real source, previously excluded outright by an `elif` that
    treated "has a code" and "has a redirect" as mutually exclusive) UNION
    every descendant term's own codes, recursively (so a REDIRECT'S target
    can itself be a codeless node whose own further redirect must be
    chased -- the original two-hop fix, preserved). `visited` guards
    against a cyclical reference by (tag, ref) identity so this always
    terminates."""
    if node is None:
        return set()
    visited = set() if visited is None else visited
    codes = set()
    direct = norm_code(node.findtext("code") or "")
    if direct:
        codes.add(direct)
    for tag, ref in reference_texts(node):
        key = (tag, ref.casefold())
        if key in visited:
            continue
        target = navigate(ref, main_terms)
        if target is not None:
            codes |= subtree_codes(target, main_terms, visited | {key})
    for child in node.findall("term"):
        codes |= subtree_codes(child, main_terms, visited)
    return codes


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/codes/icd10cm_index_terms.json")
    root = ET.parse(src).getroot()

    out: dict[str, set] = defaultdict(set)
    main_terms: dict[str, ET.Element] = {}
    # issue #6 F9-R12-B/C/C-second (Codex's independent structural check): a
    # real main-term's OWN title is sometimes a COMPOSITE of several
    # official comma-joined heading forms ("Abnormal, abnormality"), indexed
    # here under the one full-title key -- but a <see>/<seeAlso> directive
    # elsewhere in the same source routinely cites just ONE heading
    # component ("see Abnormal, ..."), which never matches that composite
    # key at all.
    #
    # THREE priority tiers, consulted in order, each one BLOCKING the next
    # for a given token the moment it has ANY claim on it (even an
    # ambiguous one) -- not merged into one flat map. The first attempt at
    # this (a single `_components` dict over every comma position at equal
    # weight) let an unrelated main term's own SECONDARY qualifier word
    # ("...abnormal Y" as a deep, narrow variant) collide with and block a
    # totally different main term's CANONICAL first heading ("Abnormal,
    # abnormality" -- both segments equally name the SAME entry) -- 102
    # distinct reference texts (449 directives) that resolve through a
    # unique canonical first heading were measured staying wrongly
    # unresolved this way.
    #   1. exact_titles  -- the full title string (main_terms itself).
    #   2. canonical_heads -- EACH main term's OWN position-0 comma segment
    #      only (its primary alphabetical heading).
    #   3. secondary_components -- EACH main term's remaining (position 1+)
    #      comma segments (narrower qualifying variants).
    # A component earns a lookup key only when it identifies a SINGLE main
    # term WITHIN ITS OWN TIER (an ambiguous component -- 5 in the real
    # source, at tier 2 -- stays unresolved rather than guessing, and BLOCKS
    # the lower tier from claiming that same token on its behalf).
    _canonical_heads: dict[str, set[ET.Element]] = defaultdict(set)
    _secondary_components: dict[str, set[ET.Element]] = defaultdict(set)
    for letter in root.findall("letter"):
        for mt in letter.findall("mainTerm"):
            title = _norm_ws(plain_title(mt)).lower()
            if title:
                main_terms.setdefault(title, mt)
                parts = [p.strip() for p in title.split(",") if p.strip()]
                if parts:
                    _canonical_heads[parts[0]].add(mt)
                    for part in parts[1:]:
                        _secondary_components[part].add(mt)
            walk(mt, [], out)
    _claimed = set(main_terms)   # tier 1 already owns these tokens
    for component, nodes in _canonical_heads.items():
        if component in _claimed:
            continue
        _claimed.add(component)          # tier 2 owns this token now, ambiguous or not
        if len(nodes) == 1:
            main_terms[component] = next(iter(nodes))
    for component, nodes in _secondary_components.items():
        if component in _claimed:
            continue
        if len(nodes) == 1:
            main_terms[component] = next(iter(nodes))

    # Cross-reference aliases: kept in a SEPARATE map (issue #6 F9-R12-A) --
    # a redirect alias is exact-Index-lookup signal only, never per-code
    # embedding text (see module docstring). issue #6 F9-R12-B: walks EVERY
    # node at every depth via `iter_index_nodes`, not just top-level
    # mainTerms, and reads every directive on each via `reference_texts`.
    cross_ref: dict[str, set] = defaultdict(set)
    directive_counts = {"total": 0, "resolved": 0,
                        "external_table_reference": 0, "unresolved": 0}
    # issue #6 F9-R12-C, second re-review: EVERY unresolved directive, not a
    # capped sample -- a directive this compiler cannot resolve is a
    # complete, auditable record, never truncated away.
    unresolved_full: set[str] = set()
    for letter in root.findall("letter"):
        for mt in letter.findall("mainTerm"):
            for node, path in iter_index_nodes(mt):
                phrase = " ".join(p for p in path if p).lower()
                if not phrase:
                    continue
                for tag, ref in reference_texts(node):
                    directive_counts["total"] += 1
                    # issue #6 F9-R12-C: "resolved" requires a NONEMPTY code
                    # set -- a navigable target that itself resolves to no
                    # codes must not count as resolved.
                    target = navigate(ref, main_terms)
                    codes = subtree_codes(target, main_terms) if target is not None else set()
                    status = classify_reference(ref, codes)
                    directive_counts[status] += 1
                    if status == "resolved":
                        for code in codes:
                            cross_ref[code].add(phrase)
                    elif status == "unresolved":
                        unresolved_full.add(f"{tag}: {ref}")

    version = root.findtext("version") or ""
    data = {"version": version.strip(),
            "source": src.name,
            "terms": {c: sorted(ps) for c, ps in sorted(out.items())},
            "cross_reference_terms": {c: sorted(ps) for c, ps in sorted(cross_ref.items())},
            "reference_directives": {**directive_counts,
                                    "unresolved_directives": sorted(unresolved_full)}}
    dst.write_text(json.dumps(data, indent=1))
    n_phrases = sum(len(v) for v in out.values())
    n_xref = sum(len(v) for v in cross_ref.values())
    print(f"{len(out)} codes, {n_phrases} direct phrases, "
          f"{len(cross_ref)} codes / {n_xref} cross-reference phrases -> {dst}")
    print(f"reference directives: {directive_counts['total']} total, "
          f"{directive_counts['resolved']} resolved, "
          f"{directive_counts['external_table_reference']} external-table, "
          f"{directive_counts['unresolved']} unresolved")


if __name__ == "__main__":
    main()
