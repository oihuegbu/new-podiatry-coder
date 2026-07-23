"""Parse the official ICD-10-CM Alphabetic Index XML (CDC/NCHS,
icd10cm-index-*.xml) into data/codes/icd10cm_index_terms.json:

    {"version": "...", "terms": {"<dotless code>": ["phrase", ...]}}

Each phrase is the full Index path that leads to the code ("cellulitis toe",
"paronychia"), i.e. the official alternate wording a clinical note may use
for that code. Two phrase sources:

1. Direct entries: every term node carrying a <code>, phrase = the chain of
   plain title texts from the mainTerm down (nonessential modifiers in
   <nemod> are dropped). Trailing '-' on codes (incomplete stems like
   L03.03-) is stripped; consumers prefix-match.
2. One-hop cross references: a mainTerm with NO code of its own whose
   <see>/<seeAlso> points at another main term ("Paronychia — see also
   Cellulitis, digit") contributes its own title as an alias phrase on
   every code under the referenced main term's subtree. This is what maps
   'paronychia' to the L03.0x cellulitis family and NOT to the L03.04x
   lymphangitis family (which lives under the 'Lymphangitis' main term).

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
        out[code].add(" ".join(p).lower())
    for child in node.findall("term"):
        walk(child, p, out)


def subtree_codes(node) -> set[str]:
    codes = set()
    c = norm_code(node.findtext("code") or "")
    if c:
        codes.add(c)
    for child in node.findall("term"):
        codes |= subtree_codes(child)
    return codes


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/codes/icd10cm_index_terms.json")
    root = ET.parse(src).getroot()

    out: dict[str, set] = defaultdict(set)
    main_terms: dict[str, ET.Element] = {}
    for letter in root.findall("letter"):
        for mt in letter.findall("mainTerm"):
            title = plain_title(mt).lower()
            if title:
                main_terms.setdefault(title, mt)
            walk(mt, [], out)

    # one-hop see/seeAlso aliases for code-less main terms
    aliases = 0
    for letter in root.findall("letter"):
        for mt in letter.findall("mainTerm"):
            title = plain_title(mt).lower()
            if not title or norm_code(mt.findtext("code") or ""):
                continue
            ref = (mt.findtext("see") or mt.findtext("seeAlso") or "").strip()
            if not ref:
                continue
            target_main = ref.split(",")[0].strip().lower()
            target = main_terms.get(target_main)
            if target is None:
                continue
            for code in subtree_codes(target):
                out[code].add(title)
                aliases += 1

    version = root.findtext("version") or ""
    data = {"version": version.strip(),
            "source": src.name,
            "terms": {c: sorted(ps) for c, ps in sorted(out.items())}}
    dst.write_text(json.dumps(data, indent=1))
    n_phrases = sum(len(v) for v in out.values())
    print(f"{len(out)} codes, {n_phrases} phrases ({aliases} via see/seeAlso aliases) -> {dst}")


if __name__ == "__main__":
    main()
