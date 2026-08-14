"""A real, minimal PDF and its matching vision-transcription payload — test support.

Every end-to-end suite that drives `run.main()` now needs an ORIGINAL DOCUMENT that
can actually be read a second way, because `app.ingestion.source_evidence` compiles the
PDF's embedded text layer into an independent read channel and the pipeline refuses to
release a claim whose quotations no independent channel confirms (issue #6 F6-R6-A).
A `b"%PDF-1.4 fixture"` placeholder is no longer a document.

The PDF is written by hand rather than with a generator library so the suites exercise
pdfplumber's real word extraction — including the per-word boxes an evidence region is
built from — without adding a build-time dependency to the deployed image.
"""
from __future__ import annotations

#: Width budget, in characters, before a logical line is wrapped onto the next visual
#: line. Purely cosmetic — reconciliation is whitespace-tokenized, so wrapping cannot
#: change what any channel reads — but it keeps the text inside the MediaBox, which is
#: what a real document looks like.
_WRAP = 88


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _wrap(line: str) -> list[str]:
    if len(line) <= _WRAP:
        return [line]
    out: list[str] = []
    current = ""
    for word in line.split(" "):
        if current and len(current) + 1 + len(word) > _WRAP:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        out.append(current)
    return out


def build_pdf(pages: list[list[str]], *, rotate: list[int] | None = None) -> bytes:
    """A syntactically complete PDF whose text layer says exactly `pages`."""
    count = len(pages)
    font_number = 3 + 2 * count
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
    objects: list[bytes] = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        f"<</Type/Pages/Kids[{kids}]/Count {count}>>".encode("latin-1"),
    ]
    for index, lines in enumerate(pages):
        turn = (rotate or [0] * count)[index]
        rotation = f"/Rotate {turn}" if turn else ""
        objects.append(
            (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]{rotation}"
             f"/Resources<</Font<</F1 {font_number} 0 R>>>>"
             f"/Contents {4 + 2 * index} 0 R>>").encode("latin-1"))
        body = "BT /F1 8 Tf 40 740 Td 12 TL\n"
        for line in lines:
            for visual in _wrap(line):
                body += f"({_escape(visual)}) Tj T*\n"
        body += "ET"
        raw = body.encode("latin-1")
        objects.append(b"<</Length " + str(len(raw)).encode() + b">>stream\n"
                       + raw + b"\nendstream")
    objects.append(b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + payload + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n"
            f"{start}\n%%EOF").encode()
    return bytes(out)


def page_images(count: int) -> list[dict]:
    """Identity of the rendered page images the vision channel was shown."""
    return [{"page_number": index, "sha256": f"sha256:{index:064d}",
             "width": 2550, "height": 3300}
            for index in range(1, count + 1)]


def vision_extraction(page_texts: list[str], *, metadata: dict | None = None,
                      statuses: list[str] | None = None,
                      document_version: str = "sha256:" + "a1" * 32,
                      extracted_text_sha256: str = "",
                      image_digests: list[str] | None = None,
                      page_separator: str = "\n\n") -> dict:
    """Exactly the shape `pdf_parser.extract_from_pdf` returns for those pages."""
    statuses = statuses or ["extracted"] * len(page_texts)
    images = ([{"page_number": index, "sha256": digest, "width": 2550, "height": 3300}
               for index, digest in enumerate(image_digests, start=1)]
              if image_digests else page_images(len(page_texts)))
    return {
        "metadata": dict(metadata or {}),
        "sections": {"full_text": page_separator.join(page_texts)},
        "page_texts": [{"page_number": index, "status": status, "text": text}
                       for index, (text, status)
                       in enumerate(zip(page_texts, statuses), start=1)],
        "note_integrity": {
            "source_pdf_sha256": document_version,
            "extracted_text_sha256": extracted_text_sha256,
            "page_count": len(page_texts),
            "page_images": images,
        },
    }
