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

import hashlib
from pathlib import Path

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


def digest_of(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def rendered_page(index: int) -> bytes:
    """Stand-in bytes for ONE rendered page image.

    The compiler now RECOMPUTES every page-image digest from the bytes the transcriber
    hands it (issue #6 F7-R5), so a fixture can no longer assert a digest out of thin
    air: it must supply bytes that produce the digest it claims. These only have to be
    distinct per page and stable — nothing reads them as an image.
    """
    return f"rendered-page-{index}".encode("utf-8")


#: The identity the compiler records for the primary reading. It is the shape
#: `pdf_parser` declares for the call it actually made — the vendor of the client that
#: answered, the model that was sent, the prompt digest — and the compiler refuses a
#: transcription that carries none, because channel independence would then be decided
#: against a guess. "claude" mirrors the deployed transcriber, so the paid OpenAI page
#: reader is genuinely independent of it in these suites exactly as in production.
VISION_CHANNEL = {
    "provider": "claude",
    "profile": "test-vision-model",
    "prompt_sha256": "sha256:" + "11" * 32,
    "schema_version": "pdf_parser/vision-1",
    "client": "tests.source_pdf.StubVisionClient",
}


def page_images(payloads: list[bytes]) -> list[dict]:
    """Identity of the rendered page images the vision channel was shown."""
    return [{"page_number": index, "sha256": digest_of(raw),
             "width": 2550, "height": 3300}
            for index, raw in enumerate(payloads, start=1)]


def vision_extraction(page_texts: list[str], *, metadata: dict | None = None,
                      statuses: list[str] | None = None,
                      pdf_path: str | Path | None = None,
                      document_version: str | None = None,
                      extracted_text_sha256: str = "",
                      image_payloads: list[bytes] | None = None,
                      vision_channel: dict | None = None,
                      page_separator: str = "\n\n") -> dict:
    """Exactly the shape `pdf_parser.extract_from_pdf` returns for those pages.

    `pdf_path` is the document this reading is OF. The compiler recomputes that file's
    digest and refuses a transcription claiming a different one, so any fixture that is
    actually compiled must say which file it read; `document_version` remains available
    for the negative case (a claim that deliberately does not match the bytes).
    """
    statuses = statuses or ["extracted"] * len(page_texts)
    payloads = (list(image_payloads) if image_payloads is not None
                else [rendered_page(index)
                      for index in range(1, len(page_texts) + 1)])
    images = page_images(payloads)
    if document_version is None:
        document_version = (digest_of(Path(pdf_path).read_bytes())
                            if pdf_path is not None else "sha256:" + "a1" * 32)
    return {
        "metadata": dict(metadata or {}),
        "sections": {"full_text": page_separator.join(page_texts)},
        "page_texts": [{"page_number": index, "status": status, "text": text}
                       for index, (text, status)
                       in enumerate(zip(page_texts, statuses), start=1)],
        # The bytes each page-image digest above was taken from. The compiler verifies
        # rather than trusts, so a fixture supplies them exactly as the parser does.
        "page_image_bytes": {index: raw
                             for index, raw in enumerate(payloads, start=1)},
        "note_integrity": {
            "source_pdf_sha256": document_version,
            "extracted_text_sha256": extracted_text_sha256,
            "page_count": len(page_texts),
            "page_images": images,
            "vision_channel": dict(vision_channel or VISION_CHANNEL),
        },
    }
