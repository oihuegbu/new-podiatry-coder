"""The Source Evidence Compiler — provider-neutral ingestion of the ORIGINAL document.

================================================================================
WHY THIS MODULE EXISTS — issue #6, finding F6-R6-A (P1), product directive §1
================================================================================
`pdf_parser.extract_from_pdf` produces ONE reading of the document: a vision model's
transcription. That reading is a candidate, never an authority — a misread side,
ordinal, unit, decimal, dose or negation in it would flow through anchoring, gates
and the certificate looking exactly like a correct one.

This module compiles the SAME document into a `SourceEvidenceDocument`
(`app/contracts/source_evidence.py`) carrying:

    * the original PDF's own digest;
    * the identity of every rendered page image the vision channel was actually
      shown (taken from the transcriber, not re-derived — see below);
    * the PDF's EMBEDDED TEXT LAYER as a second, deterministic read channel, with
      per-word bounding boxes;
    * explicit per-page outcomes for blank, rotated, duplicated, missing and
      low-quality pages.

WHY THE EMBEDDED TEXT LAYER IS THE SECOND CHANNEL (AND WHAT HAPPENS WITHOUT ONE)
--------------------------------------------------------------------------------
Two options were available: (a) the PDF's embedded text plus the vision model, or
(b) two independent vision profiles from different vendors. (a) is preferred
wherever it exists because it is *deterministic* — it is a property of the document
bytes rather than of any model, it needs no second inference call and therefore
costs nothing, and it carries word geometry, which is the only way a released fact
can resolve to an exact page REGION rather than merely a page number.

A scanned, image-only document has no text layer, so (a) is impossible for it. Such
a page is recorded as `no_embedded_text` and its read is UNREADABLE — never silently
"agreed". `IndependentVisionReader` below is option (b) for exactly those pages: a
second model read from a DIFFERENT declared vendor than the primary transcriber,
invoked lazily and only for the pages that carry a quotation justifying a released
line (`contracts.source_evidence.pages_needing_independent_read`). A document whose
text layer covers it costs nothing extra; a scanned one costs one extra read of the
few pages the claim actually rests on — not a second read of every note.

NOTHING HERE IS CLINICAL
------------------------
This module knows about bytes, pages, glyphs and boxes. It contains no medical
vocabulary and makes no coding decision; the reconciliation MECHANISM it feeds
(`contracts.source_evidence.reconcile_spans`) is likewise purely structural.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from app.contracts.source_evidence import (
    PAGE_SEPARATOR, ChannelKind, PageRead, PageStatus, ReadChannel, SourceEvidenceDocument,
    SourcePage, SourceToken, build_page_read, normalize_token,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

#: Channel ids. Stable strings: they appear in certificates and in the durable audit
#: record, so a reader years from now can tell which reading proved what.
PRIMARY_CHANNEL_ID = "vision_transcription"
EMBEDDED_TEXT_CHANNEL_ID = "pdf_embedded_text"
SECONDARY_VISION_CHANNEL_ID = "independent_vision"

#: A text layer that recovers less than this share of the tokens the primary channel
#: read on the same page is not a reading of that page — it is a header, a watermark
#: or a partial OCR artifact left in the file. Treating it as a full reading would
#: manufacture disagreements on every unrecovered word; treating it as agreement would
#: be worse. It is therefore declared UNREADABLE, which holds (system work) and lets
#: the independent vision channel cover the page.
MINIMUM_TEXT_LAYER_YIELD = 0.6

_VISION_PROMPT = (
    "Transcribe this page image VERBATIM. Copy every word exactly as printed, in "
    "reading order, preserving line breaks. Do not summarise, correct, reorder, "
    "expand abbreviations or add anything that is not printed on the page. If a word "
    "is unreadable, write [UNCLEAR] in its place. Output the transcription only."
)


class SourceEvidenceCompilationError(RuntimeError):
    """The document could not be compiled into a source-evidence record.

    Raised rather than degraded: a compiler that returned a document with one channel
    would hand the pipeline a transcription presented as if it had been checked.
    """


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedded_text_pages(pdf_path: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """The PDF's own text layer, per page, with word boxes.

    Returns `(pages, notes)`. A failure to open or parse the file is reported in
    `notes` and yields NO pages — the caller then has no independent channel and every
    quotation holds, which is the correct answer for "we could not read the document
    ourselves", not an error that loses the note.
    """
    notes: list[str] = []
    try:
        import pdfplumber
    except Exception as exc:                              # pragma: no cover - packaging
        return {}, [f"embedded-text channel unavailable: pdfplumber import failed "
                    f"({type(exc).__name__}: {exc})"]
    pages: dict[int, dict[str, Any]] = {}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    words = page.extract_words(use_text_flow=False,
                                               keep_blank_chars=False) or []
                except Exception as exc:
                    notes.append(f"page {index}: text layer could not be extracted "
                                 f"({type(exc).__name__})")
                    words = []
                tokens = _tokens_from_words(words)
                pages[index] = {
                    "text": " ".join(t.text for t in tokens),
                    "tokens": tokens,
                    "rotation": int(getattr(page, "rotation", 0) or 0),
                    "width": _as_float(getattr(page, "width", None)),
                    "height": _as_float(getattr(page, "height", None)),
                }
    except Exception as exc:
        return {}, [f"embedded-text channel unavailable: {type(exc).__name__}: {exc}"]
    return pages, notes


def _tokens_from_words(words: list) -> list[SourceToken]:
    """Word boxes as comparable tokens, with line-break hyphenation repaired.

    A word the typesetter split across two lines ("well-" / "healed") is TWO words in
    the text layer and ONE word in a reading of the page image. That is a property of
    the layout, not a difference in what the page says, so the pair is rejoined here —
    at the only place that can see it, because by the time the reconciler has a token
    stream the line structure is gone. The join is purely structural (a token whose
    RAW text ends in a hyphen, followed by another token) and its box is the union of
    the two, so the recovered word still resolves to a real region on the page.
    """
    tokens: list[SourceToken] = []
    pending: dict | None = None
    for word in words:
        raw = str(word.get("text") or "")
        if pending is not None:
            raw = pending["text"][:-1] + raw
            word = {"x0": _min_of(pending["x0"], word.get("x0")),
                    "top": _min_of(pending["top"], word.get("top")),
                    "x1": _max_of(pending["x1"], word.get("x1")),
                    "bottom": _max_of(pending["bottom"], word.get("bottom"))}
            pending = None
        elif raw.endswith("-") and len(raw) > 1:
            pending = {"text": raw, "x0": _as_float(word.get("x0")),
                       "top": _as_float(word.get("top")),
                       "x1": _as_float(word.get("x1")),
                       "bottom": _as_float(word.get("bottom"))}
            continue
        normalized = normalize_token(raw)
        if not normalized:
            continue
        tokens.append(SourceToken(
            text=raw, normalized=normalized,
            x0=_as_float(word.get("x0")), top=_as_float(word.get("top")),
            x1=_as_float(word.get("x1")), bottom=_as_float(word.get("bottom"))))
    if pending is not None:                       # a trailing hyphen with nothing after
        normalized = normalize_token(pending["text"])
        if normalized:
            tokens.append(SourceToken(
                text=pending["text"], normalized=normalized, x0=pending["x0"],
                top=pending["top"], x1=pending["x1"], bottom=pending["bottom"]))
    return tokens


def _min_of(left, right):
    values = [v for v in (_as_float(left), _as_float(right)) if v is not None]
    return min(values) if values else None


def _max_of(left, right):
    values = [v for v in (_as_float(left), _as_float(right)) if v is not None]
    return max(values) if values else None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def compile_source_evidence(pdf_path: str | Path, extraction: dict) -> SourceEvidenceDocument:
    """Compile one original document + its vision transcription into source evidence.

    `extraction` is exactly what `pdf_parser.extract_from_pdf` returned. The primary
    channel is built from its per-page transcription, and the compiled document's
    `primary_text()` is asserted to be byte-identical to the `sections["full_text"]`
    the coder reads — if it were not, every character offset an evidence span carries
    would point at the wrong page, and a "verified" region would be a fabrication.
    """
    pdf_path = Path(pdf_path)
    integrity = extraction.get("note_integrity") or {}
    page_texts = extraction.get("page_texts") or []
    if not isinstance(page_texts, list) or not page_texts:
        raise SourceEvidenceCompilationError(
            f"{pdf_path.name}: the transcription carries no per-page text, so no "
            f"quotation can be located on a page of the original document")

    document_sha256 = str(integrity.get("source_pdf_sha256") or "").strip()
    if not document_sha256:
        try:
            document_sha256 = _digest_bytes(pdf_path.read_bytes())
        except Exception as exc:
            raise SourceEvidenceCompilationError(
                f"{pdf_path.name}: the original document's digest is unavailable "
                f"({type(exc).__name__}: {exc})") from exc

    images = {int(entry.get("page_number") or 0): entry
              for entry in (integrity.get("page_images") or [])
              if isinstance(entry, dict)}
    embedded, notes = _embedded_text_pages(pdf_path)
    anomalies: list[str] = []

    ordered = sorted(page_texts, key=lambda p: int(p.get("page_number") or 0))
    if [int(p.get("page_number") or 0) for p in ordered] != list(range(1, len(ordered) + 1)):
        raise SourceEvidenceCompilationError(
            f"{pdf_path.name}: the transcription's page numbering is not a contiguous "
            f"1-based sequence; a quotation cannot be attributed to a page")
    if embedded and len(embedded) != len(ordered):
        anomalies.append(
            f"the document has {len(embedded)} page(s) but the transcription returned "
            f"{len(ordered)}")

    seen_images: dict[str, int] = {}
    pages: list[SourcePage] = []
    cursor = 0
    for entry in ordered:
        number = int(entry.get("page_number") or 0)
        text = str(entry.get("text") or "")
        primary = build_page_read(PRIMARY_CHANNEL_ID, number, text,
                                  status=(PageStatus.BLANK
                                          if str(entry.get("status") or "") == "blank"
                                          else None))
        reads = [primary]
        flags: list[str] = []

        image = images.get(number) or {}
        image_sha = str(image.get("sha256") or "")
        if not image_sha:
            anomalies.append(f"page {number}: no rendered-page image identity was "
                             f"recorded by the transcriber")
        elif image_sha in seen_images:
            # Explicit, non-fatal: two identical rendered pages are a real document
            # feature (a duplicated fax page) AND a claim risk (the same service read
            # twice). Recorded on both pages so neither can be read in isolation.
            flags.append(f"duplicate_of_page:{seen_images[image_sha]}")
            anomalies.append(f"page {number} renders identically to page "
                             f"{seen_images[image_sha]}")
        else:
            seen_images[image_sha] = number

        layer = embedded.get(number)
        rotation = int((layer or {}).get("rotation") or 0)
        if rotation:
            flags.append(f"rotated:{rotation}")
        if layer is None:
            secondary = PageRead(
                channel_id=EMBEDDED_TEXT_CHANNEL_ID, page_number=number,
                status=PageStatus.MISSING,
                detail="the document's text layer has no such page")
            flags.append("text_layer_unavailable")
        else:
            tokens = layer["tokens"]
            primary_count = len(primary.tokens)
            if not tokens:
                flags.append("no_embedded_text")
                secondary = PageRead(
                    channel_id=EMBEDDED_TEXT_CHANNEL_ID, page_number=number,
                    status=(PageStatus.BLANK if primary_count == 0
                            else PageStatus.UNREADABLE),
                    text="", text_sha256=_digest(""),
                    detail=("the page is blank in both readings" if primary_count == 0
                            else "the page carries no embedded text (image-only page); "
                                 "an independent read of the page image is required"))
            elif primary_count and len(tokens) < MINIMUM_TEXT_LAYER_YIELD * primary_count:
                flags.append("low_text_yield")
                secondary = build_page_read(
                    EMBEDDED_TEXT_CHANNEL_ID, number, layer["text"], tokens=tokens,
                    status=PageStatus.UNREADABLE,
                    detail=(f"the text layer recovers {len(tokens)} of the "
                            f"{primary_count} token(s) the transcription read; too "
                            f"partial to check a quotation against"))
            else:
                secondary = build_page_read(
                    EMBEDDED_TEXT_CHANNEL_ID, number, layer["text"], tokens=tokens)
        reads.append(secondary)

        if primary.status is PageStatus.MISSING:
            status = PageStatus.MISSING
        elif not primary.tokens and not secondary.tokens:
            status = PageStatus.BLANK
        else:
            status = PageStatus.READ

        pages.append(SourcePage(
            page_number=number,
            image_sha256=image_sha,
            width=_as_float(image.get("width")) or (layer or {}).get("width"),
            height=_as_float(image.get("height")) or (layer or {}).get("height"),
            rotation=rotation, status=status, flags=tuple(flags), reads=tuple(reads),
            char_start=cursor, char_end=cursor + len(text)))
        cursor += len(text) + len(PAGE_SEPARATOR)

    channels = [
        ReadChannel(channel_id=PRIMARY_CHANNEL_ID, kind=ChannelKind.VISION,
                    provider=_primary_provider(), profile=_primary_profile(),
                    prompt_sha256=_primary_prompt_digest(),
                    schema_version="pdf_parser/vision-1"),
        ReadChannel(channel_id=EMBEDDED_TEXT_CHANNEL_ID, kind=ChannelKind.EMBEDDED_TEXT,
                    provider="pdf", engine=_pdfplumber_identity(),
                    schema_version="embedded-text-1"),
    ]

    document = SourceEvidenceDocument(
        filename=pdf_path.name,
        document_sha256=document_sha256,
        page_count=len(pages),
        channels=tuple(channels),
        primary_channel_id=PRIMARY_CHANNEL_ID,
        pages=tuple(pages),
        anomalies=tuple(anomalies),
        compiled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        compiler_notes=tuple(notes))

    problems = document.integrity_problems()
    if problems:
        raise SourceEvidenceCompilationError(
            f"{pdf_path.name}: compiled source evidence is not self-consistent: "
            f"{'; '.join(problems)}")

    expected = str((extraction.get("sections") or {}).get("full_text") or "")
    if expected and document.primary_text() != expected:
        # Not a warning: every evidence span is anchored into `full_text` by character
        # offset, so a compiled text that differs from it by ONE character makes every
        # page attribution downstream wrong while still looking anchored.
        raise SourceEvidenceCompilationError(
            f"{pdf_path.name}: the compiled primary text is not byte-identical to the "
            f"text the coder reads; evidence offsets could not be attributed to a page")
    independently_readable = sum(
        1 for page in pages
        if (page.read_by(EMBEDDED_TEXT_CHANNEL_ID) is not None
            and page.read_by(EMBEDDED_TEXT_CHANNEL_ID).usable))
    logger.info(
        f"  Source evidence: {len(pages)} page(s), channels="
        f"{[c.channel_id for c in channels]}, "
        f"independently readable pages={independently_readable}/{len(pages)}"
        + (f", anomalies={anomalies}" if anomalies else ""))
    return document


def _digest_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _primary_provider() -> str:
    try:
        from app.core.config import LLM_PROVIDER
        return str(LLM_PROVIDER or "")
    except Exception:                                     # pragma: no cover - config
        return ""


def _primary_profile() -> str:
    try:
        from app.core.config import CLAUDE_MODEL
        return str(CLAUDE_MODEL or "")
    except Exception:                                     # pragma: no cover - config
        return ""


def _primary_prompt_digest() -> str:
    try:
        from app.ingestion.pdf_parser import EXTRACTION_SYSTEM_PROMPT
        return _digest(EXTRACTION_SYSTEM_PROMPT)
    except Exception:                                     # pragma: no cover - import
        return ""


def _pdfplumber_identity() -> str:
    try:
        import pdfplumber
        return f"pdfplumber/{getattr(pdfplumber, '__version__', 'unknown')}"
    except Exception:                                     # pragma: no cover - packaging
        return "pdfplumber/unavailable"


# --------------------------------------------------------------------------
# option (b): a genuinely independent SECOND MODEL read, for pages with no text layer
# --------------------------------------------------------------------------

class IndependentVisionReader:
    """A second read of specific page images, by a DIFFERENT declared vendor.

    Used only where option (a) is impossible — an image-only page — and only for the
    pages carrying a quotation that justified a released line. Independence is a
    checked property, not a naming convention: `channel()` declares the provider, and
    `contracts.source_evidence.independent_of` refuses to treat this channel as
    evidence about the primary one unless that provider differs.

    Failure is silence, never a fabricated agreement: a page this reader cannot read
    simply gets no read, and the quotation on it stays UNVERIFIABLE (a hold).
    """

    def __init__(self, pdf_path: str | Path, *, dpi: int = 300,
                 provider: str = "openai") -> None:
        self.pdf_path = Path(pdf_path)
        self.dpi = int(dpi)
        self.provider = str(provider)

    # ---------------------------------------------------------------- identity
    def channel(self) -> ReadChannel:
        return ReadChannel(
            channel_id=SECONDARY_VISION_CHANNEL_ID, kind=ChannelKind.VISION,
            provider=self.provider, profile=self._model(),
            prompt_sha256=_digest(_VISION_PROMPT),
            schema_version="independent-vision-1")

    def _model(self) -> str:
        try:
            from app.core.config import OPENAI_MODEL
            return str(OPENAI_MODEL or "")
        except Exception:                                 # pragma: no cover - config
            return ""

    # ------------------------------------------------------------------ reads
    def read_pages(self, page_numbers: tuple[int, ...]) -> dict[int, PageRead]:
        out: dict[int, PageRead] = {}
        for number in page_numbers:
            try:
                text = self._read_page(number)
            except Exception as exc:
                logger.warning(f"  independent page read failed for page {number} "
                               f"({type(exc).__name__}: {exc}); the quotations on that "
                               f"page stay unverified and hold")
                continue
            out[number] = build_page_read(SECONDARY_VISION_CHANNEL_ID, number, text)
        return out

    def _page_image(self, number: int) -> bytes:
        from pdf2image import convert_from_path
        images = convert_from_path(str(self.pdf_path), dpi=self.dpi,
                                   first_page=number, last_page=number)
        if not images:
            raise SourceEvidenceCompilationError(
                f"{self.pdf_path.name}: page {number} could not be rendered")
        buffer = BytesIO()
        images[0].save(buffer, format="PNG")
        return buffer.getvalue()

    def _read_page(self, number: int) -> str:
        from app.core.llm_client import get_openai_client
        payload = base64.b64encode(self._page_image(number)).decode("utf-8")
        client = get_openai_client()
        response = client.chat.completions.create(
            model=self._model(),
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _VISION_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{payload}"}},
            ]}],
        )
        return str(response.choices[0].message.content or "")
