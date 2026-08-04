from __future__ import annotations

import hashlib
import json
import mimetypes
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree


class DocumentIngestionError(RuntimeError):
    """A document cannot be converted to a stable, source-addressable text."""


@dataclass(frozen=True)
class PageBoundary:
    page: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    source_sha256: str
    text_sha256: str
    media_type: str
    text: str
    pages: tuple[PageBoundary, ...]
    extractor: str

    def page_for_offset(self, offset: int) -> int | None:
        for page in self.pages:
            if page.start_offset <= offset < page.end_offset:
                return page.page
        return None

    def verify(self) -> None:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise DocumentIngestionError("ingested text hash mismatch")
        previous = 0
        for boundary in self.pages:
            if boundary.start_offset != previous or boundary.end_offset < boundary.start_offset:
                raise DocumentIngestionError("page boundaries are not contiguous")
            previous = boundary.end_offset
        if self.pages and previous != len(self.text):
            raise DocumentIngestionError("page boundaries do not cover the document")

    def write(self, destination: Path) -> Path:
        self.verify()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["pages"] = [asdict(item) for item in self.pages]
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination


def load_ingested_document(path: Path) -> IngestedDocument:
    value = json.loads(path.read_text(encoding="utf-8"))
    document = IngestedDocument(
        document_id=value["document_id"],
        source_sha256=value["source_sha256"],
        text_sha256=value["text_sha256"],
        media_type=value["media_type"],
        text=value["text"],
        pages=tuple(PageBoundary(**item) for item in value["pages"]),
        extractor=value["extractor"],
    )
    document.verify()
    return document


class DocumentIngestor:
    def __init__(self, ocr_command: tuple[str, ...] | None = None) -> None:
        self.ocr_command = ocr_command

    def ingest(self, source: Path, document_id: str | None = None) -> IngestedDocument:
        source = source.resolve()
        if not source.is_file():
            raise DocumentIngestionError(f"document does not exist: {source}")
        raw = source.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        suffix = source.suffix.casefold()
        if suffix == ".pdf":
            pages, extractor = self._pdf(source)
        elif suffix in {".txt", ".md", ".csv"}:
            pages, extractor = [raw.decode("utf-8-sig")], "utf8"
        elif suffix == ".docx":
            pages, extractor = [self._docx(source)], "docx-wordprocessingml"
        elif media_type.startswith("image/"):
            pages, extractor = [self._ocr(source)], "configured-ocr"
        else:
            raise DocumentIngestionError(f"unsupported document media type: {media_type}")
        text, boundaries = self._join_pages(pages)
        if not text.strip():
            raise DocumentIngestionError("document extraction produced no text")
        result = IngestedDocument(
            document_id=document_id or source_hash,
            source_sha256=source_hash,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            media_type=media_type,
            text=text,
            pages=boundaries,
            extractor=extractor,
        )
        result.verify()
        return result

    def _pdf(self, source: Path) -> tuple[list[str], str]:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise DocumentIngestionError("PDF support requires the pypdf package") from error
        try:
            reader = PdfReader(str(source), strict=True)
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as error:
            raise DocumentIngestionError(f"PDF parser rejected the document: {type(error).__name__}") from error
        if not pages or any(not page.strip() for page in pages):
            if self.ocr_command:
                return [self._ocr(source)], "configured-ocr"
            raise DocumentIngestionError("one or more PDF pages have no text; configured OCR is required")
        return pages, f"pypdf:{getattr(__import__('pypdf'), '__version__', 'unknown')}"

    @staticmethod
    def _docx(source: Path) -> str:
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        try:
            with zipfile.ZipFile(source) as archive:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
        except Exception as error:
            raise DocumentIngestionError("invalid DOCX container") from error
        paragraphs = []
        for paragraph in root.findall(".//w:p", namespace):
            paragraphs.append("".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)))
        return "\n".join(paragraphs)

    def _ocr(self, source: Path) -> str:
        if not self.ocr_command:
            raise DocumentIngestionError("image-only documents require an explicitly configured OCR command")
        command = [part.replace("{input}", str(source)) for part in self.ocr_command]
        if all("{input}" not in part for part in self.ocr_command):
            raise DocumentIngestionError("OCR command must contain an {input} placeholder")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as error:
            raise DocumentIngestionError("configured OCR command failed") from error
        if not result.stdout.strip():
            raise DocumentIngestionError("configured OCR command returned no text")
        return result.stdout

    @staticmethod
    def _join_pages(pages: list[str]) -> tuple[str, tuple[PageBoundary, ...]]:
        normalized = [page.replace("\r\n", "\n").replace("\r", "\n") for page in pages]
        parts: list[str] = []
        boundaries: list[PageBoundary] = []
        offset = 0
        for number, page in enumerate(normalized, 1):
            start = offset
            if number > 1:
                parts.append("\n\f\n")
                offset += 3
            parts.append(page)
            offset += len(page)
            boundaries.append(PageBoundary(number, start, offset))
        return "".join(parts), tuple(boundaries)
