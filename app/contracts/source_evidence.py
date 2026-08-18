"""The versioned `SourceEvidenceDocument` — the ORIGINAL document as read, not as transcribed.

================================================================================
WHY THIS MODULE EXISTS — issue #6, finding F6-R6-A (P1), product directive §1
================================================================================
Until this contract existed, one vision model produced `page_texts` and every
downstream "exact" evidence span was exact only *relative to that model's own
output*. Page-coverage checks proved the model emitted a string for every page;
they proved nothing about whether the side, the ordinal, the unit, the decimal,
the measurement or the negation in that string is what the document actually
says. A misread word could therefore become a fully anchored, certified, billed
code, and the audit trail could identify the PDF but not show where the billed
fact appears in it.

The reviewer's framing is the invariant this module implements:

    An LLM transcription may be a candidate reading; it is never the authority
    against which its own correctness is proven.

So a `SourceEvidenceDocument` is a document read by MORE THAN ONE channel:

    original PDF bytes  ── document_sha256
        ├── rendered page images ── per-page image_sha256, size, rotation
        ├── channel 1 (primary)  ── the vision transcription  (a CANDIDATE reading)
        └── channel 2..n         ── the embedded text layer with word boxes, and/or
                                    a genuinely independent second model read
                                    (different declared provider)

and reconciliation is the act of proving a quotation taken from channel 1 also
appears, token for token, in an INDEPENDENT channel's reading of the same page.

WHAT IS RECONCILED, AND WHY IT IS NOT EVERY WORD
------------------------------------------------
The unit of reconciliation is the EVIDENCE SPAN: the verbatim quotation the
extraction layer used to support one clinical fact. That is precisely the
"code-changing text" the directive names — laterality, anatomy, ordinal,
measurement, dose, status and negation all live inside those quotations, and
nothing else in the document can change a code without first becoming one.
Reconciling spans rather than whole documents is also what keeps the second
channel affordable: only the PAGES carrying a span that justified a RELEASED
line ever need an independently paid-for read.

NO MEDICAL VOCABULARY APPEARS HERE, AND NONE MAY
-------------------------------------------------
The materiality rule is structural, not lexical: two readings agree when their
token sequences are equal after Unicode/case/edge-punctuation normalization, and
disagree otherwise. There is no list of laterality words, no list of units, no
negation lexicon — a perturbed side, a perturbed ordinal, a perturbed decimal
and a dropped negation are all simply *token differences inside a quotation that
justified a billed code*. That is what makes this mechanism survive every future
change to the clinical vocabulary the extraction layer uses (directive §1's
final acceptance test).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.dates import find_dates, parse_date

from .claim_bundle import content_digest

# --------------------------------------------------------------------------
# schema identity
# --------------------------------------------------------------------------

SCHEMA_ID = "source_evidence_document"
SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

#: The string joining consecutive pages into the one text the coder reads. It is
#: part of the CONTRACT because every evidence span's character offsets are taken
#: in that joined string: change it and every recorded offset silently shifts.
PAGE_SEPARATOR = "\n\n"


class SourceEvidenceError(Exception):
    """Base class: any refusal to interpret a payload as a SourceEvidenceDocument."""


class UnknownSourceEvidenceSchema(SourceEvidenceError):
    """The payload declares a schema id/version this build does not implement."""


class InvalidSourceEvidenceDocument(SourceEvidenceError):
    """The payload declares a supported schema but does not satisfy it."""


class ChannelIndependenceError(SourceEvidenceError):
    """A channel that must be INDEPENDENT of the primary reading is not (F7-R5).

    Raised, never recorded-and-continued. A same-kind channel that shares the primary
    channel's provider -- or whose provider could not be established at all -- cannot be
    evidence about the primary reading: one vendor agreeing with itself is repetition,
    not confirmation. Admitting such a channel would put a reading in the document that
    `reconcile_spans` must not credit, while making the record LOOK independently
    checked; and paying for it would buy nothing. The correct outcome for "the second
    channel we obtained is not independent" is a loud stop, exactly as for "no second
    channel could be obtained".
    """


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------

class PageStatus(str, Enum):
    """What happened to ONE page. Every page has exactly one of these — a page is
    never silently absent from the document (directive §1, blank/rotated/duplicated/
    missing/low-quality pages must have explicit outcomes)."""

    READ = "READ"
    #: Deliberately distinct from UNREADABLE: a genuinely empty page is a normal
    #: document feature and must not look like a failed read.
    BLANK = "BLANK"
    #: The renderer produced the page but no channel could obtain text from it.
    UNREADABLE = "UNREADABLE"
    #: The renderer produced the page and a channel did NOT return it at all.
    MISSING = "MISSING"


class ChannelKind(str, Enum):
    #: The PDF's own embedded text layer, with per-word boxes. Deterministic: it
    #: is a property of the document bytes, not of any model.
    EMBEDDED_TEXT = "embedded_text"
    #: A vision/multimodal model reading a rendered page image.
    VISION = "vision"
    #: A deterministic OCR engine reading a rendered page image.
    OCR = "ocr"


class ReconciliationStatus(str, Enum):
    """The outcome of proving one quotation against an independent reading."""

    #: The quotation's token sequence was found in an independent channel.
    AGREED = "AGREED"
    #: An independent channel reads the same region DIFFERENTLY.
    DISAGREED = "DISAGREED"
    #: An independent channel read the page and the quotation is not in it at all.
    NOT_LOCATED = "NOT_LOCATED"
    #: No independent channel covers this quotation's page(s) — nothing was proven
    #: either way. This is system work (obtain a second channel), not coding work.
    UNVERIFIABLE = "UNVERIFIABLE"
    #: The quotation carries no material token (punctuation/whitespace only), so
    #: there is nothing to reconcile. Recorded rather than dropped.
    VACUOUS = "VACUOUS"


#: A detected disagreement is an INTEGRITY failure: the record does not say what the
#: claim rests on. It is never retryable and never a coder's judgement call.
BLOCKING_STATUSES = frozenset({ReconciliationStatus.DISAGREED,
                               ReconciliationStatus.NOT_LOCATED})
#: An absent second channel proves nothing. Fail closed, but as SYSTEM work.
HOLDING_STATUSES = frozenset({ReconciliationStatus.UNVERIFIABLE})
#: Statuses that permit a release.
CLEARED_STATUSES = frozenset({ReconciliationStatus.AGREED,
                              ReconciliationStatus.VACUOUS})


# --------------------------------------------------------------------------
# token normalization — the ONE definition of "the same reading"
# --------------------------------------------------------------------------

#: Zero-width and soft-hyphen characters carry no reading; they are layout, not text.
_INVISIBLE = dict.fromkeys(
    map(ord, "­​‌‍⁠﻿"), None)
_DASHES = {ord(c): "-" for c in "‐‑‒–—―−"}
_APOSTROPHES = {ord(c): "'" for c in "‘’ʼ′"}
_QUOTES = {ord(c): '"' for c in "“”″"}
_TRANSLATION = {**_INVISIBLE, **_DASHES, **_APOSTROPHES, **_QUOTES}

_LEADING_PUNCT = re.compile(r"^[^0-9a-z]+")
_TRAILING_PUNCT = re.compile(r"[^0-9a-z]+$")
_ALNUM_ONLY = re.compile(r"[^0-9a-z]+")
_HAS_DIGIT = re.compile(r"\d")


def normalize_token(raw: str) -> str:
    """The comparable form of one token.

    Unicode form, letter case and EDGE punctuation are presentation, not reading:
    `"(RIGHT)"`, `"right,"` and `"right"` are the same word. INTERNAL punctuation is
    NOT stripped, deliberately — doing so would make `"3.0"` and `"30"` compare equal
    and silently erase exactly the decimal-misread class this module exists to catch.
    """
    text = unicodedata.normalize("NFKC", str(raw or "")).translate(_TRANSLATION)
    text = text.casefold()
    text = _LEADING_PUNCT.sub("", text)
    text = _TRAILING_PUNCT.sub("", text)
    return text


def is_numeric_token(normalized: str) -> bool:
    """Does this token carry a digit? Recorded on every difference so an auditor can
    see at a glance that a disagreement was on a measurement/ordinal/dose rather than
    on prose. It is an ANNOTATION, never a gate input — gating must not depend on a
    token's shape, or a misread word would become releasable."""
    return bool(_HAS_DIGIT.search(normalized))


def tokens_equal(left: str, right: str) -> bool:
    """Do two normalized tokens represent the same reading?

    Equal forms agree. Beyond that there is exactly ONE tolerance, and it is a
    typesetting artifact rather than a reading difference: a word broken across a
    line by a hyphen ("well-" / "healed") is recovered by one channel as a compound
    and by the other as a hyphenless run. Two tokens that differ ONLY in internal
    punctuation are therefore treated as equal *when neither carries a digit* — a
    numeric token's internal punctuation is its decimal point, which is claim-
    affecting and can never be normalized away.
    """
    if left == right:
        return True
    if not left or not right:
        return False
    if is_numeric_token(left) or is_numeric_token(right):
        return False
    return _ALNUM_ONLY.sub("", left) == _ALNUM_ONLY.sub("", right)


def tokenize(text: str) -> tuple[str, ...]:
    """Whitespace tokens of `text`, normalized, with empty (pure-punctuation) tokens
    dropped. Dropping them is what makes punctuation-only differences between two
    readings a NON-event (directive §1, acceptance test 2)."""
    out: list[str] = []
    for piece in str(text or "").split():
        token = normalize_token(piece)
        if token:
            out.append(token)
    return tuple(out)


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------

class _Strict(BaseModel):
    """Frozen + `extra="forbid"`, for the same reason as the claim contract: a
    source-evidence record a consumer can edit is not evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceToken(_Strict):
    """One token as ONE channel read it, with where on the page it sits.

    `x0/top/x1/bottom` are PDF user-space coordinates with the origin at the page's
    top-left (pdfplumber's convention), or all None for a channel that reports no
    geometry (a model reading a page image returns text, not boxes). A missing box
    is recorded as missing; it is never approximated.
    """

    text: str
    normalized: str
    x0: float | None = None
    top: float | None = None
    x1: float | None = None
    bottom: float | None = None
    confidence: float | None = None
    #: Competing readings this channel considered for this token, best first.
    alternatives: tuple[str, ...] = ()


class PageRead(_Strict):
    """What ONE channel obtained from ONE page."""

    channel_id: str
    page_number: int = Field(ge=1)
    status: PageStatus
    text: str = ""
    text_sha256: str = ""
    tokens: tuple[SourceToken, ...] = ()
    #: Why this read is not READ, when it is not.
    detail: str = ""

    @property
    def usable(self) -> bool:
        """Can this read be used to PROVE or DISPROVE a quotation? A blank page
        proves nothing about a quotation someone claims is on it."""
        return self.status is PageStatus.READ and bool(self.tokens)


class ReadChannel(_Strict):
    """The identity of one way of reading the document.

    Mirrors `claude_coder.extraction.ExtractionOrigin` deliberately — provider,
    profile, prompt digest and schema version are the SAME four facts that decide
    whether two model assertions are independent anywhere else in this codebase, and
    a second notion of identity here would be free to drift out of agreement with
    that one.
    """

    channel_id: str
    kind: ChannelKind
    #: Declared provider ("pdf" for the document's own text layer, an LLM vendor for
    #: a model read). Never a credential.
    provider: str = ""
    profile: str = ""
    prompt_sha256: str = ""
    schema_version: str = ""
    #: Free-form engine/tool identity (library + version) for deterministic channels.
    engine: str = ""


class SourcePage(_Strict):
    """One rendered page of the ORIGINAL document, and every channel's read of it."""

    page_number: int = Field(ge=1)
    #: sha256 of the rendered page image bytes — the "source image hash" a released
    #: fact must resolve to.
    image_sha256: str = ""
    width: float | None = None
    height: float | None = None
    rotation: int = 0
    status: PageStatus = PageStatus.READ
    #: Explicit, non-fatal observations: `rotated:90`, `duplicate_of_page:2`,
    #: `low_text_yield`, `no_embedded_text`. Recorded so a page anomaly is a stated
    #: outcome rather than a silent skip.
    flags: tuple[str, ...] = ()
    reads: tuple[PageRead, ...] = ()
    #: [char_start, char_end) of this page's PRIMARY text inside `primary_text()`.
    char_start: int = 0
    char_end: int = 0

    def read_by(self, channel_id: str) -> PageRead | None:
        for read in self.reads:
            if read.channel_id == channel_id:
                return read
        return None


class SourceEvidenceDocument(_Strict):
    """A versioned, multi-channel reading of one original document."""

    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION
    filename: str = ""
    #: sha256 of the ORIGINAL document bytes (`sha256:<hex>` form, matching the
    #: `document_version` every evidence span is already salted with).
    document_sha256: str = ""
    page_count: int = 0
    channels: tuple[ReadChannel, ...] = ()
    primary_channel_id: str = ""
    pages: tuple[SourcePage, ...] = ()
    #: Document-level anomalies (duplicate pages, page-count disagreements between
    #: channels, an unopenable text layer). Carried into the certificate.
    anomalies: tuple[str, ...] = ()
    compiled_at: str = ""
    #: Why a deterministic channel is absent, when it is. An EMPTY string with no
    #: independent channel would read as "nobody looked"; this says who looked and
    #: what they found.
    compiler_notes: tuple[str, ...] = ()

    # ------------------------------------------------------------------ views

    def channel(self, channel_id: str) -> ReadChannel | None:
        for channel in self.channels:
            if channel.channel_id == channel_id:
                return channel
        return None

    @property
    def primary_channel(self) -> ReadChannel | None:
        return self.channel(self.primary_channel_id)

    def page(self, page_number: int) -> SourcePage | None:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def primary_text(self) -> str:
        """The exact string the coder reads and every evidence span is anchored into."""
        return PAGE_SEPARATOR.join(
            (page.read_by(self.primary_channel_id).text
             if page.read_by(self.primary_channel_id) else "")
            for page in self.pages)

    def pages_for_offsets(self, start: int | None, end: int | None) -> tuple[int, ...]:
        """Which page(s) a [start, end) character range of `primary_text()` falls on.

        A quotation may straddle the page separator; returning EVERY overlapped page
        (rather than the first) is what lets such a span be reconciled against the
        concatenation of the readings that actually contain it.
        """
        if start is None or end is None or end <= start:
            return ()
        return tuple(page.page_number for page in self.pages
                     if page.char_start < end and start < page.char_end)

    def independent_channels(self) -> tuple[ReadChannel, ...]:
        """Every channel that may be used to CHECK the primary one."""
        primary = self.primary_channel
        if primary is None:
            return ()
        return tuple(c for c in self.channels if independent_of(c, primary))

    # ------------------------------------------------------------- integrity

    def integrity_problems(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.schema_id != SCHEMA_ID:
            out.append(f"schema id {self.schema_id!r} is not {SCHEMA_ID!r}")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            out.append(f"schema version {self.schema_version} is not supported")
        if not self.document_sha256:
            out.append("document carries no source-document digest")
        if self.primary_channel is None:
            out.append("document declares no primary read channel")
        if len(self.pages) != self.page_count:
            out.append(f"document declares {self.page_count} page(s) but carries "
                       f"{len(self.pages)}")
        numbers = [p.page_number for p in self.pages]
        if numbers != sorted(set(numbers)) or (numbers and numbers[0] != 1):
            out.append("pages are not a contiguous 1-based sequence")
        # The offsets are the join between a character span and a page. If they do not
        # reproduce the primary text, every recorded page number is wrong.
        text = self.primary_text()
        for page in self.pages:
            read = page.read_by(self.primary_channel_id)
            expected = read.text if read else ""
            if text[page.char_start:page.char_end] != expected:
                out.append(f"page {page.page_number} character offsets do not "
                           f"reproduce its primary text")
        return tuple(out)

    # ------------------------------------------------------- transformations

    def with_channel(self, channel: ReadChannel, reads: dict[int, PageRead],
                     *, require_independent: bool = False) -> "SourceEvidenceDocument":
        """A COPY carrying one more channel — how a lazily obtained second read is
        added without mutating an already-attested document.

        Refuses to replace an existing channel id: a channel whose reads can be
        overwritten is not evidence of anything.

        `require_independent` is set by every caller adding a channel whose PURPOSE is
        to check the primary reading. It makes independence a checked precondition of
        entering the document rather than a property the reconciler silently discovers
        is absent (issue #6 F7-R5): without it, a second read that shares the primary's
        provider is admitted, contributes nothing, and leaves an audit record naming a
        channel that proved nothing.
        """
        if self.channel(channel.channel_id) is not None:
            raise InvalidSourceEvidenceDocument(
                f"channel {channel.channel_id!r} is already part of this document; "
                f"a second read is a NEW channel, never an overwrite of an old one")
        if require_independent:
            require_independent_channel(self, channel)
        pages = tuple(
            page.model_copy(update={"reads": page.reads + (reads[page.page_number],)})
            if page.page_number in reads else page
            for page in self.pages)
        return self.model_copy(update={"channels": self.channels + (channel,),
                                       "pages": pages})

    # ----------------------------------------------------------- serialization

    def identity(self) -> dict[str, Any]:
        """The compact, certificate-sized identity of this reading.

        The full token streams are NOT included: a certificate must bind WHICH bytes
        were read by WHICH channels, not carry a second copy of the document.
        """
        return {
            "schema": f"{self.schema_id}/{self.schema_version}",
            "filename": self.filename,
            "document_sha256": self.document_sha256,
            "page_count": self.page_count,
            "primary_channel_id": self.primary_channel_id,
            "channels": [c.model_dump(mode="json") for c in self.channels],
            "pages": [{
                "page_number": p.page_number,
                "image_sha256": p.image_sha256,
                "status": p.status.value,
                "rotation": p.rotation,
                "flags": list(p.flags),
                "reads": [{"channel_id": r.channel_id, "status": r.status.value,
                           "text_sha256": r.text_sha256, "tokens": len(r.tokens),
                           "detail": r.detail}
                          for r in p.reads],
            } for p in self.pages],
            "anomalies": list(self.anomalies),
            "compiler_notes": list(self.compiler_notes),
        }

    def fingerprint(self) -> str:
        return "sha256:" + content_digest(self.identity())

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def independent_of(channel: ReadChannel, primary: ReadChannel) -> bool:
    """May `channel` be used as evidence ABOUT `primary`'s reading?

    A channel is never independent of itself. A channel of a DIFFERENT KIND is
    independent by construction — the document's own text layer and a deterministic
    OCR engine are properties of the bytes/pixels rather than of the transcriber. Two
    channels of the SAME kind are independent only when a different declared provider
    answered, which is the identical rule `claude_coder.verify.corroboration_origin`
    applies to code corroboration: one vendor agreeing with itself is repetition, not
    confirmation.
    """
    if channel.channel_id == primary.channel_id:
        return False
    if channel.kind is not primary.kind:
        return True
    return bool(channel.provider) and bool(primary.provider) and \
        channel.provider != primary.provider


def require_independent_channel(document: SourceEvidenceDocument,
                                channel: ReadChannel) -> None:
    """Fail closed unless `channel` is genuinely independent of `document`'s primary.

    `independent_of` ANSWERS the question; this one ENFORCES it, and is called by every
    path that obtains a channel in order to check the primary reading. Both refusals it
    can raise are the same defect seen from two sides: a reading credited as independent
    that is not weakens the control silently, and a reading rejected as same-provider
    that actually is independent holds the encounter for nothing -- which is why the
    identities being compared must come from the client/callable that ran, not from a
    configuration setting that may describe a different call (issue #6 F7-R5).
    """
    primary = document.primary_channel
    if primary is None:
        raise ChannelIndependenceError(
            f"channel {channel.channel_id!r} cannot be established as independent: the "
            f"document declares no primary read channel to be independent OF")
    if not independent_of(channel, primary):
        raise ChannelIndependenceError(
            f"channel {channel.channel_id!r} (kind {channel.kind.value}, provider "
            f"{channel.provider or '<undeclared>'}) is not independent of the primary "
            f"channel {primary.channel_id!r} (kind {primary.kind.value}, provider "
            f"{primary.provider or '<undeclared>'}); a reading that shares the primary "
            f"reading's provider -- or whose provider is not positively established --"
            f" cannot be evidence about it")


def load_document(payload: Any) -> SourceEvidenceDocument:
    """Parse a payload as a SourceEvidenceDocument, or refuse with a typed error."""
    if not isinstance(payload, dict):
        raise InvalidSourceEvidenceDocument(
            f"a SourceEvidenceDocument payload must be an object, got "
            f"{type(payload).__name__}")
    schema_id = payload.get("schema_id")
    if schema_id != SCHEMA_ID:
        raise UnknownSourceEvidenceSchema(
            f"payload declares schema_id {schema_id!r}, not {SCHEMA_ID!r}")
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnknownSourceEvidenceSchema(
            f"payload declares schema_version {version!r}; this build reads "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}")
    try:
        return SourceEvidenceDocument.model_validate(payload)
    except ValidationError as exc:
        raise InvalidSourceEvidenceDocument(
            f"payload declares {SCHEMA_ID}/{version} but does not satisfy it: "
            f"{exc}") from None


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------

class SpanTarget(_Strict):
    """One quotation to prove: what it says and where the primary channel put it."""

    span_id: str
    text: str
    start: int | None = None
    end: int | None = None
    #: The clinical fact this quotation supports — the join key back into the graph.
    fact_id: str = ""


class TokenDifference(_Strict):
    """One disagreement between two readings of the same place on the page."""

    #: `substituted` | `missing_from_independent_read` | `absent_from_quotation`
    kind: str
    quoted: str = ""
    independent: str = ""
    #: Annotation only (see `is_numeric_token`): does either side carry a digit?
    numeric: bool = False


class PageRegion(_Strict):
    """Where on the page a quotation was found, in PDF user space.

    Present only when the proving channel reports geometry. `None` is recorded as
    `None`: an approximate box would be worse than an absent one, because a claim
    reviewer would take it literally.
    """

    page_number: int
    x0: float
    top: float
    x1: float
    bottom: float


class SpanReconciliation(_Strict):
    """The proof (or the refusal) for ONE quotation."""

    span_id: str
    fact_id: str = ""
    status: ReconciliationStatus
    #: Which page(s) of the ORIGINAL document the quotation sits on.
    pages: tuple[int, ...] = ()
    #: sha256 of the rendered image(s) of those page(s) — the source image identity a
    #: released fact must resolve to.
    page_image_sha256: tuple[str, ...] = ()
    #: The channel that proved (or refuted) it, and the digest of the exact text it read.
    verified_by_channel_id: str = ""
    verified_text_sha256: tuple[str, ...] = ()
    region: PageRegion | None = None
    differences: tuple[TokenDifference, ...] = ()
    detail: str = ""

    @property
    def blocking(self) -> bool:
        return self.status in BLOCKING_STATUSES

    @property
    def holding(self) -> bool:
        return self.status in HOLDING_STATUSES


class SourceReconciliation(_Strict):
    """Every quotation's proof, for one encounter — bound into the certificate."""

    control_mode: str = "ENFORCED_FAIL_CLOSED"
    document_sha256: str = ""
    document_fingerprint: str = ""
    primary_channel_id: str = ""
    independent_channel_ids: tuple[str, ...] = ()
    spans: tuple[SpanReconciliation, ...] = ()
    #: Per-page explicit outcomes, including pages nothing was quoted from.
    page_outcomes: tuple[dict[str, Any], ...] = ()
    document_anomalies: tuple[str, ...] = ()

    def by_span_id(self) -> dict[str, SpanReconciliation]:
        return {s.span_id: s for s in self.spans}

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in ReconciliationStatus}
        for span in self.spans:
            counts[span.status.value] += 1
        return counts

    def certificate_record(self) -> dict[str, Any]:
        """What the release certificate binds: identities, per-span outcomes and every
        difference — never the token streams themselves."""
        return {
            "control_mode": self.control_mode,
            "document_sha256": self.document_sha256,
            "document_fingerprint": self.document_fingerprint,
            "primary_channel_id": self.primary_channel_id,
            "independent_channel_ids": list(self.independent_channel_ids),
            "summary": self.summary(),
            "document_anomalies": list(self.document_anomalies),
            "page_outcomes": [dict(p) for p in self.page_outcomes],
            "spans": [s.model_dump(mode="json") for s in self.spans],
        }


def _infix_alignment(quoted: tuple[str, ...],
                     independent: tuple[str, ...]) -> tuple[int, list[tuple[str, int, int]]]:
    """Best alignment of `quoted` ANYWHERE inside `independent`.

    Semi-global (infix) edit distance: skipping independent-channel tokens before and
    after the match is free, so a short quotation is located inside a whole page's
    reading without having to guess a window. Returns the edit cost and the operation
    trace `(op, quoted_index, independent_index)` with `op` in
    {"match", "substitute", "delete", "insert"} — "delete" means the quotation has a
    token the independent read does not, "insert" the reverse.

    O(len(quoted) x len(independent)); the quotation is a short verbatim phrase and a
    page's reading is at most a few thousand tokens, so this is cheap enough to run on
    every span of every note.
    """
    n, m = len(quoted), len(independent)
    if n == 0:
        return 0, []
    # dp[i][j] = cost of aligning quoted[:i] ending at independent[:j]; row 0 is all
    # zeros (a free prefix skip).
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for i in range(1, n + 1):
        row, prev = dp[i], dp[i - 1]
        qi = quoted[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0 if tokens_equal(qi, independent[j - 1]) else 1)
            row[j] = min(sub, prev[j] + 1, row[j - 1] + 1)
    end = min(range(m + 1), key=lambda j: (dp[n][j], -j))
    cost = dp[n][end]
    # Backtrace from (n, end) to row 0 (the free prefix skip).
    trace: list[tuple[str, int, int]] = []
    i, j = n, end
    while i > 0:
        qi = quoted[i - 1]
        if j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                0 if tokens_equal(qi, independent[j - 1]) else 1):
            trace.append(("match" if tokens_equal(qi, independent[j - 1])
                          else "substitute", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            trace.append(("delete", i - 1, -1))
            i -= 1
        else:
            trace.append(("insert", -1, j - 1))
            j -= 1
    trace.reverse()
    return cost, trace


def _region_of(tokens: list[SourceToken], indices: list[int],
               page_of_index: list[int]) -> PageRegion | None:
    """The bounding box of the matched tokens, when the proving channel reports one.

    Only tokens on ONE page contribute: a box spanning two pages is not a region on
    any page, and reporting the first page's box for a two-page quotation would state
    something false. A quotation that straddles a page break therefore resolves to its
    page NUMBERS with no box, which is recorded honestly rather than approximated.
    """
    boxed = [(page_of_index[i], tokens[i]) for i in indices
             if 0 <= i < len(tokens) and tokens[i].x0 is not None]
    if not boxed:
        return None
    pages = {p for p, _ in boxed}
    if len(pages) != 1:
        return None
    page_number = pages.pop()
    xs0 = [t.x0 for _, t in boxed if t.x0 is not None]
    tops = [t.top for _, t in boxed if t.top is not None]
    xs1 = [t.x1 for _, t in boxed if t.x1 is not None]
    bottoms = [t.bottom for _, t in boxed if t.bottom is not None]
    if not (xs0 and tops and xs1 and bottoms):
        return None
    return PageRegion(page_number=page_number, x0=min(xs0), top=min(tops),
                      x1=max(xs1), bottom=max(bottoms))


#: How bad each answer is. A quotation gets the WORST answer any independent channel
#: gave it, never the most convenient one: two independent readings that disagree with
#: each other about a word a billed line rests on is an unresolved discrepancy, and the
#: directive's instruction for an unresolved discrepancy is to block. Letting a second
#: channel that happens to repeat the primary's misreading clear the first channel's
#: disagreement would make the mechanism weaker the more channels it is given.
_SEVERITY = {
    ReconciliationStatus.DISAGREED: 0,
    ReconciliationStatus.NOT_LOCATED: 1,
    ReconciliationStatus.AGREED: 2,
}

#: Below this share of matched tokens the independent reading does not contain the
#: quotation at all (NOT_LOCATED) rather than reading it differently (DISAGREED).
#: It only splits one blocking status from another blocking status — no release
#: depends on where it sits.
_LOCATION_THRESHOLD = 0.5


def _channel_verdicts(document: SourceEvidenceDocument, target: "SpanTarget",
                      quoted: tuple[str, ...], pages: tuple[int, ...],
                      page_images: tuple[str, ...],
                      channels: tuple[ReadChannel, ...]) -> list[SpanReconciliation]:
    """One verdict per channel that could read EVERY page the quotation sits on.

    Extracted so that the two things that must be proven against a page -- a quotation
    the PRIMARY reading proposed, and a quotation an INDEPENDENT reading proposed that
    the primary one may never have contained -- are located, tokenized, aligned, boxed
    and scored by exactly one implementation. Only the SET of channels asked and the
    rule for combining their answers differ between the two, and both of those are the
    caller's, stated at the call site rather than buried here.
    """
    candidates: list[SpanReconciliation] = []
    for channel in channels:
        reads = [document.page(n).read_by(channel.channel_id) if document.page(n)
                 else None for n in pages]
        if any(r is None or not r.usable for r in reads):
            continue                      # this channel does not cover every page
        tokens: list[SourceToken] = []
        page_of_index: list[int] = []
        for page_number, read in zip(pages, reads):
            for token in read.tokens:
                tokens.append(token)
                page_of_index.append(page_number)
        independent = tuple(t.normalized for t in tokens)
        cost, trace = _infix_alignment(quoted, independent)
        matched = [j for op, _, j in trace if op == "match"]
        differences = tuple(
            TokenDifference(
                kind=("substituted" if op == "substitute"
                      else "missing_from_independent_read" if op == "delete"
                      else "absent_from_quotation"),
                quoted=(quoted[i] if i >= 0 else ""),
                independent=(independent[j] if j >= 0 else ""),
                numeric=(is_numeric_token(quoted[i]) if i >= 0 else False)
                or (is_numeric_token(independent[j]) if j >= 0 else False))
            for op, i, j in trace if op != "match")
        share = len(matched) / len(quoted)
        if cost == 0:
            status = ReconciliationStatus.AGREED
            detail = (f"every token of the quotation was read identically by "
                      f"{channel.channel_id}")
        elif share >= _LOCATION_THRESHOLD:
            status = ReconciliationStatus.DISAGREED
            detail = (f"{channel.channel_id} reads {len(differences)} token(s) of "
                      f"this quotation differently")
        else:
            status = ReconciliationStatus.NOT_LOCATED
            detail = (f"{channel.channel_id} read page(s) "
                      f"{', '.join(str(p) for p in pages)} and the quotation does "
                      f"not appear in that reading")
        candidates.append(SpanReconciliation(
            span_id=target.span_id, fact_id=target.fact_id, status=status,
            pages=pages, page_image_sha256=page_images,
            verified_by_channel_id=channel.channel_id,
            verified_text_sha256=tuple(r.text_sha256 for r in reads if r),
            region=_region_of(tokens, matched, page_of_index),
            differences=differences, detail=detail))
    return candidates


def reconcile_spans(document: SourceEvidenceDocument,
                    targets: list[SpanTarget] | tuple[SpanTarget, ...],
                    *, control_mode: str = "ENFORCED_FAIL_CLOSED") -> SourceReconciliation:
    """Prove every quotation against an INDEPENDENT reading of its own page(s).

    Fail-closed by construction: the only statuses that permit a release are AGREED
    (an independent channel read the same tokens) and VACUOUS (the quotation contains
    no material token). Everything else — a differing token, a quotation the
    independent reading does not contain, or no independent reading at all — is
    reported as what it is and stops the claim at the gate that reads this record.
    """
    primary = document.primary_channel
    independents = document.independent_channels()
    results: list[SpanReconciliation] = []

    for target in targets:
        quoted = tokenize(target.text)
        pages = document.pages_for_offsets(target.start, target.end)
        page_images = tuple(
            (document.page(n).image_sha256 if document.page(n) else "") for n in pages)
        if not quoted:
            results.append(SpanReconciliation(
                span_id=target.span_id, fact_id=target.fact_id,
                status=ReconciliationStatus.VACUOUS, pages=pages,
                page_image_sha256=page_images,
                detail="quotation carries no material token (punctuation only)"))
            continue
        if not pages:
            results.append(SpanReconciliation(
                span_id=target.span_id, fact_id=target.fact_id,
                status=ReconciliationStatus.UNVERIFIABLE,
                detail="quotation has no verified character offsets in the primary "
                       "reading, so no page of the original document can be named"))
            continue

        candidates = _channel_verdicts(document, target, quoted, pages, page_images,
                                       independents)
        # The worst answer wins (see `_SEVERITY`). Channels that could not read the
        # page at all produced no candidate and therefore contribute nothing — an
        # unreadable channel is not an opinion.
        best = min(candidates, key=lambda c: _SEVERITY[c.status], default=None)
        if best is None:
            missing = ", ".join(str(p) for p in pages)
            best = SpanReconciliation(
                span_id=target.span_id, fact_id=target.fact_id,
                status=ReconciliationStatus.UNVERIFIABLE, pages=pages,
                page_image_sha256=page_images,
                detail=(f"no independent channel produced a usable reading of page(s) "
                        f"{missing}; the transcription cannot be its own authority"))
        results.append(best)

    quoted_pages = {p for r in results for p in r.pages}
    page_outcomes = tuple({
        "page_number": page.page_number,
        "status": page.status.value,
        "image_sha256": page.image_sha256,
        "rotation": page.rotation,
        "flags": list(page.flags),
        "quoted_from": page.page_number in quoted_pages,
        "independently_read_by": [c.channel_id for c in independents
                                  if (page.read_by(c.channel_id) or None)
                                  and page.read_by(c.channel_id).usable],
    } for page in document.pages)

    return SourceReconciliation(
        control_mode=control_mode,
        document_sha256=document.document_sha256,
        document_fingerprint=document.fingerprint(),
        primary_channel_id=(primary.channel_id if primary else ""),
        independent_channel_ids=tuple(c.channel_id for c in independents),
        spans=tuple(results),
        page_outcomes=page_outcomes,
        document_anomalies=document.anomalies)


# --------------------------------------------------------------------------
# an INDEPENDENT READING of the whole document (issue #6 F7-R3, second reopen)
# --------------------------------------------------------------------------
#
# `reconcile_spans` above proves a quotation the PRIMARY reading already proposed. That
# makes the second reading a check on what the primary transcription CONTAINED, and
# nothing at all about what it LEFT OUT: a service the transcription never captured is
# absent from the only string any extractor was ever given, so every extractor misses
# it identically and the claim is silently short a line.
#
# Recall therefore needs a second reading of the DOCUMENT, not a second model over one
# transcript. The compiler already builds one for every note -- the document's own
# embedded text layer, with per-word boxes -- and the paid page reader already exists
# for pages that layer cannot cover. What was missing is the ability to hand that
# channel's reading to an extractor as a document in its own right, and then to locate
# and prove quotations expressed in ITS character offsets rather than the primary's.
# That is all this section adds.

class ChannelReading(_Strict):
    """One channel's reading of the WHOLE document, as a single anchorable string.

    Deliberately shaped exactly like `primary_text()` -- the same page separator, the
    same page-ordered concatenation -- so that a quotation anchored into it is located
    on a page by the same arithmetic, in this channel's own coordinate space. A page
    this channel could not usably read contributes an EMPTY segment and is named in
    `uncovered_pages`: a reading with a hole in it must say where the hole is, because
    "this channel found no additional service" and "this channel never saw that page"
    are different answers and only one of them is evidence.
    """

    channel_id: str
    text: str = ""
    #: (page_number, char_start, char_end) of every page inside `text`.
    page_offsets: tuple[tuple[int, int, int], ...] = ()
    covered_pages: tuple[int, ...] = ()
    uncovered_pages: tuple[int, ...] = ()

    @property
    def usable(self) -> bool:
        """Is there anything here an extractor could read?"""
        return bool(self.covered_pages) and bool(self.text.strip())

    def pages_for_offsets(self, start: int | None, end: int | None) -> tuple[int, ...]:
        """Which page(s) a [start, end) range of THIS reading falls on."""
        if start is None or end is None or end <= start:
            return ()
        return tuple(number for number, first, last in self.page_offsets
                     if first < end and start < last)


def channel_reading(document: SourceEvidenceDocument,
                    channel_id: str) -> ChannelReading:
    """Assemble one channel's reading of the document into an anchorable string."""
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    covered: list[int] = []
    uncovered: list[int] = []
    cursor = 0
    for page in document.pages:
        read = page.read_by(channel_id)
        body = read.text if (read is not None and read.usable) else ""
        (covered if body else uncovered).append(page.page_number)
        offsets.append((page.page_number, cursor, cursor + len(body)))
        parts.append(body)
        cursor += len(body) + len(PAGE_SEPARATOR)
    return ChannelReading(
        channel_id=channel_id, text=PAGE_SEPARATOR.join(parts),
        page_offsets=tuple(offsets), covered_pages=tuple(covered),
        uncovered_pages=tuple(uncovered))


def recall_channel(document: SourceEvidenceDocument) -> ReadChannel | None:
    """WHICH channel a recall reading should come from, by the document's own rules.

    The candidate set is exactly `independent_channels()` -- the one existing answer to
    "which readings may be evidence about the primary one" -- so a channel that could
    not check the transcription can never become the reading that adds to it either.
    Among those, the widest page coverage wins, and a channel that is a property of the
    document's bytes rather than of a model breaks ties ahead of one that is not: it is
    free, deterministic, reproducible from the same file forever, and it cannot invent
    a service. Ties beyond that fall to channel id so the choice is reproducible.
    """
    scored: list[tuple[int, int, str, ReadChannel]] = []
    for channel in document.independent_channels():
        reading = channel_reading(document, channel.channel_id)
        if not reading.usable:
            continue
        deterministic = 0 if channel.kind is not ChannelKind.VISION else 1
        scored.append((-len(reading.covered_pages), deterministic,
                       channel.channel_id, channel))
    if not scored:
        return None
    scored.sort(key=lambda item: item[:3])
    return scored[0][3]


def recall_reading(document: SourceEvidenceDocument) -> ChannelReading | None:
    """The independent reading of the document a recall extraction should be run over,
    or None when no channel other than the transcription could read any page."""
    channel = recall_channel(document)
    if channel is None:
        return None
    return channel_reading(document, channel.channel_id)


def reconcile_reading(document: SourceEvidenceDocument,
                      targets: list[SpanTarget] | tuple[SpanTarget, ...],
                      reading: ChannelReading,
                      *, control_mode: str = "ENFORCED_FAIL_CLOSED"
                      ) -> SourceReconciliation:
    """Prove quotations that came from `reading` -- NOT from the primary transcription.

    Two things differ from `reconcile_spans`, and both are forced by what is being
    asked rather than chosen for convenience:

      * the page is located in the READING's coordinate space, because that is the
        string these offsets were verified against;

      * the primary transcription's SILENCE cannot refute the quotation. Whether the
        transcription omitted this passage is the very hypothesis under test, so the
        channel under test does not get a veto over it. It can only CONFIRM: if the
        transcription does contain the passage after all, the passage is proven and
        what the primary reading actually missed was the extraction, not the page.

    Every OTHER channel is a full authority and the worst of their answers wins,
    exactly as everywhere else -- so two readings that both fail to find the passage
    refute it, and a lone unsupported reading is never admitted.

    The result when nothing but the transcription could be asked, and it does not
    contain the passage, is UNVERIFIABLE: two readings of one document disagree about
    whether a passage is on the page, nothing settled it, and that is system work
    (obtain a reading of that page) rather than either a confirmation or a refutation.
    `pages_needing_independent_read` then names exactly those pages.
    """
    primary = document.primary_channel
    primary_id = primary.channel_id if primary is not None else ""
    checking = tuple(c for c in document.channels
                     if c.channel_id != reading.channel_id)
    results: list[SpanReconciliation] = []

    for target in targets:
        quoted = tokenize(target.text)
        pages = reading.pages_for_offsets(target.start, target.end)
        page_images = tuple(
            (document.page(n).image_sha256 if document.page(n) else "") for n in pages)
        if not quoted:
            results.append(SpanReconciliation(
                span_id=target.span_id, fact_id=target.fact_id,
                status=ReconciliationStatus.VACUOUS, pages=pages,
                page_image_sha256=page_images,
                detail="quotation carries no material token (punctuation only)"))
            continue
        if not pages:
            results.append(SpanReconciliation(
                span_id=target.span_id, fact_id=target.fact_id,
                status=ReconciliationStatus.UNVERIFIABLE,
                detail=(f"quotation has no verified character offsets in the "
                        f"{reading.channel_id} reading, so no page of the original "
                        f"document can be named")))
            continue

        verdicts = _channel_verdicts(document, target, quoted, pages, page_images,
                                     checking)
        others = [v for v in verdicts if v.verified_by_channel_id != primary_id]
        transcription = next(
            (v for v in verdicts if v.verified_by_channel_id == primary_id), None)
        best = min(others, key=lambda c: _SEVERITY[c.status], default=None)
        if best is None and transcription is not None:
            if transcription.status is ReconciliationStatus.AGREED:
                best = transcription.model_copy(update={"detail": (
                    f"the primary transcription contains this passage as well "
                    f"({transcription.detail}); what the primary reading missed is "
                    f"the event, not the text")})
            else:
                best = SpanReconciliation(
                    span_id=target.span_id, fact_id=target.fact_id,
                    status=ReconciliationStatus.UNVERIFIABLE, pages=pages,
                    page_image_sha256=page_images,
                    detail=(f"the {reading.channel_id} reading of page(s) "
                            f"{', '.join(str(p) for p in pages)} carries this passage "
                            f"and the primary transcription does not; no third reading "
                            f"of those page(s) was available to settle which is right"))
        if best is None:
            missing = ", ".join(str(p) for p in pages)
            best = SpanReconciliation(
                span_id=target.span_id, fact_id=target.fact_id,
                status=ReconciliationStatus.UNVERIFIABLE, pages=pages,
                page_image_sha256=page_images,
                detail=(f"no channel other than {reading.channel_id} itself produced a "
                        f"usable reading of page(s) {missing}; a reading cannot be its "
                        f"own authority"))
        results.append(best)

    quoted_pages = {p for r in results for p in r.pages}
    page_outcomes = tuple({
        "page_number": page.page_number,
        "status": page.status.value,
        "image_sha256": page.image_sha256,
        "rotation": page.rotation,
        "flags": list(page.flags),
        "quoted_from": page.page_number in quoted_pages,
        "independently_read_by": [c.channel_id for c in checking
                                  if (page.read_by(c.channel_id) or None)
                                  and page.read_by(c.channel_id).usable],
    } for page in document.pages)

    return SourceReconciliation(
        control_mode=control_mode,
        document_sha256=document.document_sha256,
        document_fingerprint=document.fingerprint(),
        primary_channel_id=primary_id,
        # The primary channel is ASKED above (it can confirm a recall quotation) but is
        # never listed here: this field is what a certificate reads as "who
        # independently checked this", and the transcription is not an independent check
        # on itself in either direction.
        independent_channel_ids=tuple(c.channel_id for c in checking
                                      if c.channel_id != primary_id),
        spans=tuple(results),
        page_outcomes=page_outcomes,
        document_anomalies=document.anomalies)


def merge_reconciliations(*parts: "SourceReconciliation | None") -> "SourceReconciliation | None":
    """One reconciliation record over quotations proven in DIFFERENT readings.

    Every consumer downstream joins on `span_id`, and a span id is salted with the
    reading it was anchored in, so the union is well defined and collision-free. The
    FIRST record carrying a given span id wins, so re-reconciling after a paid page
    read is done by putting the newer record first rather than by mutating an older one.
    """
    present = [p for p in parts if p is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    spans: dict[str, SpanReconciliation] = {}
    for record in present:
        for span in record.spans:
            spans.setdefault(span.span_id, span)
    head = present[0]
    # The PRIMARY channel is deliberately excluded even though `reconcile_reading` asks
    # it: it is a legitimate CONFIRMER of a recall quotation, but it is never an
    # independent check on itself, and a merged record that listed it here would tell a
    # certificate reader the transcription had independently verified the transcription.
    channels: list[str] = []
    for record in present:
        for channel_id in record.independent_channel_ids:
            if channel_id and channel_id != head.primary_channel_id \
                    and channel_id not in channels:
                channels.append(channel_id)
    return head.model_copy(update={
        "spans": tuple(spans.values()),
        "independent_channel_ids": tuple(channels)})


def pages_needing_independent_read(
        document: SourceEvidenceDocument,
        reconciliation: SourceReconciliation,
        span_ids: set[str] | frozenset[str]) -> tuple[int, ...]:
    """Exactly the pages a SECOND PAID READ would change the answer for.

    This is the cost control the directive asks for, expressed as a query rather than
    as a policy buried in a caller: a page is worth paying to re-read only when a
    quotation that justified a RELEASED line sits on it AND no independent channel
    could read it. Pages nobody quoted from, and pages already independently read, are
    never re-read — so a document whose text layer covers it costs nothing extra at
    all, and a scanned document costs one read of the few pages that carry the claim.
    """
    wanted: set[int] = set()
    for span in reconciliation.spans:
        if span.span_id in span_ids and span.status is ReconciliationStatus.UNVERIFIABLE:
            wanted.update(span.pages)
    return tuple(sorted(n for n in wanted if document.page(n) is not None))


def build_page_read(channel_id: str, page_number: int, text: str,
                    *, tokens: list[SourceToken] | None = None,
                    status: PageStatus | None = None,
                    detail: str = "") -> PageRead:
    """One channel's read of one page, with its digest and token stream derived here.

    Centralised so every producer — the PDF text layer, the vision transcription and
    any second model read — hashes and tokenizes identically. Two producers computing
    "the same" digest differently is the drift class this codebase keeps finding.
    """
    body = str(text or "")
    if tokens is None:
        tokens = [SourceToken(text=piece, normalized=normalize_token(piece))
                  for piece in body.split() if normalize_token(piece)]
    if status is None:
        status = PageStatus.READ if tokens else PageStatus.BLANK
    return PageRead(
        channel_id=channel_id, page_number=page_number, status=status, text=body,
        text_sha256="sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
        tokens=tuple(tokens), detail=detail)


# --------------------------------------------------------------------------
# the DATE OF SERVICE, reconciled the same way every other code-changing fact is
# (issue #6 F7-R4)
# --------------------------------------------------------------------------
#
# The DOS is not a clinical fact the extraction layer quotes -- it arrives as a
# STRUCTURED METADATA FIELD of the primary vision transcription, so `reconcile_spans`
# above never saw it. It is nonetheless the single most date-versioned value on the
# claim: it selects the coverage in force, the provider's billing affiliation, the
# authorization window, the effective code edition and the claim's own service date.
# A one-character misread of it produces a fully populated, fully fingerprinted,
# confidently wrong claim.
#
# Reconciling it needs no new mechanism, only an anchor. A date is proven exactly
# like any other quotation once you can say WHERE ON THE PAGE it is written, so this
# locates the transcription's proposed date inside the transcription's own reading
# (`app.core.dates.find_dates` gives the character offsets), turns each occurrence
# into an ordinary `SpanTarget`, and hands them to `reconcile_spans`. The proof, the
# page, the page-image digest, the region and the token differences are then produced
# by the same code path, with the same fail-closed statuses, as every clinical fact.
#
# NOTHING HERE IS CLINICAL, and nothing here is a date FORMAT policy: an unrecognised
# written form simply yields no anchor, which holds.

#: The `fact_id` every service-date span carries, so the reconciliation record can be
#: joined back to the field it proves without pattern-matching a span id.
SERVICE_DATE_FACT_ID = "encounter.date_of_service"

#: Ordering used to collapse SEVERAL written occurrences of the same date into one
#: answer. It is deliberately NOT `_SEVERITY`'s "worst wins": that rule collapses two
#: CHANNELS reading ONE place, where a disagreement is unresolved and must block. Here
#: the occurrences are different PLACES on the page that all say the same date, and the
#: two failure modes are genuinely different:
#:
#:   * a CONTRADICTION (the independent reading of that place says something else, or
#:     does not contain the date at all) is evidence the transcription invented or
#:     misread a date, and one is enough to block -- worst wins among those;
#:   * an ABSENCE (that place sits on a page nothing independent could read) proves
#:     nothing either way, and must not erase a proof obtained elsewhere.
#:
#: So: any contradiction blocks; otherwise the best available proof stands.
_SERVICE_DATE_PREFERENCE = {
    ReconciliationStatus.AGREED: 0,
    ReconciliationStatus.VACUOUS: 1,
    ReconciliationStatus.UNVERIFIABLE: 2,
    ReconciliationStatus.NOT_LOCATED: 3,
    ReconciliationStatus.DISAGREED: 4,
}


class ServiceDateEvidence(_Strict):
    """What the ORIGINAL DOCUMENT says its date of service is, and how that was proven.

    `status is AGREED` is the only outcome that lets a document-derived DOS bind to a
    claim. Every other outcome is recorded with the reason, and the encounter holds --
    a date nobody could confirm against the page is not a safer default than no date,
    it is a worse one, because it looks exactly like a confirmed date downstream.
    """

    #: The ISO date the primary transcription proposed, normalized. Empty when it
    #: proposed nothing parseable.
    candidate: str = ""
    #: The date exactly as it is WRITTEN on the page ("March 14, 2026", "3/14/26").
    located_text: str = ""
    #: How many times that date is written in the primary reading.
    occurrences: int = 0
    status: ReconciliationStatus = ReconciliationStatus.UNVERIFIABLE
    detail: str = ""
    span_id: str = ""
    pages: tuple[int, ...] = ()
    page_image_sha256: tuple[str, ...] = ()
    verified_by_channel_id: str = ""
    verified_text_sha256: tuple[str, ...] = ()
    differences: tuple[TokenDifference, ...] = ()
    region: PageRegion | None = None
    document_sha256: str = ""
    document_fingerprint: str = ""

    @property
    def reconciled(self) -> bool:
        """May this date bind to a claim on the document's authority alone?"""
        return self.status is ReconciliationStatus.AGREED

    def record(self) -> dict[str, Any]:
        """The compact form carried in the encounter context and the certificate."""
        return {
            "candidate": self.candidate,
            "located_text": self.located_text,
            "occurrences": self.occurrences,
            "status": self.status.value,
            "detail": self.detail,
            "span_id": self.span_id,
            "pages": list(self.pages),
            "page_image_sha256": list(self.page_image_sha256),
            "verified_by_channel_id": self.verified_by_channel_id,
            "document_sha256": self.document_sha256,
            "document_fingerprint": self.document_fingerprint,
            "differences": [d.model_dump(mode="json") for d in self.differences],
        }


def reconcile_service_date(document: SourceEvidenceDocument, candidate: Any,
                           *, control_mode: str = "ENFORCED_FAIL_CLOSED"
                           ) -> ServiceDateEvidence:
    """Prove the transcription's date of service against the ORIGINAL document.

    Three refusals, each a different fact and each recorded as itself:

      * the transcription proposed nothing parseable -> UNVERIFIABLE (nothing to prove);
      * it proposed a date that is written NOWHERE in its own reading of the document
        -> NOT_LOCATED. This is the metadata-only misread: the structured field says
        one date, the pages say another, and no page of the original can be named for
        the value the claim would carry;
      * it proposed a date that IS written on a page, but an independent reading of
        that page reads those characters differently -> DISAGREED. This is the
        transcription-wide misread, and it is exactly the perturbation case a
        single-channel read can never detect.
    """
    base = {"document_sha256": document.document_sha256,
            "document_fingerprint": document.fingerprint()}
    parsed = parse_date(str(candidate or "").strip())
    if parsed is None:
        return ServiceDateEvidence(
            status=ReconciliationStatus.UNVERIFIABLE,
            detail="the document's reading proposed no parseable date of service, so "
                   "there is no date to locate on a page of the original",
            **base)
    text = document.primary_text()
    hits = [(start, end) for start, end, found in find_dates(text) if found == parsed]
    if not hits:
        return ServiceDateEvidence(
            candidate=parsed.isoformat(), status=ReconciliationStatus.NOT_LOCATED,
            detail=(f"the date of service reported for this encounter "
                    f"({parsed.isoformat()}) is written nowhere in the document's own "
                    f"reading, so no page of the original states it"),
            **base)
    targets = [SpanTarget(span_id=f"{SERVICE_DATE_FACT_ID}@{start}",
                          text=text[start:end], start=start, end=end,
                          fact_id=SERVICE_DATE_FACT_ID)
               for start, end in hits]
    reconciliation = reconcile_spans(document, targets, control_mode=control_mode)
    # Any contradiction blocks (worst of them is reported); absent that, the best
    # proof any occurrence obtained stands. See `_SERVICE_DATE_PREFERENCE`.
    contradicting = [span for span in reconciliation.spans
                     if span.status in BLOCKING_STATUSES]
    chosen = (max(contradicting, key=lambda span: _SERVICE_DATE_PREFERENCE[span.status])
              if contradicting
              else min(reconciliation.spans,
                       key=lambda span: _SERVICE_DATE_PREFERENCE[span.status]))
    located = next((text[start:end] for start, end in hits
                    if f"{SERVICE_DATE_FACT_ID}@{start}" == chosen.span_id), "")
    return ServiceDateEvidence(
        candidate=parsed.isoformat(), located_text=located, occurrences=len(hits),
        status=chosen.status, detail=chosen.detail, span_id=chosen.span_id,
        pages=chosen.pages, page_image_sha256=chosen.page_image_sha256,
        verified_by_channel_id=chosen.verified_by_channel_id,
        verified_text_sha256=chosen.verified_text_sha256,
        differences=chosen.differences, region=chosen.region, **base)
