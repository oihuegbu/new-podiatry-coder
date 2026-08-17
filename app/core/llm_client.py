from __future__ import annotations

import json
import random
import threading
import time

import uuid

from openai import OpenAI
from app.core.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_EFFORT,
    ANTHROPIC_USE_BATCH,
    ANTHROPIC_BATCH_MAX_WAIT_S,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# One HTTP client PER THREAD, cached in thread-local storage. The SDK clients
# wrap httpx, whose httpcore connection pool guards its internal state with a
# single lock; sharing ONE client across many worker threads serializes on —
# and, observed live, hard-DEADLOCKS on — that pool lock. A 16-worker
# batch-synonym run froze indefinitely with every worker parked in
# httpcore _synchronization.Lock.__enter__ (connection_pool.handle_request),
# their batches long since ended and nothing actually in flight; the per-
# request wall-clock timeout could not rescue it because the block is on the
# pool LOCK, not on a socket read or a connection-slot wait. Giving each
# thread its own client — hence its own pool and lock — removes the shared
# state entirely, so concurrency can never deadlock on it. Built once per
# thread (ThreadPoolExecutor reuses a fixed worker set), not per call.
_tls = threading.local()


# Hard per-request wall clock. The SDKs' own retry layer is disabled
# (max_retries=0) because chat_completion below owns retries with its own
# backoff — stacking the two multiplied worst-case latency. Observed live:
# a consistency batch sat 5+ hours on one note's NER call with all three
# workers asleep on socket reads; a request that hasn't answered in 20
# minutes is dead and must surface as a retryable timeout, not a hang.
_REQUEST_TIMEOUT_S = 1200.0


def get_openai_client() -> OpenAI:
    client = getattr(_tls, "openai", None)
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY,
                        timeout=_REQUEST_TIMEOUT_S, max_retries=0)
        _tls.openai = client
        logger.info("OpenAI client initialized (thread=%s)",
                    threading.current_thread().name)
    return client


def get_anthropic_client():
    client = getattr(_tls, "anthropic", None)
    if client is None:
        import anthropic  # lazy import so openai-only installs still work
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY,
                                     timeout=_REQUEST_TIMEOUT_S,
                                     max_retries=0)
        _tls.anthropic = client
        logger.info("Anthropic client initialized (thread=%s)",
                    threading.current_thread().name)
    return client


#: The canonical provider id for the SDK package that implements a client object.
#: Independence is decided by comparing provider ids that arrive from several places --
#: `config.LLM_PROVIDER`, `chat_completion(provider=...)`, `ReadChannel.provider` and
#: `ExtractionOrigin.provider` -- so a client must resolve to the SAME vocabulary those
#: use ("claude", not the SDK distribution name "anthropic"), or the comparison is
#: between two spellings of one vendor and silently reads as independence.
_CLIENT_PACKAGE_PROVIDER = {"anthropic": "claude", "openai": "openai"}


def provider_of_client(client) -> str:
    """The canonical provider id of the client object that ACTUALLY answers a call.

    Configuration says which vendor a component is SUPPOSED to call; this says which
    vendor's SDK object is about to be called. The two can disagree -- a component may
    call one vendor unconditionally while a generic setting names another -- and when
    they do, an independence decision taken on the configured value is a decision about
    a call that never happened (issue #6 F7-R5). Every channel identity that feeds
    `contracts.source_evidence.independent_of` or `claude_coder.verify`'s
    `corroboration_origin` is therefore derived from the object that ran, exactly as
    `verify.declare_model_profile` derives a callable's identity from the callable.

    FAIL-CLOSED: an unrecognised client yields "" -- identity unestablished -- which
    every independence check reads as "not independent", never as "independent".
    """
    root = str(getattr(type(client), "__module__", "") or "").split(".")[0]
    return _CLIENT_PACKAGE_PROVIDER.get(root, "")


def client_identity(client) -> str:
    """The auditable, credential-free identity of a client object (never its keys)."""
    return f"{type(client).__module__}.{type(client).__qualname__}"


# Transient provider-side failures: capacity ("Overloaded"), rate limits and
# 5xx gateway blips. These are retryable by definition — the request is valid,
# the service just couldn't take it at that moment. Without retries a single
# capacity blip aborts an entire note mid-pipeline (observed live: 7 of 55
# notes in one batch died on Anthropic 'overloaded_error').
_RETRYABLE_MARKERS = (
    "overloaded", "rate_limit", "rate limit", "429", "500", "502", "503",
    "504", "529", "timeout", "timed out", "connection", "server_error",
    "internal error", "service unavailable", "capacity",
)
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 5.0


class _TruncatedResponse(Exception):
    """The model hit max_tokens mid-response. With structured outputs the
    grammar guarantees well-formed JSON only if generation COMPLETES — a
    truncated stream is a syntactically broken prefix that downstream
    parsing can only discard (observed live: note 008's Opus verify pass
    truncated, parse failed, and the note silently lost its entire
    verification audit via the fallback). Retried with a doubled budget."""


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in _RETRYABLE_MARKERS)


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.05,
    max_tokens: int = 4096,
    json_mode: bool = True,
    effort: str | None = None,
    json_schema: dict | None = None,
    use_batch: bool | None = None,
    provider: str | None = None,
) -> tuple[str, dict]:
    """`json_schema`: a strict JSON Schema the response must conform to,
    enforced by the provider's structured-output API (Anthropic
    output_config.format / OpenAI json_schema response_format). `effort`:
    per-call reasoning-effort override (Claude only; defaults to
    CLAUDE_EFFORT). `use_batch`: per-call override of ANTHROPIC_USE_BATCH —
    pass False for an interactive/low-latency call (the Batches API is ~50%
    cheaper but adds minutes of latency, unsuitable for a sequential loop)."""
    selected_provider = str(provider or LLM_PROVIDER).strip().lower()
    if selected_provider not in {"openai", "claude"}:
        raise ValueError(f"unsupported LLM provider: {selected_provider}")
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            if selected_provider == "claude":
                return _claude_chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    effort=effort,
                    json_schema=json_schema,
                    use_batch=use_batch,
                )
            return _openai_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                json_schema=json_schema,
            )
        except _TruncatedResponse as exc:
            if attempt == _MAX_ATTEMPTS:
                raise
            last_exc = exc
            max_tokens *= 2
            logger.warning(
                f"LLM response truncated at max_tokens — retry "
                f"{attempt}/{_MAX_ATTEMPTS - 1} with doubled budget "
                f"({max_tokens} pre-headroom)"
            )
            continue
        except Exception as exc:  # noqa: BLE001 — classified below
            if not _is_retryable(exc) or attempt == _MAX_ATTEMPTS:
                raise
            last_exc = exc
            # Exponential backoff with jitter: 5s, 10s, 20s, 40s (+0-25%).
            delay = _BASE_DELAY_S * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            logger.warning(
                f"LLM call failed with transient error ({exc}) — "
                f"retry {attempt}/{_MAX_ATTEMPTS - 1} in {delay:.0f}s"
            )
            time.sleep(delay)
    raise last_exc  # pragma: no cover — loop always returns or raises


# Newer OpenAI models (the GPT-5 family and o-series reasoning models) changed the
# chat-completions parameter contract: `max_tokens` was renamed to
# `max_completion_tokens`, and they reject a non-default `temperature`. Rather than
# hardcode a model-name list that silently goes stale as OpenAI ships new models, we
# LEARN each model's quirks from the API's own 400 signal on first use and cache them,
# so subsequent calls send the correct shape directly. For models that accept the
# classic parameters (e.g. gpt-4o) the request is byte-for-byte unchanged.
_OPENAI_UNSUPPORTED_PARAMS: dict[str, set[str]] = {}


def _openai_unsupported_param(exc: Exception) -> str | None:
    """If `exc` is an OpenAI 400 rejecting a parameter we know how to adapt
    (`max_tokens` -> `max_completion_tokens`, or a non-default `temperature`),
    return that parameter name; otherwise None (caller re-raises)."""
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else {}
    code = (err or {}).get("code")
    param = (err or {}).get("param")
    if code in ("unsupported_parameter", "unsupported_value") and param in (
            "max_tokens", "temperature"):
        return param
    msg = str(exc).lower()
    if "unsupported" in msg or "not supported" in msg:
        for p in ("max_tokens", "temperature"):
            if p in msg:
                return p
    return None


def _openai_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    json_schema: dict | None = None,
) -> tuple[str, dict]:
    client = get_openai_client()
    mdl = model or OPENAI_MODEL

    def _build(unsupported: set[str]) -> dict:
        kw: dict = {
            "model": mdl,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if "temperature" not in unsupported:
            kw["temperature"] = temperature
        # `max_tokens` was renamed to `max_completion_tokens` on newer models.
        kw["max_completion_tokens" if "max_tokens" in unsupported else "max_tokens"] = max_tokens
        if json_schema is not None:
            # strict=True is what makes OpenAI grammar-enforce the schema; without
            # it the schema is advisory only. Our schemas already satisfy strict
            # mode's constraints (all-required, additionalProperties: false).
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": json_schema},
            }
        elif json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    unsupported = set(_OPENAI_UNSUPPORTED_PARAMS.get(mdl, ()))
    response = None
    # Bounded by the number of parameters we know how to adapt (max_tokens,
    # temperature): each iteration either succeeds, adapts one newly-rejected
    # parameter and retries, or re-raises an error we cannot adapt.
    for _ in range(3):
        try:
            response = client.chat.completions.create(**_build(unsupported))
            break
        except Exception as exc:  # noqa: BLE001 — only param-shape 400s are adapted; others re-raise
            param = _openai_unsupported_param(exc)
            if param is None or param in unsupported:
                raise
            unsupported.add(param)
            _OPENAI_UNSUPPORTED_PARAMS[mdl] = set(unsupported)
            logger.warning(
                f"OpenAI model {mdl!r} rejected '{param}'; adapting request "
                f"(max_tokens->max_completion_tokens / drop temperature) and retrying"
            )
    assert response is not None  # loop only exits via break or a raise

    if response.choices[0].finish_reason == "length":
        raise _TruncatedResponse(f"finish_reason=length at {max_tokens} tokens")
    content = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return content, usage


def _extract_json_from_text(text: str) -> str:
    """Pull the outermost JSON object or array from a string.

    Claude sometimes wraps the response in prose or markdown fences even when
    instructed not to. This function finds the first '{' or '[' and the matching
    closing delimiter so downstream JSON parsers always get clean input.
    """
    text = text.strip()

    # Fast path: already clean JSON
    if text.startswith("{") or text.startswith("["):
        return text

    # Strip a single ```json … ``` or ``` … ``` fence
    import re
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Last resort: find first { or [ and walk to the matching closer
    for start_char, end_char in (("{", "}"), ("[", "]")):
        idx = text.find(start_char)
        if idx == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[idx:], start=idx):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[idx : i + 1]

    # Nothing found — return as-is and let the caller handle the parse error
    return text


def _claude_message_via_batch(client, body: dict):
    """Submit one Messages request through the Message Batches API and block
    until its result is available.

    Why a single-request batch: the Batches API prices every token at 50% of
    the interactive rate with an identical model and output distribution —
    the discount buys Anthropic scheduling flexibility, nothing else. Wrapping
    each call individually keeps the pipeline's architecture (sequential
    passes, chat_completion's retry/truncation loop) completely unchanged;
    the only trade is latency, which a scheduled batch pipeline can absorb.
    """
    custom_id = f"req-{uuid.uuid4().hex}"
    batch = client.messages.batches.create(
        requests=[{"custom_id": custom_id, "params": body}])
    logger.info(f"Batch {batch.id} submitted — polling for completion")

    deadline = time.monotonic() + ANTHROPIC_BATCH_MAX_WAIT_S
    delay = 5.0
    while batch.processing_status != "ended":
        if time.monotonic() > deadline:
            try:
                client.messages.batches.cancel(batch.id)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            # "timeout" marker makes this retryable in chat_completion.
            raise TimeoutError(
                f"batch {batch.id} not finished after "
                f"{ANTHROPIC_BATCH_MAX_WAIT_S:.0f}s — canceled (timeout)")
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)
        batch = client.messages.batches.retrieve(batch.id)

    for entry in client.messages.batches.results(batch.id):
        if entry.custom_id != custom_id:
            continue
        kind = entry.result.type
        if kind == "succeeded":
            return entry.result.message
        if kind == "errored":
            # Surface the provider error text so _is_retryable can classify
            # it (overloaded/rate_limit/5xx → retry; invalid_request → raise).
            raise RuntimeError(
                f"batch request errored: {entry.result.error}")
        # canceled/expired — expired means unprocessed after 24h.
        raise RuntimeError(f"batch request {kind} (timeout)")
    raise RuntimeError(f"batch {batch.id} returned no result for {custom_id}")


def _claude_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None,
    max_tokens: int,
    json_mode: bool,
    effort: str | None = None,
    json_schema: dict | None = None,
    use_batch: bool | None = None,
) -> tuple[str, dict]:
    client = get_anthropic_client()

    full_user_prompt = user_prompt
    if json_mode:
        full_user_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Respond with valid JSON only. No markdown fences, no prose outside the JSON object."
        )

    # With adaptive thinking, thinking tokens count against max_tokens.
    # Ensure there is always enough headroom for the actual JSON output.
    effective_max_tokens = max(max_tokens * 3, 16384)

    # output_config carries both the per-call effort override and (when a
    # schema is given) the structured-output format. output_config.format is
    # grammar-enforced by the API and documented compatible with adaptive
    # thinking — we pass no tools, so the known thinking+tools+format
    # instability doesn't apply. NOTE: forced tool_choice is NOT thinking-
    # compatible; output_config.format is the supported path.
    output_config: dict = {"effort": effort or CLAUDE_EFFORT}
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}

    # Prompt-caching breakpoints: one on the system prompt (static per pass —
    # shared by every note in a batch) and one on the user turn (note + RAG
    # context — identical across the 3 consistency runs of the same note, so
    # runs that start after the first write re-read the whole prefix at 10%
    # of the input price). Cache hits inside the Batches API are best-effort
    # but the breakpoints cost nothing when they miss beyond the 25% write
    # premium on the first run.
    body: dict = {
        "model": model or CLAUDE_MODEL,
        "max_tokens": effective_max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
        "system": [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": full_user_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
        }],
    }

    if (ANTHROPIC_USE_BATCH if use_batch is None else use_batch):
        response = _claude_message_via_batch(client, body)
    else:
        # Streaming supports long-running interactive requests (>10 min)
        # required by the SDK for extended thinking on complex notes.
        with client.messages.stream(**body) as stream:
            response = stream.get_final_message()

    # A max_tokens stop means the JSON stream was cut mid-generation —
    # structurally invalid by construction. Raise so the retry loop reruns
    # the call with a doubled budget instead of handing a broken prefix to
    # the parser (whose fallback silently drops the whole pass's output).
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise _TruncatedResponse(
            f"stop_reason=max_tokens at {effective_max_tokens} tokens")

    # Extract text from response content blocks (skip thinking blocks)
    content = ""
    for block in response.content:
        if block.type == "text":
            content = block.text
            break

    # For JSON mode: extract the JSON object/array regardless of surrounding prose or fences
    if json_mode:
        content = _extract_json_from_text(content)

    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        # Cache economics: reads bill at 10% of input price, writes at 125%.
        # Surfaced so per-note cost accounting reflects the true spend.
        "cache_read_tokens": getattr(
            response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0) or 0,
    }
    return content, usage

