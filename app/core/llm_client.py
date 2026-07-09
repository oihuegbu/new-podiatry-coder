from __future__ import annotations

import json

from openai import OpenAI
from app.core.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_EFFORT,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

_openai_client: OpenAI | None = None
_anthropic_client = None  # anthropic.Anthropic, lazy-imported


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    return _openai_client


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # lazy import so openai-only installs still work
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized")
    return _anthropic_client


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.05,
    max_tokens: int = 4096,
    json_mode: bool = True,
) -> tuple[str, dict]:
    if LLM_PROVIDER == "claude":
        return _claude_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    return _openai_chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )


def _openai_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> tuple[str, dict]:
    client = get_openai_client()
    kwargs: dict = {
        "model": model or OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
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


def _claude_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None,
    max_tokens: int,
    json_mode: bool,
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

    # Use streaming to support long-running requests (>10 min) required by the SDK
    # for extended thinking on complex notes.
    content = ""
    input_tokens = 0
    output_tokens = 0
    with client.messages.stream(
        model=model or CLAUDE_MODEL,
        max_tokens=effective_max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": CLAUDE_EFFORT},
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": full_user_prompt}],
    ) as stream:
        response = stream.get_final_message()

    # Extract text from response content blocks (skip thinking blocks)
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
    }
    return content, usage


