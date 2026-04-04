from openai import OpenAI
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
from app.core.logger import get_logger

logger = get_logger(__name__)

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    return _client


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.05,
    max_tokens: int = 4096,
    json_mode: bool = True,
) -> tuple[str, dict]:
    client = get_openai_client()
    kwargs = {
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


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    client = get_openai_client()
    model = model or OPENAI_EMBEDDING_MODEL

    batch_size = 2048
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings
