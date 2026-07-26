import os
import json
import re
import asyncio
from openai import AsyncOpenAI

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY"
)
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)

_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def _strip_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _extract_json_object(text: str):
    """Tolerate leading/trailing prose around the JSON object/array."""
    text = _strip_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


async def call_llm_json(prompt: str, timeout: float = 15.0) -> dict:
    """
    Calls OpenRouter LLM and parses JSON output.
    Returns parsed dict or list.

    Unchanged - other questions (q9, etc.) depend on this exact signature.
    """
    try:
        response = await asyncio.wait_for(
            _client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            ),
            timeout=timeout,
        )
        text = (response.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        return json.loads(text)
    except Exception as e:
        print(f"WARNING: OpenRouter LLM call failed or timed out: {e}", flush=True)
        return {}


async def chat_json(messages, max_tokens: int = 2048, timeout: float = 35.0) -> dict:
    """
    Used by q10.py. Takes a full OpenAI-style messages list
    ([{"role": "system"/"user", "content": ...}, ...]) instead of a single
    prompt string, since q10's batch prompts need a separate system message.
    Reuses the same OpenRouter client/model as call_llm_json above.

    Returns {} on any failure so callers fall back to their own heuristics
    instead of crashing.
    """
    try:
        response = await asyncio.wait_for(
            _client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            ),
            timeout=timeout,
        )
        text = response.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as e:
        print(f"WARNING: OpenRouter chat_json call failed or timed out: {e}", flush=True)
        return {}