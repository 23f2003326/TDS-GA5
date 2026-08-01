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


class LLMUnavailable(Exception):
    """Raised/importable by callers (e.g. q9.py) that need to distinguish
    "no provider configured" from other failures. llm.py itself never
    raises this - call_llm_json/chat_json always return {} on failure so
    they never crash a caller. It exists purely so `from llm import
    LLMUnavailable` succeeds."""
    pass


# IMPORTANT: build the client lazily / defensively. openai>=1.0's client
# constructor raises immediately if no api_key is resolvable (neither the
# explicit arg nor OPENAI_API_KEY env var). Since llm.py is imported at
# module load time by q10.py and q11.py, and those are imported at module
# load time by main.py, an eager crash here takes down the ENTIRE monolith
# (every question, not just the ones using the LLM) - which is what "all
# categories 0/7" almost always means. Never let a missing key crash import.
_client = None
if OPENROUTER_API_KEY:
    try:
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    except Exception as e:
        print(f"WARNING: could not construct OpenRouter client: {e}", flush=True)
        _client = None
else:
    print("WARNING: OPENROUTER_API_KEY not set - LLM calls will fall back "
          "to heuristics everywhere.", flush=True)


def available() -> bool:
    """True iff a usable LLM client is configured. Callers (e.g. q9.py's
    run_model) check this before doing any batching/chunking work, so it
    must never raise and must reflect the real client state."""
    return _client is not None


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
    Returns parsed dict or list. Returns {} (never raises) if no client is
    configured or the call/parse fails, so callers can rely on their own
    heuristic fallback.
    """
    if _client is None:
        return {}
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
        return _extract_json_object(text)
    except Exception as e:
        print(f"WARNING: OpenRouter LLM call failed or timed out: {e}", flush=True)
        return {}


async def chat_json(messages, max_tokens: int = 2048, timeout: float = 35.0) -> dict:
    """
    Used by q9.py / q10.py. Takes a full OpenAI-style messages list
    ([{"role": "system"/"user", "content": ...}, ...]) instead of a single
    prompt string, since batch prompts need a separate system message.
    Reuses the same OpenRouter client/model as call_llm_json above.

    Returns {} on any failure (including no client configured) so callers
    fall back to their own heuristics instead of crashing.
    """
    if _client is None:
        return {}
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