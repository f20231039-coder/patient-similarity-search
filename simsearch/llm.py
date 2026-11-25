"""Thin LLM client, used in exactly one place: turning a plain-language search
into structured filters (see query_parser.py). Groq, Gemini, Ollama (Qwen) and
OpenAI all speak the OpenAI chat-completions protocol, so one HTTP client covers
them. No other part of the product calls an LLM."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import httpx

from . import config

log = logging.getLogger(__name__)

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash-lite"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
}


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM:
    def __init__(self, provider: str, model: str = "", api_key: str = "", base_url: str = "",
                 timeout: float = 30.0, retries: int = 1):
        self.retries = retries  # the app keeps this low (fast fallback to rules); evals raise it
        default_url, default_model = PROVIDERS[provider]
        self.provider = provider
        self.model = model or default_model
        self.name = f"{provider}:{self.model}"
        self.base_url = (base_url or default_url).rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(timeout=timeout, headers=headers)

    def complete_json(self, system: str, user: str) -> dict:
        """One deterministic call that must return a JSON object."""
        body: dict[str, Any] = {"model": self.model, "temperature": 0,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                "response_format": {"type": "json_object"}}
        r = None
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=body)
            except httpx.TimeoutException as e:
                if attempt < self.retries:
                    continue
                raise LLMError(f"{self.name} timed out") from e
            except httpx.HTTPError as e:
                raise LLMError(f"{self.name} unreachable: {e}") from e
            if r.status_code == 400 and "response_format" in body:  # provider without JSON mode
                body.pop("response_format")
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < self.retries:
                # rate limit / overload: honour Retry-After, else back off 5s, 15s, 30s
                wait = r.headers.get("retry-after")
                time.sleep(float(wait) if wait and wait.replace(".", "").isdigit() else (5, 15, 30)[min(attempt, 2)])
                continue
            break
        if r is None or r.status_code >= 400:
            raise LLMError(f"{self.name} returned HTTP {r.status_code if r is not None else '?'}")
        text = r.json()["choices"][0]["message"].get("content") or ""
        return parse_json_object(text)


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a reply (tolerates ```json fences)."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise LLMError("reply contained no JSON object")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("JSON reply is not an object")
    return obj


def get_llm(provider: str | None = None) -> Optional[OpenAICompatLLM]:
    """The configured LLM, or None (then the rule-based parser is used)."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider in ("none", "", "offline", "rules"):
        return None
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider}")
    if provider != "ollama" and not config.LLM_API_KEY:
        log.warning("LLM_PROVIDER=%s but LLM_API_KEY is empty; using the rule-based parser", provider)
        return None
    return OpenAICompatLLM(provider, config.LLM_MODEL, config.LLM_API_KEY, config.LLM_BASE_URL,
                           timeout=min(config.REQUEST_TIMEOUT_S, 30.0))
