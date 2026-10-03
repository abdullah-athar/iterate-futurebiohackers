"""Minimal, dependency-free LLM clients (Anthropic, Gemini, OpenAI-compatible) + a mock.

Select with a spec string: "anthropic[:model]", "gemini[:model]", "openai[:model]", "mock".
Credentials come from ANTHROPIC_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), OPENAI_API_KEY;
OPENAI_BASE_URL overrides the OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, ...).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-5",
}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM:
    name = "llm"

    def complete(self, system: str, user: str, max_tokens: int = 6000) -> LLMResponse:
        raise NotImplementedError


def _post(url: str, headers: dict[str, str], body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {e.code}: {e.read().decode()[:500]}") from None


def _env(*names: str) -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise RuntimeError(f"Set one of {names} to use this LLM backend.")


class AnthropicLLM(LLM):
    def __init__(self, model: str) -> None:
        self.model, self.name = model, f"anthropic:{model}"

    def complete(self, system, user, max_tokens=6000):
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": _env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
                     {"model": self.model, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": user}]})
        text = "".join(b.get("text", "") for b in data.get("content", []))
        u = data.get("usage", {})
        return LLMResponse(text, u.get("input_tokens", 0), u.get("output_tokens", 0))


class GeminiLLM(LLM):
    def __init__(self, model: str) -> None:
        self.model, self.name = model, f"gemini:{model}"

    def complete(self, system, user, max_tokens=6000):
        key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
        data = _post(url, {}, {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        })
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        u = data.get("usageMetadata", {})
        return LLMResponse(text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0))


class OpenAICompatLLM(LLM):
    def __init__(self, model: str) -> None:
        self.model, self.name = model, f"openai:{model}"
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def complete(self, system, user, max_tokens=6000):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        body["max_completion_tokens" if "api.openai.com" in self.base else "max_tokens"] = max_tokens
        data = _post(f"{self.base}/chat/completions",
                     {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"}, body)
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage", {})
        return LLMResponse(text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class MockLLM(LLM):
    """Deterministic scripted proposer for tests and dry runs (no network)."""

    name = "mock"

    def __init__(self, scripts: list[tuple[str, str]] | None = None) -> None:
        from .seeds import mock_proposals
        self.scripts = scripts or mock_proposals.PROPOSALS
        self.i = 0

    def complete(self, system, user, max_tokens=6000):
        hypothesis, source = self.scripts[self.i % len(self.scripts)]
        self.i += 1
        text = f"HYPOTHESIS: {hypothesis}\n```python\n{source}\n```"
        return LLMResponse(text, prompt_tokens=len(user) // 4, completion_tokens=len(text) // 4)


def make_llm(spec: str) -> LLM:
    backend, _, model = spec.partition(":")
    backend = backend.lower()
    model = model or os.environ.get("AUTORESEARCH_MODEL") or DEFAULT_MODELS.get(backend, "")
    if backend == "mock":
        return MockLLM()
    if backend == "anthropic":
        return AnthropicLLM(model)
    if backend == "gemini":
        return GeminiLLM(model)
    if backend in ("openai", "openrouter", "vllm", "ollama"):
        return OpenAICompatLLM(model)
    raise ValueError(f"Unknown LLM backend '{backend}'. Use anthropic|gemini|openai|mock[:model].")
