import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class GenerationResult:
    assistant_text: str
    tool_calls: List[ToolCall]


class ModelBackend:
    """Interface for model backends."""

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        decoding: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        raise NotImplementedError


class OpenRouterBackend(ModelBackend):
    """OpenAI-compatible backend targeting OpenRouter endpoints."""

    _RETRYABLE_400_MARKERS = (
        "developer instruction is not enabled",
        "provider returned error",
        "no providers available",
        "temporarily unavailable",
        "upstream error",
        "try again",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        tool_call_extractor=None,
        max_retries: int = 8,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "qwen2.5-coder-0.5b-instruct")
        self.base_url = base_url.rstrip("/")
        self.tool_call_extractor = tool_call_extractor or self._extract_tool_calls_from_text
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff_s = max(0.0, float(initial_backoff_s))
        self.max_backoff_s = max(0.0, float(max_backoff_s))
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        # OpenRouter expects provider-prefixed model ids (e.g., qwen/qwen2.5-coder-0.5b-instruct).
        if "/" not in self.model and self.model.startswith("qwen"):
            self.model = f"qwen/{self.model}"

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        decoding: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if decoding:
            payload.update({k: v for k, v in decoding.items() if v is not None})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            data: Optional[Dict[str, Any]] = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                except httpx.RequestError:
                    if attempt >= self.max_retries:
                        raise
                    self._sleep_before_retry(attempt)
                    continue

                if response.status_code >= 400:
                    detail = response.text[:2000]
                    retryable = self._is_retryable_status(response.status_code, detail)
                    if retryable and attempt < self.max_retries:
                        self._sleep_before_retry(attempt)
                        continue

                    if retryable:
                        message = (
                            f"OpenRouter error after {attempt + 1} attempts "
                            f"({response.status_code}): {detail}"
                        )
                    else:
                        message = f"OpenRouter error {response.status_code}: {detail}"
                    raise httpx.HTTPStatusError(
                        message,
                        request=response.request,
                        response=response,
                    )

                try:
                    parsed = response.json()
                except json.JSONDecodeError:
                    if attempt >= self.max_retries:
                        raise
                    self._sleep_before_retry(attempt)
                    continue
                if isinstance(parsed, dict):
                    data = parsed
                    break
                if attempt >= self.max_retries:
                    raise ValueError("OpenRouter returned non-dict JSON response")
                self._sleep_before_retry(attempt)

            if data is None:
                raise RuntimeError("OpenRouter request exhausted retries without a valid response")

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        assistant_text = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls: List[ToolCall] = []
        if raw_tool_calls:
            for tc in raw_tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = func.get("name")
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                if name:
                    tool_calls.append(ToolCall(name=name, arguments=args or {}))
        elif tools:
            # Fallback: extract from assistant text
            tool_calls = self.tool_call_extractor(assistant_text)

        return GenerationResult(assistant_text=assistant_text, tool_calls=tool_calls)

    @classmethod
    def _is_retryable_status(cls, status_code: int, detail: str) -> bool:
        if status_code in {408, 409, 425, 429} or status_code >= 500:
            return True
        if status_code != 400:
            return False
        lowered = detail.lower()
        return any(marker in lowered for marker in cls._RETRYABLE_400_MARKERS)

    def _sleep_before_retry(self, attempt: int) -> None:
        base = self.initial_backoff_s * (2**attempt)
        wait_s = min(self.max_backoff_s, base)
        if wait_s <= 0:
            return
        jitter_max = min(1.0, wait_s * 0.25)
        wait_s += random.uniform(0.0, jitter_max)
        time.sleep(wait_s)

    @staticmethod
    def _extract_tool_calls_from_text(text: str) -> List[ToolCall]:
        """Heuristic extraction for Qwen-style inline tool calls."""
        tool_calls: List[ToolCall] = []

        # Pattern 1: fenced JSON blocks
        for match in re.finditer(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE):
            try:
                payload = json.loads(match.group(1))
                if isinstance(payload, dict):
                    name = payload.get("name") or payload.get("function", {}).get("name")
                    args = payload.get("arguments") or payload.get("function", {}).get("arguments")
                    if isinstance(args, str):
                        args = json.loads(args)
                    if name:
                        tool_calls.append(ToolCall(name=name, arguments=args or {}))
            except json.JSONDecodeError:
                continue

        # Pattern 2: <tool_call name="..."> ... </tool_call>
        for match in re.finditer(r"<tool_call\s+name=\"([^\"]+)\">([\s\S]*?)</tool_call>", text):
            name = match.group(1)
            body = match.group(2).strip()
            try:
                args = json.loads(body)
            except json.JSONDecodeError:
                args = {"raw": body}
            tool_calls.append(ToolCall(name=name, arguments=args))

        return tool_calls


class NoToolBackend(ModelBackend):
    """Backend stub that never emits tool calls (for patch-only mode)."""

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        decoding: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        # Echo back last user message as a noop response.
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        content = last_user.get("content", "")
        return GenerationResult(assistant_text=content, tool_calls=[])
