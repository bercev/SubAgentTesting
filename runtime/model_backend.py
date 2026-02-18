import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx


@dataclass
class ToolCall:
    """One structured tool invocation emitted by the model backend."""

    name: str
    arguments: Dict[str, Any]


@dataclass
class GenerationResult:
    """Backend generation output consumed by the runtime loop."""

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
    """OpenAI-compatible backend targeting OpenRouter endpoints with retries."""

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
        max_retries: int = 8,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 10.0,
        event_logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize backend with strict model requirements and retry policy."""

        api_key_value = api_key or os.getenv("OPENROUTER_API_KEY")
        model_value = model or os.getenv("OPENROUTER_MODEL")
        if not api_key_value:
            raise ValueError("OPENROUTER_API_KEY is required")
        if not model_value:
            raise ValueError("backend.model or OPENROUTER_MODEL is required")

        self.api_key = api_key_value
        self.model = model_value
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff_s = max(0.0, float(initial_backoff_s))
        self.max_backoff_s = max(0.0, float(max_backoff_s))
        self.event_logger = event_logger

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        decoding: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Call OpenRouter chat completions and return structured tool calls only."""

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if decoding:
            payload.update({k: v for k, v in decoding.items() if v is not None})

        endpoint = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            data: Optional[Dict[str, Any]] = None
            for attempt in range(self.max_retries + 1):
                attempt_no = attempt + 1
                self._emit_log(
                    "api_request"
                    f" provider=openrouter"
                    f" model={self.model}"
                    f" attempt={attempt_no}/{self.max_retries + 1}"
                    f" method=POST"
                    f" url={endpoint}"
                    f" payload_bytes={self._json_size_bytes(payload)}"
                    f" payload_preview={self._preview_json(payload)}"
                )
                started = time.monotonic()
                try:
                    response = client.post(
                        endpoint,
                        json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                except httpx.RequestError as exc:
                    latency_ms = int(max(0.0, (time.monotonic() - started) * 1000.0))
                    self._emit_log(
                        "api_error"
                        f" provider=openrouter"
                        f" model={self.model}"
                        f" attempt={attempt_no}/{self.max_retries + 1}"
                        f" kind=request_error"
                        f" latency_ms={latency_ms}"
                        f" detail={self._preview_text(str(exc), limit=1200)}"
                    )
                    if attempt >= self.max_retries:
                        raise
                    wait_s = self._sleep_before_retry(attempt)
                    self._emit_log(
                        "api_retry"
                        f" provider=openrouter"
                        f" model={self.model}"
                        f" attempt={attempt_no}/{self.max_retries + 1}"
                        f" wait_s={wait_s:.2f}"
                        " reason=request_error"
                    )
                    continue

                latency_ms = int(max(0.0, (time.monotonic() - started) * 1000.0))
                response_text_obj = getattr(response, "text", "")
                response_text = response_text_obj if isinstance(response_text_obj, str) else str(response_text_obj)
                self._emit_log(
                    "api_response"
                    f" provider=openrouter"
                    f" model={self.model}"
                    f" attempt={attempt_no}/{self.max_retries + 1}"
                    f" status_code={response.status_code}"
                    f" latency_ms={latency_ms}"
                    f" body_bytes={len(response_text.encode('utf-8', errors='ignore'))}"
                    f" body_preview={self._preview_text(response_text, limit=1200)}"
                )
                if response.status_code >= 400:
                    detail = response_text[:2000]
                    retryable = self._is_retryable_status(response.status_code, detail)
                    if retryable and attempt < self.max_retries:
                        wait_s = self._sleep_before_retry(attempt)
                        self._emit_log(
                            "api_retry"
                            f" provider=openrouter"
                            f" model={self.model}"
                            f" attempt={attempt_no}/{self.max_retries + 1}"
                            f" wait_s={wait_s:.2f}"
                            f" reason=http_{response.status_code}"
                        )
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
                    self._emit_log(
                        "api_error"
                        f" provider=openrouter"
                        f" model={self.model}"
                        f" attempt={attempt_no}/{self.max_retries + 1}"
                        " kind=json_decode_error"
                    )
                    if attempt >= self.max_retries:
                        raise
                    wait_s = self._sleep_before_retry(attempt)
                    self._emit_log(
                        "api_retry"
                        f" provider=openrouter"
                        f" model={self.model}"
                        f" attempt={attempt_no}/{self.max_retries + 1}"
                        f" wait_s={wait_s:.2f}"
                        " reason=json_decode_error"
                    )
                    continue
                if isinstance(parsed, dict):
                    data = parsed
                    self._emit_log(
                        "api_parsed"
                        f" provider=openrouter"
                        f" model={self.model}"
                        f" attempt={attempt_no}/{self.max_retries + 1}"
                        " parsed_type=dict"
                    )
                    break
                if attempt >= self.max_retries:
                    raise ValueError("OpenRouter returned non-dict JSON response")
                self._emit_log(
                    "api_error"
                    f" provider=openrouter"
                    f" model={self.model}"
                    f" attempt={attempt_no}/{self.max_retries + 1}"
                    " kind=non_dict_response"
                )
                wait_s = self._sleep_before_retry(attempt)
                self._emit_log(
                    "api_retry"
                    f" provider=openrouter"
                    f" model={self.model}"
                    f" attempt={attempt_no}/{self.max_retries + 1}"
                    f" wait_s={wait_s:.2f}"
                    " reason=non_dict_response"
                )

            if data is None:
                raise RuntimeError("OpenRouter request exhausted retries without a valid response")

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        assistant_text = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls: List[ToolCall] = []
        for tc in raw_tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = func.get("name")
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    self._emit_log(
                        "api_tool_args_parse_error"
                        f" provider=openrouter"
                        f" model={self.model}"
                        f" tool_name={name or 'unknown'}"
                        f" raw_preview={self._preview_text(args, limit=800)}"
                    )
                    args = {"raw": args}
            if name:
                tool_calls.append(ToolCall(name=name, arguments=args or {}))

        self._emit_log(
            "api_result"
            f" provider=openrouter"
            f" model={self.model}"
            f" assistant_chars={len(assistant_text)}"
            f" tool_calls={len(tool_calls)}"
        )
        return GenerationResult(assistant_text=assistant_text, tool_calls=tool_calls)

    @classmethod
    def _is_retryable_status(cls, status_code: int, detail: str) -> bool:
        """Mark transport/provider failures as retryable without masking fatal 4xx errors."""

        if status_code in {408, 409, 425, 429} or status_code >= 500:
            return True
        if status_code != 400:
            return False
        lowered = detail.lower()
        return any(marker in lowered for marker in cls._RETRYABLE_400_MARKERS)

    @staticmethod
    def _json_size_bytes(payload: Any) -> int:
        """Approximate payload size as UTF-8 JSON bytes for logging."""

        try:
            serialized = json.dumps(payload, default=str, ensure_ascii=False)
        except Exception:
            serialized = str(payload)
        return len(serialized.encode("utf-8", errors="ignore"))

    @staticmethod
    def _preview_text(text: str, limit: int = 2000) -> str:
        """Compact and truncate free-text fields before logging."""

        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "...[truncated]"

    def _preview_json(self, payload: Dict[str, Any], limit: int = 2000) -> str:
        """Serialize payloads safely for run-log diagnostics."""

        try:
            serialized = json.dumps(payload, ensure_ascii=False)
        except Exception:
            serialized = str(payload)
        return self._preview_text(serialized, limit=limit)

    def _emit_log(self, message: str) -> None:
        """Best-effort sink for backend diagnostics that must not break runs."""

        if not self.event_logger:
            return
        try:
            self.event_logger(message)
        except Exception:
            return

    def _sleep_before_retry(self, attempt: int) -> float:
        """Sleep using bounded exponential backoff with jitter and return wait seconds."""

        base = self.initial_backoff_s * (2**attempt)
        wait_s = min(self.max_backoff_s, base)
        if wait_s <= 0:
            return 0.0
        jitter_max = min(1.0, wait_s * 0.25)
        wait_s += random.uniform(0.0, jitter_max)
        time.sleep(wait_s)
        return wait_s


class NoToolBackend(ModelBackend):
    """Backend stub that never emits tool calls (for patch-only mode)."""

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        decoding: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Return last user content directly to keep patch-only smoke flows simple."""

        # Echo back last user message as a noop response.
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        content = last_user.get("content", "")
        return GenerationResult(assistant_text=content, tool_calls=[])
