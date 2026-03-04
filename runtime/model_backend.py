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
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


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
        full_log_previews: bool = False,
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
        self.full_log_previews = bool(full_log_previews)

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
                    f" payload_preview={self._preview_json(payload, limit=self._preview_limit(2000))}"
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
                    f" body_preview={self._preview_text(response_text, limit=self._preview_limit(1200))}"
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
                    self._emit_usage_log(data, attempt_no=attempt_no, total_attempts=self.max_retries + 1)
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
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None

        usage = data.get("usage")
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        if isinstance(usage, dict):
            prompt_raw = usage.get("prompt_tokens")
            completion_raw = usage.get("completion_tokens")
            total_raw = usage.get("total_tokens")
            if isinstance(prompt_raw, bool):
                prompt_raw = None
            if isinstance(completion_raw, bool):
                completion_raw = None
            if isinstance(total_raw, bool):
                total_raw = None
            if isinstance(prompt_raw, (int, float)):
                prompt_tokens = int(prompt_raw)
            if isinstance(completion_raw, (int, float)):
                completion_tokens = int(completion_raw)
            if isinstance(total_raw, (int, float)):
                total_tokens = int(total_raw)

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
                        f" raw_preview={self._preview_text(args, limit=self._preview_limit(800))}"
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
        return GenerationResult(
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

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
    def _preview_text(text: str, limit: Optional[int] = 2000) -> str:
        """Compact and truncate free-text fields before logging."""

        compact = " ".join((text or "").split())
        if limit is None or len(compact) <= limit:
            return compact
        return compact[:limit] + "...[truncated]"

    def _preview_json(self, payload: Dict[str, Any], limit: Optional[int] = 2000) -> str:
        """Serialize payloads safely for run-log diagnostics."""

        try:
            serialized = json.dumps(payload, ensure_ascii=False)
        except Exception:
            serialized = str(payload)
        return self._preview_text(serialized, limit=limit)

    def _preview_limit(self, default_limit: int) -> Optional[int]:
        """Return preview truncation limit, or None when full previews are enabled."""

        return None if self.full_log_previews else default_limit

    def _emit_log(self, message: str) -> None:
        """Best-effort sink for backend diagnostics that must not break runs."""

        if not self.event_logger:
            return
        try:
            self.event_logger(message)
        except Exception:
            return

    def _emit_usage_log(self, data: Dict[str, Any], *, attempt_no: int, total_attempts: int) -> None:
        """Emit compact usage/cost diagnostics for OpenRouter responses when available."""

        usage = data.get("usage")
        if not isinstance(usage, dict):
            return

        fields = [
            "api_usage",
            "provider=openrouter",
            f"model={self.model}",
            f"attempt={attempt_no}/{total_attempts}",
        ]

        response_id = data.get("id")
        if isinstance(response_id, str) and response_id:
            fields.append(f"response_id={response_id}")

        upstream_provider = data.get("provider")
        if isinstance(upstream_provider, str) and upstream_provider:
            fields.append(f"upstream_provider={upstream_provider}")

        upstream_model = data.get("model")
        if isinstance(upstream_model, str) and upstream_model:
            fields.append(f"upstream_model={upstream_model}")

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                fields.append(f"{key}={int(value)}")

        cost_value = usage.get("cost")
        if isinstance(cost_value, bool):
            cost_value = None
        if isinstance(cost_value, (int, float)):
            fields.append(f"cost_usd={float(cost_value):.12g}")

        is_byok = usage.get("is_byok")
        if isinstance(is_byok, bool):
            fields.append(f"is_byok={str(is_byok)}")

        self._emit_log(" ".join(fields))

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
