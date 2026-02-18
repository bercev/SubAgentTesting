from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from runtime.model_backend import ModelBackend, OpenRouterBackend


def build_backend(
    backend_config: Mapping[str, Any],
    event_logger: Optional[Callable[[str], None]] = None,
) -> ModelBackend:
    """Construct backend implementation from agent backend config."""

    backend_type = backend_config.get("type", "openrouter")
    if backend_type == "openrouter":
        model_id = backend_config.get("model")
        if not model_id:
            raise ValueError("Missing model id in agent spec backend.model")
        return OpenRouterBackend(
            model=model_id,
            base_url=backend_config.get("base_url", "https://openrouter.ai/api/v1"),
            max_retries=backend_config.get("max_retries", 8),
            initial_backoff_s=backend_config.get("initial_backoff_s", 1.0),
            max_backoff_s=backend_config.get("max_backoff_s", 10.0),
            event_logger=event_logger,
        )
    raise ValueError(f"Unsupported backend type: {backend_type}")
