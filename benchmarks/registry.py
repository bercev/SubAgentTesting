from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from benchmarks.discovery import discover_benchmark_adapters


class BenchmarkRegistry:
    """Lookup table for benchmark adapters discovered at runtime."""

    # Reserved for hard-coded adapter substitutions in exceptional cases.
    DEFAULT_OVERRIDES: Mapping[str, type[Any]] = {}

    def __init__(self, overrides: Optional[Mapping[str, type[Any]]] = None) -> None:
        """Load discovered adapters and apply optional override mapping."""

        self._registry: Dict[str, type[Any]] = discover_benchmark_adapters()
        self._registry.update(dict(self.DEFAULT_OVERRIDES))
        if overrides:
            self._registry.update(dict(overrides))

    def get_adapter(self, name: str):
        """Return adapter class for a benchmark name or raise a deterministic error."""

        if name not in self._registry:
            supported = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Unknown benchmark '{name}'. Supported benchmarks: {supported}")
        return self._registry[name]

    def list_benchmarks(self) -> list[str]:
        """List all available benchmark names in stable order."""

        return sorted(self._registry.keys())
