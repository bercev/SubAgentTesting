from typing import Dict

from benchmarks.swebench_verified.adapter import SWEbenchVerifiedAdapter


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, type] = {
            "swebench_verified": SWEbenchVerifiedAdapter,
        }

    def get_adapter(self, name: str):
        if name not in self._registry:
            raise KeyError(f"Unknown benchmark {name}")
        return self._registry[name]
