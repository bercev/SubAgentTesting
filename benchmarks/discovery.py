from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Type


def _is_adapter_candidate(value: Any) -> bool:
    """Check whether a class exposes the required benchmark adapter surface."""

    if not inspect.isclass(value):
        return False
    required_methods = (
        "from_config",
        "load_tasks",
        "workspace_root_for_task",
        "to_prediction_record",
        "get_evaluator",
    )
    if not isinstance(getattr(value, "benchmark_name", None), str):
        return False
    return all(callable(getattr(value, method, None)) for method in required_methods)


def discover_benchmark_adapters() -> Dict[str, Type[Any]]:
    """Import `benchmarks/*/adapter.py` modules and register adapter classes."""

    discovered: Dict[str, Type[Any]] = {}
    root = Path(__file__).resolve().parent
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name in {"__pycache__"}:
            continue
        adapter_file = child / "adapter.py"
        if not adapter_file.exists():
            continue

        module_name = f"benchmarks.{child.name}.adapter"
        module = importlib.import_module(module_name)
        adapter_cls = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if _is_adapter_candidate(obj):
                adapter_cls = obj
                break

        if adapter_cls is None:
            raise ValueError(
                f"Module {module_name} must define a benchmark adapter class with "
                "benchmark_name and required adapter methods"
            )

        benchmark_name = getattr(adapter_cls, "benchmark_name")
        if benchmark_name in discovered:
            raise ValueError(
                f"Duplicate benchmark_name '{benchmark_name}' discovered in adapters "
                f"{discovered[benchmark_name].__name__} and {adapter_cls.__name__}"
            )
        discovered[benchmark_name] = adapter_cls
    return discovered
