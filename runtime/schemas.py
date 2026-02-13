from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkTask:
    """One benchmark task consumed by the runtime loop."""

    task_id: str
    instruction: str
    resources: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    expected_output_type: str = "text"


@dataclass
class AgentResult:
    """Terminal agent output captured per benchmark task."""

    task_id: str
    final_artifact: str
    metadata: Optional[Dict[str, Any]] = None
