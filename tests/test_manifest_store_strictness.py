from __future__ import annotations

from pathlib import Path

import pytest

from runtime.manifest_store import write_manifest


def test_write_manifest_fails_on_non_serializable_payload(tmp_path: Path):
    payload = {"run_id": "2026-02-13_010203", "config_snapshot": {"bad": object()}}
    with pytest.raises(TypeError):
        write_manifest(tmp_path / "manifest.json", payload)
