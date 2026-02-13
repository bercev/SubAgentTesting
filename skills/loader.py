from pathlib import Path
from typing import List, Set, Tuple


def load_skills(skill_dirs: List[Path]) -> Tuple[str, Set[str]]:
    """Load skill markdown files and aggregate allowed-tool declarations."""

    combined_sections: List[str] = []
    allowed: Set[str] = set()
    for skill_dir in sorted(skill_dirs, key=lambda p: p.name):
        skill_path = skill_dir / "SKILL.md"
        if not skill_path.exists():
            continue
        text = skill_path.read_text(encoding="utf-8")
        combined_sections.append(f"[Skill: {skill_dir.name}]\n{text}")
        block = _extract_allowed_tools(text)
        allowed.update(block)
    return "\n\n".join(combined_sections), allowed


def _extract_allowed_tools(text: str) -> Set[str]:
    """Parse `Allowed Tools:` block and return normalized tool name set."""

    allowed: Set[str] = set()
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "allowed tools:":
            start_idx = idx + 1
            break
    if start_idx is None:
        return allowed

    # Consume contiguous bullet lines until next non-bullet/non-empty section line.
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            tool = stripped.lstrip("-").strip()
            if tool:
                allowed.add(tool)
            continue
        break
    return allowed
