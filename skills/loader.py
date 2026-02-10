import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_skills(skill_dirs: List[Path]) -> Tuple[str, Set[str]]:
    """Load SKILL.md files and return combined text + allowed tools."""
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
    allowed: Set[str] = set()
    match = re.search(r"Allowed Tools:\s*(.+?)(\n\S|\Z)", text, flags=re.DOTALL)
    if not match:
        return allowed
    body = match.group(1)
    for line in body.splitlines():
        line = line.strip(" -*")
        if line:
            allowed.add(line)
    return allowed
