import json
import re
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ArtifactPolicyResult:
    """Result of output normalization/validation for one artifact."""

    artifact: str
    valid: bool
    reason: str


def apply_artifact_policy(raw_artifact: str, output_type: str) -> ArtifactPolicyResult:
    """Dispatch artifact handling based on expected output type."""

    normalized_type = (output_type or "text").strip().lower()
    policy = _POLICIES.get(normalized_type, _normalize_text_artifact)
    return policy(raw_artifact or "")


def _normalize_patch_artifact(raw_artifact: str) -> ArtifactPolicyResult:
    """Extract and validate unified-diff patch output."""

    text = _normalize_newlines(raw_artifact).strip()
    if not text:
        return ArtifactPolicyResult(artifact="", valid=False, reason="empty_output")

    candidate = _extract_diff_candidate(text)
    if not candidate:
        return ArtifactPolicyResult(artifact="", valid=False, reason="no_diff_found")

    candidate = _truncate_non_patch_tail(candidate)
    candidate = candidate.strip()
    if not candidate:
        return ArtifactPolicyResult(artifact="", valid=False, reason="empty_after_sanitize")
    if not candidate.startswith("diff --git "):
        return ArtifactPolicyResult(artifact="", valid=False, reason="missing_diff_header")
    if not re.search(r"^---\s+.+$", candidate, flags=re.MULTILINE):
        return ArtifactPolicyResult(artifact="", valid=False, reason="missing_old_file_header")
    if not re.search(r"^\+\+\+\s+.+$", candidate, flags=re.MULTILINE):
        return ArtifactPolicyResult(artifact="", valid=False, reason="missing_new_file_header")
    if not re.search(r"^@@\s+.+\s+@@", candidate, flags=re.MULTILINE):
        return ArtifactPolicyResult(artifact="", valid=False, reason="missing_hunk_header")
    if not candidate.endswith("\n"):
        candidate += "\n"
    return ArtifactPolicyResult(artifact=candidate, valid=True, reason="ok")


def _normalize_text_artifact(raw_artifact: str) -> ArtifactPolicyResult:
    """Normalize plain-text artifacts with newline and edge whitespace cleanup."""

    normalized = _normalize_newlines(raw_artifact).strip()
    return ArtifactPolicyResult(artifact=normalized, valid=True, reason="ok")


def _normalize_json_artifact(raw_artifact: str) -> ArtifactPolicyResult:
    """Validate and canonicalize JSON output into stable serialized form."""

    text = _normalize_newlines(raw_artifact).strip()
    if not text:
        return ArtifactPolicyResult(artifact="", valid=False, reason="empty_output")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ArtifactPolicyResult(artifact=text, valid=False, reason="invalid_json")
    normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return ArtifactPolicyResult(artifact=normalized, valid=True, reason="ok")


def _extract_diff_candidate(text: str) -> str:
    """Find the first diff payload candidate in raw model output."""

    if text.startswith("diff --git "):
        return text

    for fenced_match in re.finditer(r"```(?:diff|patch)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        body = fenced_match.group(1).strip()
        diff_index = body.find("diff --git ")
        if diff_index >= 0:
            return body[diff_index:]

    diff_index = text.find("diff --git ")
    if diff_index >= 0:
        return text[diff_index:]
    return ""


def _truncate_non_patch_tail(candidate: str) -> str:
    """Stop patch output at first clear non-patch tail after hunk body starts."""

    allowed_prefixes = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
        "@@",
        "new file mode ",
        "deleted file mode ",
        "old mode ",
        "new mode ",
        "rename from ",
        "rename to ",
        "similarity index ",
        "dissimilarity index ",
        "Binary files ",
        "\\ No newline at end of file",
    )
    lines = candidate.splitlines()
    kept = []
    saw_diff = False
    saw_patch_body = False

    for line in lines:
        if line.startswith("diff --git "):
            saw_diff = True
            kept.append(line)
            continue
        if not saw_diff:
            continue
        if not line:
            kept.append(line)
            continue
        if line.startswith(("+", "-", " ")):
            kept.append(line)
            saw_patch_body = True
            continue
        if line.startswith(allowed_prefixes):
            kept.append(line)
            continue
        if saw_patch_body:
            break
        kept.append(line)

    return "\n".join(kept)


def _normalize_newlines(text: str) -> str:
    """Normalize CRLF/CR newlines to LF for deterministic downstream parsing."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


_POLICIES: Dict[str, Callable[[str], ArtifactPolicyResult]] = {
    "patch": _normalize_patch_artifact,
    "text": _normalize_text_artifact,
    "json": _normalize_json_artifact,
}
