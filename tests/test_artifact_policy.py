from runtime.artifact_policy import apply_artifact_policy


def test_patch_policy_accepts_valid_unified_diff():
    patch = (
        "diff --git a/example.py b/example.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    result = apply_artifact_policy(patch, "patch")
    assert result.valid is True
    assert result.reason == "ok"
    assert result.artifact == patch


def test_patch_policy_extracts_diff_from_chatter():
    raw = (
        "I will investigate now.\n\n"
        "```diff\n"
        "diff --git a/example.py b/example.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
        "More explanation.\n"
    )
    result = apply_artifact_policy(raw, "patch")
    assert result.valid is True
    assert result.reason == "ok"
    assert result.artifact.startswith("diff --git a/example.py b/example.py\n")
    assert "More explanation." not in result.artifact


def test_patch_policy_rejects_non_diff_text():
    raw = "Let me inspect files first with workspace_list."
    result = apply_artifact_policy(raw, "patch")
    assert result.valid is False
    assert result.artifact == ""
    assert result.reason == "no_diff_found"


def test_patch_policy_rejects_truncated_patch():
    raw = (
        "diff --git a/example.py b/example.py\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
    )
    result = apply_artifact_policy(raw, "patch")
    assert result.valid is False
    assert result.artifact == ""
    assert result.reason == "missing_hunk_header"
