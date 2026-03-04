Tool Use Protocol

- Use tools when they reduce uncertainty.
- Treat tool outputs as intermediate state. Final output in tools mode must be a submit(final_artifact) call.
- Never end with prose/explanations in tools mode. Always use submit(...).
- If you cannot produce a valid patch, call submit("CANNOT PRODUCE OUTPUT {reason}").

Tool Signatures (human-readable)

- workspace_list(path)
- workspace_open(path, start_line=1, end_line=None)
  - Prefer bounded reads and page through files with start_line/end_line.
- workspace_search(query, glob="**/*")
  - Supports only query and optional glob.
  - Do NOT pass start_line/end_line.
- workspace_apply_patch(unified_diff)
- workspace_write(path, content)
- bash(cmd, timeout_s=None)
- submit(final_artifact)

Search Usage

- Prefer recursive globs when looking for code:
  - workspace_search(query="separability_matrix", glob="**/*.py")
  - workspace_search(query="test_name", glob="**/tests/*.py")
- If a search returns nothing, change query or glob and retry.

Invalid vs Valid Examples

- Invalid: workspace_search(query="foo", start_line=1, end_line=40)
- Valid: workspace_search(query="foo", glob="**/*.py")
- Valid: workspace_open(path="pkg/file.py", start_line=120, end_line=220)

Paging Pattern for workspace_open

- Start with a targeted range near matches.
- If you need more context, request the next range.
- Avoid opening whole files when line ranges are sufficient.

Allowed Tools:
- workspace_list
- workspace_open
- workspace_search
- workspace_apply_patch
- workspace_write
- bash
- submit
