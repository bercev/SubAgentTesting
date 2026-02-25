Repo Triage

- Identify key files, tests, and failure modes before editing.
- Prefer reading over writing in the first loop.
- First-pass workflow:
  1. workspace_list(".")
  2. workspace_search(...) with a recursive code glob such as "**/*.py"
  3. workspace_open(...) with bounded line ranges around matches
  4. Inspect nearby definitions/callers before editing
  5. Only then patch/apply changes
- If the first search is too broad, narrow the query or glob and retry.
- Use test globs when relevant (for example "**/tests/*.py").

Allowed Tools:
- workspace_list
- workspace_open
- workspace_search
- bash
