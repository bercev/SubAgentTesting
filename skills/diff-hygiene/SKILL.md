Diff Hygiene

- Keep patches minimal and localized.
- Avoid whitespace-only edits unless needed.
- Provide context lines for patches.
- Pre-submit checklist:
  - Starts with `diff --git`
  - Includes `---` and `+++` file headers
  - Includes at least one `@@` hunk header
  - Contains no explanation text before/after the diff
  - Ends with a trailing newline
- If the patch is malformed or incomplete, fix it before submit(...).
- If you cannot produce a valid patch, submit("").

Allowed Tools:
- workspace_apply_patch
- workspace_write
- workspace_open
- workspace_search
- submit
