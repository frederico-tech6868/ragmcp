## Project Progress

### Done: Tests for edge cases in curation logic
- Cleaned up `src/curation.py` (966 -> ~250 lines): duplicated constants, functions and
  copied `curate_skills` bodies removed. `src/memory.py` duplicates removed too.
- Data paths are relative to the project root, so scripts and tests work from any
  directory.
- `curate_skills` now appends every run (or an `error` entry for a failed run) to the
  curation history in `data/persistent_memory/curation_history.json`.
- `curate_skills()` with no argument curates everything in `skills/` (used by
  `start_curation_loop`).
- `src/test_curation.py` rewritten: 9 tests, all passing, run in a temporary directory
  (they no longer write to `data/`). Run with `.venv/bin/python -m pytest -q src`.

### Done: Natural language parser tests and fixes
- `src/test_natural_language_parser.py` is now 6 pytest tests (no writes to `data/`).
- `parse_natural_language_task` splits on "and"/"then" before matching, allows digits
  ("font size to 14"), handles "enable X", and no longer duplicates "then" steps.
  Conditionals ("if ..., then ...") are saved as unrecognized tasks.

### Done: Scheduler without apscheduler
- `src/scheduler.py` uses only the standard library: `start_scheduler(interval, run)` runs
  the job in a daemon thread and returns an object whose `stop()` takes effect at once.
  A failing job is logged and the loop keeps going. `src/test_scheduler.py`: 3 tests.

### Done: tool_parser split
- `src/tool_parser.py` has only `parse_tool_call`; the dynamic tool registry
  (`DYNAMIC_TOOLS`, `verify_code_safety_isolated`, `secure_register_tool`) is in
  `src/tool_registry.py`. Fixed `secure_register_tool` (`local_scope = {{}}` raised
  TypeError). Tests: `src/test_tool_parser.py`, `src/test_tool_registry.py`.
- Note: `verify_code_safety_isolated` only checks that the code compiles and runs in a
  subprocess; `secure_register_tool` then `exec`s it in-process. It is not a sandbox.

### Known issues
- None recorded.

### Next Steps
1. Proceed with Slack integration
2. Continue with Persistent Memory updates
