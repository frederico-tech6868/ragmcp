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

### Known issues
- `src/tool_parser.py` holds two modules pasted together.

### Next Steps
1. Proceed with Slack integration
2. Continue with Persistent Memory updates
