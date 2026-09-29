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

### Known issues
- `src/scheduler.py` imports `apscheduler`, which is not a dependency.
- `src/test_memory.py` and `src/test_natural_language_parser.py` are scripts, not pytest
  tests.
- `src/tool_parser.py` holds two modules pasted together.

### Next Steps
1. Proceed with Slack integration
2. Continue with Persistent Memory updates
