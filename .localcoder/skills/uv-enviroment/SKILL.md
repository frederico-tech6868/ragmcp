# Skill: Python UV Environment Manager

## Purpose
Optimize the development workflow by enforcing the use of the high-performance `uv` package manager. This skill eliminates traditional, slower tools (`pip`, `venv`, `pipx`, `poetry`) and ensures consistent, modern, reproducible Python project setups.

## Core Capabilities
1. Project Initialization & Structure (`uv init`)
2. Dependency & Dev-Dependency Management (`uv add`, `uv remove`)
3. Environment Synchronization (`uv sync`)
4. Ephemeral Script & Tool Execution (`uv run`, `uvx`)

## Execution Guidelines & Command Mapping

Always prefer `uv` native project commands. Do NOT use standard `pip install` or manual `source .venv/bin/activate` unless explicitly requested.

### 1. Project Initialization
When starting a new Python project or application, initialize it natively to generate a structured `pyproject.toml` and a locked Python version.
* **Command:** `uv init <project-name>` (or `uv init .` inside an existing directory)
* **Specific Python Version:** `uv init --python 3.14`

### 2. Dependency Management
Never use `pip install` for project-level tracking. Use `uv add` to automatically modify `pyproject.toml` and generate a `uv.lock` file.
* **Production Dependencies:** `uv add package_name`
* **Development Dependencies:** `uv add --dev pytest ruff mypy`
* **Removing Packages:** `uv remove package_name`

### 3. Running Scripts and Applications
Do not require explicit activation of virtual environments. Rely on `uv run` to auto-encapsulate dependencies.
* **Run a script:** `uv run script.py`
* **Run an application framework:** `uv run python -m flask run`

### 4. Ephemeral Tools (Replacing pipx)
To execute standalone CLI tools without cluttering local project dependencies, use `uvx`.
* **Linting / Formatting:** `uvx ruff check .`
* **Ad-hoc tools:** `uvx cowsay "Hello"`

### 5. Legacy Migration
If an existing project contains a legacy setup, migrate it immediately:
* **From requirements.txt:** Run `uv init .` then `uv add -r requirements.txt` to lock everything into the modern format.

## Troubleshooting Rules
* If a virtual environment vanishes or gets corrupted, execute `uv sync` to force-rebuild the environment from the lockfile.
* If a script requires a specific Python version not present on the host system, `uv` will download it seamlessly—do not attempt manual system python installs.
