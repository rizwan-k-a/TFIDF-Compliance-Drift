# Copilot instructions for this repository

Purpose
- Short, concrete guidance for AI coding agents working on this repo.

What I found
- Python project for TF-IDF based Compliance Drift Monitoring System.
- Folder structure: `backend/` (UI-agnostic logic), `frontend/` (Streamlit UI entrypoint at `frontend/app.py`), `data/` (regulatory + internal documents), `src/` (legacy/educational modules), `tests/`.

Actionable next steps for humans (populate these so agents can work automatically)
- Replace placeholder text files in `data/` with real regulatory and internal documents.
- Implement a main entrypoint (e.g., `main.py`) to run the full pipeline: load data -> preprocess -> vectorize -> similarity -> drift -> alerts -> save results.
- Add tests under `tests/` for core modules and run `pytest -q`.
- Populate `notebooks/experiments.ipynb` with hyperparameter sweeps and diagnostics.

How AI agents should proceed (when repo is populated)
- Start by locating these files in order and report back if missing: `README.md`, `requirements.txt` or `pyproject.toml`, `Dockerfile`, `src/` or `app/`, `tests/`.
- Extract the primary entrypoint (look for `if __name__ == "__main__"`, console scripts in `pyproject.toml`, or `scripts/`).
- Identify data flow by tracing: data loaders -> preprocessing -> model/train/eval code (search for `fit`, `transform`, `TfidfVectorizer`, or `sklearn`).

Project conventions to document here (examples to replace)
- Packaging: Python modules live under `backend/`, `frontend/`, and `src/`.
- Virtualenv: prefer `python -m venv .venv` and `pip install -r requirements.txt`.
- Tests: use `pytest` with tests under `tests/` and run `pytest -q`.
- Notebooks: keep exploratory work out of the production path; prefer adding utilities under `scripts/`.
- Folder structure: `data/` for regulatory texts and internal docs; `backend/` for processing + ML; `frontend/` for Streamlit UI; `src/` for legacy/educational modules; `tests/` for pytest.

Developer workflows (examples to update for this project)
- Setup dev env:

  1. `python -m venv .venv`
  2. Windows: `.venv\\Scripts\\Activate.ps1`; then `pip install -r requirements.txt`

- Run tests: `pytest -q` (or the project's test command from CI if present).
- Lint/format: follow repo config (black, isort, flake8) if present; otherwise use `black .`.

Integration points and external dependencies
- Document any external APIs, data stores, or model registries used (e.g., S3 buckets, databases, MLflow). Agents should not assume presence of credentials — ask for them.

When to ask the human
- If any of the key files listed above are missing, or if CI scripts are unclear, request the `README.md` and the main entrypoints.
- If you need environment variables or secrets, stop and ask rather than guessing.

How to update these instructions
- Replace sections above with concrete project paths and commands once those files exist. Keep this file concise (20–50 lines).

If you want, I can create a minimal README and placeholder files to help bootstrap the repo — tell me what language/runtime you expect.
