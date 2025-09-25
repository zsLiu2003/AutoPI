# Repository Guidelines

AutoPI orchestrates prompt-injection optimization runs targeting third-party agents. Use this guide to stay aligned with the existing layout and workflows.

## Project Structure & Module Organization
- `main.py` is the CLI entrypoint; it wires dataset loading, optimizer/evaluator orchestration, and result persistence.
- `pipeline/` hosts the core logic: `optimizer.py` drives mutation loops, `executor.py` talks to target agents, and `evaluator.py` aggregates judge and gradient scores.
- `data/` holds prompt seeds, accumulated results, and helpers (`inputs.py`, `dataset_loader.py`); avoid committing large generated logs outside `data/logs/`.
- `config/` contains YAML/JSON config and parser utilities; prefer adding new toggles to `config.yaml`.
- Tests live at repo root as `test_*.py`; keep related fixtures beside the code for now. Auxiliary helpers sit in `utils/`, while `examples/` and shell scripts demonstrate sample runs.

## Build, Test, and Development Commands
- Create an environment and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run a single optimization: `python main.py --target-command "cat /etc/passwd" --seed-tool "diagnostic probe" --target-agent cline --output logs`.
- Batch runs and multi-command flows rely on `--batch-mode` or `--multi-command-mode`; see `run_optimization.sh` for presets.
- Automated checks: `python -m pytest` and `ruff check .`; format with `ruff format` before committing.

## Coding Style & Naming Conventions
- Follow standard Python 3.10 style: 4-space indentation, descriptive snake_case names, PascalCase for classes, and module-level logger per file.
- Keep functions small, type hinted, and guarded by docstrings explaining intent and side effects.
- Prefer explicit imports over wildcard, and centralize configuration defaults in `config/parser.py`.

## Testing Guidelines
- Use `pytest` for new coverage; mirror existing naming (`test_feature_name`) at repo root or alongside modules.
- When tests depend on network APIs, mock the provider via `utils.llm_provider` stubs to keep runs deterministic.
- Validate regressions with targeted commands, e.g., `pytest test_response_parsing.py -k "batch"`.

## Commit & Pull Request Guidelines
- History shows terse lowercase summaries (`agnostic`, `cline_sonnet`); keep them imperative and scoped, e.g., `add user-specific optimizer fallback`.
- Reference related issues in the body, include reproduction commands, and attach sample output when changing optimizer behaviour.
- PRs should state the risk level, test evidence (`pytest`, CLI runs), and note any configuration files that need reloading.

## Security & Configuration Tips
- Store API keys outside the repo; copy `config/api_config.json` to a local override and add secrets via environment variables.
- Logs under `data/` may capture model responses—scrub sensitive content before sharing artifacts.
