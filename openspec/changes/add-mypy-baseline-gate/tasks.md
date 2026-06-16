## 1. Dependencies & configuration

- [x] 1.1 Add `mypy` and `mypy-baseline` to the `dev` dependency group in `pyproject.toml`; run `uv sync --group dev`.
- [x] 1.2 Add a `[tool.mypy]` block targeting `src/sleap_roots_analyze`: `python_version = "3.11"`, `ignore_missing_imports = true`, `disallow_untyped_defs = true` (the single starting ratchet knob).

## 2. Freeze existing debt

- [x] 2.1 Run `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline sync --baseline-path .mypy-baseline.txt` to generate `.mypy-baseline.txt` (329 errors frozen).
- [x] 2.2 Verify `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt` exits 0 on a clean tree (baseline fully absorbs current errors: new=0).
- [x] 2.3 Commit `.mypy-baseline.txt`.

## 3. CI gate

- [x] 3.1 Add a dedicated `type-check` job to `.github/workflows/ci.yml` (ubuntu-latest, `uv sync --group dev`, then `set +o pipefail` + `mypy ... | mypy-baseline filter`), modeled on the existing `reproducibility-gates` job.
- [x] 3.2 Confirmed the gate fails when a new untyped public def is introduced (filter exit 1, new=1) and passes when absent.

## 4. Documentation

- [x] 4.1 Added a "Type Checking (mypy ratchet)" section to `docs/CONTRIBUTING.md` (local command, frozen-baseline meaning, new-defs-must-be-typed, `mypy-baseline sync` to regenerate) plus a pre-submit checklist entry.

## 5. Validation

- [x] 5.1 Ran `openspec validate add-mypy-baseline-gate --strict` — valid.
- [x] 5.2 Ran `uv run black --check` and `uv run ruff check src/sleap_roots_analyze` — both pass, no regressions.
