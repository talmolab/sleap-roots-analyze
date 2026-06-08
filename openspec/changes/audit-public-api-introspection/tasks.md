# Tasks: Audit `__all__` Exports for Introspection-Readiness

## 1. Audit script + test harness (write the check first)

- [x] 1.1 Write `scripts/check_public_api_docs.py`: iterate
  `sleap_roots_analyze.__all__`; for each callable function assert (a) every param
  + return annotated, (b) `typing.get_type_hints()` resolves, (c) docstring with
  `Args:`/`Returns:`, (d) every param named in the docstring, (e) `Raises:` present
  when the function body contains a non-trivial `raise`; for each class assert a
  class docstring and annotated/documented non-trivial `__init__` params. Print a
  per-symbol report; `exit(1)` on any violation, `exit(0)` when clean. Expose a
  `run_audit() -> list[str]` returning violation strings so tests can call it.
- [x] 1.2 Write `tests/test_public_api_docs.py` that imports `run_audit()` and
  asserts it returns no violations. Confirm it FAILS now (script reports the 7
  known failures) — red state.

## 2. Fix the 7 failing functions (no behavior change)

- [x] 2.1 Add `from typing import Any` to `src/sleap_roots_analyze/visualization.py`
  (fixes `get_type_hints` `NameError` on `create_trait_boxplots_by_genotype`,
  `create_exploratory_summary_plots`, `create_trait_by_genotype_boxplots`).
- [x] 2.2 Add a return annotation and a `Returns:` section to `cli.main`.
- [x] 2.3 Add `Returns:` sections to `save_viz_config` and `validate_viz_config`
  (`pipeline/config/utils.py`).
- [x] 2.4 Add a `Returns:` section to `create_publication_figure`.
- [x] 2.5 Re-run the audit — expect 0 violations across all 112 entries (green).

## 3. Documentation

- [x] 3.1 Write `docs/public_api_audit_2026.md`: methodology + criteria, the
  pre-change failures (7 functions) and their fixes, and the post-change result
  (0 violations / 112 entries).

## 4. Validation

- [x] 4.1 `uv run python scripts/check_public_api_docs.py` exits 0.
- [x] 4.2 `uv run pytest tests/test_public_api_docs.py tests/test_public_api.py`
  pass.
- [x] 4.3 `uv run black --check` + `uv run ruff check` clean on changed files.
- [x] 4.4 `openspec validate audit-public-api-introspection --strict` passes.
