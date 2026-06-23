## 1. Expose step-02 / step-03 functions (extract + expose, #116 pattern)

- [x] 1.1 Extract step-03's inline NaN validation into `data_cleanup.py`:
      `build_clean_validation_report(df, trait_cols) -> dict` (same report keys as
      `validate_clean.py:64-79`) and `validate_clean_traits(df, trait_cols) -> dict`
      (builds report, raises the canonical `ValueError` via a single shared
      `_format_nan_validation_error(report)`, else returns the report).
- [x] 1.2 Refactor `ValidateCleanStep.execute` to: build report via
      `build_clean_validation_report` → save `03_validation_report.json` (unchanged) → raise
      via `_format_nan_validation_error` on failure. Preserve save-then-raise ordering and
      `StepResult.metadata` keys (`valid_trait_names`, `trait_names`).
- [x] 1.3 Add `apply_data_cleanup_filters`, `validate_clean_traits`,
      `build_clean_validation_report` (and, after §2, `clean_traits_for_analysis`) to
      `__init__.py` imports and `__all__`.
- [x] 1.4 Ensure Google-style docstrings (Args/Returns/Raises) + resolvable type hints on
      all newly-public functions; add a one-line comment at `data_cleanup.py:154` noting the
      trait-column ordering comes from `df.columns`, not the `set()`.

## 2. Tests first for the entry point (red)

- [x] 2.1 New file `tests/test_clean_traits_entry_point.py`. Add a seeded
      (`np.random.seed(42)`) fixture builder modeled on `tests/fixtures.py` `mixed_problem_data`,
      with a couple of NaN-heavy traits + good traits, sized so good traits clear
      `min_samples_per_trait` and NaN-heavy traits exceed `max_nans_per_trait`; pass explicit
      thresholds rather than relying on defaults tuned for ~150-sample data.
- [x] 2.2 Test: returns a 3-tuple `(clean_df, trait_cols, cleanup_log)`; `trait_cols` ==
      `[c for c in input_traits if c in clean_df.columns]` (survivor derivation pinned).
- [x] 2.3 Test: output has **no NaNs** in returned trait cols.
- [x] 2.4 Test: **sample loss minimized** — `len(clean_df) > len(df.dropna())`.
- [x] 2.5 Test: `perform_pca_analysis(clean_df[trait_cols])` runs and its reported sample
      count == `len(clean_df)` (row dropna is a no-op).
- [x] 2.6 Tests (separate, distinct messages): raises on (a) empty input — entry point's own
      message, not PCA's; (b) <2 surviving samples — message names the count; (c) single
      `var(ddof=0)==0` trait — "no non-constant trait remains". Plus a **passing** case:
      multiple traits where ≥1 varies returns successfully.
- [x] 2.7 Test: caller-supplied `trait_cols` bypasses `get_trait_columns`.
- [x] 2.8 Test: `cleanup_kwargs` pass through — tightening `max_nans_per_trait` changes the
      surviving traits; `cleanup_log["effective_thresholds"]` records the value used.
- [x] 2.9 Test: default column names (`geno`/`rep`) work on a `geno`/`rep` fixture, and
      `replicate_col=None` is honored.
- [x] 2.10 Test (public API, #116 style): `apply_data_cleanup_filters`,
      `validate_clean_traits`, `build_clean_validation_report`, `clean_traits_for_analysis`
      importable from `sleap_roots_analyze`, present in `__all__`, identity-equal to module
      definitions, and `get_type_hints` resolves on each.
- [x] 2.11 Test: `validate_clean_traits` on residual-NaN input raises the **byte-exact**
      canonical message (`"Validation failed: {n} NaN values found in trait columns!\n
      Affected traits: [...]"`).
- [x] 2.12 Regression test for "no behavior change to steps 01–03": reuse/extend
      `TestQCPipelineIntegration` (`tests/test_qc_pipeline.py`, the 187→158 sample / 38-trait
      / `validation_passed` baseline). Assert cleaned-data frame equality, saved
      `03_validation_report.json` contents, and `StepResult.metadata` unchanged. Capture the
      pre-refactor baseline from `git stash`/pre-refactor `HEAD` if a frozen snapshot is used
      — do not hand-author it.

## 3. Implement the entry point (green)

- [x] 3.1 Implement `clean_traits_for_analysis(df, trait_cols=None, *, barcode_col="Barcode",
      genotype_col="geno", replicate_col="rep", **cleanup_kwargs)`:
      empty-input guard → resolve trait cols (`get_trait_columns` if None) →
      `apply_data_cleanup_filters` → derive surviving cols → `validate_clean_traits` →
      assert ≥2 samples → assert ≥1 `var(ddof=0)>0` trait → enrich `cleanup_log` with
      `effective_thresholds` + `validation_summary` → return `(clean_df, trait_cols, cleanup_log)`.
- [x] 3.2 Google-style docstring: document the 4 ordered checks, that defaults are the
      cleanup function's (not the pipeline config's), the parity caveat, and that ≥2 samples
      is a runnability floor.

## 4. Docs + verify

- [x] 4.1 Update `docs/API.md` with the 4 new public functions (signatures matching code).
- [x] 4.2 Add a `docs/CHANGELOG.md` `[Unreleased] → ### Added` entry.
- [x] 4.3 `uv run pytest tests/ -q` green; new tests deterministic.
- [x] 4.4 `uv run black --check . && uv run ruff check .` clean.
- [x] 4.5 `openspec validate add-clean-traits-entry-point --strict` passes.
- [x] 4.6 Confirm the change can ride the `0.1.0a3` cut (#163); else flag a `0.1.0a4`
      follow-up. Note: per #163, the a3 cut does not currently list #164 — this must merge
      before that release runs.
