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
- [x] 2.12 "No behavior change to steps 01–03" coverage. The transparent refactor is guarded
      by: (a) the **existing** `tests/test_step_validate_clean.py` (drives `ValidateCleanStep`
      directly; passes unchanged), (b) the **existing** `TestQCPipelineIntegration`
      (`tests/test_qc_pipeline.py`, steps 01→03; passes unchanged), and (c) a **new** focused
      `test_step02_to_step03_uses_shared_functions_and_passes` in
      `tests/test_clean_traits_entry_point.py` that runs `CleanupTraitsStep` → `ValidateCleanStep`
      through the extracted functions and asserts `validation_passed` / no trait NaNs.
      (Corrected: no new test was added to `test_qc_pipeline.py`; the existing one is the
      integration guard.)
- [x] 2.13 Behavior tests added with §3 changes: default thresholds deliver a clean frame on
      ordinary sparse data (no raise); residual NaN rows dropped while clean rows kept;
      cleanup-path (not pre-shrunk input) trips the <2-samples gate; explicit `trait_cols`
      missing / non-numeric and duplicate column names raise actionable errors; `UserWarning`
      in the p > n regime.

## 3. Implement the entry point (green)

- [x] 3.1 Implement `clean_traits_for_analysis(df, trait_cols=None, *, barcode_col="Barcode",
      genotype_col="geno", replicate_col="rep", **cleanup_kwargs)`:
      empty-input guard → duplicate-column + explicit-trait_cols (missing/non-numeric) guards →
      resolve trait cols (`get_trait_columns` if None) → thresholds defaulted from
      `inspect.signature(apply_data_cleanup_filters)` (no hardcoded copies) →
      `apply_data_cleanup_filters` → derive surviving cols → **drop residual NaN rows in
      surviving traits** → `validate_clean_traits` (defensive) → assert
      ≥`MIN_SAMPLES_FOR_ANALYSIS` samples → assert ≥1 `var(ddof=0)>0` trait → INFO-log
      effective thresholds + pipeline-divergence note → `UserWarning` if p > n → enrich
      `cleanup_log` with `effective_thresholds` + `validation_summary` → return tuple.
- [x] 3.2 Google-style docstring: document the cleanup order (drop bad traits, then residual
      NaN rows), the 4 ordered checks, that defaults are the cleanup function's signature
      defaults (not the pipeline config's), that name-sanitization is NOT applied so output is
      not byte-equivalent to the pipeline, and that ≥2 samples is a runnability floor.

## 5. mypy baseline gate (repo-health, required for green CI)

- [x] 5.1 Annotate `**cleanup_kwargs: Any` + add `Any` to the typing import in
      `data_cleanup.py` (fixes the one new error this PR introduced).
- [x] 5.2 `mypy-baseline sync` `.mypy-baseline.txt`: the baseline was already stale on `main`
      (63 entries no longer reproduce, from #159/#162 merging without a sync) and `main` is
      mypy-red, so the gate fails regardless of #166. Sync regenerates it; verify
      `mypy … | mypy-baseline filter` exits 0 (new:0, fixed:0).
- [ ] 5.3 Follow-up issue: fix the inherited #159 type-lies frozen in the baseline
      (`data_utils.py` `convert_to_json_serializable` untyped; `reduce_trait_redundancy.py`
      `files_generated` `list[str]` vs `list[Path]`). Out of scope for #166.

## 4. Docs + verify

- [x] 4.1 Update `docs/API.md` with the 4 new public functions (signatures matching code).
- [x] 4.2 Add a `docs/CHANGELOG.md` `[Unreleased] → ### Added` entry.
- [x] 4.3 `uv run pytest tests/ -q` green; new tests deterministic.
- [x] 4.4 `uv run black --check . && uv run ruff check .` clean.
- [x] 4.5 `openspec validate add-clean-traits-entry-point --strict` passes.
- [x] 4.6 Confirm the change can ride the `0.1.0a3` cut (#163); else flag a `0.1.0a4`
      follow-up. Note: per #163, the a3 cut does not currently list #164 — this must merge
      before that release runs.
