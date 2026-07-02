# Tasks: Public Outlier-Plotting Entry Point — `plot_outlier_analysis`

TDD order: behavior/oracle tests first (red), implement to green, then the transparent step
refactor, then docs/gates. B1 + D3 ratified in the updated #173 (2026-07-01).

## 1. Behavior tests (`tests/test_plot_outlier_analysis.py`)

- [x] 1.1 Clean, unique-indexed trait fixture with injected outliers + `geno` column (mirrors #165).
- [x] 1.2 Returns `dict[str, plt.Figure]`; mahalanobis keys + `outliers_per_genotype`, no
      `pca_outlier`; isolation_forest key + per-genotype, no mahalanobis keys.
- [x] 1.3 Determinism/match: re-detected `outlier_indices` (captured via a wrapped `create_*` spy)
      equal `remove_outlier_samples`'s, for both `mahalanobis` and the seed-load-bearing
      `isolation_forest`.
- [x] 1.4 `which`: list narrows; bare string works; `None` = full set; unavailable/misspelled key
      raises naming the available keys; per-genotype key rejected when no genotype column.
- [x] 1.5 No IO (no files written; figures returned open).
- [x] 1.6 Per-genotype figure present only when a genotype column exists.
- [x] 1.7 Preconditions/misuse: NaN → error mentioning `clean_traits_for_analysis`; non-unique index;
      empty frame; missing `trait_cols`; unknown `method`; cross-method `detect_kwargs`;
      `random_state=None` accepted.
- [x] 1.8 Detector-failure: raises before any `create_*` is called (spied).
- [x] 1.9 Public API: importable, in `__all__` once, `get_type_hints` resolves, Google
      Args/Returns/Raises; `test_public_api_docs` audit passes.
- [x] 1.10 Lower-level helper returns core-only figures when `genotype_col` is not given.

## 2. Implementation (`outlier_visualization.py`)

- [x] 2.1 `select_outlier_figures(df, results, method, which=None, genotype_col=None)` — the single
      per-method selection: dispatch `results[method]` to `create_mahalanobis_outlier_plots` /
      `create_isolation_forest_plots`; add `outliers_per_genotype` (over full `results`) when
      `genotype_col` present; filter by `which`. Fully type-annotated.
- [x] 2.2 `plot_outlier_analysis(clean_df, trait_cols=None, *, method="mahalanobis",
      barcode_col="Barcode", genotype_col="geno", replicate_col="rep", random_state=42, which=None,
      **detect_kwargs)`: misuse + NaN + unique-index + unknown-`method`/`detect_kwargs` guards;
      re-detect; raise on detector error before delegating; `select_outlier_figures(clean_df,
      {method: result}, method, which=which, genotype_col=genotype_col)`.
- [x] 2.3 Google docstring; `**detect_kwargs: Any` with `Any` imported (guards the `get_type_hints`
      `NameError`). `random_state: Optional[int] = 42` (matches docs/tests; the seed reaches the
      detector via a `Dict[str, Any]` splat so the detectors' `int` annotation is not tripped).

## 3. Single-source-of-truth step refactor (transparent)

- [x] 3.1 Regression test in `tests/test_step_visualize_outliers.py` with **realistic** detector
      output: asserts the step's `mahalanobis`/`isolation_forest` filenames equal the `create_*`
      keys (prefixed as before), and that comparison + a single cross-method per-genotype figure are
      unchanged.
- [x] 3.2 Step's `mahalanobis` / `isolation_forest` blocks call
      `select_outlier_figures(df, outlier_results, method)` (no re-detection); `pca`, `kmeans`,
      `gmm`, `hierarchical`, comparison, and cross-method per-genotype blocks unchanged.

## 4. Public API + docs

- [x] 4.1 Import + `__all__` entry for `plot_outlier_analysis` (no dup; `create_*` stay exported).
- [x] 4.2 `docs/API.md`: new `outlier_visualization` section for `plot_outlier_analysis` **and**
      backfill of `create_mahalanobis_outlier_plots` / `create_isolation_forest_plots` /
      `create_outliers_per_genotype_plot` (module had no section); TOC entry added.
- [x] 4.3 `docs/CHANGELOG.md` `[Unreleased] → Added` entry with the `(#173)` suffix.
- [~] 4.4 `docs/public_api_audit_2026.md` count left as-is — it is a dated point-in-time #117 audit
      snapshot (no test/spec asserts the number); rewriting its measured counts would misrepresent
      the historical audit. Noted rather than edited.

## 5. Reproducibility gate

- [x] 5.1 `plot_outlier_analysis` added to `tests/reproducibility_cases.py` `EXCLUDED` with the reason
      "composes figures over already-swept `detect_outliers_*`; adds no new stochastic step and
      returns Figures (not sweep-comparable)". `EXPECTED_QUALNAMES` / `len(CASES)` unchanged (it is
      `EXCLUDED`, not `CASES`); `test_excluded_set_is_consistent` covers it.

## 6. Verification

- [x] 6.1 `openspec validate add-outlier-plotting-entry-point --strict` — valid.
- [x] 6.2 `uv run pytest` green for the affected suites (plot, step, reproducibility,
      public_api_docs, public_api, remove_outlier_samples) — 173 passed.
- [x] 6.3 `uv run black --check` + `uv run ruff check` clean; mypy baseline delta 0 (`new: 0`).

## 7. PR #175 review follow-ups (single-detection reuse, correctness, lifecycle)

- [x] 7.1 Promote the selection helper to **public** `select_outlier_figures` + add to `__all__`;
      neutral `select_outlier_figures:` error prefix (not `plot_outlier_analysis:`).
- [x] 7.2 Fix the `which`-narrow figure leak: skip the per-genotype render when not requested and
      `plt.close` any built-but-excluded figures (and close all before raising on a missing key).
      Regression test: `len(plt.get_fignums()) == len(returned)` after a narrowed call.
- [x] 7.3 `plot_outlier_analysis` `random_state -> Optional[int] = 42` (matches docs/tests).
- [x] 7.4 Add `barcode_col`/`genotype_col`/`replicate_col` to `plot_outlier_analysis`, forwarded to
      `get_trait_columns` and used for the per-genotype figure (drop hardcoded `"geno"`). Tests:
      metadata cols forwarded; custom `genotype_col` drives the per-genotype figure.
- [x] 7.5 `remove_outlier_samples(return_detector_result=False)` additive flag → 3-tuple when `True`;
      default 2-tuple unchanged. Tests: flag returns raw dict; it feeds `select_outlier_figures`.
- [x] 7.6 Edge-case tests: duplicate columns, non-numeric explicit `trait_cols`, all-NaN column.
- [x] 7.7 Reword "byte-identical" → "same figure keys/filenames" (step test + `visualize_outliers.py`
      comments). Document the figure lifecycle in `docs/API.md`; document `select_outlier_figures` +
      `return_detector_result` in API.md/CHANGELOG.
