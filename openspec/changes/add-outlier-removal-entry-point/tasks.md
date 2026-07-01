## 1. Tests first (red)

- [x] 1.1 New file `tests/test_remove_outlier_samples.py`. Add a seeded (`np.random.seed(42)`)
      fixture builder: a clean (NaN-free) `geno`/`rep`/`Barcode` + several numeric-trait frame
      of ~40 samples with a handful of injected outlier rows (a few samples pushed ~8 SD out on
      2–3 traits). **Pin the fixture against the default detector**: it MUST be validated so that
      `detect_outliers_mahalanobis(chi2_percentile=97.5)` flags *exactly* the injected indices
      (verified during authoring — Mahalanobis on this fixture is deterministic and exact).
- [x] 1.2 Test: `remove_outlier_samples(clean_df)` returns a 2-tuple `(trimmed_df, report)`;
      `len(trimmed_df) == len(clean_df) - report["n_outliers"]`; columns of `trimmed_df` equal
      columns of `clean_df` (rows dropped, not columns).
- [x] 1.3 Test: for the **default Mahalanobis** path, `report["outlier_indices"]` equals the
      injected labels exactly and those labels are absent from `trimmed_df`. For
      `method="isolation_forest"`, assert only `set(injected) <= set(report["outlier_indices"])`
      (contamination is a quota, so it may flag extra rows — do NOT assert equality).
- [x] 1.4 Test: every label in `report["outlier_indices"]` is a member of `clean_df.index`
      (alignment), and `report["outlier_barcodes"]` lists the removed rows' `Barcode` values.
- [x] 1.5 Test: method selection — default is `"mahalanobis"` (`report["method"]`);
      `method="isolation_forest", contamination=0.2` sets `report["method"] == "isolation_forest"`
      (the dispatch key, not `"IsolationForest"`) and records `method_params["contamination"] ==
      0.2`; unknown `method` raises `ValueError` naming the supported methods before any detection.
- [x] 1.6 Test: clean-input precondition — a frame with a NaN in a trait column raises
      `ValueError` whose message names the trait(s) **and** contains `"clean_traits_for_analysis"`
      (the entry point must WRAP `validate_clean_traits`'s message, which does not itself mention
      the entry point); use `pytest.raises(..., match=r"clean_traits_for_analysis")`. Assert it
      raises before any detector runs.
- [x] 1.7 Test: unique-index precondition — a frame whose index has duplicate labels raises
      `ValueError` stating the index must be unique, before any detector runs; a default
      `RangeIndex` frame passes the check.
- [x] 1.8 Test: output readiness —
      (a) a fixture where trimming would leave <2 samples raises `ValueError` naming the count,
          and the raised error carries `outlier_report` (assert the attribute is present);
      (b) the constant-trait gate is unit-tested directly on the readiness helper (constructing a
          "trait varies in clean_df but is constant only after the detector removes exactly the
          varying rows" fixture is not reliably achievable through real detection, so test the
          gate in isolation and document it as white-box);
      (c) happy path: `perform_pca_analysis(trimmed_df[trait_cols])` runs and reports sample count
          == `len(trimmed_df)`;
      (d) a varying trait surviving alongside a constant one passes the gate (returns successfully).
- [x] 1.9 Test: p > n warning — a fixture where trimming leaves more surviving traits than
      samples emits a `UserWarning` mentioning `p > n` (use `pytest.warns`) and still returns.
- [x] 1.10 Test: over-removal rail — use n≈30 with `method="mahalanobis", use_chi_squared=False,
      distance_threshold=<tuned>` so >50% of samples are removed yet ≥2 varying samples survive;
      assert a `UserWarning` about the large removal fraction fires AND the call still returns.
      Add a sibling case where over-removal breaches readiness (<2 survive) and assert BOTH the
      `UserWarning` fires AND a `ValueError` is then raised (order: warn before gate).
- [x] 1.11 Test: report schema — keys present and consistent
      (`n_input_samples == n_outliers + n_output_samples`,
      `removal_fraction == n_outliers / n_input_samples`); for Mahalanobis `threshold_type`,
      `threshold_value`, `n_components`, `goodness_of_fit` populated; for isolation forest those
      four are `None`; `outlier_barcodes` is `None` when no `Barcode` column. **JSON adversarial**:
      build the fixture so a missing cast would surface numpy types — assert
      `all(type(i) in (int, str) for i in report["outlier_indices"])`,
      `type(report["threshold_value"]) is float` (Mahalanobis), and `json.dumps(report)` succeeds.
- [x] 1.12 Test: determinism —
      (a) default Mahalanobis: two calls with `random_state=7` return identical `outlier_indices`
          and equal `trimmed_df`; `report["random_state"] == 7` (records the seed);
      (b) **seed-sensitive path**: `method="isolation_forest"` (or `robust_covariance=True`) — a
          re-run with the SAME seed yields identical `outlier_indices` (proves the seed is
          actually threaded and load-bearing, not vacuously constant).
- [x] 1.13 Test: detector-failure surfacing — monkeypatch the chosen detector to return a result
      with an `error` key / no `outlier_indices`; assert `remove_outlier_samples` raises
      `ValueError` surfacing it (NOT a silent `(clean_df, n_outliers=0)`).
- [x] 1.14 Test: input misuse — empty input; duplicate column names; explicit `trait_cols`
      missing from `clean_df`; explicit non-numeric `trait_cols` — each raises a distinct,
      actionable `ValueError` (not a bare pandas error).
- [x] 1.15 Test: caller-supplied `trait_cols` bypasses `get_trait_columns`; `replicate_col=None`
      is honored on a frame with no replicate column.
- [x] 1.16 Test (single-source-of-truth): spy/monkeypatch
      `sleap_roots_analyze.outlier_removal.detect_outliers_mahalanobis` and
      `remove_outliers_from_data`; assert `remove_outlier_samples(clean_df)` calls them, and that
      `inspect.getsource(remove_outlier_samples)` contains no independent distance/score
      thresholding logic.
- [x] 1.17 Test (public API): `remove_outlier_samples` importable from `sleap_roots_analyze`,
      present in `__all__` (alongside the already-present `detect_outliers_mahalanobis`,
      `detect_outliers_isolation_forest`, `remove_outliers_from_data`), no duplicate `__all__`
      entries, identity-equal to the module definition, and `get_type_hints` resolves on it.

## 2. Implement the entry point (green)

- [x] 2.1 New module `src/sleap_roots_analyze/outlier_removal.py`. Implement
      `remove_outlier_samples(clean_df, trait_cols=None, *, method="mahalanobis",
      barcode_col="Barcode", genotype_col="geno", replicate_col="rep", random_state=42,
      **detect_kwargs) -> tuple[pd.DataFrame, dict]`:
      empty-input guard → duplicate-column + explicit-`trait_cols` (missing/non-numeric) guards
      → unknown-`method` guard → **unique-index guard** → resolve trait cols (`get_trait_columns`
      if `None`) → **NaN-free precondition via `validate_clean_traits`, wrapped to add the
      `clean_traits_for_analysis` pointer** → dispatch to `detect_outliers_mahalanobis` /
      `detect_outliers_isolation_forest` (thread `random_state`, forward `**detect_kwargs`;
      **raise if the result carries `error` / lacks `outlier_indices`**) →
      `remove_outliers_from_data(clean_df, outlier_indices, keep_metadata=True,
      return_outliers=True)` → assemble `outlier_report` (spec "Auditable Outlier Report" schema;
      `method` from the dispatch key; threshold/`n_components`/`goodness_of_fit` from the detector
      for Mahalanobis, `None` for isolation forest) → **over-removal `UserWarning` if
      `removal_fraction > 0.5` (before the readiness gates)** → output readiness gates
      (≥`MIN_SAMPLES_FOR_ANALYSIS` samples, then ≥1 `var(ddof=0) > 0` trait), raising a
      `ValueError` that **carries `outlier_report`** → **`p > n` `UserWarning`** → return
      `(trimmed_df, outlier_report)`.
- [x] 2.2 Reuse the shared helpers from `data_cleanup.py` (`get_trait_columns`,
      `validate_clean_traits`, `MIN_SAMPLES_FOR_ANALYSIS`) and the detectors/remover from
      `outlier_detection.py` — no re-implementation of detection or removal.
- [x] 2.3 Ensure `outlier_report` values are plain Python types (cast numpy scalars with
      `int()` / `float()`, indices/barcodes to lists) so `json.dumps` succeeds; omit large
      per-sample arrays (`mahalanobis_distances`, `anomaly_scores`).
- [x] 2.4 Google-style docstring: document the composition (detect → remove), the `method`
      choices and per-method `**detect_kwargs` (without pinning the detectors' own default values,
      which live in `outlier_detection.py`), the clean-input + unique-index preconditions and why
      (PCA `dropna` index alignment), the output readiness gates (and that the raise carries the
      report), the over-removal and p > n warnings, the report schema, the ~2.5%-by-design trim of
      the default `chi2_percentile=97.5`, and determinism via `random_state` (noting when the seed
      is/ isn't load-bearing).

## 3. Public API + types

- [x] 3.1 Add `remove_outlier_samples` to `src/sleap_roots_analyze/__init__.py` imports and
      `__all__` (verify the detect/remove functions are already present; do not duplicate).
- [x] 3.2 Resolvable type hints (`typing.get_type_hints` must not raise); annotate
      `**detect_kwargs: Any` and import `Any` if needed. The package `test_public_api_docs` audit
      gate enforces complete hints + Google Args/Returns/Raises for every `__all__` entry, so this
      is gate-checked, not optional.

## 4. Reproducibility + mypy gates (repo-health, required for green CI)

- [x] 4.1 **Register `remove_outlier_samples` in the reproducibility registry**: add it to
      `tests/reproducibility_cases.py` `CASES` (or `EXCLUDED` with a documented reason), and
      update the pinned `EXPECTED_QUALNAMES` and case-count anchors in
      `tests/test_reproducibility.py` in lockstep. Without this the package-wide
      `test_sweep_covers_all_stochastic_functions` gate auto-discovers the new `random_state`
      function and fails (CI red). Run `uv run pytest tests/test_reproducibility.py` to confirm.
- [x] 4.2 Run `uv run mypy … | mypy-baseline filter`; resolve any new errors the new module
      introduces (new:0). Do not mask real type errors with the baseline.

## 5. Docs + verify

- [x] 5.1 Add `remove_outlier_samples` to `docs/API.md` (signature matching code). While there,
      add the missing `detect_outliers_isolation_forest` and `remove_outliers_from_data` entries
      it cross-references (currently absent from the `outlier_detection` section), so the new
      entry's references resolve.
- [x] 5.2 Add a `docs/CHANGELOG.md` `[Unreleased] → ### Added` entry with a `(#165)` suffix,
      noting it composes after `clean_traits_for_analysis`.
- [x] 5.3 `uv run pytest tests/test_remove_outlier_samples.py -q` green; new tests deterministic
      (seeded). Then run the **full** suite `uv run pytest -m "not integration" tests/` once — the
      new `__all__` entry + module are picked up by `test_public_api.py`,
      `test_public_api_docs.py`, `test_packaging.py`, and `test_reproducibility.py`, none of which
      are "outlier suites".
- [x] 5.4 `uv run black --check . && uv run ruff check .` clean.
- [ ] 5.5 `openspec validate add-outlier-removal-entry-point --strict` passes. (NOT RUN:
      `openspec` CLI not installed in the dev WSL env; the change's proposal/spec/design markdown
      is unchanged from authoring and structurally intact — run this on a machine with the CLI.)
- [x] 5.6 Confirm the change rides the `0.1.0a4` pre-release cut for the bloom-mcp quality tool
      to consume.
