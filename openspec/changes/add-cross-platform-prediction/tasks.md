> Full rationale for every design choice referenced below (PLS component count, clustering
> aggregation level, permutation runtime, CV-metric aggregation, result-type shape) is in
> `design.md`'s Decisions section. Flag **Decision 4** (aggregate vs. per-genotype CV metrics) to
> the user/reviewer explicitly before implementing Section 3 — it's an interpretation of ambiguous
> roadmap language, not a confirmed fact.

## 1. Fixtures (test-first)

- [ ] 1.1 Add `cross_platform_planted_signal_fixture` to `tests/fixtures.py` — implements
      theory.md §4.2's `make_planted_signal_fixture` exactly: `n_genotypes=19`, `n_traits=10`,
      `signal_strength=0.8`, `seed=42`. Returns `(X, y, genotypes)`. Use this file's existing
      global-state RNG convention (`np.random.seed` + `np.random.normal`) if it diverges from
      theory.md's `np.random.default_rng` pseudo-code — verify the resulting LOGO-CV R² is still
      close to `signal_strength` under whichever RNG API is actually used, and pin whichever
      concrete numbers result (do not assume theory.md's illustrative code produces
      CI-appropriate numbers without running it first, per this repo's established precedent of
      empirically verifying fixture parameters before locking them).
- [ ] 1.2 Add `cross_platform_pure_noise_fixture` to `tests/fixtures.py` — same shape as 1.1 but
      `X` and `y` independently drawn `np.random.normal`, no planted relationship (equivalently,
      `signal_strength` → 0 in the same generator, or a fully independent draw — pick whichever
      keeps the fixture simplest to reason about and document the choice). Expected LOGO-CV R²
      ≈ 0 (within tolerance established empirically, not assumed).
- [ ] 1.3 Add `cross_platform_synthetic_non_edpie_fixture` to `tests/fixtures.py` — a fixture
      with different genotype count (e.g. 25, not 19), different trait count/names (not
      EDPIE-style column names), and a known planted signal, confirming `logo_cv_predict`/
      `fit_pca_on_fold` make no assumption about EDPIE-specific shapes or names. Document the
      chosen parameters and seed.
- [ ] 1.4 **Verify real-data availability for the trait-set identity oracle (Section 5).**
      Inspect `tests/data/Turface_all_traits_2024.csv`, a Field-platform equivalent, and
      `Wheat_EDPIE_cylinder_master_data.xlsx` to determine whether they are the same
      genotype-mean/BLUP-level processed form used in the real Section 3.4 pipeline run (external
      reference only, not committed to this repo: the vault's
      `cross_platform_field_v2/.../exp{1,2}_trait_clusters.csv`, dated 2026-03-30, contains the
      actual verified `trait`/`cluster_id`/`is_representative`/`variance` output). If the
      committed `tests/data/` files are not already in the right form, this task blocks on either
      (a) processing them into genotype-mean form within the test itself, or (b) the manual
      real-data validation task (Section 8) producing a fresh, correctly-shaped CSV that the
      oracle test can then be pointed at. Record the outcome and the resulting trait counts here
      before writing Section 5's test.

## 2. `fit_pca_on_fold` (test-first)

- [ ] 2.1 Write failing test `test_fit_pca_on_fold_fits_on_train_only`: build `X_train`,
      `X_test` where perturbing `X_test` (holding `X_train` fixed) does not change the returned
      projection's underlying components — assert by calling `fit_pca_on_fold` twice with the
      same `X_train` but different `X_test`, then independently fitting
      `sklearn.decomposition.PCA(n_components=1).fit(X_train)` in the test and confirming the
      function's output equals that independently-fit PCA's `.transform(X_test)`.
- [ ] 2.2 Write failing test `test_fit_pca_on_fold_output_shape`: `X_train` shape
      `(n_train, n_traits)`, `X_test` shape `(n_test, n_traits)`, assert output shape is
      `(n_test, n_components)` for `n_components in (1, 2)`.
- [ ] 2.3 Write failing test `test_fit_pca_on_fold_raises_when_n_traits_less_than_n_components`:
      `X_train` with 2 traits, `n_components=3` — assert `ValueError` raised before any sklearn
      call (verify via a mock/spy on `sklearn.decomposition.PCA` that it is never constructed).
- [ ] 2.4 Write failing test `test_fit_pca_on_fold_deterministic`: call twice with identical
      `X_train`/`X_test`/`n_components=1`; assert identical output (no random-state sensitivity
      for a full-rank `X_train`).
- [ ] 2.5 Write failing test `test_fit_pca_on_fold_does_not_mutate_inputs`: assert `X_train`/
      `X_test` arrays are unchanged after the call (no in-place sklearn side effects leak
      through).
- [ ] 2.6 Implement `fit_pca_on_fold(X_train, X_test, n_components=1) -> np.ndarray` in new module
      `src/sleap_roots_analyze/cross_platform_prediction.py` per theory.md §5's contract exactly:
      fresh `PCA(n_components=n_components).fit(X_train)`, return `.transform(X_test)`. Google
      docstring with Args/Returns/Raises (required by the existing `public-api-introspection`
      enforced contract once this function is added to `__all__` — see Section 7). Make 2.1-2.5
      green.

## 3. `logo_cv_predict` core + CV-hygiene (test-first)

> **Before writing this section's tests, confirm Decision 4 (design.md) with the user/reviewer:**
> CV R²/RMSE/Spearman ρ are aggregate metrics over concatenated leave-one-out predictions, not a
> distinct score per single held-out genotype (which is statistically undefined for R²/ρ at
> n_test=1). If this reading is wrong, this section's assertions need to change before, not after,
> implementation.

- [ ] 3.1 Write failing test `test_logo_cv_predict_pipeline_instantiated_inside_fold`
      (`reduction_method="representatives"`, the simplest branch): using a mock/spy on
      `sklearn.pipeline.Pipeline.__init__` or `Ridge.fit`, assert a fresh model instance is fit
      exactly once per fold (19 folds → 19 distinct fit calls on 18-genotype training data each),
      not once on the full 19-genotype set before the loop.
- [ ] 3.2 Write failing test `test_logo_cv_predict_representatives_scaler_never_sees_held_out_genotype`:
      instrument `StandardScaler.fit` (mock/spy) to record the training data it was fit on each
      fold; assert the held-out genotype's row is never present in any fold's scaler-fit data.
- [ ] 3.3 Write failing test `test_logo_cv_predict_pls_latent_uses_fixed_n_components_1`: assert
      the `PLSRegression` instance constructed inside each fold has `n_components == 1` (per
      design.md Decision 1) — inspect via mock/spy on `PLSRegression.__init__`.
- [ ] 3.4 Write failing test `test_logo_cv_predict_pls_latent_never_sees_held_out_genotype_y`:
      same pattern as 3.2, but for the `PLSRegression` fit call's `y_train` argument (supervised
      step — must never include the held-out genotype's target value).
- [ ] 3.5 Write failing test `test_logo_cv_predict_representative_indices_fixed_pre_loop`:
      `reduction_method="representatives"` with `representative_indices` passed in; assert the
      same indices are used for every fold (no re-selection inside the loop) — this is the one
      reduction step theory.md §2.2 confirms is safe to fix up front (unsupervised).
- [ ] 3.6 Write failing test `test_logo_cv_predict_pc1_calls_fit_pca_on_fold_per_fold`:
      `reduction_method="pc1"`; mock/spy `fit_pca_on_fold` (Section 2) and assert it is called
      once per fold with that fold's `X_train`/`X_test` only — never with data from other folds
      or the held-out genotype folded into `X_train`.
- [ ] 3.7 Write failing test `test_logo_cv_predict_returns_one_prediction_per_genotype`: output
      length equals `len(genotypes)`, in the same order as the input `genotypes` sequence.
- [ ] 3.8 Write failing test `test_logo_cv_predict_planted_signal_recovers_expected_r2`: using
      `cross_platform_planted_signal_fixture` (1.1), assert the aggregate R² (concatenated
      predictions, per Decision 4) is within an empirically-derived tolerance of the fixture's
      `signal_strength`, for both `ridge`/`representatives`-style and `pls_latent` reduction
      methods.
- [ ] 3.9 Write failing test `test_logo_cv_predict_pure_noise_r2_near_zero`: using
      `cross_platform_pure_noise_fixture` (1.2), assert aggregate R² ≈ 0 within tolerance.
- [ ] 3.10 Write failing test `test_logo_cv_predict_synthetic_non_edpie_fixture_generalizes`:
      using `cross_platform_synthetic_non_edpie_fixture` (1.3), assert the planted signal is
      recovered similarly to 3.8 — confirms no hidden EDPIE-specific coupling.
- [ ] 3.11 Write failing test `test_logo_cv_predict_computes_rmse_and_spearman_rho`: assert RMSE
      and Spearman ρ (with its p-value) are returned alongside R², all three computed over the
      same concatenated predictions.
- [ ] 3.12 Implement `logo_cv_predict(X, y, genotypes, reduction_method="pls_latent",
      representative_indices=None) -> LOGOCVResult` in `cross_platform_prediction.py` per
      theory.md §3/§3.1's pattern: `LeaveOneOut` over genotypes; inside each fold, instantiate a
      fresh `Pipeline` (`StandardScaler` + one of `PLSRegression(n_components=1)` /
      `Ridge()` depending on `reduction_method`); for `representatives`, reduce `X_train`/`X_test`
      to `representative_indices` before the pipeline; for `pc1`, reduce via `fit_pca_on_fold`
      per fold; for `pls_latent`, fit the pipeline on the full (scaled) trait matrix with no
      separate reduction step. Compute aggregate R² (`sklearn.metrics.r2_score`), RMSE
      (`root_mean_squared_error` or `mean_squared_error(squared=False)` per the pinned
      scikit-learn version), and Spearman ρ/p (`scipy.stats.spearmanr`) over the concatenated
      leave-one-out predictions. Return a small structured result (dataclass or dict — decide
      during implementation which is more ergonomic for Section 6's `CrossPlatformPredictionResult`
      adapter to consume) containing per-genotype `y_true`/`y_pred` plus the three aggregate
      metrics. Google docstring with Args/Returns/Raises. Make 3.1-3.11 green.

## 4. Explicit leakage regression test (test-first)

- [ ] 4.1 Write failing test `test_leakage_detectable_ratio_at_least_1_10`
      (`tests/test_cross_platform_prediction.py`): implement theory.md §4.3's `logo_cv_r2(...,
      fit_inside_fold=True/False)` pattern (either as a small test-local helper, or by exercising
      `logo_cv_predict`'s internals if it exposes an equivalent toggle — decide during
      implementation which avoids duplicating the CV-hygiene logic) using
      `cross_platform_planted_signal_fixture` (1.1); assert
      `r2_outside_fold / max(r2_inside_fold, 1e-6) >= 1.10`. If this fails, per theory.md §4.4:
      do not adjust the threshold — the fixture needs revisiting.
- [ ] 4.2 Write failing test `test_leakage_not_falsely_flagged_on_pure_noise`: same
      inside/outside-fold comparison on `cross_platform_pure_noise_fixture` (1.2) — both should
      produce R² ≈ 0 (no signal for leakage to inflate), confirming 4.1 isn't a fixture artifact
      that fires regardless of actual leakage.
- [ ] 4.3 Ensure production `logo_cv_predict` (Section 3) has no code path equivalent to the
      `fit_inside_fold=False` branch — 3.1/3.2/3.4's mock/spy tests already assert this
      structurally; cross-reference here rather than duplicating.

## 5. Trait-set identity oracle (test-first, real-data-dependent — see task 1.4)

- [ ] 5.1 Once task 1.4 resolves the real-data question, write failing test
      `test_cluster_representatives_reproduce_section_3_4_trait_set`: call
      `cluster_correlated_traits`/`select_cluster_representatives`
      (`cross_experiment_analysis.py`, reused unchanged) on the verified real EDPIE
      genotype-mean/BLUP-level cylinder and field matrices at the existing default
      `threshold=0.8`; assert the resulting representative-trait counts are 28 (cylinder) and 14
      (field), and — if task 1.4 confirms specific trait names are stably reproducible — assert
      the exact trait-name sets match (pin the list; do not derive it analytically). This is a
      **deterministic identity check**, not a numeric R² threshold, per the roadmap/issue.
- [ ] 5.2 Write failing test `test_cluster_representatives_deterministic_given_same_input`:
      calling `cluster_correlated_traits`/`select_cluster_representatives` twice on the same
      genotype-mean matrix produces identical cluster assignments and representative selections
      — confirms 5.1's reproduction isn't incidentally order-dependent.

## 6. `CrossPlatformPredictionResult` (test-first)

- [ ] 6.1 Write failing test `test_cross_platform_prediction_result_round_trips_through_json`:
      build a `CrossPlatformPredictionResult` (with a small `TargetPrediction` list) and assert
      `json.dumps(dataclasses.asdict(result))` succeeds, round-trips as native Python types,
      matching the `BLUPResult`/`HeritabilityResult` precedent exactly (`allow_nan=False` finite
      contract).
- [ ] 6.2 Write failing test `test_cross_platform_prediction_result_no_sklearn_objects`: assert
      no sklearn/numpy object appears in `dataclasses.asdict(result)`.
- [ ] 6.3 Write failing test `test_cross_platform_prediction_result_from_logo_cv_adapter`: an
      adapter (e.g. `CrossPlatformPredictionResult.from_predictions(...)` or equivalent — name
      decided during implementation) builds the frozen result from one or more
      `logo_cv_predict` outputs (Section 3) plus pair/method metadata; assert field values match
      the source data exactly.
- [ ] 6.4 Write failing test `test_cross_platform_prediction_result_pc1_reported_separately`:
      a result containing both representative-trait `TargetPrediction` entries and a
      `target_name="PC1"` entry; assert PC1's R² is a distinct field from any representative
      trait's R² (never averaged/combined into a single scalar).
- [ ] 6.5 Implement `CrossPlatformPredictionResult` and `TargetPrediction` in `result_types.py`
      following the `BLUPResult` template: `@dataclass(frozen=True)`, `to_dict()`,
      `to_json(**kwargs)` (`allow_nan=False` default), a `from_*` adapter, Google docstrings.
      Make 6.1-6.4 green.
- [ ] 6.6 Export `CrossPlatformPredictionResult` and `TargetPrediction` from
      `sleap_roots_analyze/__init__.py`, add to `__all__`. Confirm the existing
      `public-api-introspection` enforced contract passes for both new names (full type hints,
      Google docstrings with Args/Returns, `Raises:` if applicable) — run
      `scripts/check_public_api_docs.py` locally before relying on CI to catch a gap.

## 7. Public API export for `fit_pca_on_fold` / `logo_cv_predict`

- [ ] 7.1 Write failing test `test_cross_platform_prediction_functions_importable_from_package_root`
      (mirroring `statistics-api`'s "Public Statistics API Surface" requirement): assert
      `from sleap_roots_analyze import fit_pca_on_fold, logo_cv_predict` succeeds and both are
      identity-equal (`is`) to the functions defined in `cross_platform_prediction.py`.
- [ ] 7.2 Add both names to `__all__` in `sleap_roots_analyze/__init__.py`. Confirm
      `public-api-introspection`'s enforced contract passes (type hints resolve via
      `typing.get_type_hints`, Google docstrings with Args/Returns/Raises,
      `scripts/check_public_api_docs.py` exits 0 including the two new names).

## 8. Manual real-data validation (non-CI, pre-merge gate)

- [ ] 8.1 **Manual, not part of `pytest`.** Regenerate `08_blup_adjusted_means.csv` for all 4
      real EDPIE platforms (Turface19, Turface150, Cylinder, Field) by rerunning Tier 1's merged
      pipeline against the real platform configs (paths to be supplied by Elizabeth when this
      task is executed).
- [ ] 8.2 **Manual, not part of `pytest`.** Using the Python API directly (no pipeline/CLI
      wiring — that's Tier 3.5), run `logo_cv_predict` for the 4 directed pairs (Turface19→
      Cylinder, Turface19→Field, Cylinder→Field, Turface150→Turface19) with
      `reduction_method="pls_latent"` (primary) and `representatives` (comparison). Inspect the
      resulting R²/RMSE/ρ.
- [ ] 8.3 **Manual, not part of `pytest`.** Sanity-check the results against known correlation
      numbers already in the roadmap (e.g. Turface19→Cylinder correlation = 0.67, p = 0.002 —
      note this is a *correlation* depletion statistic, not directly comparable to a prediction
      R², but should not be wildly implausible in relation to it — e.g. a near-perfect predictive
      R² would be surprising given the near-null correlation result). Record findings; this gates
      PR readiness alongside `/pre-merge-check` — Elizabeth reviews and approves before the PR is
      opened.

## 9. Docs

- [ ] 9.1 Add `## cross_platform_prediction Module` section to `docs/API.md` documenting
      `fit_pca_on_fold` and `logo_cv_predict` signatures/defaults.
- [ ] 9.2 Add a `docs/CHANGELOG.md` `[Unreleased]` `### Added` entry: `fit_pca_on_fold()`,
      `logo_cv_predict()`, `CrossPlatformPredictionResult`/`TargetPrediction`, with a one-line
      rationale (Tier 3 of the cross-platform genotype-prediction program, #194).
- [ ] 9.3 Add a `CrossPlatformPredictionResult` row to `docs/result-types.md`, following the
      existing per-type documentation pattern (including any caveat bullets, e.g. Decision 4's
      aggregate-vs-per-genotype metric interpretation if still relevant post-review).

## 10. Validation

- [ ] 10.1 `openspec validate add-cross-platform-prediction --strict` — resolve every reported
      issue before requesting review.
- [ ] 10.2 `/lint` (black + ruff) on all changed files.
- [ ] 10.3 Full `uv run pytest --cov --cov-branch` — confirm no regressions, and that all new
      tests (Sections 2-7) pass, including on whichever CI platforms actually run them.
- [ ] 10.4 `/review-openspec` — adversarial proposal review, ≥1 round, reconcile literally into
      `design.md`. This task is not satisfied until the user has reviewed and approved the
      reconciled proposal — required before implementation (Sections 1-9 above) begins, per the
      roadmap's per-tier loop.
- [ ] 10.5 Complete Section 8's manual real-data validation and get Elizabeth's explicit sign-off
      before opening the PR.
