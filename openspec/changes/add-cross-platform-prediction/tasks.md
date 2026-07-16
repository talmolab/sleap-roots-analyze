> Full rationale for every design choice referenced below (PLS component count, clustering
> aggregation level, permutation runtime, CV-metric aggregation, result-type shape) is in
> `design.md`'s Decisions section. Flag **Decision 4** (aggregate vs. per-genotype CV metrics) to
> the user/reviewer explicitly before implementing Section 3 — it's an interpretation of ambiguous
> roadmap language, not a confirmed fact.
>
> **`/review-openspec` round 1 reconciled into this file** (full findings in design.md's
> "Adversarial Review Reconciliation (round 1)" section): fixtures redesigned as N-seed averages
> (1.1/1.2, empirically verified — do not reuse theory.md's literal single-seed recipe, which was
> independently confirmed NOT to recover its claimed R² at this program's scale); `representative_
> indices` renamed `representative_names` throughout (Decision 7); input-validation tests added
> (new 3.13-3.18); Section 5 (trait-set identity oracle) is **BLOCKED** pending a handoff
> investigation — do not start it; Section 6's export gained a test-first task; Section 9 gained a
> `docs/CROSS_PLATFORM_ANALYSIS.md` task.

## 1. Fixtures (test-first)

- [ ] 1.1 Add `cross_platform_planted_signal_fixture` to `tests/fixtures.py` — **redesigned during
      `/review-openspec` round 1; do not use theory.md §4.2's literal single-seed recipe as-is**
      (independently verified: at `n_genotypes=19, n_traits=10, signal_strength=0.8, seed=42`,
      LOGO-CV R² = 0.469 (ridge) / 0.448 (pls), not "≈0.8"; sweeping seeds 1-7 ranges -0.67 to
      0.87 — a fundamental small-sample LOGO-CV variance property at this scale, not an
      implementation bug). Use `n_genotypes=19`, **`n_traits=3`** (reduced from theory.md's
      illustrative 10 — empirically shown to reduce per-seed variance meaningfully, though not
      eliminate it), `signal_strength=0.8`, and a **fixed, deterministic list of 20 seeds**
      (`1..20`). The fixture returns a **list of 20 `(X, y, genotypes)` tuples** (or an equivalent
      structure — decide during implementation), not a single tuple. The oracle (task 3.8) uses
      the **mean** LOGO-CV R² across all 20 realizations, empirically verified in this review
      round: Ridge mean≈0.717 (sd≈0.149), PLS(1) mean≈0.658 (sd≈0.172) — re-verify these exact
      figures during implementation with the actual RNG API/draw order used
      (`np.random.default_rng` vs. this file's legacy `np.random.seed`+`np.random.normal`
      convention — pick whichever this file already uses elsewhere and re-run the verification
      under that API, per this repo's established precedent that the two APIs produce different
      streams for the same seed number) and pin whatever the real numbers are — do not assume
      today's exploratory verification numbers transfer byte-for-byte to a different RNG call
      order.
- [ ] 1.2 Add `cross_platform_pure_noise_fixture` to `tests/fixtures.py` — same N=20-seed
      averaged structure as 1.1, but `X` and `y` independently drawn with no planted relationship
      (use a distinct seed offset, e.g. `seed + 10000`, to guarantee independence from 1.1's
      draws). Expected **mean** LOGO-CV R² is negative (empirically verified: Ridge mean≈-0.279,
      PLS mean≈-0.260 at `n_traits=3` — LOGO-CV R² going negative on pure noise is a known,
      correct property of `r2_score` at small n, not a bug). The oracle (task 3.9, and the
      leakage test's noise counterpart, task 4.2) asserts a clear separation between this
      fixture's mean R² and 1.1's mean R² (e.g. a gap of at least 0.5), not that this fixture's
      mean is "≈0" in isolation.
- [ ] 1.3 Add `cross_platform_synthetic_non_edpie_fixture` to `tests/fixtures.py` — a fixture
      with different genotype count (e.g. 25, not 19), different trait count/names (not
      EDPIE-style column names), and a known planted signal, confirming `logo_cv_predict`/
      `fit_pca_on_fold` make no assumption about EDPIE-specific shapes or names. Follow the same
      N-seed-averaged design as 1.1 for the same statistical reasons. Document the chosen
      parameters and seeds.
- [ ] 1.4 **BLOCKED — trait-set identity oracle (Section 5). Do not start until this returns.**
      `/review-openspec` round 1 found the original design (reproduce 28 cylinder + 14 field
      traits directly from `select_cluster_representatives()` on BLUP-adjusted means) tests the
      wrong substrate and very likely the wrong quantity — see design.md Decision 2's full
      revision. Confirmed: a real, already-committed fixture in this repo
      (`tests/fixtures/real/wheat_edpie/expected/cross_platform/root_core_vs_cylinder/
      exp{1,2}_trait_clusters.csv`) shows `select_cluster_representatives` alone gives 28
      field / 121 cylinder representatives — neither matches 14/28. A handoff investigation has
      been requested (external vault access needed to confirm the real Section 3.4 pipeline
      mechanism: clustering alone, or clustering plus cross-platform correlation filtering at
      |ρ|≥0.55, per `results_3.4_draft_20260330.md`'s literal wording). **Section 5 cannot be
      correctly specced or implemented until this returns — proceed with Sections 2, 3, 4, 6-10
      in the meantime; they do not depend on this resolution.**

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
- [ ] 3.5 Write failing test `test_logo_cv_predict_representative_names_fixed_pre_loop`:
      `reduction_method="representatives"` with `representative_names` passed in; assert the
      same indices are used for every fold (no re-selection inside the loop) — this is the one
      reduction step theory.md §2.2 confirms is safe to fix up front (unsupervised).
- [ ] 3.6 Write failing test `test_logo_cv_predict_pc1_calls_fit_pca_on_fold_per_fold`:
      `reduction_method="pc1"`; mock/spy `fit_pca_on_fold` (Section 2) and assert it is called
      once per fold with that fold's `X_train`/`X_test` only — never with data from other folds
      or the held-out genotype folded into `X_train`.
- [ ] 3.7 Write failing test `test_logo_cv_predict_returns_one_prediction_per_genotype`: output
      length equals `len(genotypes)`, in the same order as the input `genotypes` sequence.
- [ ] 3.8 Write failing test `test_logo_cv_predict_planted_signal_recovers_expected_r2`: using
      `cross_platform_planted_signal_fixture` (1.1, **N=20-seed averaged design — see design.md
      Decision 6**), run `logo_cv_predict` once per seed realization and assert the **mean**
      aggregate R² across all 20 is comfortably positive and within the empirically-verified range
      (≈0.5-0.8, re-verify exact figures during implementation), for both `representatives` and
      `pls_latent` reduction methods. Do NOT assert closeness on a single seed's R² — per
      design.md Decision 6, single-seed LOGO-CV R² at this scale is too high-variance to pin
      reliably.
- [ ] 3.9 Write failing test `test_logo_cv_predict_pure_noise_r2_clearly_below_signal`: using
      `cross_platform_pure_noise_fixture` (1.2, same N=20-seed averaged design), assert the
      **mean** aggregate R² across all 20 seeds is comfortably separated (below, by an
      empirically-justified margin — e.g. at least 0.5) from 3.8's planted-signal mean R², for
      both reduction methods. Do not assert the noise fixture's mean is "≈0" in isolation — it is
      expected to be negative (a known, correct LOGO-CV R² property at small n, not a bug).
- [ ] 3.10 Write failing test `test_logo_cv_predict_synthetic_non_edpie_fixture_generalizes`:
      using `cross_platform_synthetic_non_edpie_fixture` (1.3), assert the planted signal is
      recovered similarly to 3.8 — confirms no hidden EDPIE-specific coupling.
- [ ] 3.11 Write failing test `test_logo_cv_predict_computes_rmse_and_spearman_rho`: assert RMSE
      and Spearman ρ (with its p-value) are returned alongside R², all three computed over the
      same concatenated predictions.
- [ ] 3.13 Write failing test `test_logo_cv_predict_rejects_mismatched_lengths`: `len(X) !=
      len(y)` or `len(X) != len(genotypes)` — assert a clean `ValueError`, not an obscure
      downstream shape error.
- [ ] 3.14 Write failing test `test_logo_cv_predict_rejects_invalid_reduction_method`:
      `reduction_method="not_a_real_method"` — assert `ValueError` listing the three valid values.
- [ ] 3.15 Write failing test `test_logo_cv_predict_representatives_requires_representative_names`:
      `reduction_method="representatives"` with `representative_names=None` (the default) —
      assert a clean `ValueError` raised upfront, not a `TypeError` surfacing deep inside the fold
      loop.
- [ ] 3.16 Write failing test `test_logo_cv_predict_rejects_too_few_genotypes`: `len(genotypes) <
      2` (LOGO-CV cannot form a fold with an empty training set) — assert `ValueError`.
- [ ] 3.17 Write failing test `test_logo_cv_predict_constant_y_does_not_crash`: `y` with zero
      variance (all identical values) — assert the function does not raise (R²/Spearman ρ may be
      `NaN`/degenerate per `sklearn`/`scipy`'s own documented behavior; assert whatever that
      documented behavior actually is, don't invent a new contract).
- [ ] 3.18 Write failing test `test_logo_cv_predict_rejects_nan_in_X`: `X` containing a `NaN`
      value (concretely reachable in production use, per Tier 1's own contract: a failed-trait
      column in `08_blup_adjusted_means.csv` is entirely `NaN` — see `extract_blup_table`'s
      documented behavior in `statistics-api`'s spec) — assert a clean `ValueError` at the top of
      `logo_cv_predict`, not a silent `NaN`-propagating fit deep inside a fold.
- [ ] 3.19 Implement `logo_cv_predict(X, y, genotypes, reduction_method="pls_latent",
      representative_names=None) -> LOGOCVResult` in `cross_platform_prediction.py`. `X` is a
      `pandas.DataFrame` (`(n_genotypes, n_traits)`, columns named by trait, index by genotype —
      see design.md Decision 7); `representative_names` is a list of trait names (column labels),
      used directly as `X[representative_names]` — no index bookkeeping. Validate inputs upfront
      (3.13-3.18) before entering the fold loop. Per theory.md §3/§3.1's pattern: `LeaveOneOut`
      over genotypes; inside each fold, instantiate a fresh `Pipeline` (`StandardScaler` + one of
      `PLSRegression(n_components=1)` / `Ridge()` depending on `reduction_method`); for
      `representatives`, reduce `X_train`/`X_test` to `representative_names` before the pipeline;
      for `pc1`, reduce via `fit_pca_on_fold` per fold; for `pls_latent`, fit the pipeline on the
      full (scaled) trait matrix with no separate reduction step. Explicit docstring precondition
      (design.md Decision 9): `X`'s columns must never include the target trait's own values —
      this is a caller responsibility, not independently verifiable from `X` alone. Compute
      aggregate R² (`sklearn.metrics.r2_score`), RMSE (`root_mean_squared_error` or
      `mean_squared_error(squared=False)` per the pinned scikit-learn version — document that
      RMSE is not comparable across differently-scaled traits, design.md Decision 9), and Spearman
      ρ/p (`scipy.stats.spearmanr` — document that its p-value is an asymptotic approximation,
      imprecise below n≈20-30, per design.md Decision 9) over the concatenated leave-one-out
      predictions. Document `Ridge()`'s default `alpha=1.0` as an accepted, undiscussed choice
      (design.md Decision 9), not a tuned value. Return a small structured result (dataclass or
      dict — decide during implementation which is more ergonomic for Section 6's
      `CrossPlatformPredictionResult` adapter to consume) containing per-genotype `y_true`/`y_pred`
      plus the three aggregate
      metrics. Google docstring with Args/Returns/Raises. Make 3.1-3.18 green.

## 4. Explicit leakage regression test (test-first)

- [ ] 4.1 Write failing test `test_leakage_detectable_ratio_at_least_1_10`
      (`tests/test_cross_platform_prediction.py`): implement theory.md §4.3's `logo_cv_r2(...,
      fit_inside_fold=True/False)` pattern against the **N=20-seed averaged planted-signal
      fixture** (1.1, design.md Decision 6 — not theory.md's literal single-seed fixture),
      computing `mean(r2_outside_across_seeds)` and `mean(r2_inside_across_seeds)`; assert the
      ratio of means is at least 1.10 (either as a small test-local helper, or by exercising
      `logo_cv_predict`'s internals if it exposes an equivalent toggle — decide during
      implementation which avoids duplicating the CV-hygiene logic). If this fails, per theory.md
      §4.4: do not adjust the threshold — the fixture needs revisiting (re-check the redesigned
      parameters in design.md Decision 6, not the original theory.md recipe).
- [ ] 4.2 Write failing test `test_leakage_not_falsely_flagged_on_pure_noise`: same
      mean-of-20-seeds inside/outside-fold comparison on `cross_platform_pure_noise_fixture`
      (1.2) — both should produce comparably low (near-zero-or-negative) mean R², confirming 4.1
      isn't a fixture artifact that fires regardless of actual leakage.
- [ ] 4.3 Ensure production `logo_cv_predict` (Section 3) has no code path equivalent to the
      `fit_inside_fold=False` branch — 3.1/3.2/3.4's mock/spy tests already assert this
      structurally; cross-reference here rather than duplicating.

## 5. Trait-set identity oracle — **BLOCKED, do not start** (see task 1.4 / design.md Decision 2)

> `/review-openspec` round 1 found the original design below tests the wrong substrate and very
> likely the wrong quantity (confirmed: a real committed fixture shows `select_cluster_
> representatives` alone gives 28 field / 121 cylinder representatives, not 14/28 — see task 1.4
> and design.md Decision 2 for the full finding). **5.1/5.2 below are the pre-review draft,
> retained for reference only — do not implement against them as written.** A handoff
> investigation into the real Section 3.4 pipeline has been requested; this section will be
> rewritten once it returns, before any implementation here begins.

- [ ] 5.1 **(superseded, pending rewrite)** Once task 1.4 resolves the real mechanism, write a
      failing test reproducing whatever the actual Section 3.4 pipeline does — likely clustering
      *plus* cross-platform correlation filtering at |ρ|≥0.55, not clustering alone (see task
      1.4's reconstruction). Do not assume the pre-review draft's "call
      `select_cluster_representatives` on genotype-mean/BLUP-level input, assert 28/14" design is
      correct — it has been empirically shown not to match.
- [ ] 5.2 **(superseded, pending rewrite)** `test_cluster_representatives_deterministic_given_same_input`
      may still be a valid, independent check once 5.1 is rewritten (determinism of
      `cluster_correlated_traits`/`select_cluster_representatives` alone is unaffected by which
      quantity 5.1 ultimately targets) — re-confirm once 5.1's design is finalized.

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
- [ ] 6.6a **(New — added after `/review-openspec` round 1)** Write failing test
      `test_cross_platform_prediction_result_importable_from_package_root` (mirroring
      `tests/test_blup_result.py`'s `test_blupresult_importable_from_root`/
      `test_listed_in_result_types_all` precedent): assert
      `from sleap_roots_analyze import CrossPlatformPredictionResult, TargetPrediction` succeeds
      and both appear in `sleap_roots_analyze.__all__` with no duplicates. Section 6 previously
      went straight to implementing the export (6.6b) with no preceding failing test — inconsistent
      with Section 7's own 7.1-before-7.2 pattern for the same kind of export.
- [ ] 6.6b Export `CrossPlatformPredictionResult` and `TargetPrediction` from
      `sleap_roots_analyze/__init__.py`, add to `__all__` **following the existing
      grouping-by-comment-header convention** (e.g. the `# PC-correlation & trait-enrichment
      workflows` block at `__init__.py:362`, which groups a prior tier's 2 functions + 2 result
      types from one new module under one comment header) — add a
      `# Cross-platform prediction (Tier 3, #194)`-style header grouping
      `fit_pca_on_fold`/`logo_cv_predict`/`CrossPlatformPredictionResult`/`TargetPrediction`
      together, rather than scattering them into the generic result-types/functions blocks. Make
      6.6a green. Confirm the existing `public-api-introspection` enforced contract passes for
      both new names (full type hints, Google docstrings with Args/Returns, `Raises:` if
      applicable) — run `scripts/check_public_api_docs.py` locally before relying on CI to catch a
      gap.

## 7. Public API export for `fit_pca_on_fold` / `logo_cv_predict`

- [ ] 7.1 Write failing test `test_cross_platform_prediction_functions_importable_from_package_root`
      (mirroring `statistics-api`'s "Public Statistics API Surface" requirement): assert
      `from sleap_roots_analyze import fit_pca_on_fold, logo_cv_predict` succeeds and both are
      identity-equal (`is`) to the functions defined in `cross_platform_prediction.py`.
- [ ] 7.2 Add both names to `__all__` in `sleap_roots_analyze/__init__.py`, in the **same
      `# Cross-platform prediction (Tier 3, #194)` comment-header group as task 6.6b** (one
      grouped block for all four new names, matching the `pc_correlations` precedent's "2
      functions + 2 result types from one new module, added together" shape — not two separate
      scattered blocks). Confirm `public-api-introspection`'s enforced contract passes (type hints
      resolve via `typing.get_type_hints`, Google docstrings with Args/Returns/Raises,
      `scripts/check_public_api_docs.py` exits 0 including all four new names).

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
      opened. **(Added after `/review-openspec` round 1's git-workflow reviewer)** If this
      surfaces an actual implementation bug, fix it in this PR (Tier 1/2 precedent). If it instead
      surfaces a design-level surprise (e.g. `n_components=1` looking implausible against real
      data), that is grounds for a follow-up issue per design.md's Risks section, not a reason to
      block or split this PR.

## 9. Docs

- [ ] 9.1 Add `## cross_platform_prediction Module` section to `docs/API.md`, **including its
      entry in the file's existing Table of Contents** (the TOC is currently a strict 1:1 index of
      every `##` module heading — found during `/review-openspec` round 1 that the original task
      draft would have broken this invariant for the first time), documenting `fit_pca_on_fold`
      and `logo_cv_predict` signatures/defaults.
- [ ] 9.2 Add a `docs/CHANGELOG.md` `[Unreleased]` `### Added` entry: `fit_pca_on_fold()`,
      `logo_cv_predict()`, `CrossPlatformPredictionResult`/`TargetPrediction`, with a one-line
      rationale (Tier 3 of the cross-platform genotype-prediction program, #194).
- [ ] 9.3 Add a `CrossPlatformPredictionResult` row to `docs/result-types.md`, following the
      existing per-type documentation pattern (including any caveat bullets, e.g. Decision 4's
      aggregate-vs-per-genotype metric interpretation if still relevant post-review, and Decision
      9's RMSE-cross-trait-scale / Spearman-p-approximation caveats).
- [ ] 9.4 **(New — added after `/review-openspec` round 1)** Add a new section to
      `docs/CROSS_PLATFORM_ANALYSIS.md` documenting `logo_cv_predict`/`fit_pca_on_fold`, mirroring
      the existing `## Public PC-Correlation and Trait-Enrichment Workflows` section's shape
      (description + code example) — found during review to be the actual narrative home for
      cross-platform program additions (the immediately-preceding `pc_correlations` tier added its
      module this way; `docs/API.md`/`docs/result-types.md` alone were not sufficient for that
      tier either).

## 10. Validation

- [ ] 10.1 `openspec validate add-cross-platform-prediction --strict` — resolve every reported
      issue before requesting review.
- [ ] 10.2 `/lint` (black + ruff) on all changed files.
- [ ] 10.3 Full `uv run pytest --cov --cov-branch` — confirm no regressions, and that all new
      tests (Sections 2-7) pass, including on whichever CI platforms actually run them.
- [ ] 10.4 `/review-openspec` — adversarial proposal review, ≥1 round, reconcile literally into
      `design.md`. **Round 1 complete** (5 parallel reviewers; 2 BLOCKING + 9 IMPORTANT findings,
      reconciled into `design.md`'s "Adversarial Review Reconciliation (round 1)" section, this
      file, and `proposal.md` — one BLOCKING finding, the trait-set identity oracle's mechanism,
      remains genuinely open pending a handoff investigation; see task 1.4). This task is not
      satisfied until the user has reviewed and approved the reconciled proposal — required before
      implementation (Sections 1-9 above) begins, per the roadmap's per-tier loop. Section 5 also
      requires the handoff investigation's return before it can be implemented, independent of
      general approval on Sections 1-4/6-10.
- [ ] 10.5 Complete Section 8's manual real-data validation and get Elizabeth's explicit sign-off
      before opening the PR.
