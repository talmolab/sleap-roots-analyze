> Full rationale for every design choice referenced below (self-contained `permutation_test()`,
> per-target (not per-permutation) `joblib.Parallel` strategy and its empirical basis, separate
> `CrossPlatformPermutationResult` type, additive `PredictCrossPlatformStep` extension, config
> field placement, figure scope) is in `design.md`'s Decisions section and the brainstormed design
> at `docs/superpowers/specs/2026-07-16-prediction-permutation-and-figure-design.md`.
>
> **`/review-openspec` round 1 reconciled into this file** (full findings in design.md's
> "Adversarial Review Reconciliation (round 1)" section): a dropped-scenario bug in the MODIFIED
> "Predict Cross-Platform..." requirement, a missing non-finite-null guard, the `2q/n` top-
> quartile-recovery math (not the roadmap's unverified "≈25%"), a backwards `docs/API.md` task
> premise, a missed stale-sentence doc fix, per-target/method seed independence via
> `SeedSequence.spawn`, `assert_allclose` (not bit-identical) for parallel-vs-serial, the `joblib`
> dependency task moved before the section that imports it, and several smaller test/doc gaps.
>
> **`/review-openspec` round 2 (fresh, no memory of round 1) reconciled into this file** (full
> findings in design.md's "Adversarial Review Reconciliation (round 2)" section): two new BLOCKING
> issues — the `numpy.random.Generator(random_state)` API call in round 1's own fix was invalid
> (that constructor rejects a bare `int`; fixed to `numpy.random.default_rng(random_state)`, which
> accepts `int`/`SeedSequence`/`Generator` uniformly, resolving the `SeedSequence.spawn()` →
> `permutation_test(random_state=...)` handoff round 1 left unspecified), and Section 6 (now 7)
> depended on `create_prediction_figure()` (Section 8, now 6) without that section being sequenced
> first — fixed by moving the figure-content module ahead of the step and restructuring the step's
> implementation into 3 genuine red→green increments instead of "8 tests, then 1 commit at the
> end." Also fixed: an intrinsic CI-timeout risk in the oracle tests (missing explicit reduced
> `n_permutations`), a factually-wrong claim about Tier 3.5's manual-validation timing (corrected
> against real commit timestamps), missing config validation for the 2 previously-untested
> `PredictionConfig` fields, unspecified seed-enumeration order, unspecified partial-failure/atomic-
> write semantics, an unstated fail-fast-vs-complete rationale, a documentation anti-pattern
> (cross-referencing `design.md`, which moves to `openspec/changes/archive/...` on archival) fixed
> by stating key numbers directly in shipped docs instead, and several missing edge-case
> scenarios/tests (explicit invalid `q`, PC1-only targets, `comparison_methods=[]`, an
> `observed_top_quartile_recovery` wiring check).

## 1. Fixtures (test-first)

- [ ] 1.1 Reuse Tier 3's N=20-seed-averaged planted-signal fixture (`n_genotypes=19, n_traits=3,
      signal_strength=0.8`, seeds `1..20`) and its pure-noise counterpart, per Tier 3 Decision 6 —
      do not re-derive from `theory.md`'s single-seed recipe. Confirm these fixtures (or a thin
      wrapper reusing their generator function) are importable from this tier's test module.
- [ ] 1.2 Add a small set of independent pure-noise fixtures (real, non-degenerate `X`; `y`
      independently drawn, no planted relationship) for the K-S permutation-calibration oracle —
      **exactly 40** independent realizations (pinned to one exact number, not "~30-50", per
      `/review-openspec` round 3's request for exact, measurable quantities), each with a **fixed,
      committed literal seed** (not regenerated per test run) and a **fixed, committed literal
      `permutation_test` `random_state`** for the oracle itself, `n_genotypes=19` (matches the
      other fixtures' scale). Pinning both to literals makes task 9.1's K-S test a deterministic
      golden check rather than a live stochastic
      draw with an intrinsic ~5%+ false-failure rate across 3 OSes × every PR.
- [ ] 1.3 Add a small 2-platform synthetic BLUP fixture pair with `prediction.visualize: true`
      wired into a harness YAML (extending Tier 3.5's own harness fixture at
      `tests/fixtures/harness/cross_platform/`, or a sibling file) — used for wiring-correctness
      tests in Sections 6-8, not for statistical oracle assertions (mirrors Tier 3.5 task 1.1/1.2's
      "wiring correctness, not statistical claim" framing).

## 2. `permutation_test()`/`top_quartile_recovery()` (test-first)

- [ ] 2.1 Write failing test `test_top_quartile_recovery_perfect_prediction_recovers_all`: a
      strictly monotonic `y_pred` (e.g. `y_pred == y_true`) gives `top_quartile_recovery == 1.0`.
- [ ] 2.2 Write failing test `test_top_quartile_recovery_uses_top_2q_predicted_set`: construct a
      case where the true top-`Q` genotypes are NOT in the predicted top-`Q` but ARE in the
      predicted top-`2Q` — assert full recovery (proves the `2Q` window, not `Q`, is used).
- [ ] 2.3 Write failing test `test_top_quartile_recovery_default_q_is_quarter_of_n`: with
      `len(y_true) == 19` and `q` omitted, assert the effective `q` used equals `round(19 / 4)`
      (inspect via a case constructed so a wrong `q` produces a different recovery fraction).
- [ ] 2.3a Write failing test `test_top_quartile_recovery_small_n_gives_at_least_one_and_not_over_n`:
      at `len(y_true)=3` (this program's smallest real scale), assert the effective *default* `q`
      used is `>= 1` and `2 * q <= len(y_true)` (guards against a vacuous `q=0` or an out-of-range
      window).
- [ ] 2.3b Write failing test `test_top_quartile_recovery_rejects_explicit_invalid_q`: a caller-
      supplied `q=0`, a negative `q`, or a `q` with `2 * q > len(y_true)` raises `ValueError` naming
      the invalid `q` and `len(y_true)` — a stricter contract than 2.3a's *default*-`q` case, which
      the function itself computes and guarantees valid.
- [ ] 2.4 Implement `top_quartile_recovery(y_true, y_pred, q=None)` in
      `cross_platform_prediction.py`. Make 2.1-2.3b green.

> **Commit boundary**: 2.1-2.4 (`top_quartile_recovery`, no dependency on `permutation_test`) is a
> natural standalone commit; 2.5 onward (`permutation_test`, which calls `top_quartile_recovery`)
> is a second.

- [ ] 2.5 Write failing test `test_permutation_test_observed_matches_direct_logo_cv_predict_call`:
      `permutation_test(X, y, genotypes, method).observed_r2` (etc.) exactly matches an
      independent `logo_cv_predict(X, y, genotypes, method)` call's `r2` (etc.) on the same inputs;
      `observed_top_quartile_recovery` exactly equals
      `top_quartile_recovery(y, that_call.y_pred)` — the same observed call's predictions, not a
      separately-shuffled or stale value (this metric has no `LOGOCVResult`/`TargetPrediction`
      analog to cross-check against, unlike the other three, so this test is its only wiring
      oracle).
- [ ] 2.6 Write failing test `test_permutation_test_null_distributions_have_length_n_permutations`:
      `null_r2`/`null_rmse`/`null_spearman_rho`/`null_top_quartile_recovery` each have length `N`
      for `n_permutations=N`.
- [ ] 2.7 Write failing test `test_permutation_test_shuffles_y_not_x_or_genotypes`: spy on
      `logo_cv_predict` (or inspect its call arguments) to confirm each permutation iteration's `X`
      and `genotypes` arguments are unchanged from the original inputs, only `y` differs.
- [ ] 2.8 Write failing test `test_permutation_test_deterministic_given_same_random_state`: two
      calls in the same process, with identical arguments (including `random_state`), produce
      bit-identical null arrays. Also parametrize over `random_state` being a plain `int` and a
      `numpy.random.SeedSequence` instance — both must work, since `VisualizePredictionStep`
      (Section 7) passes `SeedSequence` children, not raw ints.
- [ ] 2.9 Write failing test `test_permutation_test_different_random_state_differs`: two calls
      differing only in `random_state` produce non-identical `null_r2` arrays.
- [ ] 2.10 Write failing test
      `test_permutation_test_null_top_quartile_recovery_uses_shuffled_y_as_truth`: construct a
      case where using the *original* `y` as truth (instead of that permutation's shuffled `y`)
      would produce a detectably different recovery value — assert the shuffled-`y`-as-truth
      behavior (Decision 2 in design.md).
- [ ] 2.11 Write failing test `test_permutation_test_p_value_formula`: for a hand-constructed
      `null` array and `observed` value, assert `p_value_r2` (etc.) equals
      `(count(v >= observed) + 1) / (n_permutations + 1)` exactly.
- [ ] 2.12 Write failing test `test_permutation_test_rejects_non_positive_n_permutations`:
      `n_permutations=0` and `n_permutations=-1` both raise `ValueError`, before any
      `logo_cv_predict` call (spy to confirm zero calls made).
- [ ] 2.12a Write failing test `test_permutation_test_accepts_n_permutations_equal_1`:
      `n_permutations=1` does not raise; `null_r2`/etc. each have length exactly `1`; the resulting
      p-value formula degenerates correctly to `(count + 1) / 2` (either `0.5` or `1.0`) — a
      boundary distinct from the general `n_permutations<=0` rejection in 2.12.
- [ ] 2.13 Write failing test `test_permutation_test_surfaces_logo_cv_predict_validation_errors`:
      an invalid `reduction_method` (or mismatched-length `X`/`y`/`genotypes`, or duplicate
      `genotypes`) raises the same `ValueError` `logo_cv_predict` itself would raise, from the
      observed-value call, before any permutation runs.
- [ ] 2.13a Write failing test `test_permutation_test_rejects_non_finite_null_values_with_named_error`:
      construct a permutation iteration whose LOGO-CV fold structure produces a non-finite
      `spearman_rho` (e.g. inject via a monkeypatched `logo_cv_predict` returning a degenerate
      result for one specific permutation index, independent of whether that permutation's
      shuffled `y` was itself constant) — assert `ValueError` naming both the offending metric and
      the permutation index, raised only after all `n_permutations` calls complete (fail-fast on
      the first occurrence was considered and rejected: a genuinely non-finite-producing bug is
      expected to affect many permutations within one target, not a rare one-off, so failing fast
      saves little wall-clock time while complicating which-permutations-ran accounting), not a
      downstream `to_json()` crash with no indication of which permutation caused it.
- [ ] 2.14 Implement `permutation_test(X, y, genotypes, reduction_method="pls_latent",
      representative_names=None, n_permutations=1000, random_state=42)` in
      `cross_platform_prediction.py`, building its RNG via
      `numpy.random.default_rng(random_state)` (not `numpy.random.Generator(random_state)`
      directly, which requires a `BitGenerator` instance and rejects a bare `int` — `default_rng`
      accepts `int`/`SeedSequence`/`Generator` uniformly, which is what makes 2.8's parametrized
      `SeedSequence` case and Section 7's per-target `SeedSequence` children both work with no
      int-extraction step). Returns a `PermutationResult` (Section 3), including the non-finite-
      null scan from 2.13a. Make 2.5-2.13a green.

## 3. `PermutationResult`/`CrossPlatformPermutationResult` (test-first)

- [ ] 3.1 Write failing test `test_permutation_result_round_trips_through_json_as_native_types`:
      `json.dumps(dataclasses.asdict(result))` succeeds; parsed-back numeric fields (including
      every element of every null-distribution list) are Python `float`, not `np.float64`.
- [ ] 3.2 Write failing test `test_permutation_result_null_lists_have_length_n_permutations`.
- [ ] 3.3 Write failing test
      `test_cross_platform_prediction_result_has_no_permutation_result_field`: inspect
      `dataclasses.fields(CrossPlatformPredictionResult)` and `TargetPrediction`, assert neither
      references `PermutationResult`/`CrossPlatformPermutationResult` (Decision 3 — types stay
      structurally independent).
- [ ] 3.3a Write failing test `test_permutation_result_has_no_sklearn_or_numpy_object`:
      `dataclasses.asdict(result)` contains no sklearn `Pipeline`/`PLSRegression`/`Ridge`/`PCA`/
      `StandardScaler` object and no raw `numpy.ndarray` — every null-distribution field is a
      plain Python `list` of `float`.
- [ ] 3.4 Implement `PermutationResult`/`CrossPlatformPermutationResult` in `result_types.py`,
      mirroring `TargetPrediction`/`CrossPlatformPredictionResult`'s `to_dict()`/`to_json()`
      pattern exactly. Make 3.1-3.3a green.

> **Commit boundary**: 3.1-3.4 (the dataclasses themselves) have no import-time dependency on
> `permutation_test` and can land before Section 2 finishes; 3.5-3.8 (the adapter, which calls
> `permutation_test` in its own test) must land after Section 2 is green.

- [ ] 3.5 Write failing test
      `test_cross_platform_permutation_result_adapter_maps_fields_from_real_output`: build a
      `CrossPlatformPermutationResult` from real `permutation_test()` outputs for multiple targets,
      assert every field matches exactly.
- [ ] 3.6 Implement the `from_permutation_test_results`-style adapter (naming to match
      `CrossPlatformPredictionResult.from_logo_cv_results`'s convention). Make 3.5 green.
- [ ] 3.7 Write failing test
      `test_permutation_result_types_importable_from_package_root`:
      `from sleap_roots_analyze import CrossPlatformPermutationResult, PermutationResult`
      succeeds; both names in `__all__`, no duplicates.
- [ ] 3.8 Add both names to `__init__.py`'s import block and `__all__` (grouped by comment header,
      matching the existing `pc_correlations`/cross-platform-prediction grouping convention). Make
      3.7 green.

## 4. `PredictionConfig` new fields + `CrossPlatformConfig` cross-check (test-first)

- [ ] 4.1 Write failing test `test_prediction_config_visualize_defaults_to_false_and_no_op`:
      `PredictionConfig()` has `visualize=False`, `n_permutations=1000`,
      `permutation_random_state=42`, `permutation_n_jobs=8`; construction with these defaults does
      not raise.
- [ ] 4.2 Write failing test `test_cross_platform_config_rejects_visualize_true_with_enabled_false`:
      `CrossPlatformConfig(..., prediction=PredictionConfig(enabled=False, visualize=True))`
      raises `ValueError` at construction time.
- [ ] 4.3 Write failing test `test_prediction_config_permutation_fields_validation_skipped_when_visualize_false`:
      `enabled=True, visualize=False` with `n_permutations=0`, `permutation_n_jobs=0`, and
      `permutation_random_state=-1` (all simultaneously invalid) does not raise — none of the 3
      permutation-related fields are validated unless `visualize=True`.
- [ ] 4.4 Write failing test `test_prediction_config_rejects_non_positive_n_permutations_when_visualize_true`:
      `enabled=True, visualize=True, n_permutations=0` (and `-1`) raises `ValueError`.
- [ ] 4.4a Write failing test `test_prediction_config_rejects_non_positive_permutation_n_jobs_when_visualize_true`:
      `enabled=True, visualize=True, permutation_n_jobs=0` (and `-1`) raises `ValueError` naming
      the field, not `joblib.Parallel`'s own raw error surfacing later inside
      `VisualizePredictionStep`.
- [ ] 4.4b Write failing test `test_prediction_config_rejects_invalid_permutation_random_state_when_visualize_true`:
      `enabled=True, visualize=True, permutation_random_state=-1` (and a non-`int` value) raises
      `ValueError` naming the field, not `numpy.random.SeedSequence`'s own raw error.
- [ ] 4.5 Extend `PredictionConfig`/`CrossPlatformConfig.__post_init__` in
      `pipeline/config/components.py` with the 4 new fields and the 4.2/4.4/4.4a/4.4b validations.
      Make 4.1-4.4b green.

## 5. `PredictCrossPlatformStep` additive extension (test-first)

- [ ] 5.1 Write failing test `test_predict_step_exposes_predictor_matrices_in_step_result_data`:
      after a normal run, `StepResult.data["predictor_matrices"]` holds `source_clean`/
      `target_clean` (DataFrames matching the step's own internal computation) and
      `source_representative_names`/`target_representatives`.
- [ ] 5.2 Write failing test
      `test_predict_step_existing_data_metadata_files_unchanged_by_predictor_matrices_addition`: a
      full backward-compat regression test — every existing key in `StepResult.data`/`metadata`
      and every path in `files_generated` is byte-for-byte/value-for-value identical to this step's
      pre-Tier-4 behavior on the same fixture (guards against Decision 6's additive-only promise).
      Explicitly assert `set(result.data.keys()) - {"predictor_matrices"}` equals the exact
      pre-Tier-4 key set (method names only) — not just that pre-existing keys are unchanged, but
      that `"predictor_matrices"` is the *only* addition, catching any accidental extra key leakage.
- [ ] 5.3 Extend `PredictCrossPlatformStep.execute()` in `predict_cross_platform.py` to populate
      `predictor_matrices`. Make 5.1-5.2 green.

## 5a. `joblib` dependency (must land before Section 7b)

- [ ] 5a.1 Add `joblib` to `pyproject.toml`'s direct dependencies (design.md Decision 5), pinned to
      the version already resolved transitively via `scikit-learn` in this environment's lockfile,
      and regenerate `uv.lock` accordingly (found during `/review-openspec` round 3: the original
      task didn't mention the lockfile needs regenerating alongside the `pyproject.toml` edit) —
      ordered ahead of Section 7b, which imports `joblib.Parallel` directly at module level (this
      task was originally worded around "Section 6," which meant `VisualizePredictionStep` before
      round 2's Section 6/7 restructuring moved the step to Section 7 and made Section 6 the
      figure-content module instead, which does not import `joblib` — corrected during round 3).
      Adding the dependency after that import would exist is backwards.

## 6. Figure content: `visualize_prediction.py` module (test-first)

> **Moved ahead of the step that consumes it** (found during `/review-openspec` round 2: the
> step's own tests mock/call `create_prediction_figure`, which didn't exist yet when this section
> was numbered after the step — a real sequencing bug, not just a stylistic reordering).

- [ ] 6.1 Write failing test `test_create_prediction_figure_scatter_panel_uses_pc1_target_only`:
      given multiple targets' data, the obs-vs-pred scatter panel's plotted points correspond only
      to the `PC1` target's `y_true`/`y_pred`.
- [ ] 6.2 Write failing test `test_create_prediction_figure_violin_panel_pools_all_targets_nulls`:
      the violin/strip panel's null data is the concatenation of every target's `null_r2`, and its
      observed-points data is every target's `observed_r2` (one point per target).
- [ ] 6.3 Write failing test `test_create_prediction_figure_bar_chart_shows_observed_vs_null_mean`:
      the two bars' heights equal the mean observed and mean null top-quartile-recovery across all
      targets.
- [ ] 6.4 Write failing test `test_create_prediction_figure_returns_a_figure_with_three_axes`.
- [ ] 6.4a Write failing test `test_create_prediction_figure_handles_single_target`: given only one
      target's data (e.g. the PC1-only degenerate case Section 7 also handles — found during
      `/review-openspec` round 3: task 7c.7 will call this function for that same fixture, and
      nothing previously verified the violin/bar-chart panels degrade gracefully to `N=1` target
      rather than erroring on a single-element distribution), assert the figure is still built
      successfully with all 3 panels present, not a crash.
- [ ] 6.5 Implement `create_prediction_figure(...)` (and any supporting per-panel helper functions)
      in new `src/sleap_roots_analyze/visualize_prediction.py`, following
      `cross_experiment_analysis.py`'s plotting-function convention (pure functions returning a
      `matplotlib.Figure`, no file I/O). Make 6.1-6.4a green.

## 7. `VisualizePredictionStep` (test-first)

> Restructured during `/review-openspec` round 2: round 1's "3 test-group commits, 1 implementation
> commit at the end" pattern doesn't achieve real atomicity — each test-only commit would leave
> `uv run pytest` red until the final implementation commit lands (verified against Tier 3.5's own
> real commit history, which never split tests from implementation this way). Restructured into 3
> genuine red→green pairs, each landing its own working implementation increment.

**7a. Wiring and predictor-matrix reuse (red→green pair 1)**

> **Mocking-across-process-boundary note (found during `/review-openspec` round 3):** every test
> below that mocks/spies on `permutation_test` or a BLUP-loading/aggregation function MUST fix
> `config.prediction.permutation_n_jobs=1` in its fixture. `joblib.Parallel(n_jobs=1)` runs
> sequentially in-process (never touching the `loky` backend), so a `unittest.mock.patch` in the
> test process stays valid; at `n_jobs>1`, `loky` dispatches to separate worker processes where a
> parent-process mock is invisible (the worker calls the real, unmocked function, or a spy
> implemented as a non-picklable `Mock` raises `PicklingError`) — either way the test would stop
> testing what it claims, or break outright, the moment real parallelization is exercised. This
> applies to 7a.1, 7a.3, 7a.4, and 7b.4 below; 7b.1/7b.2 avoid the issue by inspecting the
> `joblib.Parallel`/`delayed` construction itself rather than mocking `permutation_test`.

- [ ] 7a.1 Write failing test `test_visualize_prediction_step_reuses_task6_predictor_matrices`
      (`permutation_n_jobs=1`, see note above): spy on any BLUP-loading/genotype-mean-aggregation
      function to confirm zero calls when `predictor_matrices` is supplied via
      `kwargs["06_predict_cross_platform"]`.
- [ ] 7a.2 Write failing test `test_visualize_prediction_step_handles_pc1_only_targets`: with zero
      representative-trait targets (only `target_name="PC1"` present, e.g. a fixture where
      `select_cluster_representatives` returned empty — Tier 3.5's own documented degenerate case),
      the step's target/method enumeration produces exactly `N=1` unit per method, not a crash.
      This test only exercises the pre-`joblib` serial path built in 7a.5 — task 7b.7 below
      re-verifies this same PC1-only fixture through the real `joblib.Parallel` dispatch added in
      7b.6, since the "Step still runs with only the PC1 target" scenario explicitly requires the
      degenerate case to work when *dispatched through `joblib.Parallel`*, not merely when called
      serially.
- [ ] 7a.3 Write failing test
      `test_visualize_prediction_step_calls_permutation_test_once_per_target_per_method`
      (`permutation_n_jobs=1`, see note above): for `N` targets × `M` methods (`reduction_method` +
      `comparison_methods`), `permutation_test` is called exactly `N * M` times.
- [ ] 7a.4 Write failing test
      `test_visualize_prediction_step_calls_permutation_test_n_times_when_comparison_methods_empty`
      (`permutation_n_jobs=1`, see note above): with `comparison_methods=[]` (`K=0`),
      `permutation_test` is called exactly `N` times (`N` targets × 1 method) — an explicit `K=0`
      case, not just the general `N * M` formula in 7a.3.
- [ ] 7a.5 Implement a minimal `VisualizePredictionStep(BaseStep)` in new
      `src/sleap_roots_analyze/pipeline/steps/visualize_prediction.py`: reads `predictor_matrices`
      and task 6's results, enumerates `(target_name, method)` combinations in the canonical order
      (methods first, `[reduction_method] + comparison_methods`; then `target_names` in task 6's
      `CrossPlatformPredictionResult.predictions` order — representative traits, then `"PC1"`
      last), and calls `permutation_test()` serially (no `joblib` yet) for each. Make 7a.1-7a.4
      green.

**7b. `joblib` parallelization across targets (red→green pair 2 — the riskiest, most novel piece)**

- [ ] 7b.1 Write failing test
      `test_visualize_prediction_step_parallelizes_across_target_method_units_not_within_one`:
      inspect the `joblib.Parallel`/`delayed` call structure (or the list of dispatched callables)
      to confirm one `delayed(...)` unit per `(target, method)` combination, not per permutation
      iteration.
- [ ] 7b.2 Write failing test
      `test_visualize_prediction_step_joblib_n_jobs_and_backend_match_config`:
      `joblib.Parallel` is constructed with `n_jobs=config.prediction.permutation_n_jobs,
      backend="loky"` (per the cross-platform-analysis spec's explicit backend choice).
- [ ] 7b.3 Write failing test
      `test_visualize_prediction_step_derives_independent_seed_per_target_method`: for `N`
      `(target, method)` combinations enumerated in the canonical order (7a.5), assert
      `numpy.random.SeedSequence(config.prediction.permutation_random_state).spawn(N)` derives `N`
      distinct seeds, the `i`-th child assigned to the `i`-th combination in that order — no two
      combinations receive the same seed, and re-running with the same `permutation_random_state`
      and the same combinations reproduces the same derived seeds.
- [ ] 7b.4 Write failing test `test_visualize_prediction_step_permutation_test_receives_derived_seed`
      (`permutation_n_jobs=1`, see the note at the top of Section 7a): spy on `permutation_test` to
      confirm each call's `random_state` argument is that combination's derived `SeedSequence`
      child from 7b.3 (passed through unchanged — no int-extraction step, per `default_rng`'s
      uniform acceptance of `SeedSequence`), not the raw
      `config.prediction.permutation_random_state`.
- [ ] 7b.5 Write failing test
      `test_visualize_prediction_step_parallel_vs_serial_results_agree_within_tolerance`: on a
      small fixture with an explicitly small, stated `n_permutations` (e.g. 50, to bound the
      elementwise-comparison surface and keep this test CI-fast), `permutation_n_jobs=1` and
      `permutation_n_jobs=4` produce `CrossPlatformPermutationResult`s that agree via
      `numpy.testing.assert_allclose(..., rtol=1e-6, atol=1e-9)` for every numeric field, including
      every element of every null distribution — **not bit-identical** (loky worker processes may
      resolve a different default BLAS thread count than the main process, a documented source of
      ULP-level floating-point differences per `docs/reproducibility.md`'s established cross-BLAS
      tolerance convention, independent of this step's own correctness). This test calls the real
      `permutation_test` (no mocking) at both `n_jobs` settings, so it is unaffected by the
      mocking-across-process-boundary note above.
- [ ] 7b.6 Extend `VisualizePredictionStep` to dispatch the 7a.5 enumeration through
      `joblib.Parallel(n_jobs=config.prediction.permutation_n_jobs, backend="loky")`, deriving
      per-combination seeds via `SeedSequence.spawn(N)` (7b.3). Make 7b.1-7b.5 green.
- [ ] 7b.7 Write failing test `test_visualize_prediction_step_handles_pc1_only_targets_via_joblib`:
      re-run 7a.2's PC1-only (zero representative-trait) fixture at `permutation_n_jobs=4` (real
      multi-process dispatch, no mocking), asserting the step still runs successfully through the
      full `joblib.Parallel` dispatch with `N=1` total unit per method, producing a valid
      `CrossPlatformPermutationResult` — not a crash. Required per the `cross-platform-analysis`
      spec's "Step still runs with only the PC1 target when zero representative traits are
      selected" scenario, which explicitly requires this to work when dispatched through
      `joblib.Parallel`, not merely when called serially (found during `/review-openspec` round 3:
      7a.2 alone only tests the pre-`joblib` serial path built in 7a.5, never re-verified after
      real parallelization is wired in by 7b.6). Make green (should already pass once 7b.6 lands;
      this is a regression/completeness check, not new production code).

**7c. JSON/figure output (red→green pair 3)**

- [ ] 7c.1 Write failing test `test_visualize_prediction_step_saves_one_json_per_method`: `K + 1`
      `07_permutation_<method>.json` files for `reduction_method` + `K` `comparison_methods`.
- [ ] 7c.2 Write failing test
      `test_visualize_prediction_step_saves_one_json_when_comparison_methods_empty`: with `K=0`,
      exactly 1 `07_permutation_<method>.json` file is saved, not `0` (a distinct check from
      7c.1's general formula, guarding the `K=0` edge specifically).
- [ ] 7c.3 Write failing test
      `test_visualize_prediction_step_permutation_observed_matches_task6_prediction_exactly`: for
      each target/method, the permutation JSON's `observed_r2`/`observed_rmse`/
      `observed_spearman_rho` exactly matches task 6's `06_prediction_<method>.json`'s `r2`/
      `rmse`/`spearman_rho` for the same target.
- [ ] 7c.4 Write failing test `test_visualize_prediction_step_saves_one_figure_using_primary_method_only`:
      exactly one `07_prediction_figure.png`, and (via a spy/mock on `create_prediction_figure`,
      Section 6) confirm it was called with only the primary `reduction_method`'s results, not
      `comparison_methods`'.
- [ ] 7c.5 Write failing test
      `test_visualize_prediction_step_rejects_non_finite_permutation_result_with_named_error`:
      when `permutation_test` (or its underlying `logo_cv_predict` calls) would produce a
      non-finite null value for one target/method, the step surfaces `ValueError` naming the
      target/method/permutation-index (propagated from `permutation_test`'s own guard, task
      2.13a), before attempting to write any `07_permutation_<method>.json`.
- [ ] 7c.6 Write failing test
      `test_visualize_prediction_step_writes_no_partial_json_files_when_any_combination_fails`:
      one `(target, method)` combination fails (per 7c.5) while every other combination for the
      pair would have individually succeeded — assert **zero** `07_permutation_<method>.json`
      files exist after the exception propagates, including for methods whose own combinations all
      succeeded (all-or-nothing per pair, not a partial write).
- [ ] 7c.7 Extend `VisualizePredictionStep` to save one `CrossPlatformPermutationResult` JSON per
      method (only after every combination for the pair has succeeded — 7c.6's all-or-nothing
      contract) and call `create_prediction_figure()` (Section 6) with the primary method's
      results only, saving `07_prediction_figure.png` with `dpi=300, bbox_inches="tight"` matching
      `visualize_cross_platform.py`'s convention. Make 7c.1-7c.6 green.

## 8. `CrossPlatformPipeline` task wiring (test-first)

- [ ] 8.1 Write failing test `test_cross_platform_pipeline_appends_visualize_prediction_task_when_visualize_enabled`:
      `create_tasks()` includes a 7th task, `depends_on=["06_predict_cross_platform"]`, when
      `config.prediction.visualize=True`.
- [ ] 8.2 Write failing test `test_cross_platform_pipeline_omits_visualize_prediction_task_when_disabled`:
      `create_tasks()` returns exactly 6 tasks when `config.prediction.visualize=False` (the
      default) — including when `config.prediction.enabled=True` (prediction alone, no
      visualization).
- [ ] 8.3 Add `_run_visualize_prediction` runner method + the conditional `Task(...)` entry to
      `CrossPlatformPipeline.create_tasks()`. Make 8.1-8.2 green.
- [ ] 8.4 Write failing test `test_cli_cross_platform_dry_run_lists_visualize_prediction_step_when_enabled`
      and `test_cli_cross_platform_dry_run_omits_it_when_disabled` (mirroring Tier 3.5 tasks 7.1-7.2).
- [ ] 8.5 Update `cli.py`'s dry-run steps list (conditional 7th entry) and docstring. Make 8.4 green.

## 9. Oracle tests (test-first, per design.md Decision 11)

> **CI-timeout note (found during `/review-openspec` round 2):** every oracle test below MUST
> state an explicit, small `n_permutations` for CI — at the production default (`1000`) across
> the N=20-seed fixture (9.2/9.3) or the 40-fixture set (9.1), several of these would individually
> cost minutes and collectively threaten the shared 30-minute CI job budget across 3 OSes. Only
> 9.1 stated this explicitly in an earlier draft; 9.2/9.3/9.4b now do too.
>
> **Estimated (not yet measured) total added CI time (found during `/review-openspec` round 3):**
> ~8 minutes serial, at ~20ms/`logo_cv_predict` call, `n_permutations=200`, across 9.1 (40 fixtures
> × 201 calls) + 9.2 + 9.3 + 9.4b (20-seed fixture, ×201 calls each). This is an estimate, not a
> measurement — task 9.6 MUST record the actual measured wall time for Section 9's test suite (per
> OS, since Windows runners are typically slower for process/import-heavy work) and, if it
> meaningfully threatens the shared 30-minute budget once Sections 2-8's other new tests and 7b.5's
> real multi-process `loky` test are also counted, reduce `n_permutations` further and/or the
> fixture count, rather than assuming the estimate above holds.

- [ ] 9.1 Write failing test `test_permutation_test_p_values_are_uniform_under_null` (K-S
      calibration oracle): run `permutation_test()` on all 40 pure-noise fixtures (task 1.2) with
      `n_permutations=200` (explicit, CI-fast — distinct from the `n_permutations=1000` production
      default), collect the resulting `p_value_r2`s, K-S-test against `Uniform(0,1)`, assert
      `p > 0.05`.
- [ ] 9.1a Write failing test `test_permutation_test_p_value_rmse_is_uniform_under_null`: the same
      K-S calibration procedure as 9.1, but for `p_value_rmse` — a separate, explicit check that
      the RMSE-specific left-tail p-value formula (per the Permutation Test requirement's
      per-metric-direction scenario) is itself correctly calibrated, not just correctly directed
      (found during `/review-openspec` round 3: no oracle anywhere previously checked
      `p_value_rmse`'s calibration or direction — only `p_value_r2` was covered by 9.1).
- [ ] 9.2 Write failing test `test_permutation_test_signal_r2_exceeds_its_own_null_median`: on the
      N=20-seed planted-signal fixture (task 1.1), with `n_permutations=200` (explicit, CI-fast),
      the mean observed R² across seeds is comfortably above the mean permutation-null median
      across the same seeds.
- [ ] 9.2a Write failing test `test_permutation_test_signal_p_value_rmse_reads_as_significant`: on
      the same planted-signal fixture, assert `p_value_rmse` is small (comfortably below `0.5`),
      proving the RMSE-specific left-tail formula is actually in effect for a genuinely good
      (low-RMSE) result — direct regression test for the "A good (low-RMSE) result does not read
      as non-significant" scenario, guarding against the exact directional-inversion bug found
      during `/review-openspec` round 3 (the naive R²/ρ-style right-tail formula would instead
      produce `p_value_rmse ≈ 1.0` here, reading as non-significant).
- [ ] 9.3 Write failing test `test_permutation_test_noise_r2_falls_within_its_own_null_band`: on the
      pure-noise fixture (task 1.1), with `n_permutations=200` (explicit, CI-fast), mean observed
      R² falls within mean-null-median ± 1σ.
- [ ] 9.4a **Spike, not a test** — compute (via a real run, recorded here and in design.md's
      Decision 11 note): the pure-noise fixture's actual mean null top-quartile-recovery value at
      `n_permutations=200`. Cross-check against the theoretical chance-level baseline `2q / n`
      (found during `/review-openspec` round 1: at `n=19, q=5`, `2q/n ≈ 52.6%`, not the roadmap's
      originally-estimated "≈25%" — the roadmap's number was never verified against the actual
      `top-q`-in-`top-2q` window definition; independently re-derived and confirmed during round 2
      via the hypergeometric mean). This step produces a number to write a real assertion against
      in 9.4b; it is not itself a red/green TDD step (a value that doesn't exist yet cannot be
      asserted against).
- [ ] 9.4b Write failing test `test_permutation_test_top_quartile_recovery_signal_vs_noise`, using
      9.4a's now-known empirical value (`n_permutations=200`, explicit, CI-fast): signal fixture's
      observed recovery ≥ 80%; noise fixture's observed recovery is comfortably separated from the
      signal's, close to the empirically-determined null value (not a blindly-pinned 25% or an
      untested theoretical `2q/n`).
- [ ] 9.5 Write failing test `test_visualize_prediction_step_figure_provenance`: run the step
      twice (different input CSV content between runs, e.g. a perturbed fixture), assert the two
      resulting `07_prediction_figure.png` files differ (via content hash), and that a given run's
      figure's mtime is at or after its input CSVs' mtimes.
- [ ] 9.6 Implement whatever `permutation_test()`/fixture adjustments are needed to make 9.1, 9.1a,
      9.2, 9.2a, 9.3, and 9.4b-9.5 pass, recording the actual empirically-determined values from
      9.4a in this file and in `design.md`'s Decision 11 note (per this program's "verify, don't
      assume" convention), and measuring and recording Section 9's actual total wall time per OS
      (per the CI-timeout note above) here.

## 10. `theory.md` addendum

> `joblib` dependency addition lives in Section 5a (must precede Section 7b, which imports it).

- [ ] 10.1 Add a permutation-null pseudo-code section to
      `c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\theory.md` (external
      vault), matching its existing LOGO-CV/per-fold-PCA pseudo-code style, including the
      per-target (not per-permutation) `joblib.Parallel` lesson from this tier's benchmark
      (Decision 4) as a documented erratum/addendum for any future tier that loops
      `logo_cv_predict`. **This file lives in a separate external repository/git history from
      `sleap-roots-analyze`** — it is committed separately, in the vault's own repo, and will NOT
      appear in this repo's PR diff; a reviewer of the sleap-roots-analyze PR should not expect to
      see this change there. No pointer to this addendum exists anywhere in this repo (confirmed
      during `/review-openspec` round 2: it would be discoverable only via archived OpenSpec
      history or vault access) — accepted as a known, non-blocking gap, not fixed in this tier.

## 11. Manual real-data validation gate (non-CI, pre-merge, sign-off required)

> Mirrors Tier 3/3.5's Section 8 exactly — see design.md Section 5 for full rationale. Not
> complete until Elizabeth has reviewed the findings and explicitly signed off; a green CI run is
> necessary but not sufficient. **Timing, corrected during `/review-openspec` round 2** (round 1's
> claim that validation ran "between two rounds of review" was checked against Tier 3.5's actual
> commit timestamps and found factually wrong): Tier 3.5's real history shows validation landing
> **after both** 5-subagent review rounds, with a further CI-driven fix landing even after
> Elizabeth's sign-off. This section is therefore a late-stage gate that can trail every review
> round in `/pre-merge-check` (task 13.7), not a step reliably sandwiched between two of them.

- [ ] 11.1 Reuse Tier 3.5's Section 8 real BLUP tables and 4 directed-pair `CrossPlatformConfig`
      YAMLs (`pipeline_runs/section8_manual_validation_20260716/`) if still valid; rebuild via the
      same non-committed script pattern if the underlying QC vintage has moved. Extend each YAML's
      `prediction:` block with `visualize: true`, `n_permutations: 1000`, `permutation_n_jobs: 8`.
- [ ] 11.2 Run all 4 directed pairs through the full 7-task pipeline, including
      `Turface19→Cylinder` (worst-case pair, ~129 representative traits).
- [ ] 11.3 Re-measure and record real wall time, per pair and total — superseding the synthetic-
      fixture-derived 27.4-minute estimate (design.md Decision 4/Risks). If any pair or the total
      exceeds 30 minutes, apply a documented fallback (design.md Risks) before proceeding to 11.7.
- [ ] 11.4 Sanity-check permutation-based p-values against Tier 3.5's Section 8.2/8.3 nominally-
      significant findings (e.g. Cylinder→Field `Root Count 20cm`, asymptotic R²=0.25/ρ=+0.49/
      p=0.033; Turface19→Cylinder `Seminal Angles Proximal Max Max`, asymptotic R²=0.38/ρ=+0.62/
      p=0.004) — record whether these still look significant under the permutation-based p-value.
- [ ] 11.5 Cross-check against Tier 3.5's Section 8.5 multiple-testing/power caveat (raw p<0.05
      count 63/354, FDR-corrected count 9/354, all 9 negative ρ) — record whether permutation-based
      inference reinforces or changes that caveat's conclusion.
- [ ] 11.6 Visually inspect all 4 `07_prediction_figure.png` outputs for legibility and correctness.
- [ ] 11.7 Present findings to Elizabeth; record her explicit sign-off here before this task (and
      Section 13's `/pre-merge-check`) is considered complete.

## 12. Docs

- [ ] 12.1 Add a `docs/CHANGELOG.md` `[Unreleased]` `### Added` entry.
- [ ] 12.2 Extend `docs/CROSS_PLATFORM_ANALYSIS.md`'s existing `## Cross-Platform Genotype-Effect
      Prediction` section (Tier 3.5 already extended it once) with a new `###` subsection covering
      `permutation_test()` and `top_quartile_recovery()`, `VisualizePredictionStep`, and **all 4**
      new `PredictionConfig` fields (`visualize`, `n_permutations`, `permutation_random_state`,
      `permutation_n_jobs` — found during `/review-openspec` round 2 that an earlier draft of this
      task risked an incomplete YAML example naming only `visualize: true`) shown together in one
      concrete YAML example under the existing `prediction:` block. State usage (what/how) only,
      but **do not cross-reference `design.md` for key numbers** (found during round 2: `design.md`
      moves to `openspec/changes/archive/<change-id>/design.md` on archival, so a shipped-doc
      pointer to it goes stale/dangling the moment this change archives — a documentation
      anti-pattern, not a DRY win). Instead, state directly in this new subsection:
      - The one-line parallelization headline (e.g. "full per-representative-trait permutation
        nulls take ~27 minutes worst-case across all 4 pairs via `joblib.Parallel` across
        targets, verified against real EDPIE data — see Section 11's findings for the actual
        measured number").
      - The `2q/n` chance-level baseline for top-quartile recovery (e.g. "chance-level recovery is
        `2q/n`, not 25% — ≈52.6% at n=19, q=5" — found during round 2 that this math previously
        existed only in the OpenSpec proposal, not any shipped doc a user would actually see).
      Also:
      - **Correct the existing section's stale closing sentence** (found during round 1: it
        currently reads "The permutation null and its figures (Tier 4) remain a separate, later
        change" — stale now that this change *is* Tier 4).
      - **Document the new output-file naming convention** (`07_permutation_<method>.json`,
        `07_prediction_figure.png`) in an "Output:" paragraph, mirroring the existing section's own
        precedent for Tier 3.5's `06_prediction_<method>.json`.
      - **Extend the existing `### Current Limitations` subsection's #197 bullet in place** (found
        during round 2: a separate new bullet would read as a near-duplicate) to also note that
        `CrossPlatformSummaryGenerator` doesn't surface permutation/visualization output either.
- [ ] 12.3 **No `docs/API.md` entry.** Verified directly (found during `/review-openspec` round 1
      that the original task's premise was backwards): `LOGOCVResult`/`CrossPlatformPredictionResult`/
      `TargetPrediction` are all in `__all__`, but **none** has an `API.md` entry — the
      `cross_platform_prediction` module section documents only its two functions
      (`fit_pca_on_fold`, `logo_cv_predict`); `__all__` membership does not predict `API.md`
      inclusion for these dataclasses. `PermutationResult`/`CrossPlatformPermutationResult` follow
      that same (undocumented-dataclass) precedent — no `API.md` change needed. `permutation_test`/
      `top_quartile_recovery` DO get entries in the `cross_platform_prediction` module section,
      matching the existing `fit_pca_on_fold`/`logo_cv_predict` entries' heading/signature/prose
      format exactly.

## 13. Validation

- [ ] 13.1 `openspec validate add-prediction-permutation-and-figure --strict` — resolve every
      reported issue.
- [ ] 13.2 `/lint` (black + ruff) on all changed files.
- [ ] 13.3 Full `uv run pytest --cov --cov-branch` — no regressions, all new tests (Sections
      2-9) green.
- [ ] 13.4 `/review-openspec` — adversarial proposal review, budget for multiple independent
      rounds (this program's established pattern: BLOCKING findings diminishing round over round —
      NOT yet the case after round 2 here, which found 2 new BLOCKING + several IMPORTANT; a
      further round is warranted before this is a natural stopping point). Reconcile literally
      into `design.md`.
- [ ] 13.5 Wait for Elizabeth's explicit approval of the fully-reconciled proposal before starting
      implementation (Sections 1-10).
- [ ] 13.6 Section 11's manual real-data validation gate, complete with sign-off.
- [ ] 13.7 Post-implementation code review: `/pre-merge-check` including a 5-subagent `/review-pr`
      self-review pass before the PR opens, and a second fresh pass once CI runs on the open PR
      (mirroring Tier 3/3.5's precedent — both passes have caught real bugs every tier so far).
