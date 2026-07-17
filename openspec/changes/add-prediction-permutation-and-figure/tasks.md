> Full rationale for every design choice referenced below (self-contained `permutation_test()`,
> per-target (not per-permutation) `joblib.Parallel` strategy and its empirical basis, separate
> `CrossPlatformPermutationResult` type, additive `PredictCrossPlatformStep` extension, config
> field placement, figure scope) is in `design.md`'s Decisions section and the brainstormed design
> at `docs/superpowers/specs/2026-07-16-prediction-permutation-and-figure-design.md`.

## 1. Fixtures (test-first)

- [ ] 1.1 Reuse Tier 3's N=20-seed-averaged planted-signal fixture (`n_genotypes=19, n_traits=3,
      signal_strength=0.8`, seeds `1..20`) and its pure-noise counterpart, per Tier 3 Decision 6 —
      do not re-derive from `theory.md`'s single-seed recipe. Confirm these fixtures (or a thin
      wrapper reusing their generator function) are importable from this tier's test module.
- [ ] 1.2 Add a small set of independent pure-noise fixtures (real, non-degenerate `X`; `y`
      independently drawn, no planted relationship) for the K-S permutation-calibration oracle —
      ~30-50 independent realizations, each with a distinct seed, `n_genotypes=19` (matches the
      other fixtures' scale).
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
- [ ] 2.4 Implement `top_quartile_recovery(y_true, y_pred, q=None)` in
      `cross_platform_prediction.py`. Make 2.1-2.3 green.
- [ ] 2.5 Write failing test `test_permutation_test_observed_matches_direct_logo_cv_predict_call`:
      `permutation_test(X, y, genotypes, method).observed_r2` (etc.) exactly matches an
      independent `logo_cv_predict(X, y, genotypes, method)` call's `r2` (etc.) on the same inputs.
- [ ] 2.6 Write failing test `test_permutation_test_null_distributions_have_length_n_permutations`:
      `null_r2`/`null_rmse`/`null_spearman_rho`/`null_top_quartile_recovery` each have length `N`
      for `n_permutations=N`.
- [ ] 2.7 Write failing test `test_permutation_test_shuffles_y_not_x_or_genotypes`: spy on
      `logo_cv_predict` (or inspect its call arguments) to confirm each permutation iteration's `X`
      and `genotypes` arguments are unchanged from the original inputs, only `y` differs.
- [ ] 2.8 Write failing test `test_permutation_test_deterministic_given_same_random_state`: two
      calls with identical arguments (including `random_state`) produce bit-identical null arrays.
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
- [ ] 2.13 Write failing test `test_permutation_test_surfaces_logo_cv_predict_validation_errors`:
      an invalid `reduction_method` (or mismatched-length `X`/`y`/`genotypes`, or duplicate
      `genotypes`) raises the same `ValueError` `logo_cv_predict` itself would raise, from the
      observed-value call, before any permutation runs.
- [ ] 2.14 Implement `permutation_test(X, y, genotypes, reduction_method="pls_latent",
      representative_names=None, n_permutations=1000, random_state=42)` in
      `cross_platform_prediction.py`, returning a `PermutationResult` (Section 3). Make 2.5-2.13
      green.

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
- [ ] 3.4 Implement `PermutationResult`/`CrossPlatformPermutationResult` in `result_types.py`,
      mirroring `TargetPrediction`/`CrossPlatformPredictionResult`'s `to_dict()`/`to_json()`
      pattern exactly. Make 3.1-3.3 green.
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
- [ ] 4.3 Write failing test `test_prediction_config_n_permutations_validation_skipped_when_visualize_false`:
      `enabled=True, visualize=False, n_permutations=0` does not raise.
- [ ] 4.4 Write failing test `test_prediction_config_rejects_non_positive_n_permutations_when_visualize_true`:
      `enabled=True, visualize=True, n_permutations=0` (and `-1`) raises `ValueError`.
- [ ] 4.5 Extend `PredictionConfig`/`CrossPlatformConfig.__post_init__` in
      `pipeline/config/components.py` with the 4 new fields and the 4.2/4.4 validations. Make
      4.1-4.4 green.

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
- [ ] 5.3 Extend `PredictCrossPlatformStep.execute()` in `predict_cross_platform.py` to populate
      `predictor_matrices`. Make 5.1-5.2 green.

## 6. `VisualizePredictionStep` (test-first)

- [ ] 6.1 Write failing test `test_visualize_prediction_step_reuses_task6_predictor_matrices`: spy
      on any BLUP-loading/genotype-mean-aggregation function to confirm zero calls when
      `predictor_matrices` is supplied via `kwargs["06_predict_cross_platform"]`.
- [ ] 6.2 Write failing test
      `test_visualize_prediction_step_calls_permutation_test_once_per_target_per_method`: for `N`
      targets × `M` methods (`reduction_method` + `comparison_methods`), `permutation_test` is
      called exactly `N * M` times.
- [ ] 6.3 Write failing test
      `test_visualize_prediction_step_parallelizes_across_target_method_units_not_within_one`:
      inspect the `joblib.Parallel`/`delayed` call structure (or the list of dispatched callables)
      to confirm one `delayed(...)` unit per `(target, method)` combination, not per permutation
      iteration.
- [ ] 6.4 Write failing test
      `test_visualize_prediction_step_joblib_n_jobs_matches_config`:
      `joblib.Parallel` is constructed with `n_jobs=config.prediction.permutation_n_jobs`.
- [ ] 6.5 Write failing test
      `test_visualize_prediction_step_produces_identical_results_parallel_vs_serial`: on a small
      fixture, `permutation_n_jobs=1` and `permutation_n_jobs=4` produce bit-identical
      `CrossPlatformPermutationResult`s (parallelization must not change results, only wall time).
- [ ] 6.6 Write failing test `test_visualize_prediction_step_saves_one_json_per_method`: `K + 1`
      `07_permutation_<method>.json` files for `reduction_method` + `K` `comparison_methods`.
- [ ] 6.7 Write failing test
      `test_visualize_prediction_step_permutation_observed_matches_task6_prediction_exactly`: for
      each target/method, the permutation JSON's `observed_r2`/`observed_rmse`/
      `observed_spearman_rho` exactly matches task 6's `06_prediction_<method>.json`'s `r2`/
      `rmse`/`spearman_rho` for the same target.
- [ ] 6.8 Write failing test `test_visualize_prediction_step_saves_one_figure_using_primary_method_only`:
      exactly one `07_prediction_figure.png`, and (via a spy/mock on the figure-building function)
      confirm it was called with only the primary `reduction_method`'s results, not
      `comparison_methods`'.
- [ ] 6.9 Implement `VisualizePredictionStep(BaseStep)` in new
      `src/sleap_roots_analyze/pipeline/steps/visualize_prediction.py`. Make 6.1-6.8 green.

## 7. `CrossPlatformPipeline` task wiring (test-first)

- [ ] 7.1 Write failing test `test_cross_platform_pipeline_appends_visualize_prediction_task_when_visualize_enabled`:
      `create_tasks()` includes a 7th task, `depends_on=["06_predict_cross_platform"]`, when
      `config.prediction.visualize=True`.
- [ ] 7.2 Write failing test `test_cross_platform_pipeline_omits_visualize_prediction_task_when_disabled`:
      `create_tasks()` returns exactly 6 tasks when `config.prediction.visualize=False` (the
      default) — including when `config.prediction.enabled=True` (prediction alone, no
      visualization).
- [ ] 7.3 Add `_run_visualize_prediction` runner method + the conditional `Task(...)` entry to
      `CrossPlatformPipeline.create_tasks()`. Make 7.1-7.2 green.
- [ ] 7.4 Write failing test `test_cli_cross_platform_dry_run_lists_visualize_prediction_step_when_enabled`
      and `test_cli_cross_platform_dry_run_omits_it_when_disabled` (mirroring Tier 3.5 tasks 7.1-7.2).
- [ ] 7.5 Update `cli.py`'s dry-run steps list (conditional 7th entry) and docstring. Make 7.4 green.

## 8. Figure content: `visualize_prediction.py` module (test-first)

- [ ] 8.1 Write failing test `test_create_prediction_figure_scatter_panel_uses_pc1_target_only`:
      given multiple targets' data, the obs-vs-pred scatter panel's plotted points correspond only
      to the `PC1` target's `y_true`/`y_pred`.
- [ ] 8.2 Write failing test `test_create_prediction_figure_violin_panel_pools_all_targets_nulls`:
      the violin/strip panel's null data is the concatenation of every target's `null_r2`, and its
      observed-points data is every target's `observed_r2` (one point per target).
- [ ] 8.3 Write failing test `test_create_prediction_figure_bar_chart_shows_observed_vs_null_mean`:
      the two bars' heights equal the mean observed and mean null top-quartile-recovery across all
      targets.
- [ ] 8.4 Write failing test `test_create_prediction_figure_returns_a_figure_with_three_axes`.
- [ ] 8.5 Implement `create_prediction_figure(...)` (and any supporting per-panel helper functions)
      in new `src/sleap_roots_analyze/visualize_prediction.py`, following
      `cross_experiment_analysis.py`'s plotting-function convention (pure functions returning a
      `matplotlib.Figure`, no file I/O). Make 8.1-8.4 green.
- [ ] 8.6 Wire `create_prediction_figure()` into `VisualizePredictionStep` (Section 6), saving with
      `dpi=300, bbox_inches="tight"` matching `visualize_cross_platform.py`'s convention.

## 9. Oracle tests (test-first, per design.md Decision 11)

- [ ] 9.1 Write failing test `test_permutation_test_p_values_are_uniform_under_null` (K-S
      calibration oracle): run `permutation_test()` on ~30-50 independent pure-noise fixtures
      (task 1.2) with a reduced `n_permutations` (e.g. 200, CI-fast), collect the resulting
      `p_value_r2`s, K-S-test against `Uniform(0,1)`, assert `p > 0.05`.
- [ ] 9.2 Write failing test `test_permutation_test_signal_r2_exceeds_its_own_null_median`: on the
      N=20-seed planted-signal fixture (task 1.1), the mean observed R² across seeds is
      comfortably above the mean permutation-null median across the same seeds.
- [ ] 9.3 Write failing test `test_permutation_test_noise_r2_falls_within_its_own_null_band`: on the
      pure-noise fixture (task 1.1), mean observed R² falls within mean-null-median ± 1σ.
- [ ] 9.4 **Empirically determine** (via a real computation, run and recorded during
      implementation — not assumed from the roadmap's "≈25%" estimate) the pure-noise fixture's
      actual mean null top-quartile-recovery value. Write failing test
      `test_permutation_test_top_quartile_recovery_signal_vs_noise`: signal fixture's observed
      recovery ≥ 80%; noise fixture's observed recovery is comfortably separated from the
      signal's, near the empirically-determined null value (not a blindly-pinned 25%).
- [ ] 9.5 Write failing test `test_visualize_prediction_step_figure_provenance`: run the step
      twice (different input CSV content between runs, e.g. a perturbed fixture), assert the two
      resulting `07_prediction_figure.png` files differ (via content hash), and that a given run's
      figure's mtime is at or after its input CSVs' mtimes.
- [ ] 9.6 Implement whatever `permutation_test()`/fixture adjustments are needed to make 9.1-9.5
      pass, recording the actual empirically-determined values from 9.4 in this file and in
      `design.md`'s Decision 11 note (per this program's "verify, don't assume" convention).

## 10. Dependency + `theory.md` addendum

- [ ] 10.1 Add `joblib` to `pyproject.toml`'s direct dependencies (Decision 5), pinned to the
      version already resolved transitively via `scikit-learn` in this environment's lockfile.
- [ ] 10.2 Add a permutation-null pseudo-code section to
      `c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\theory.md` (external
      vault), matching its existing LOGO-CV/per-fold-PCA pseudo-code style, including the
      per-target (not per-permutation) `joblib.Parallel` lesson from this tier's benchmark
      (Decision 4) as a documented erratum/addendum for any future tier that loops
      `logo_cv_predict`.

## 11. Manual real-data validation gate (non-CI, pre-merge, sign-off required)

> Mirrors Tier 3/3.5's Section 8 exactly — see design.md Section 5 for full rationale. Not
> complete until Elizabeth has reviewed the findings and explicitly signed off; a green CI run is
> necessary but not sufficient.

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
      Section 12's `/pre-merge-check`) is considered complete.

## 12. Docs

- [ ] 12.1 Add a `docs/CHANGELOG.md` `[Unreleased]` `### Added` entry.
- [ ] 12.2 Extend `docs/CROSS_PLATFORM_ANALYSIS.md`'s existing `## Cross-Platform Genotype-Effect
      Prediction` section (Tier 3.5 already extended it once) with a new `###` subsection covering
      `permutation_test()`, `VisualizePredictionStep`, the new `PredictionConfig` fields, and a
      concrete YAML example with `visualize: true`.
- [ ] 12.3 No `docs/API.md` entry — `PermutationResult`/`CrossPlatformPermutationResult` follow
      the same `__all__`-driven pattern as their siblings and DO get an entry there (unlike
      Configs/Steps, per Tier 3.5 task 9.1's finding); confirm and add if the existing
      `CrossPlatformPredictionResult`/`TargetPrediction` entries set that precedent.

## 13. Validation

- [ ] 13.1 `openspec validate add-prediction-permutation-and-figure --strict` — resolve every
      reported issue.
- [ ] 13.2 `/lint` (black + ruff) on all changed files.
- [ ] 13.3 Full `uv run pytest --cov --cov-branch` — no regressions, all new tests (Sections 2-9)
      green.
- [ ] 13.4 `/review-openspec` — adversarial proposal review, budget for multiple independent
      rounds (this program's established pattern: 3-5 BLOCKING findings round 1, diminishing each
      round). Reconcile literally into `design.md`.
- [ ] 13.5 Wait for Elizabeth's explicit approval of the fully-reconciled proposal before starting
      implementation (Sections 1-10).
- [ ] 13.6 Section 11's manual real-data validation gate, complete with sign-off.
- [ ] 13.7 Post-implementation code review: `/pre-merge-check` including a 5-subagent `/review-pr`
      self-review pass before the PR opens, and a second fresh pass once CI runs on the open PR
      (mirroring Tier 3/3.5's precedent — both passes have caught real bugs every tier so far).
