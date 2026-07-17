## Why

Tier 3 (`add-cross-platform-prediction`, merged #195) shipped `logo_cv_predict()`,
`fit_pca_on_fold()`, and `CrossPlatformPredictionResult` as plain, stateless Python-API functions
— deliberately written (Tier 3 Decision 3) so a future permutation loop could wrap them without
refactoring. Tier 3.5 (`add-prediction-pipeline-step`, merged #199) wired the *observed*
(non-permuted) prediction into `CrossPlatformPipeline` as an optional `PredictCrossPlatformStep`.

Neither tier answers the question Wolfgang's original ask (roadmap.md Goal section) actually
requires: is a given cross-platform R² distinguishable from chance at n≈19 genotypes? Tier 3's own
Decision 9 documents `spearman_p`'s asymptotic p-value as unreliable below n≈20-30 — exactly the
gap this tier closes with an empirical, shuffled-label permutation null. Neither tier produces a
paper-ready figure either. This is Tier 4 (`add-prediction-permutation-and-figure`) of the wheat
EDPIE cross-platform genotype-prediction program: a permutation-null significance test (Python API,
hard-depends on Tier 3 only) and `VisualizePredictionStep` (a new optional pipeline step,
soft-depends on Tier 3.5) that renders a 3-panel figure per directed pair.

Full design, alternatives considered, and empirical runtime benchmarks (this program's own
"verify, don't assume" convention) are recorded in
`docs/superpowers/specs/2026-07-16-prediction-permutation-and-figure-design.md`, brainstormed and
approved by Elizabeth before this proposal was drafted.

## What Changes

- **New `permutation_test()`** (`cross_platform_prediction.py`, Tier 3's existing stateless
  module — Python API only, no pipeline dependency): given `X`, `y`, `genotypes`,
  `reduction_method`, `representative_names`, `n_permutations=1000`, `random_state=42`, computes
  the observed R²/RMSE/Spearman ρ/top-quartile-recovery via one `logo_cv_predict()` call on the
  real `y`, then the same four metrics' null distributions via `n_permutations` shuffled-`y`
  `logo_cv_predict()` calls, plus one-sided p-values for R²/RMSE/Spearman ρ. Self-contained: a
  caller gets a complete result from one call.
- **New `top_quartile_recovery(y_true, y_pred, q=None)`** (same module): the roadmap's settled
  metric — fraction of the true top-`q` genotypes recovered in the predicted top-`2q`
  (`q` defaults to `round(n/4)`).
- **New `PermutationResult`/`CrossPlatformPermutationResult`** (`result_types.py`), mirroring the
  `LOGOCVResult`/`TargetPrediction`/`CrossPlatformPredictionResult` pattern exactly (frozen
  dataclasses, `to_dict()`/`to_json()` with the finite-floats contract). Kept **separate** from
  `CrossPlatformPredictionResult` (not nested inside it) — permutation is an optional, heavier
  concern orthogonal to the observed-result type.
- **Parallelization strategy, empirically verified this session** (not assumed): `joblib.Parallel`
  parallelizing across independent *targets* (each worker runs one target's own full, serial
  1000-permutation loop) — individual permutation calls are too fast (~16-27ms) for process-based
  parallelism to pay off, and measured *slower* than serial when parallelized directly. Worst-case
  real-EDPIE estimate (Cylinder-as-target pair, ~129 representative traits, both `reduction_method`
  + one `comparison_methods` entry, N=1000, all 4 pairs): ~105.5 min serial → ~27.4 min parallel
  (`n_jobs=8`) — under the roadmap's 30-minute feasibility gate. `joblib` becomes an explicit
  direct dependency in `pyproject.toml` (previously only transitive via `scikit-learn`).
- **Additive extension to `PredictCrossPlatformStep`** (Tier 3.5, already merged): `StepResult.data`
  gains a new `predictor_matrices` key holding the already-computed `source_clean`/`target_clean`
  matrices and `source_representative_names`/`target_representatives` — no existing `data`/
  `metadata`/`files_generated` key changes shape.
- **New `VisualizePredictionStep`**, an optional 7th task on `CrossPlatformPipeline`
  (`depends_on=["06_predict_cross_platform"]`, for both data and ordering), entirely absent from
  `create_tasks()` unless `config.prediction.visualize=True`. For every `(target, method)`
  combination, runs `permutation_test()` via the parallel strategy above, saves one
  `CrossPlatformPermutationResult` JSON per method (`07_permutation_<method>.json`), and builds one
  composite 3-panel figure per pair using only the primary `reduction_method`
  (`07_prediction_figure.png`): PC1 obs-vs-pred scatter, all-targets R²-vs-pooled-null
  violin/strip, and an aggregate top-quartile-recovery bar chart.
- **`PredictionConfig` gains 4 new fields** (nested, not a new sibling config — `visualize` is
  meaningless without `enabled`): `visualize: bool = False`, `n_permutations: int = 1000`,
  `permutation_random_state: int = 42`, `permutation_n_jobs: int = 8`.
  `CrossPlatformConfig.__post_init__` gains one more cross-check: `visualize=True` with
  `enabled=False` raises `ValueError` at construction time.
- **`theory.md` (Tier 0) gains a permutation-null pseudo-code addendum**, including the
  per-target (not per-permutation) parallelization lesson from this tier's benchmark — a Tier 0
  erratum contribution, matching the "carry lessons forward" convention already used for Tier 3's
  own single-seed-fixture correction.
- **Manual, non-CI, sign-off-gated real-EDPIE-data validation** (Section 5 of the design doc,
  mirroring Tier 3/3.5's own manual validation gate exactly): re-measures real wall time against
  the worst-case pair (superseding the synthetic-fixture-derived 27.4-minute estimate), sanity-
  checks permutation-based p-values against Tier 3.5's Section 8 findings, and requires Elizabeth's
  explicit sign-off before merge.

No changes to `logo_cv_predict`, `fit_pca_on_fold`, `CrossPlatformPredictionResult`, or any of
`PredictCrossPlatformStep`'s existing (Tier 3.5) behavior/outputs — all extensions are additive.

## Impact

### Affected specs

- `cross-platform-prediction` (ADDED) — `Permutation Test`, `Top-Quartile Recovery Metric`,
  `Permutation Test Input Validation` requirements (Python-API-only, Tier 3's module).
- `serializable-result-types` (ADDED) — `Serializable Cross-Platform Permutation Result Type`,
  `CrossPlatformPermutationResult Adapter From permutation_test Output`,
  `CrossPlatformPermutationResult Public Export` requirements.
- `cross-platform-analysis` (MODIFIED) — `Cross-Platform Prediction Configuration` gains the 4 new
  `PredictionConfig` fields and the `visualize`-requires-`enabled` cross-check;
  `Predict Cross-Platform Genotype Values Pipeline Step` gains the additive `predictor_matrices`
  exposure. (ADDED) — new `Visualize Cross-Platform Prediction Pipeline Step` requirement.

### Affected code

- `src/sleap_roots_analyze/cross_platform_prediction.py` — new `permutation_test()`,
  `top_quartile_recovery()`.
- `src/sleap_roots_analyze/result_types.py` — new `PermutationResult`,
  `CrossPlatformPermutationResult`.
- `src/sleap_roots_analyze/__init__.py` — new exports.
- `src/sleap_roots_analyze/pipeline/steps/predict_cross_platform.py` — additive
  `predictor_matrices` extension to `StepResult.data`.
- `src/sleap_roots_analyze/pipeline/steps/visualize_prediction.py` (new) —
  `VisualizePredictionStep`.
- `src/sleap_roots_analyze/visualize_prediction.py` (new) — `create_prediction_figure()` and
  supporting plotting functions.
- `src/sleap_roots_analyze/pipeline/config/components.py` — 4 new `PredictionConfig` fields;
  extended `CrossPlatformConfig.__post_init__` cross-check.
- `src/sleap_roots_analyze/pipeline/pipelines/cross_platform_pipeline.py` — conditional 7th task.
- `src/sleap_roots_analyze/cli.py` — dry-run step list gains a conditional 7th entry.
- `pyproject.toml` — `joblib` added as an explicit direct dependency.
- `tests/fixtures.py` / `tests/fixtures/` — new synthetic fixtures (planted-signal/noise reuse
  from Tier 3, pure-noise fixtures for the K-S calibration test).
- `tests/test_cross_platform_prediction.py` (extended), `tests/test_result_types.py` (extended),
  `tests/test_predict_cross_platform.py` (extended), `tests/test_visualize_prediction.py` (new),
  `tests/test_cross_platform_config.py` (extended).
- `docs/API.md`, `docs/CHANGELOG.md`, `docs/CROSS_PLATFORM_ANALYSIS.md` — new function/step/config
  entries.
- `c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\theory.md` (external vault) —
  permutation-null pseudo-code addendum.

### Explicitly out of scope

- Any change to `logo_cv_predict`, `fit_pca_on_fold`, `CrossPlatformPredictionResult`,
  `cluster_correlated_traits`, `select_cluster_representatives`, or `PredictCrossPlatformStep`'s
  existing (Tier 3.5) behavior/outputs — all reused/extended additively.
- `CrossPlatformSummaryGenerator`/`.claude/commands/cross-platform-summary.md` not surfacing
  prediction *or* permutation results — still follow-up
  [#197](https://github.com/talmolab/sleap-roots-analyze/issues/197).
- `/configure-run-all`, `/dry-run`, `/validate-config` cross-platform/prediction coverage gaps —
  still follow-up [#198](https://github.com/talmolab/sleap-roots-analyze/issues/198), pre-existing.
- Heritability-based `representative_selection_metric` — still deferred (Tier 3.5 Decision 7).
