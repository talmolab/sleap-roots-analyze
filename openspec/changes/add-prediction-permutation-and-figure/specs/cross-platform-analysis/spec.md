## MODIFIED Requirements

### Requirement: Cross-Platform Prediction Configuration

The system SHALL provide a `PredictionConfig` dataclass (nested as the `prediction` field on
`CrossPlatformConfig`) with the following parameters:

- `enabled`: bool, default `False`. When `False`, `__post_init__` SHALL perform no validation at
  all (not even structural checks on other fields), so every existing `CrossPlatformConfig` that
  predates this requirement remains valid unchanged.
- `predictor_source`: `"blup"` or `"genotype_means"`, default `"blup"`.
- `reduction_method`: the primary dimensionality-reduction method passed to `logo_cv_predict`
  (`"pls_latent"`, `"representatives"`, or `"pc1"`), default `"pls_latent"`.
- `comparison_methods`: list of additional reduction methods for robustness reporting, each drawn
  from the same `{"pls_latent", "representatives", "pc1"}` set as `reduction_method`, default
  `["representatives"]`. SHALL NOT contain `reduction_method`'s own value (would silently produce
  two methods writing to the same output file).
- `representative_selection_metric`: `"variance"` only for this tier. `"heritability"` is not a
  valid value here — `select_cluster_representatives` (reused unchanged) has no metric parameter to
  select by heritability, so this option is deferred to a future change.
- `platform_pairs`: list of `{"source": str, "target": str}` dicts, default empty. When
  `enabled=True`, SHALL contain **exactly one** entry (not zero, not more than one) naming which of
  the enclosing `CrossPlatformConfig`'s `exp1_name`/`exp2_name` is the predictor and which is
  predicted.
- `blup_refit_per_fold`: bool, default `False`. Present in the schema for forward compatibility with
  a future heritability-based `representative_selection_metric`, but currently inert in this tier —
  no valid `representative_selection_metric` value triggers any auto-force or validation on it.
- `source_blup_path` / `target_blup_path`: `Optional[str]`, default `None`. Required and
  existence-checked on disk only when `enabled=True` and `predictor_source="blup"`. Not required
  when `predictor_source="genotype_means"`.
- `visualize`: bool, default `False`. Gates whether `VisualizePredictionStep` (see the Visualize
  Cross-Platform Prediction Pipeline Step requirement) is added to `CrossPlatformPipeline`.
  Requires `enabled=True` — validated in `CrossPlatformConfig.__post_init__` (see below), since
  visualization has nothing to visualize without a prior prediction run.
- `n_permutations`: int, default `1000`. Number of shuffled-label permutations `permutation_test()`
  draws per (target, method) combination when `visualize=True`. Only meaningful (and only
  validated) when `visualize=True`.
- `permutation_random_state`: int, default `42`. Seed for the permutation shuffles.
- `permutation_n_jobs`: int, default `8`. Number of `joblib.Parallel` workers `VisualizePredictionStep`
  uses to parallelize permutation testing across independent prediction targets.

`PredictionConfig.__post_init__` SHALL raise `ValueError` (not a new exception type) for any
validation failure, matching every other config dataclass's existing convention in this codebase.
`CrossPlatformConfig.__post_init__` SHALL additionally raise `ValueError` when
`prediction.visualize=True` and `prediction.enabled=False`.

#### Scenario: Validation is a full no-op when disabled

- **WHEN** `PredictionConfig(enabled=False, predictor_source="not_a_real_value",
  source_blup_path="/does/not/exist")` is constructed
- **THEN** no exception SHALL be raised

#### Scenario: Invalid enum field rejected when enabled

- **GIVEN** `enabled=True`
- **WHEN** `predictor_source`, `reduction_method`, `representative_selection_metric`, or any entry
  in `comparison_methods` is not one of its documented valid values
- **THEN** `ValueError` SHALL be raised naming the invalid field and value

#### Scenario: heritability metric is rejected, not accepted, in this tier

- **GIVEN** `enabled=True`
- **WHEN** `representative_selection_metric="heritability"`
- **THEN** `ValueError` SHALL be raised (same as any other invalid enum value) — this tier only
  supports `"variance"`

#### Scenario: comparison_methods rejects a duplicate of reduction_method

- **GIVEN** `enabled=True, reduction_method="pls_latent"`
- **WHEN** `comparison_methods` contains `"pls_latent"`
- **THEN** `ValueError` SHALL be raised at construction time

#### Scenario: comparison_methods rejects a duplicate entry within itself

- **GIVEN** `enabled=True`
- **WHEN** `comparison_methods` contains the same method twice (e.g.
  `["representatives", "representatives"]`), independent of `reduction_method`'s value
- **THEN** `ValueError` SHALL be raised at construction time (the same silent output-overwrite risk
  as the cross-field case above, just self-inflicted within the list)

#### Scenario: blup predictor_source requires resolvable paths (pre-flight guard)

- **GIVEN** `enabled=True, predictor_source="blup"`
- **WHEN** `source_blup_path` or `target_blup_path` does not resolve to an existing file
- **THEN** `ValueError` SHALL be raised at config-construction time, before any pipeline step runs

#### Scenario: genotype_means predictor_source does not require BLUP paths

- **GIVEN** `enabled=True, predictor_source="genotype_means"`
- **WHEN** `source_blup_path` and `target_blup_path` are both `None`
- **THEN** no exception SHALL be raised

#### Scenario: visualize=True requires enabled=True

- **GIVEN** `prediction.visualize=True` on a `CrossPlatformConfig`
- **WHEN** `prediction.enabled=False`
- **THEN** `CrossPlatformConfig.__post_init__` SHALL raise `ValueError`

#### Scenario: visualize field validation is skipped when visualize is False

- **GIVEN** `enabled=True, visualize=False`
- **WHEN** `n_permutations` is set to an invalid value (e.g. `0` or negative)
- **THEN** no exception SHALL be raised from `PredictionConfig`/`CrossPlatformConfig`
  construction — `n_permutations` is only meaningful, and only validated, when `visualize=True`

#### Scenario: n_permutations must be positive when visualize is True

- **GIVEN** `enabled=True, visualize=True`
- **WHEN** `n_permutations <= 0`
- **THEN** `ValueError` SHALL be raised at construction time

### Requirement: Predict Cross-Platform Genotype Values Pipeline Step

The system SHALL provide `PredictCrossPlatformStep`, an optional pipeline step consuming
`PredictionConfig` and Tier 3's `logo_cv_predict`/`fit_pca_on_fold`/`CrossPlatformPredictionResult`
(all unchanged), wired as task 6 (`depends_on=["01_load_cross_platform_data",
"05_visualize_cross_platform"]`). The step SHALL read data from task 1's result only; the
dependency on task 5 exists solely to guarantee ordering (steps 1-5 complete before prediction
runs), not for data. The task SHALL be entirely absent from `create_tasks()`'s return value — not
merely skipped at run time — when `config.prediction.enabled=False`.

For a given directed pair, the step SHALL:
1. Build the source and target predictor matrices per `predictor_source`: BLUP CSVs
   (`source_blup_path`/`target_blup_path`, with the genotype column resolved as `"Genotype"` then
   `"genotype"` — distinct from `exp1_genotype_col`/`exp2_genotype_col`, which govern the unrelated
   raw per-sample CSVs for steps 1-5; a clear `ValueError` naming both attempted column names if
   neither is present), or task 1's own raw `exp1`/`exp2` data — selected via task 1's already-
   `exclude_cols`-filtered `exp1_trait_names`/`exp2_trait_names` metadata **before** aggregating by
   genotype mean (`predictor_source="genotype_means"` — reading task 1's result directly, so this
   ablation always uses the full raw trait set even when `trait_reduction_method="clustering"` has
   reduced the data by the time it reaches later steps; task 1's raw DataFrame is NOT trait-only, so
   this trait-name selection step is required, not optional). Any trait column containing any `NaN`
   value among the common-genotype set SHALL be dropped before further use, on both source and
   target sides; if this leaves the source matrix with zero trait columns, the step SHALL raise a
   clear `ValueError`.
2. Derive `X`, every per-target `y`, and the `genotypes` list from one canonical, sorted, explicitly-
   indexed common-genotype list — never from incidental row-order agreement between independently-
   loaded/joined DataFrames.
3. Select the **target** platform's cluster-representative traits (via the existing
   `cluster_correlated_traits`/`select_cluster_representatives`, unchanged) as the primary
   prediction targets, per `representative_selection_metric` (`"variance"` only, this tier).
4. Compute one additional target, `target_name="PC1"`: the **target** platform's own first
   principal component via `pca.fit_pca()` with `StandardScaler` applied first and
   `random_state=42` fixed, called directly (not `fit_pca_on_fold`, which remains reserved for
   reducing the **source** predictor matrix per-fold when `reduction_method="pc1"`; not
   `PCAAnalysisStep`, which is config-driven via a `PCAConfig` this pipeline does not have).
5. Call `logo_cv_predict` once per target trait × per method (`reduction_method` plus each of
   `comparison_methods` — guaranteed distinct from each other), assembling one
   `CrossPlatformPredictionResult` per method.
6. Save each `CrossPlatformPredictionResult` as JSON to the run directory, one file per method.
7. Additionally include, under a `predictor_matrices` key in the returned `StepResult.data`
   (alongside the existing, unchanged `data`/`metadata`/`files_generated` contents), the already-
   computed `source_clean`/`target_clean` predictor matrices and `source_representative_names`/
   `target_representatives` computed in steps 1-4 above — so that a downstream consumer (e.g.
   `VisualizePredictionStep`) can reuse them without independently rebuilding them.

If the common-genotype count between source and target is below `logo_cv_predict`'s own minimum,
the step SHALL raise a clear `ValueError` naming the pair and the common-genotype count, rather than
passing through `logo_cv_predict`'s generic message.

The existing `cross-platform` CLI command's `--dry-run` output SHALL list this step when enabled,
and SHALL NOT list it when disabled. No new CLI command or flag SHALL be introduced.

#### Scenario: Step present only when enabled

- **WHEN** `CrossPlatformPipeline(config).create_tasks()` is called
- **THEN** a 6th task SHALL be present if and only if `config.prediction.enabled=True`

#### Scenario: Predictor matrix built from BLUP CSVs when predictor_source is blup

- **GIVEN** `predictor_source="blup"`
- **WHEN** the step runs
- **THEN** `source_blup_path`/`target_blup_path` SHALL be loaded as the predictor matrices

#### Scenario: Predictor matrix built from genotype means when predictor_source is genotype_means

- **GIVEN** `predictor_source="genotype_means"`
- **WHEN** the step runs
- **THEN** task 1's raw `exp1`/`exp2` data SHALL be filtered to `exp1_trait_names`/`exp2_trait_names`
  (task 1's own already-`exclude_cols`-filtered metadata) before aggregating via genotype-mean
  grouping, read directly from task 1's own result — not aggregated over every column in task 1's
  raw DataFrame (which also contains `genotype`, `replicate`, and other non-trait columns)

#### Scenario: genotype_means ablation is unaffected by trait_reduction_method=clustering

- **GIVEN** `predictor_source="genotype_means"` and `trait_reduction_method="clustering"` are both
  set on the same `CrossPlatformConfig`
- **WHEN** the step runs
- **THEN** the predictor matrix's columns SHALL exactly equal task 1's `exp1_trait_names`/
  `exp2_trait_names` (the full, already-filtered trait set), not the cluster-representative-reduced
  subset task 2 (`ReduceTraitRedundancyStep`) produces, and not task 1's raw DataFrame's every column

#### Scenario: Target-side cluster-representative trait selection

- **WHEN** the step selects the target platform's prediction targets
- **THEN** `select_cluster_representatives` SHALL be applied to the target platform's aligned
  predictor matrix (BLUP or genotype-mean, per `predictor_source`), independent of and using a
  separate application from the **source** platform's own representative selection (used only when
  a method is `"representatives"`, per the "logo_cv_predict called once per target trait per
  method" scenario below)

#### Scenario: X, y, and genotypes are derived from one canonical common-genotype index

- **GIVEN** the source and target predictor matrices have their genotype rows in different orders
  (same genotype set, different order)
- **WHEN** the step builds `X`, any per-target `y`, and `genotypes` for `logo_cv_predict`
- **THEN** each SHALL be indexed from one canonical, sorted, common-genotype list, so that source
  and target values for the same genotype are correctly paired regardless of either input's
  original row order — including for the PC1 target, not only representative-trait targets

#### Scenario: Task 5's dependency is for ordering only, never data

- **GIVEN** `kwargs["05_visualize_cross_platform"]` holds any value (including a sentinel or
  otherwise-unusable `TaskResult`), while `kwargs["01_load_cross_platform_data"]` is a normal, valid
  result
- **WHEN** the step runs
- **THEN** it SHALL produce a correct `CrossPlatformPredictionResult`, never reading
  `kwargs["05_visualize_cross_platform"].data`

#### Scenario: Trait columns containing any NaN are dropped before use

- **GIVEN** a source or target predictor matrix with one trait column containing a `NaN` value for
  at least one common genotype (e.g. a failed-model trait in a real `08_blup_adjusted_means.csv`)
- **WHEN** the step builds `X` or selects target-side candidate traits
- **THEN** that column SHALL be dropped before `logo_cv_predict` is called, rather than passed
  through to raise `logo_cv_predict`'s generic NaN-rejection error

#### Scenario: Clear error when the source matrix is empty after dropping NaN columns

- **WHEN** every trait column in the source predictor matrix contains at least one `NaN` value
  among the common genotypes
- **THEN** the step SHALL raise a clear `ValueError`, distinct from the zero-target-representative-
  traits case (which still has PC1 to fall back on)

#### Scenario: BLUP CSV genotype column resolved by fixed convention

- **GIVEN** `predictor_source="blup"`
- **WHEN** the step loads `source_blup_path`/`target_blup_path`
- **THEN** it SHALL resolve the genotype column as `"Genotype"` first, falling back to `"genotype"`
  — not `exp1_genotype_col`/`exp2_genotype_col`, which govern the unrelated raw per-sample CSVs
- **AND** if neither `"Genotype"` nor `"genotype"` is present, it SHALL raise a clear `ValueError`
  naming both attempted column names, not a bare pandas `KeyError`

#### Scenario: Step still runs with only the PC1 target when zero representative traits are selected

- **WHEN** `select_cluster_representatives` returns an empty list for the target platform
- **THEN** the step SHALL still run successfully, producing a `CrossPlatformPredictionResult` with
  only the PC1 target, not a crash

#### Scenario: blup_refit_per_fold has no observable effect

- **GIVEN** two otherwise-identical configs differing only in `blup_refit_per_fold` (`True` vs.
  `False`)
- **WHEN** the step runs for each
- **THEN** the resulting `CrossPlatformPredictionResult`s SHALL be identical

#### Scenario: logo_cv_predict called once per target trait per method

- **GIVEN** N target traits (representatives + PC1) and M methods (`reduction_method` +
  `comparison_methods`)
- **WHEN** the step runs
- **THEN** `logo_cv_predict` SHALL be called exactly N × M times

#### Scenario: One JSON result file saved per method

- **WHEN** the step completes
- **THEN** one `CrossPlatformPredictionResult` JSON file SHALL be written to the run directory per
  method, with no filename collisions (guaranteed by `comparison_methods` never duplicating
  `reduction_method`)

#### Scenario: Clear error when common genotypes are below the minimum

- **WHEN** the source and target predictor matrices share fewer common genotypes than
  `logo_cv_predict` requires (including zero overlap)
- **THEN** the step SHALL raise `ValueError` naming the source/target platforms and the
  common-genotype count, not a bare pass-through of `logo_cv_predict`'s generic message

#### Scenario: Backward compatible when disabled

- **GIVEN** an existing `CrossPlatformConfig` YAML with no `prediction:` key
- **WHEN** `CrossPlatformPipeline` runs
- **THEN** the run's analysis output (file list and content for the 5 existing steps — correlation
  CSVs, alignment summary, figures, `pipeline_summary.json`) SHALL be byte-identical to the same run
  before this requirement existed
- **AND** `config.yaml` is exempted from this comparison: it SHALL gain a new `prediction: {...}`
  block reflecting the `prediction` field's existence, regardless of `enabled`, since the pipeline's
  config-provenance serialization (`cli-pipeline`'s "Pipeline Run Config Provenance" requirement)
  serializes every field of the resolved config, including nested dataclasses at their default
  values — this is an expected, harmless side effect, not a behavior change

#### Scenario: PC1 target uses whole-dataset PCA, not per-fold

- **WHEN** the step computes the `target_name="PC1"` value
- **THEN** it SHALL use `pca.fit_pca()` (with `StandardScaler` applied first, `random_state=42`
  fixed) on the full common-genotype set
- **AND** `fit_pca_on_fold` SHALL NOT be called for this purpose
- **AND** the computed values SHALL match an independently-computed
  `pca.fit_pca(StandardScaler().fit_transform(X), n_components=1, random_state=42)` on the same data

#### Scenario: Dry-run lists the prediction step when enabled

- **WHEN** `sleap-roots-analyze cross-platform <config> --dry-run` runs with
  `config.prediction.enabled=True`
- **THEN** the printed step list SHALL include a 6th entry for the prediction step

#### Scenario: Dry-run omits the prediction step when disabled

- **WHEN** `sleap-roots-analyze cross-platform <config> --dry-run` runs with
  `config.prediction.enabled=False` (or the `prediction:` key absent)
- **THEN** the printed step list SHALL contain exactly the existing 5 entries

#### Scenario: predictor_matrices is exposed for downstream reuse without changing existing outputs

`StepResult.data` is a plain dict keyed by method name (one `CrossPlatformPredictionResult.to_dict()`
per `reduction_method`/`comparison_methods` entry). Adding a sibling `"predictor_matrices"` key is
safe only because no valid `reduction_method`/`comparison_methods` value can itself equal
`"predictor_matrices"` (both are constrained to `{"pls_latent", "representatives", "pc1"}`) — this
is a load-bearing invariant on the reduction-method enum, not an incidental fact.

- **WHEN** the step runs to completion
- **THEN** `StepResult.data["predictor_matrices"]` SHALL hold the `source_clean`/`target_clean`
  DataFrames and `source_representative_names`/`target_representatives` computed during this run
- **AND** every method-keyed entry in `StepResult.data` (one per `reduction_method`/
  `comparison_methods` value) SHALL be unchanged in shape and content relative to this step's
  behavior before this key was added, and `StepResult.metadata`/`StepResult.files_generated` SHALL
  likewise be unchanged
- **AND** `"predictor_matrices"` SHALL NOT collide with any valid `reduction_method`/
  `comparison_methods` value, now or if that enum is ever extended by a future change

## ADDED Requirements

### Requirement: Visualize Cross-Platform Prediction Pipeline Step

The system SHALL provide `VisualizePredictionStep`, an optional pipeline step wired as task 7,
`depends_on=["06_predict_cross_platform"]` (for both data and ordering). The task SHALL be
entirely absent from `create_tasks()`'s return value — not merely skipped at run time — when
`config.prediction.visualize=False` (the default). Per Cross-Platform Prediction Configuration's
scenario, `visualize=True` implies `enabled=True` (enforced at config-construction time), so this
step never runs without task 6 having already run in the same pipeline.

For a given directed pair, the step SHALL:
1. Read `predictor_matrices` and the observed `CrossPlatformPredictionResult`(s) from task 6's
   `StepResult.data`.
2. Derive one independent `random_state` per `(target_name, method)` combination from
   `config.prediction.permutation_random_state` via
   `numpy.random.SeedSequence(permutation_random_state).spawn(N)` (`N` = the total number of
   `(target_name, method)` combinations for this pair), rather than reusing the single configured
   seed identically for every combination — reusing one seed would make every target draw the
   identical permutation-index sequence, correlating (not independently sampling) the null draws
   this step later pools across targets into one figure panel (see the pooled-violin scenario
   below). The whole run remains deterministic given `permutation_random_state`, since
   `SeedSequence.spawn` is itself deterministic.
3. For every `(target_name, method)` combination present in task 6's results (both
   `reduction_method` and every `comparison_methods` entry), run `permutation_test()`
   (`cross_platform_prediction.py`) with `n_permutations=config.prediction.n_permutations` and that
   combination's derived `random_state` (step 2). These calls SHALL be parallelized via
   `joblib.Parallel(n_jobs=config.prediction.permutation_n_jobs, backend="loky")`, with each
   parallel unit of work being one complete `(target_name, method)` permutation test (not
   individual permutation iterations within one test). Because each `permutation_test()` call is a
   pure function of its own arguments with no shared mutable state, `loky`'s process-based
   isolation cannot introduce cross-call RNG interference or change *which* values are computed —
   results at different `permutation_n_jobs` settings SHALL agree within this codebase's
   established cross-BLAS numerical tolerance (`rtol=1e-6, atol=1e-9`, `docs/reproducibility.md`),
   not necessarily bit-for-bit, since `loky` worker processes may resolve a different default BLAS
   thread count than the main process, a documented source of ULP-level floating-point differences
   independent of this step's own logic.
4. Save one `CrossPlatformPermutationResult` per method as JSON to the run directory:
   `07_permutation_<method>.json`.
5. Build one composite figure per pair (not per method) using only the primary
   `reduction_method`'s results: a PC1 observed-vs-predicted scatter, an all-targets observed-R²-
   vs-pooled-permutation-null strip/violin plot, and an aggregate top-quartile-recovery bar chart
   (observed mean vs. null mean). Save as `07_prediction_figure.png`.

#### Scenario: Step present only when visualize is enabled

- **WHEN** `CrossPlatformPipeline(config).create_tasks()` is called
- **THEN** a 7th task SHALL be present if and only if `config.prediction.visualize=True`

#### Scenario: Step reuses task 6's predictor matrices without rebuilding them

- **GIVEN** task 6 has already run and its `StepResult.data["predictor_matrices"]` holds the
  computed `source_clean`/`target_clean` matrices
- **WHEN** task 7 runs
- **THEN** it SHALL use those matrices directly (via `kwargs["06_predict_cross_platform"]`),
  not independently reload BLUP CSVs or re-aggregate genotype means

#### Scenario: Permutation tests are parallelized across targets, not within one target's permutations

- **WHEN** task 7 runs permutation tests for `N` `(target, method)` combinations
- **THEN** `joblib.Parallel` SHALL distribute the `N` combinations across
  `config.prediction.permutation_n_jobs` workers, with each worker running one combination's
  complete, internally-serial `permutation_test()` call

#### Scenario: Each (target, method) combination draws an independent permutation seed

- **GIVEN** `N` `(target_name, method)` combinations for a pair
- **WHEN** task 7 derives each combination's `random_state` via
  `numpy.random.SeedSequence(config.prediction.permutation_random_state).spawn(N)`
- **THEN** no two combinations SHALL receive the same derived seed
- **AND** re-running the step with the same `permutation_random_state` and the same `N`
  combinations (same order) SHALL reproduce identical derived seeds, and therefore identical
  results, for every combination

#### Scenario: One permutation JSON per method

- **WHEN** task 7 runs for a pair with `reduction_method` plus `K` `comparison_methods` entries
- **THEN** exactly `K + 1` `07_permutation_<method>.json` files SHALL be saved, one per method

#### Scenario: One figure per pair, using the primary method only

- **WHEN** task 7 runs for a pair with `comparison_methods` non-empty
- **THEN** exactly one `07_prediction_figure.png` SHALL be saved, built from the primary
  `reduction_method`'s permutation results only

#### Scenario: Figure PNG is non-empty and reflects the current run's inputs

- **WHEN** task 7 completes
- **THEN** `07_prediction_figure.png` SHALL exist, be non-empty, and (via a hash or timestamp
  comparison in the acceptance test) SHALL be verifiably produced from the current run's input
  CSVs, not a stale file from a prior run

#### Scenario: Observed values in the permutation JSON match task 6's prediction JSON exactly

- **WHEN** task 7's `07_permutation_<method>.json` and task 6's `06_prediction_<method>.json` are
  compared for the same target and method
- **THEN** the permutation result's `observed_r2`/`observed_rmse`/`observed_spearman_rho` SHALL
  exactly match the prediction result's `r2`/`rmse`/`spearman_rho` for that target
