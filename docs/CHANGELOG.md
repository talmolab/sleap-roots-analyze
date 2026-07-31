# Changelog

All notable changes to `sleap-roots-analyze` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `permutation_test(X, y, genotypes, reduction_method="pls_latent",
  representative_names=None, n_permutations=1000, random_state=42)` and
  `top_quartile_recovery(y_true, y_pred, q=None)` (#200): a permutation-null
  significance test for cross-platform LOGO-CV prediction, closing the gap
  Tier 3 Decision 9 flagged (`spearman_p`'s asymptotic p-value is unreliable
  below n≈20-30). Self-contained: computes the observed R²/RMSE/Spearman ρ/
  top-quartile-recovery via one `logo_cv_predict()` call, then null
  distributions via `n_permutations` shuffled-`y` calls. One-sided p-values
  are right-tailed for R²/ρ (higher is better) and **left-tailed** for RMSE
  (lower is better — the opposite convention). `top_quartile_recovery`'s
  chance-level baseline is `2*q/n`, not a fixed 25%. Serializable
  `PermutationResult` / `CrossPlatformPermutationResult` dataclasses and
  `CrossPlatformPermutationResult.from_permutation_test_results(...)` adapter,
  mirroring `TargetPrediction`/`CrossPlatformPredictionResult`'s pattern.
- `VisualizePredictionStep`, an optional 7th task on `CrossPlatformPipeline`
  (#200), gated on `PredictionConfig.visualize` (requires `enabled=True`
  too): runs `permutation_test()` for every `(target, method)` combination
  via `joblib.Parallel` across independent targets (parallelizing individual
  permutation calls measured *slower* than serial — empirically verified,
  not assumed), saves one `07_permutation_<method>.json` per method, and a
  composite 3-panel `07_prediction_figure.png` per pair (PC1 obs-vs-pred
  scatter, all-targets R²-vs-pooled-null violin, top-quartile-recovery bar
  chart) from the primary `reduction_method`'s results. Additive extension to
  `PredictCrossPlatformStep`'s `StepResult.data` (`predictor_matrices` key)
  lets this step reuse task 6's already-computed matrices instead of
  rebuilding BLUP-loading/alignment logic. Tier 4 of the cross-platform
  genotype-prediction program (Tier 3: #194, Tier 3.5: #196).
- `PredictionConfig` (nested on `CrossPlatformConfig` as `prediction`, default
  disabled) and `PredictCrossPlatformStep`, an optional 6th task on
  `CrossPlatformPipeline` (#196): wires Tier 3's `logo_cv_predict`/
  `CrossPlatformPredictionResult` into the same per-pair `cross-platform`
  command already used for correlation, so prediction runs alongside
  correlation for the same `exp1`/`exp2` pair rather than only being reachable
  via direct Python calls. `predictor_source` selects `"blup"` (Tier 1's
  `08_blup_adjusted_means.csv`) or `"genotype_means"` (a plain per-genotype
  mean of the same raw data correlation already loads) as the predictor
  matrix; `platform_pairs` names the predictor/predicted direction. No new CLI
  command or flag — the existing `cross-platform` command's `--dry-run`
  output lists the step when enabled. Byte-identical backward compatibility
  when disabled (the default). Tier 3.5 of the cross-platform
  genotype-prediction program (Tier 3: #194).
- `logo_cv_predict(X, y, genotypes, reduction_method="pls_latent",
  representative_names=None)` and `fit_pca_on_fold(X_train, X_test,
  n_components=1)` (#194): leave-one-genotype-out (LOGO) cross-validated
  ridge/PLS prediction machinery, implementing the CV-hygiene contract (a
  fresh `sklearn.Pipeline` fit inside each fold; `fit_pca_on_fold` fits PCA on
  training data only, distinct from the pipeline-level `PCA` step, to avoid
  leaking the held-out genotype's position into component loadings). Reports
  aggregate R², RMSE, and Spearman ρ over concatenated leave-one-out
  predictions. Three `reduction_method` values: `pls_latent` (default,
  `PLSRegression(n_components=1)`, fixed rather than inner-CV-searched),
  `representatives` (variance-based cluster representatives, selected once
  before the fold loop), `pc1` (a per-fold principal-component score). Tier 3
  of the cross-platform genotype-prediction program (Tier 1: #109, Tier 2:
  #114) — reframes the wheat EDPIE cross-platform result from correlation to
  predictability.
- Serializable `CrossPlatformPredictionResult` / `TargetPrediction` dataclasses
  and `CrossPlatformPredictionResult.from_logo_cv_results(...)` adapter
  (#194): one result per (platform pair, reduction method), with one
  `TargetPrediction` per prediction target (each cluster-representative trait,
  plus `"PC1"`, reported as an independent entry, never averaged with the
  representative-trait targets). Provides `to_dict()` / `to_json()` (strict
  `allow_nan=False`).
- `calculate_heritability_estimates(fixed_effects=...)` (#114): optional list of
  metadata-style covariate columns (experiment, wave, batch, scanner) added as
  fixed effects to the mixed model, changing the formula from `value ~ 1` to
  `value ~ C(fe_1) + C(fe_2) + ...` — every name is `C(...)`-wrapped
  unconditionally (always categorical). Corrects a heritability-inflation bug
  where a batch/experiment confound gets absorbed into the genotype term when
  genotypes aren't balanced across batches. Default `None` reproduces
  pre-existing behavior exactly. Tier 2 of the cross-platform
  genotype-prediction program (Tier 1: #109).
- `StatisticsConfig.fixed_effects` (default `None`): threads the above into the
  QC/Viz pipeline's `StatisticalAnalysisStep`.
- `extract_blup_table(heritability_results)` (#109): builds a genotype x trait
  BLUP-adjusted-means `pd.DataFrame` from a `calculate_heritability_estimates` result —
  `adjusted_mean = intercept + blup[genotype]` for each trait whose mixed model
  succeeded, with a genuine `NaN` column (not dropped, not zero-filled) for any
  trait that failed. Importable from `sleap_roots_analyze`. Tier 1 of the
  cross-platform genotype-prediction program: BLUP-adjusted means are the predictor
  substrate later tiers use to test whether one phenotyping platform's genotype
  effects predict another's.
- Serializable `BLUPResult` dataclass and `BLUPResult.from_blup_table(df, *,
  intercepts=None)` adapter (#109): holds the genotype x trait adjusted-means matrix
  for traits whose model succeeded, plus the names of traits that failed
  (`failed_traits`) — a column needs to be entirely finite to count as succeeded, so
  even a single cell-level gap reclassifies a trait the same as an outright model
  failure. Provides `to_dict()` / `to_json()` (strict `allow_nan=False`).
- `StatisticsConfig.generate_blup_table` (default `True`): controls whether the QC/Viz
  pipeline's `StatisticalAnalysisStep` writes `08_blup_adjusted_means.csv` alongside
  `08_heritability_results.csv`. Only takes effect when `calculate_heritability` is
  also `True` — free once the model is fit.
- `StatisticsConfig.fixed_effects` must now be `None` or a `list[str]` — a bare
  string is rejected with `ValueError` at config-construction time instead of
  silently degrading to a per-character `fixed_effects` list (PR #193 review).
- A `UserWarning` is now emitted when a fixed effect is confounded with genotype
  (every observation for some genotype confined to a single level of that fixed
  effect), even when the fit converges cleanly with zero `ConvergenceWarning`s —
  diagnostic only, does not reclassify the trait's result (PR #193 review).
- A duplicate name within `fixed_effects` (e.g. `["experiment", "experiment"]`) is
  now rejected with a structural `{"error": ...}`, matching the existing
  missing-column/reused-name error shape, instead of an obscure `patsy` failure
  (PR #193 review).

### Fixed
- `create_pca_biplot` now honors `feature_selection="top_variance"` (#202): the
  `feature_selection`→`method` mapping had `elif` branches for `extreme`/
  `top_absolute`/`top_contribution`/`vector_length` but none for `top_variance`,
  so it silently fell into `else: method = "vector_length"` — a materially
  different selection criterion (eigenvalue-weighted variance contribution vs.
  unweighted Euclidean norm in the PC plane). An explicit `top_variance` branch
  now passes `pc_indices=None` (rather than the biplot's 2 displayed PCs) since
  `select_top_features_from_pca`'s `top_variance` method ranks across all
  retained PCs and ignores `pc_indices` entirely. Any other `feature_selection`
  value now raises `ValueError` instead of silently substituting
  `vector_length`. `create_feature_contribution_plot` **removes** its
  `feature_selection` parameter (BREAKING): the parameter was never referenced
  in the function body — every code path always ranked by total variance
  contribution regardless of what was passed — and wiring it up would have made
  the chart's own title (which asserts the displayed traits are the top
  contributors) misdescribe non-contribution-selected content, so the
  parameter is removed rather than fixed, matching the parameter-free
  `create_feature_contribution_heatmap` precedent. Its on-the-fly ranking
  branch now delegates to `select_top_features_from_pca(method="top_variance")`
  instead of duplicating the formula inline. No in-repo caller nor the
  verified downstream `bloom` consumer passed `feature_selection` to this
  function, so removal is backward-compatible in practice.
- `ExploratoryAnalysisStep.execute()` and `GenerateStaticFiguresStep` no longer accumulate every
  step-4/static figure in memory before saving any of them — an OOM (`bad allocation`) on large
  experiments with many genotypes and traits (#110). Peak concurrently-open figures dropped from
  45/40 (measured on a 480-genotype x 300-trait fixture) to 4/5. Both steps now save and close each
  figure immediately after it's produced, via new private generator functions
  (`_generate_trait_histogram_batches`, `_generate_trait_boxplot_batches`) that the existing public
  `create_trait_histograms_batched()`/`create_trait_boxplots_by_genotype_batched()` now wrap.
  Additionally, `create_trait_boxplots_by_genotype()`'s horizontal-orientation branch — previously
  unbounded, unlike the vertical branch's existing 20" width cap — now caps subplot height at 20"
  (`max_subplot_height`, mirroring the vertical cap), and
  `create_trait_boxplots_by_genotype_batched()` gains genotype pagination
  (`max_genotypes_per_page`, auto-derived from the height cap: ~66 genotypes/page horizontal, ~40
  vertical): datasets with more genotypes than fit in one readable, capped figure are split across
  multiple pages instead of producing one memory-safe-but-illegible chart. Visual output changes
  (capped height, more boxplot figures for very high genotype counts) are intentional, not a
  regression.
- `PCAAnalysisStep` now updates `metadata["trait_names"]`/`valid_trait_names`
  to the zero-variance-filtered feature set after PCA, instead of leaving
  them as the pre-filter list while only recording
  `excluded_zero_variance_traits`/`n_traits_after_filtering` alongside them
  (#80). The pre-filter list is preserved under a new
  `metadata["original_trait_names"]` key. Downstream, `UMAPAnalysisStep`
  inherited the stale list directly (silently re-including excluded traits
  in `feature_cols` and the logged `n_traits`), and the `VizPipeline`
  orchestrator's `_run_generate_static_figures` cherry-picks `trait_names`/
  `original_trait_names` from the PCA branch alongside `pca_results`/
  `top_features` (previously omitted), so static figures now use the
  corrected set too — this also fixes a latent bug in `create_pca_biplot`,
  which indexes its `trait_names` argument positionally against
  `pca_results["loadings"]` and would silently mislabel feature arrows when
  an excluded trait wasn't the last column of the original trait list.
- `VizPipelineConfig` now automatically unions `statistics.fixed_effects` into
  `data.additional_exclude_cols` at construction time, so a fixed-effect column
  named outside the hardcoded metadata-substring list (e.g. `"block"`) is no
  longer silently treated as a phenotypic trait by the pipeline's upstream
  `trait_cols` scan. The existing `remove_low_h2=True` fix for direct API callers
  never reached the pipeline, since `StatisticalAnalysisStep` always calls with
  `remove_low_h2=False` (PR #193 review).

### Changed
- `calculate_heritability_estimates` additively returns `blup` (`dict[str, float]`) and
  `intercept` (`float`) keys per trait when its mixed model succeeds (#109) — both
  existing return shapes (plain dict, or the `remove_low_h2=True` 4-tuple) are
  unchanged. A trait solved via the ANOVA-based or no-variance path carries no such
  keys, since no fitted mixed-model result exists for those paths.
- When `fixed_effects` is used (#114), the `intercept` key becomes an empirical,
  sample frequency-weighted value instead of the raw model intercept — a
  sample-composition-dependent quantity that can differ trait-to-trait, not a
  population-typical value. A captured `ConvergenceWarning` during the fit (only
  when `fixed_effects` is set) is now treated as a trait-level failure, since
  `statsmodels` does not reliably raise on a fixed effect confounded with
  genotype. No change when `fixed_effects` is unset.

## [0.1.0a5] - 2026-07-13 (Pre-release)

### Added
- `UMAPResult` (#180): frozen JSON-serializable dataclass (sibling to `PCAResult`):
  `embedding`, `n_neighbors`, `min_dist`, `n_components`, `feature_names`, `n_samples`,
  `standardized`, `random_state`. Excludes the fitted `reducer`/`scaler`. Provides
  `to_dict()`/`to_json()` (strict `allow_nan=False`).
  `UMAPResult.from_umap_dict(d, *, random_state=None)` derives `n_components`/`n_samples`
  from the embedding shape and resolves the seed from the argument or the echoed dict
  key. `perform_umap_analysis` additionally returns `feature_names` + `random_state`
  (additive, non-breaking). Completes the result-types epic (#130) across
  PCA/heritability/clustering/UMAP.
- Public hierarchical-clustering entry point `hierarchical_cluster_labels` (#179):
  the **labeled** counterpart to `perform_hierarchical_clustering` (which returns only
  the dendrogram). Importable from `sleap_roots_analyze`, it composes the existing
  `perform_hierarchical_clustering` → `calculate_optimal_clusters_hierarchical` (when
  `n_clusters` is omitted) → `cut_dendrogram` into a single labeled dict (cluster
  labels/sizes, the three quality metrics, hierarchical provenance, and `data_indices`
  mapping labels back to source rows), suitable for building a `ClusterResult`. Every
  invalid argument (`method`, `metric`, `optimization_method`, `n_clusters`) surfaces as
  a single `ValueError`. Follow-up to #129; unblocks bloom-mcp.
- `HierarchicalResult` result type and the `ClusterResult.from_hierarchical_dict(d)`
  adapter (#179): a frozen, JSON-serializable view of a hierarchical run. The adapter
  takes no `random_state` (hierarchical clustering is deterministic) and stamps `None`.
- `ALGORITHM_HIERARCHICAL` discriminator constant (#179), exported from `result_types`
  alongside `ALGORITHM_KMEANS` / `ALGORITHM_GMM`.

### Changed
- Data cleanup now drops **constant (zero-variance) traits** and names them at cleaning
  time instead of letting PCA drop them silently later (#177). A new internal filter
  `remove_zero_variance_traits` runs as the **final** step of `apply_data_cleanup_filters`
  (after sample removal, so variance is measured on the reduced frame) and is re-applied
  inside `clean_traits_for_analysis` after its own residual-NaN `dropna`, guaranteeing the
  analysis-ready frame is constant-free on both the standard and loosened-NaN paths. Each
  dropped trait is logged in `cleanup_log["removed_traits"]` with `reason="zero_variance"`
  (plus `variance` and `threshold`), matching the sibling trait filters. The threshold is
  configurable via the new `CleanupConfig.min_variance` (default `0.0`, forwarded by
  `CleanupTraitsStep`); `0.0` drops exactly-constant traits and a negative value disables
  the filter. **Behavior note:** for any dataset with a genuinely-constant trait, that
  column is now absent from cleaned output and appears in the cleanup log; PCA / UMAP /
  clustering results are unchanged (those paths already dropped constants before fitting),
  and the QC PCA step's `excluded_zero_variance_traits` becomes empty when fed a cleaned
  frame. Set `min_variance` negative to retain the previous behavior.
  **Statistics note:** a constant-but-nonzero trait (e.g. always `3.0`) previously survived
  cleanup and produced a degenerate row in the heritability table (`{'heritability': 0.0,
  'model_type': 'no_variance', ...}`) and the ANOVA table (`f_statistic`/`p_value` = `NaN`);
  it is now dropped at cleanup, so those rows no longer appear and `n_traits_analyzed`
  drops accordingly. The removed entries were statistically degenerate (H²=0, p=NaN), so
  this is a correctness improvement, not a loss of real signal. (An all-zeros trait was
  already removed earlier by the zero-inflation filter, so only constant-nonzero traits are
  affected.)
- `ClusterResult.random_state` is now `Optional[int]` (default `None`) (#179), matching
  `PCAResult.random_state`, so a deterministic algorithm (hierarchical) can omit the
  seed. Source-compatible for producers (KMeans/GMM still stamp the `int` seed); a
  reader that assumed an always-`int` value must now handle `None`. `KMeansResult` and
  `GMMResult` reject `random_state=None` at construction (`TypeError`), restoring the
  pre-widening guarantee that a seeded algorithm's result always carries a real seed —
  only `HierarchicalResult` may omit it.
- `calculate_optimal_clusters_hierarchical` additively returns the winning candidate's
  `cut_result` (#179) from its `k`-scan, so `hierarchical_cluster_labels`'s auto-`k`
  path no longer re-cuts the dendrogram for the same `k` it already computed.

### Fixed
- `perform_kmeans_clustering`, `perform_gmm_clustering`, and
  `perform_hierarchical_clustering` (`clustering.py`) now re-derive `feature_names` from
  the columns actually used for fitting, on both `standardize=True` and
  `standardize=False` (#183). Previously `feature_names` was snapshotted **before**
  constant/non-numeric columns were filtered out, silently mislabeling
  `cluster_centers`/`means`/`data_processed` for such inputs — not a length mismatch a
  caller would notice, a positional mislabeling. `standardize=False` additionally now
  applies the same numeric + non-zero-variance filter `standardize=True` always used
  internally; previously it applied no filtering at all, so a non-numeric column reached
  the estimator directly and failed with a raw sklearn "could not convert string to float"
  error instead of the clear "No numeric columns with non-zero variance found" message.
  This is separate from the `#177` cleanup change above: `#177` keeps constant traits from
  ever reaching these functions through the standard cleanup step; `#183` fixes the
  functions' own label bookkeeping for any caller that doesn't go through it — e.g. direct
  callers, or `standardize=False`. `KMeansResult`/`GMMResult` and
  `detect_outliers_kmeans`/`_gmm`/`_hierarchical` (`outlier_detection.py`) inherit the
  corrected values automatically, with no adapter code changes needed.
- `perform_hierarchical_clustering` now requires the euclidean metric for `centroid`
  and `median` linkage, not only `ward` (#179) — scipy's `linkage()` enforces this for
  all three methods; the other two previously raised a wrapped `RuntimeError` instead
  of the clear `ValueError` `ward` already got.

## [0.1.0a4] - 2026-07-02 (Pre-release)

### Added
- Public outlier-plotting entry point `plot_outlier_analysis` (#173): the
  **plotting** sibling of `remove_outlier_samples`. Importable from
  `sleap_roots_analyze`, it **re-detects** outliers with the same detector, seed, and
  per-method parameters (so, under the shared NaN-free + unique-index preconditions,
  it flags the same samples `remove_outlier_samples` removes), then composes the
  existing public `create_*_outlier` figure functions and returns a
  `Dict[str, plt.Figure]` — **IO-free** (the caller saves/persists). Covers the two
  `remove_outlier_samples` methods (`mahalanobis`, `isolation_forest`); the
  `detect_outliers_pca`/`_kmeans`/`_gmm`/`_hierarchical` plots stay pipeline-only. A
  `which` selector narrows the returned figures, and metadata-column args
  (`barcode_col`/`genotype_col`/`replicate_col`) mirror `remove_outlier_samples` so the
  plotted set matches the trimmed one. The pipeline's `VisualizeOutliersStep` shares the
  selection via the public `select_outlier_figures(df, results, method, ...)` helper on
  its own pre-computed results (no behavior change).
- `select_outlier_figures` (#173): public no-detection figure-selection layer (used by
  both `plot_outlier_analysis` and the pipeline), so a consumer holding a detector
  result can plot without re-detecting.
- `remove_outlier_samples` gains an additive `return_detector_result=False` flag (#173):
  when `True` it also returns the raw detector dict, so a consumer (the bloom-mcp
  `remove_outliers` tool) detects once and feeds both the trim and the plots. Default
  preserves the compact-report 2-tuple contract.
- Public outlier-removal entry point `remove_outlier_samples` (#165): the
  **quality**-step follow-up to `clean_traits_for_analysis`. Takes a clean
  (NaN-free) trait table and detects + removes outlier samples so PCA/UMAP/
  clustering can optionally run on outlier-trimmed data, composing the existing
  public `detect_outliers_mahalanobis` / `detect_outliers_isolation_forest` and
  `remove_outliers_from_data` primitives (no new detection/removal algorithm).
  Importable from `sleap_roots_analyze`, it returns `(trimmed_df, outlier_report)`
  — an auditable, JSON-serializable report — and composes after
  `clean_traits_for_analysis`, before `perform_pca_analysis` / UMAP / clustering.
  Enforces NaN-free + unique-index preconditions, rejects unknown / cross-method
  `detect_kwargs` with an actionable error, re-applies the readiness gates, and warns
  on over-removal (> 50%), the `p > n` regime, and — on the Mahalanobis path — a small
  sample (`n < 30`) or a violated chi-squared goodness-of-fit.

### Changed
- `apply_data_cleanup_filters` bare-default thresholds tightened to the canonical
  QC values — `max_nans_per_trait` `0.3 → 0.2` and `max_nans_per_sample`
  `0.2 → 0.0` — so they equal `CleanupConfig()`'s defaults (with `max_nan_fraction`
  ↔ `max_nans_per_sample` name mapping). The function signature is now the single
  source of truth for canonical cleanup, and `clean_traits_for_analysis` inherits
  it instead of carrying a hardcoded copy (#167). **Behavior note:** code calling
  `apply_data_cleanup_filters` with the *bare* defaults now cleans more strictly —
  `max_nans_per_sample=0.0` drops every sample that still has any NaN in a surviving
  trait, which can remove more plants than before. The QC pipeline
  (`CleanupTraitsStep` passes explicit `config.cleanup.*`) and the
  `clean_traits_for_analysis` entry point (already pinned to `0.2`/`0.0` since #164)
  are reproducibility-neutral — their effective thresholds are unchanged.

## [0.1.0a3] - 2026-06-24 (Pre-release)

### Added
- Serializable analysis result types in the public API: `PCAResult`
  (#127/#149), `HeritabilityResult` (#128/#150), and `ClusterResult` with its
  `KMeansResult` / `GMMResult` subclasses (#129/#151). Each is a `frozen`
  dataclass exported from the top-level `sleap_roots_analyze` namespace and
  `__all__`, with `to_dict()` / `to_json()` adapters for lossless
  serialization, so downstream consumers (e.g. bloom-mcp) can import and
  round-trip typed results instead of parsing untyped `perform_*` dict returns.
- Public minimal-QC entry point `clean_traits_for_analysis` (#164): turns a raw
  wide trait table into a clean, analysis-ready frame by composing the QC step-02
  cleanup with the step-03 validation, then gating on no-NaN, ≥2 samples, and ≥1
  non-constant trait — so `perform_pca_analysis` no longer silently drops rows.
  Returns `(clean_df, trait_cols, cleanup_log)`. As part of the same change, the
  step-02/step-03 functions are now importable from `sleap_roots_analyze`:
  `apply_data_cleanup_filters`, `validate_clean_traits`, and
  `build_clean_validation_report` (the latter two extracted from
  `ValidateCleanStep`'s inline check, which now calls them — no pipeline behavior
  change). All four are in `__all__`.
- Numerical-stability golden gate (`tests/test_numerical_stability.py`): a
  golden-vs-committed drift detector that pins the UMAP (Procrustes on aligned
  coordinates, `atol=1e-6`), clustering (ARI `>0.95` + pinned `n_clusters`), and pandas
  trait-aggregation (`assert_frame_equal rtol=1e-10`) paths for the `turface_19`
  reference slice — catching `numba`/`numpy`/`umap-learn`/`pandas` upgrade drift that the
  same-machine determinism sweep is structurally blind to. Golden artifacts +
  `golden_provenance.json` live under
  `tests/fixtures/real/wheat_edpie/expected/numerical_stability/`, regenerated by
  `scripts/regenerate_numerical_stability_golden.py`. Runs single-OS (macOS) in a
  dedicated `numerical-stability` CI job and self-skips elsewhere. See
  [docs/reproducibility.md](reproducibility.md).
- Public cross-platform PC-correlation and trait-enrichment workflows (#119).
  Two new top-level functions wrap the wheat-EDPIE analyses as single calls:
  `cross_platform_pc_correlations` (PC-level; loads per-platform PCA outputs,
  aggregates to genotype-mean PC scores, correlates every PC across platform
  pairs with combined/per-pair FDR; returns a typed `CrossPlatformPCResult`) and
  `trait_correlation_enrichment` (binomial enrichment over existing
  `cross_platform_correlations.csv`, returning typed `EnrichmentResult`s). Both
  exported in `__all__`. A new `pc-correlations` CLI subcommand drives the PC
  workflow.
- Optional, config-gated trait-enrichment step in the cross-platform pipeline
  (`CrossPlatformConfig.enrichment_enabled`, default `False`), with
  `enrichment_p_value_column` validated against `correlation_method`. When
  enabled it runs a per-pair exact binomial test (nominal p, no FDR) and writes
  `trait_enrichment.csv`; existing runs are unchanged.
- Analysis-input contract conformance tests (`tests/test_contract_conformance.py`):
  validate a canonicalized **copy** of each of the four EDPIE post-QC fixtures and every
  packaged `sleap_roots_contracts` canonical example against `validate_analysis_input`,
  under both lenient and `strict=True` modes, with a purity guard and a reason-checked
  negative control. Adds `sleap-roots-contracts[pandas]>=0.1.0a1` as a dev dependency.
  No `src/` changes; the #146/#120 reproduction goldens stay green (#147).
- Optional input-contract validation at the QC data-load boundary
  (`LoadDataStep`). A new `data.validate_input: off | warn | strict` flag
  (default `warn`) validates the entry input via the optional
  `sleap-roots-contracts` dependency before any analysis runs. Install with
  `pip install "sleap-roots-analyze[contracts]"`; when the extra is absent,
  validation degrades to a logged no-op (never an `ImportError`). Validation runs
  on a discarded copy of the entry frame, so enabling it never changes results —
  proven equivalent to the #120/#146 `turface_19` golden across `off`/`warn`/`strict`.
  Under the default `warn`, a NaN/blank in the `genotype` role is now a structural
  hard-fail (alongside missing `genotype`, no numeric trait, and bad role dtype), so
  exports carrying blank genotype cells will error unless `validate_input` is set to
  `off` or the column is cleaned first. (#144)
- Extend the same optional input-contract validation to the **cross-platform**
  load boundary (`LoadCrossPlatformDataStep`). New `validate_input: off | warn |
  strict` flag (default `warn`) on `CrossPlatformConfig` validates each loaded
  experiment frame on a discarded copy; no golden output change. Aligned frames carry
  no per-sample id, so `strict` injects a synthetic positional `sample_id` into the
  discarded copy (rather than failing on the structurally-absent role) and otherwise
  enforces the full contract. Rows with a blank/NaN genotype are now dropped during
  alignment, so `off`/`warn`/`strict` stay output-identical. (#154)
- CI reproducibility gates (#133): a whole-package coverage guard that fails CI if any
  function accepting `random_state` is missing from the determinism sweep
  (`tests/reproducibility_cases.py`), and an opt-in result-object JSON round-trip gate
  (`tests/test_result_serialization.py`) that asserts losslessly when a registered
  function returns a dataclass (and is guarded so a new registered case can't silently
  drop). The determinism sweep runs single-OS (same-machine comparison); the
  serialization round-trip runs on the full OS matrix. See
  [docs/reproducibility.md](reproducibility.md).
- Expose the eight `statistics.py` functions through the top-level
  `sleap_roots_analyze` namespace and `__all__`, so they can be imported directly
  (e.g. `from sleap_roots_analyze import calculate_heritability_estimates`) instead
  of reaching into the internal `statistics` submodule:
  `calculate_trait_statistics`, `perform_anova_by_genotype`,
  `calculate_heritability_estimates`, `identify_high_heritability_traits`,
  `analyze_heritability_thresholds`, `analyze_trait_variance`,
  `diagnose_heritability_issues`, and `compare_trait_heritabilities` (Part of #116).

### Fixed
- Pipeline provenance manifests now emit forward-slash (POSIX) paths on every OS.
  Steps previously hand-stringified paths with `str(path)` before serialization,
  which baked in backslashes on Windows (e.g. `out\a.csv` instead of `out/a.csv`)
  in the `files_generated` and `metadata` fields of `pipeline_summary.json` and in
  the standalone `*_manifest.json` files. Producers now store `Path` and let the
  central serializers normalize once via `Path.as_posix()` (`convert_to_json_serializable`
  and the `save_json` default hook); the `files_generated` field type was tightened
  to `List[Path]` so the divergence can't recur. Completes the single-site fix from
  #156. (#157)
- Add the missing `from typing import Any` import in `statistics.py` so
  `typing.get_type_hints()` no longer raises `NameError` on the three functions that
  annotate `Dict[str, Any]`, unblocking downstream tool-schema generation.

## [0.1.0a2] - 2026-03-18 (Pre-release)

### Fixed
- Forward sanitized column names (`Genotype`, `Replicate`) from `CleanupTraitsStep`
  to `apply_data_cleanup_filters()` so `02_removed_samples_detail.csv` contains
  correct genotype and replicate values instead of empty strings
- Fix `removed_samples` cleanup log key to be an independent deep copy of
  `removed_samples_detail` (was a mutable alias to the same list object)
- Remove dead `removed_sample_indices` code from `cleanup_traits.py`
- Fix `from src.sleap_roots_analyze` import path in test files
- **NaN-removed sample traceability** — Corrected a dictionary key mismatch in
  `apply_data_cleanup_filters()` where `removal_stats.get("removed_samples_detail")`
  should have been `removal_stats.get("removal_details")`, causing
  `02_removed_samples_detail.csv` to always be written as a header-only file.
  Also fixed a secondary key mismatch in `CleanupTraitsStep` (`sample_info["index"]`
  → `sample_info["sample_index"]`) and aligned the empty-DataFrame fallback column
  schema. Affected datasets: Turface 19 (29 samples, 15.5% of dataset) and
  Turface 150 (1 sample).

## [0.1.0a1] - 2026-03-17 (Pre-release)

### Added
- **FDR Correction for Cross-Platform Correlations** (PR #45)
  - Configurable False Discovery Rate (FDR) correction via `fdr_correction_method` config parameter
  - Three correction methods: `fdr_bh` (Benjamini-Hochberg), `fdr_by` (Benjamini-Yekutieli, default), `none`
  - New CSV output columns: `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
  - Updated visualization to show FDR-corrected significance counts in summary plots
  - Comprehensive documentation in `docs/CROSS_PLATFORM_ANALYSIS.md` with mathematical formulations
  - Pipeline summary JSON now includes FDR metadata (`fdr_correction_method`, `significant_correlations`)
- **Visualization Pipeline** with DAG-based architecture for automated visualization workflows
  - 10 new pipeline steps (PCAAnalysisStep, LoadDataAndImagesStep, UMAPAnalysisStep, ClusterAnalysisStep, etc.)
  - 4 configuration presets: minimal, standard, comprehensive, publication
  - Example scripts and comprehensive documentation
- **Unified pipeline architecture** with modular configuration system
  - Reorganized config into reusable components (25 components in `config/components.py`)
  - Composition-based config for QC and Viz pipelines
  - All steps unified in single `pipeline/steps/` directory
  - Pipeline orchestrators in `pipeline/pipelines/` subdirectory
- **Adaptive sizing utilities** (`viz_utils.py`) for automatic plot dimension calculations
  - `calculate_figure_size()` with layout-aware sizing (single, horizontal, vertical, grid)
  - `calculate_grid_dimensions()` for optimal subplot layouts
  - `calculate_subplot_grid_size()` for multi-trait plots
  - `calculate_correlation_matrix_size()` and `calculate_barplot_size()` for specific plot types
  - 347 comprehensive tests for adaptive sizing functions
- **Comprehensive test coverage improvements**
  - PCAAnalysisStep: 31% → 100% coverage (11 new tests)
  - RemoveOutliersStep: 55% → 100% coverage (17 new tests)
  - Overall pipeline coverage improved from 66% to 68%
  - Tests cover all removal strategies, feature selection methods, edge cases, and file outputs
- **Effect size-based goodness-of-fit evaluation** for large samples (n > 500) in Mahalanobis outlier detection
  - K-S test becomes hypersensitive with large n, now uses K-S statistic magnitude instead of p-values
  - New thresholds: excellent (<0.05), good (<0.10), acceptable (<0.15), poor (<0.20), very poor (≥0.20)
  - `print_goodness_of_fit_summary()` function for formatted console output with interpretation
  - References to Massey (1951) and Sullivan & Feinn (2012) for statistical methodology
- **Configurable ID column** in interactive visualization functions
  - Added `id_col` parameter to `create_interactive_scatter_with_images()`, `create_interactive_pca_with_images()`, and `create_interactive_umap_with_images()`
  - Fixes hardcoded "Barcode" column assumption, now supports lowercase or custom column names
- **Pipeline validation warnings** for outlier detection configuration (Issue #20)
  - Early detection when no outlier detection methods are configured
  - Clear, actionable warning messages in pipeline output
  - Graceful handling with pipeline continuing successfully
- Comprehensive test suite with 1900+ tests achieving 97%+ coverage across all modules
- Complete PCA module with mathematical validation (88 tests)
  - Per-feature variance explained calculations with configurable ddof
  - Mathematical validation test suite (11 properties verified)
  - `calculate_pca_metrics()` for comprehensive PCA metrics
  - `build_feature_metrics_df()` for per-feature analysis
  - Edge case handling for single samples and constant features
- **Outlier Detection Module** (`sleap_roots_analyze.outlier_detection`) with three complementary methods:
  - `detect_outliers_mahalanobis()`: Statistical detection using Mahalanobis distance with chi-squared and custom thresholds
  - `detect_outliers_pca()`: Outlier detection based on PCA reconstruction error
  - `detect_outliers_isolation_forest()`: Tree-based anomaly detection for complex, non-linear patterns
  - `remove_outliers_from_data()`: Utility to remove outliers while preserving DataFrame structure and metadata
  - `calculate_outlier_threshold()`: Calculate chi-squared or direct distance thresholds
  - `identify_outliers_from_distances()`: Identify outliers from pre-calculated distances
  - Support for robust covariance estimation using MinCovDet
  - Automatic index preservation through NaN removal
  - Comprehensive test suite with 94% coverage (74 tests)
- Improved PCA documentation with scikit-learn references and mathematical proofs
- Numerical accuracy tests with known correct answers
- Edge case fixtures for boundary condition testing
- `.gitattributes` file for consistent line endings across platforms
- Integrated heritability filtering in `calculate_heritability_estimates()`
- Optional saving of removed samples in `remove_nan_samples()`
- Detailed removal statistics and metadata tracking
- Modular data cleanup functions: `remove_zero_inflated_traits()`, `remove_traits_with_many_nans()`, `remove_low_sample_traits()`
- Claude commands for PR review (`.claude/commands/review-pr.md`) and changelog updates (`.claude/commands/update-changelog.md`)
- **Visualization Module** (`sleap_roots_analyze.visualization`):
  - `create_feature_contribution_heatmap()`: Heatmap showing feature contributions to principal components
  - `save_publication_figure()`: Save figures in publication-ready formats (PDF, EPS, PNG, SVG)
  - `identify_extreme_phenotypes()`: Identify genotypes with extreme phenotypes for each trait
  - `create_phenotype_variation_plot()`: Box plots with jittered points showing phenotypic variation
  - `create_feature_contribution_plot()`: Now uses pre-calculated contributions from `run_pca_and_export_artifacts` for efficiency
  - All visualization functions now use Google-style docstrings for consistency
- **PCA Module Enhancements**:
  - `run_pca_and_export_artifacts()`: Comprehensive PCA analysis with CSV export functionality
    - Exports loadings, trait variance contributions, PC scores, and variance explained
    - Calculates trait fractional contributions that sum to 1.0
    - Integration with existing visualization functions
  - Added tests verifying fractional contributions sum to 1 in all scenarios
  - Added metadata hygiene tests for `trait_cols=None` behavior
- **Outlier Visualization Module** (`sleap_roots_analyze.outlier_visualization`):
  - Support for all three outlier detection methods
  - `create_comprehensive_outlier_comparison()`: Compare results from multiple detection methods
  - Integration with new PCA artifact export functionality

### Changed
- **Adaptive boxplot layout** for trait visualizations (Issue #73):
  - Auto-switch from vertical to horizontal orientation when genotype count exceeds threshold (default: 8)
  - Configurable via `orientation` ("vertical", "horizontal", "auto") and `horizontal_threshold` parameters
  - Consistent unfilled boxplot styling across both orientations: blue (`#1f77b4`) outlines, green (`#2ca02c`) medians, gridlines
  - Adaptive figure sizing: subplot width scales with genotype count (0.5 in/genotype, min 4.0, max 20.0 inches)
  - Font scaling for x-axis labels when genotype count exceeds 10 (min 6pt)
  - `tight_layout()` called by batched wrapper after suptitle (not in base function) to prevent overlap
  - Replaced seaborn horizontal boxplot with matplotlib for consistent styling
- **PC boxplot layout** now stacks vertically (1 column) instead of grid layout for better display with many genotypes
  - Default figsize updated from (16, 10) to (20, 6) for wider genotype labels
- **Goodness-of-fit display** removed from outlier detection plots (too crowded)
  - Results still available in JSON output and via `print_goodness_of_fit_summary()`
  - Cleaner, more focused visualization
- **Outlier Detection Refactoring**:
  - Removed redundant validation checks from outlier detection functions (now handled by `perform_pca_analysis`)
  - Isolation Forest now uses shared `standardize_data` function for consistency
  - Standardized feature naming convention across all methods (using "Feature_" prefix)
- Made `statsmodels` a required dependency (removed `mixed_model_available` checks)
- Integrated `save_nan_removed_rows` functionality into `remove_nan_samples()`
- Moved utility functions to `data_utils.py` module
- Improved test fixtures organization with categories (heritability, ANOVA, edge cases)
- Updated documentation to reflect actual implementation
- Renamed `link_images_to_samples()` to `link_rhizovision_images_to_samples()` for clarity
- Made `_convert_to_json_serializable()` public API by removing underscore prefix
- Added configurable `alpha` parameter to `perform_anova_by_genotype()` (default: 0.05)
- Changed metadata key from `_metadata` to `__calculation_metadata__` to avoid trait name conflicts
- Refactored `apply_data_cleanup_filters()` to use new modular functions

### Fixed
- **JSON serialization in pipeline summaries** (PR #45)
  - Added Path object handling with `as_posix()` in `convert_to_json_serializable()`
  - Excluded non-serializable sklearn PCA object from StepResult metadata
  - Fixed numpy type serialization (int64, float64) in pipeline summary JSON
- **TypeError in interactive image gallery** when image paths are None
  - Added null check before Path conversion in `create_interactive_image_gallery()`
- **ipykernel hanging bug** in VS Code Jupyter notebooks
  - Pinned ipykernel to <7.0.0 (ipykernel 7.x has known kernel hanging issues)
- Line ending consistency issues across different platforms
- Test accuracy for heritability calculations with mixed models
- Handling of infinity values in statistical calculations
- Edge case handling for insufficient data conditions
- Duplicate imports in `test_statistics.py` (PR #2 review)
- Misplaced docstring between test classes (PR #2 review)
- Brittle test dependency in heritability tests (PR #2 review)

### Development
- Added `black` code formatter configuration
- Added `ruff` linter with Google docstring convention
- Improved test organization and fixture management
- Enhanced numerical stability tests

## [0.0.1] - 2024-12-01

### Added
- **Core Modules**:
  - `data_cleanup.py`: Data loading and cleaning utilities
  - `statistics.py`: Statistical analysis including heritability estimation
  - `data_utils.py`: Utility functions for data processing
  - `outlier_detection.py`: Placeholder for outlier detection (in development)

- **Data Cleaning Features**:
  - `load_trait_data()`: Load CSV/Excel files with validation
  - `get_trait_columns()`: Automatic metadata detection and exclusion
  - `remove_nan_samples()`: Sample filtering based on missing data
  - `remove_zero_inflated_traits()`: Detection and removal of zero-inflated traits
  - `remove_low_variance_traits()`: Filter traits with insufficient variation
  - `link_images_to_samples()`: Connect trait data to image files

- **Statistical Analysis**:
  - `calculate_heritability_estimates()`: Broad-sense heritability using mixed models
  - `perform_anova_by_genotype()`: ANOVA analysis for genotype effects
  - `calculate_trait_statistics()`: Comprehensive trait statistics
  - `identify_high_heritability_traits()`: Threshold-based trait identification
  - `analyze_heritability_thresholds()`: Threshold sensitivity analysis

- **Testing Infrastructure**:
  - Centralized fixtures in `tests/fixtures.py`
  - Test data files for various experimental designs
  - Coverage reporting configuration
  - Edge case and numerical accuracy testing

- **Documentation**:
  - Comprehensive README with examples
  - Testing guide with best practices
  - Release process documentation
  - Claude AI development guidelines

- **Development Tools**:
  - `uv` package manager support with dependency groups
  - `black` code formatting configuration
  - `ruff` linting with Google docstring convention
  - `pytest` with coverage reporting

## Version History

### Versioning Scheme

We use [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Authors

* **Elizabeth Berrigan** - *Initial work* - [GitHub Profile](https://github.com/eberrigan)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](../LICENSE) file for details.
