## Why

Fixes #80. Follow-up to #74/#76: `PCAAnalysisStep` correctly computes a
zero-variance-filtered trait list (`pca_results["feature_names"]`) but never
writes it back into `metadata["trait_names"]`/`valid_trait_names"`
(`pca_analysis.py:166-174`) — the shared "traits still in play" contract
every other filtering step in the pipeline maintains
(`cleanup_traits.py`, `filter_heritability.py`, `remove_outliers.py`,
`detect_outliers.py`). Downstream steps that read `trait_names` silently get
the pre-PCA list instead. See `design.md` for the full investigation:
why the fix must touch two files, not one (`UMAPAnalysisStep` and
`GenerateStaticFiguresStep` sit on asymmetric DAG branches relative to
`PCAAnalysisStep`), and the severity finding that one plotting function
(`create_pca_biplot`) has a latent positional-mislabeling bug from this same
root cause, beyond what issue #80 describes.

## What Changes

- `pca_analysis.py` (`PCAAnalysisStep.execute`, ~L166-174): update
  `metadata["trait_names"]` and `metadata["valid_trait_names"]` (mirrored
  key, per the existing pipeline-wide convention) to the post-filter
  `feature_names` list. Add `metadata["original_trait_names"] = list(trait_cols)`
  to preserve the pre-filter list for traceability. `excluded_zero_variance_traits`
  and `n_traits_after_filtering` are unchanged.
- `viz_pipeline.py` (`_run_generate_static_figures`, ~L427-442): extend the
  existing PCA-branch cherry-pick block to also copy `trait_names` and
  `original_trait_names` into `combined_metadata`, so Step 9 uses the
  PCA-corrected list instead of the pre-PCA list relayed from
  Step 8 → Step 6 → Step 2. This merge is only applied `if pca_task_result:`
  (existing guard) — when PCA doesn't run, Step 9 keeps the
  `08_genotype_aggregation` branch's own value unchanged.
- `generate_static_figures.py` (~L121): no logic change — once the
  orchestrator forwards the corrected key, the existing
  `metadata.get("trait_names", metadata.get("valid_trait_names", []))`
  lookup and its downstream uses (lines 345, 409, 537) automatically resolve
  to the filtered, correctly-aligned set. This corrects the real, verified
  positional-mislabeling bug in `create_pca_biplot` (arrows silently
  mislabeled when an excluded trait isn't the last original column, see
  `design.md`). It does **not** change `create_feature_contribution_plot`'s
  behavior: that function always takes the "pre-calculated contributions"
  branch (`pca_results["feature_contributions"]`, unconditionally set by
  `perform_pca_analysis()` at `pca.py:889`, already indexed by the correct
  filtered `feature_names`) and never uses its own `trait_names` parameter
  positionally in current pipeline usage — passing the corrected list is
  still strictly more correct, just not a behavior change for this function
  today.
- Do **not** touch `generate_summary_viz.py`. Step 12 deliberately sources
  `trait_names` from `02_calculate_statistics` (pre-PCA) via
  `viz_pipeline.py:504`, for an "how many traits were in the input" report
  number (`n_traits_final`), not "how many made it into PCA/UMAP." Changing
  that source would change the meaning of an existing summary-report field;
  it isn't flagged in #80 and the heritability/aggregation branch's own
  trait-filtering logic (steps 6-8) is untouched by this proposal.
- Add regression coverage (see `tasks.md` for full detail, including reuse
  of existing interleaved-trait fixtures instead of hand-rolling new ones):
  - `test_step_pca_analysis.py`: assert the corrected metadata contract on
    an **interleaved** (non-trailing) zero-variance fixture, and the
    unchanged-value case when nothing is excluded.
  - `test_step_umap_analysis.py`: assert `UMAPAnalysisStep`, driven by an
    actually-executed `PCAAnalysisStep` result (not a hand-mocked one) that
    excluded traits, uses the filtered `trait_names` for `feature_cols` and
    the logged `n_traits`.
  - `test_viz_pipeline_zero_variance.py`: an interleaved-constant-trait
    pipeline variant (catching the biplot mislabeling the existing
    trailing-only fixture cannot), a `config.umap.enabled = True` assertion
    that `umap_parameters.json`'s `n_traits` is 4 not 8, an assertion that
    `09_generate_static_figures`'s effective `trait_names` matches the
    PCA-filtered set, and a regression test that Step 9 keeps the
    pre-PCA value when the PCA step doesn't run.

## Impact

- Affected specs: `visualization-pipeline` — modify the "Pipeline Step
  Parameter Passing" requirement to add `trait_names`/`original_trait_names`
  propagation rules for `PCAAnalysisStep`, and add a new requirement
  covering the orchestrator's metadata merge for `GenerateStaticFiguresStep`.
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/pca_analysis.py` (~L166-174)
  - `src/sleap_roots_analyze/pipeline/pipelines/viz_pipeline.py`
    (`_run_generate_static_figures`, ~L427-450)
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py`
    (~L121) — no logic change, benefits once the orchestrator forwards the
    corrected key
  - `tests/test_step_pca_analysis.py`, `tests/test_step_umap_analysis.py`,
    `tests/test_viz_pipeline_zero_variance.py`
  - `docs/CHANGELOG.md` `[Unreleased]` — new `### Fixed` entry
- No breaking changes: `trait_names`/`valid_trait_names` keys already exist
  on every step's metadata; only their *value* changes, and only for inputs
  that have zero-variance traits — a case the issue documents as already
  producing wrong values downstream today. `original_trait_names` is a new,
  additive key. No change to step ordering, DAG shape, or any dict's key
  set otherwise.
- Explicitly out of scope: `generate_summary_viz.py`'s pre-PCA trait count
  (see above); `HierarchicalResult`/clustering `feature_names` (separate bug
  class, already handled for clustering producers under #183); the
  heritability/aggregation branch's own trait-filtering logic (steps 6-8),
  which is independent of PCA's zero-variance filter and stays untouched;
  reordering or renaming any trait — this proposal only propagates a list
  that `perform_pca_analysis()` already computes correctly. **Known
  limitation, not fixed here:** none of the trait-name metadata this
  proposal corrects (`trait_names`, `original_trait_names`,
  `excluded_zero_variance_traits`) is persisted to a run's output directory
  today — it only lives in the in-memory `StepResult` chain for the
  duration of one pipeline execution, so a scientist auditing a completed
  run's files after the fact still can't see which traits PCA excluded.
  Persisting it (e.g. a small `pca_metadata.json` alongside
  `data/pca/loadings.csv`) is a reasonable follow-up but is additional new
  scope beyond what #80 asks for, so it's left for a separate issue/proposal.
- Related: #74/#76 (original PCA zero-variance fix; this proposal is its
  direct follow-up for the metadata side-effect it left unaddressed); #183
  (same "positional relabeling from an unfiltered name list" bug class,
  fixed for the clustering producers' own return dicts rather than pipeline
  metadata).
