## Context

Issue #80 reports that `PCAAnalysisStep` filters zero-variance traits into
`pca_results["feature_names"]` but never writes that filtered list back into
pipeline metadata, so downstream steps reading `metadata["trait_names"]` get
the pre-PCA list. The issue's own "Proposed Fix" section suggests a single
change: update `pca_analysis.py`'s metadata dict. This document records why
that alone is insufficient, and what the actual fix requires.

`VizPipeline` (`viz_pipeline.py`) runs 12 steps as a NetworkX DAG, not a
strict linear chain. Two steps read `trait_names` downstream of PCA, and
they reach it through different paths:

```
02_calculate_statistics ─┬─→ 03_pca_analysis ─┬─→ 04_umap_analysis
                          │                    └─→ 05_cluster_analysis
                          └─→ 06_heritability_analysis ─→ 08_genotype_aggregation ─→ 09_generate_static_figures
```

`04_umap_analysis` receives `prev_result` **directly** from `03_pca_analysis`
(`viz_pipeline.py:333-344`), and `UMAPAnalysisStep.execute` spreads
`**prev_result.metadata` verbatim (`umap_analysis.py:161-165`). So once
`PCAAnalysisStep` writes the corrected `trait_names`, UMAP inherits it for
free — no change needed in `viz_pipeline.py` for that path.

`09_generate_static_figures` does **not** receive `prev_result` from PCA at
all. Its primary metadata source is `08_genotype_aggregation`
(`viz_pipeline.py:414-465`), which is a pure passthrough
(`genotype_aggregation.py:31`) of `06_heritability_analysis`'s metadata,
itself forked from `02_calculate_statistics` — a branch that never touches
PCA. The orchestrator already knows it needs specific values from the PCA
branch and hand-picks them into `combined_metadata`
(`pca_results`/`top_features`/`n_pca_components`/`pca_explained_variance` at
`viz_pipeline.py:427-442`, `umap_results` from the UMAP branch at
443-450) — but `trait_names` isn't one of the picked keys. Fixing
`pca_analysis.py` alone is therefore a no-op for every static figure: Step 9
would keep reading the pre-PCA (heritability-filtered, not
zero-variance-filtered) trait list regardless.

**Severity finding beyond the issue's own description:** `generate_static_figures.py`
forwards its `trait_cols` variable as `trait_names=` into two plotting
functions in `visualization.py` that accept a `pca_results` dict alongside
it: `create_pca_biplot` and `create_feature_contribution_plot`. Both were
initially suspected of the same bug — positionally indexing into
`pca_results["loadings"]` using a `trait_names` list that, without this fix,
is longer than `loadings.shape[0]` whenever a trait was excluded. Verified
against current source:

- `create_pca_biplot` (`visualization.py:2092`,
  `n_features = min(len(trait_names), loadings.shape[0])`) **is** affected:
  `trait_names[idx]` labels each plotted feature arrow where `idx` indexes
  into `loadings` rows (post-filter). If the unfiltered, longer `trait_cols`
  is passed and the excluded trait isn't the last column of the original
  list, arrows get labeled with the wrong trait name — the same "positional
  misalignment, not a length mismatch a caller would notice" bug class
  already fixed for the clustering producers under #183, just reached via
  the pipeline metadata path instead of a producer's own return dict.
  `tests/test_viz_pipeline_zero_variance.py`'s existing fixture can't catch
  this: its 4 constant traits are appended *after* the 4 variable ones, so
  `trait_names[0:n_features]` happens to equal the correct filtered set by
  coincidence of column order — the mislabeling is latent, not currently
  exercised by any test.
- `create_feature_contribution_plot` is **not** affected in current usage:
  `perform_pca_analysis()` unconditionally sets
  `result["feature_contributions"]` (`pca.py:889`), a DataFrame already
  indexed by the correct, filtered `feature_names`. `create_feature_contribution_plot`
  always takes its "pre-calculated contributions" branch
  (`visualization.py:1837-1921`) given that key is present, and never reaches
  the on-the-fly branch that would positionally index the passed-in
  `trait_names` (that branch is dead code given current pipeline data flow).
  Passing the corrected `trait_names` is still strictly more correct
  defensively, but is not fixing an active bug for this function today.

## Goals / Non-Goals

- Goals: make `metadata["trait_names"]`/`valid_trait_names` reflect the
  traits PCA actually used, for every step that can reach them; preserve the
  pre-filter list under a new, additive key; fix the verified
  `create_pca_biplot` mislabeling as a consequence.
- Non-Goals: restructuring the `VizPipeline` DAG (e.g. routing Step 9 through
  the PCA branch directly) — the existing cherry-pick pattern is the
  established convention for this orchestrator and extending it is the
  minimal, consistent change; persisting excluded-trait metadata to disk
  (see "Known limitation" in `proposal.md`); touching the
  heritability/aggregation branch's own (unrelated) trait filtering.

## Decisions

- **Decision:** Fix `pca_analysis.py`'s own metadata output AND extend
  `viz_pipeline.py`'s existing PCA-branch cherry-pick block, rather than
  restructuring the DAG so Step 9 depends on PCA metadata directly.
  **Why:** the cherry-pick pattern already exists and is exactly how this
  orchestrator wires cross-branch data (see the `umap_results` merge
  immediately below it); duplicating that pattern for `trait_names` is
  consistent with the codebase's own convention and touches the fewest
  lines. Restructuring the DAG's dependency edges would be a larger,
  riskier change for no additional benefit here.
- **Decision:** Do not modify `create_feature_contribution_plot` or add a
  test asserting it mislabels today — it doesn't. Document why in this
  proposal instead of silently dropping the (incorrect) initial claim, so a
  future reader doesn't wonder why only `create_pca_biplot` gets a
  regression test.
- **Alternative considered:** Have `PCAAnalysisStep` return a *new* key
  (e.g. `pca_filtered_trait_names`) instead of overwriting `trait_names`/
  `valid_trait_names` in place. Rejected: every other filtering step
  (`cleanup_traits.py`, `filter_heritability.py`, `remove_outliers.py`,
  `detect_outliers.py`) already treats `trait_names`/`valid_trait_names` as
  the single mutable "traits still in play" contract, updating it in place
  each time a step filters. Introducing a differently-named key for PCA
  alone would be inconsistent with that established convention and would
  require every downstream consumer to know to check yet another key.

## Risks / Trade-offs

- Risk: a caller outside this pipeline's tests relies on `PCAAnalysisStep`'s
  `trait_names` being the *unfiltered* list. Mitigation: grepped all
  consumers of `PCAAnalysisStep` output within `src/` and `tests/`; the only
  consumers are `UMAPAnalysisStep`, `ClusterAnalysisStep`,
  `IdentifyInterestingGenotypesStep` (via the DAG), and this repo's own
  tests — no external/public API depends on the old (buggy) value.
- Trade-off: as noted above, the excluded-trait diagnostics remain
  in-memory-only (not persisted to disk). Accepted for this proposal's
  scope; flagged as a follow-up.

## Migration Plan

None required — this is a same-run metadata correction with no persisted
state or schema to migrate. No breaking changes to any dict's key set.

## Open Questions

- Should excluded-trait metadata be persisted to a run's output directory
  (e.g. `data/pca/pca_metadata.json`) for post-hoc auditability? Deferred to
  a follow-up issue per the "Known limitation" note in `proposal.md`.
