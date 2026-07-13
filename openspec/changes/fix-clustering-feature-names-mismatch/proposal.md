## Why

`perform_kmeans_clustering`, `perform_gmm_clustering`, and
`perform_hierarchical_clustering` (all in `clustering.py`) each snapshot
`feature_names = df_clean.columns.tolist()` **before** calling
`standardize_data`, then return that snapshot unchanged even when
`standardize_data` goes on to drop non-numeric and exact-zero-variance
columns (`pca.py:624-654`). The result is a silent **positional**
misalignment between `feature_names` and every feature-indexed array in the
same return dict (`cluster_centers`, `means`, `data_processed`) — not a
length mismatch a caller would notice, a mislabeling. All three producers
discard `standardize_data`'s filtered `df_clean` via `_` (e.g.
`X_processed, scaler, _ = standardize_data(df_clean)`), so nothing in the
`standardize=True` branch ever re-derives names from what was actually
clustered. The `standardize=False` branch is worse: it does no filtering at
all (`X_processed = df_clean.values`), so a non-numeric column reaches
`KMeans.fit()` / `GaussianMixture.fit()` / `linkage()` directly and either
mislabels `feature_names` (constant column) or fails with sklearn's own
`could not convert string to float` error (non-numeric column) instead of
the clearer "No numeric columns with non-zero variance found" message the
`standardize=True` path already gives for the equivalent all-filtered case.
(Verified empirically: both cases are already wrapped in a method-specific
`RuntimeError` by these producers' existing broad `except Exception`
handlers — this is not an uncontrolled crash today, just a confusing
message and, independently, a `feature_names` mismatch whenever at least
one valid column survives.)

This is the same bug class already fixed for PCA in #74/#76:
`perform_pca_analysis` (`pca.py:793-812`) re-derives `feature_names` from the
post-filter `df_clean` on the `standardize=True` path, and manually re-applies
the same numeric + zero-variance filter to derive `feature_names` on the
`standardize=False` path. The clustering producers never got either half of
that fix.

`KMeansResult` and `GMMResult` both carry `feature_names` as a base
`ClusterResult` field (`result_types.py:421`), so the mismatch propagates
through the typed, JSON-serialized public API too. `src/sleap_roots_analyze
/outlier_detection.py`'s `detect_outliers_kmeans`, `detect_outliers_gmm`,
and `detect_outliers_hierarchical` each `return {**cluster_result, ...}` /
`{**hier_result, ...}`, re-exporting the same buggy `feature_names`
verbatim into their own public return dict — including through the QC
pipeline's `DetectOutliersStep` (`pipeline/steps/detect_outliers.py`), which
calls all three directly.

GitHub Issue: #183

**Sequencing note re: #179/PR #182.** Issue #183 also flags that the new
`hierarchical_cluster_labels` entry point and `HierarchicalResult` dataclass
(#179, PR #182) will inherit this same bug. As of this proposal, **PR #182 is
still open** (not merged to `main`) — `HierarchicalResult`,
`from_hierarchical_dict`, and `hierarchical_cluster_labels` do not exist on
`main` or on this proposal's branch. This proposal fixes `clustering.py`'s
three producers directly, which is independent of PR #182 and unblocks
nothing that depends on it. Whichever of #182/#183 merges second SHALL add a
short regression test confirming `HierarchicalResult.feature_names` reflects
the corrected values (no code change is needed there, since the adapter
passes `feature_names` through verbatim) — tracked as a small follow-up, not
a blocker for either PR.

## What Changes

- Port the `perform_pca_analysis` pattern (`pca.py:793-812`) to all three
  clustering producers, fixing **both** branches:
  - `standardize=True`: use the `df_clean` that `standardize_data` actually
    returns (currently discarded as `_`) and re-derive
    `feature_names = df_clean.columns.tolist()` from it, after filtering.
  - `standardize=False`: apply the same numeric-only + non-zero-variance
    filter `standardize_data` uses internally (`select_dtypes` +
    `var(ddof=0) > 0`) before deriving `feature_names` and `X_processed`,
    instead of passing `df_clean.values` through unfiltered. Raise
    `ValueError("No numeric columns with non-zero variance found")` when
    nothing survives, matching `standardize_data`'s own guard — this stays
    inside the existing `try`/`except Exception as e: raise RuntimeError(...)`
    block (unchanged), so the effective result is the same `RuntimeError`
    the `standardize=True` path already raises for the identical condition
    today, not a new/different exception type for only one branch.
- No change needed in `ClusterResult.from_kmeans_dict` / `from_gmm_dict`, or
  the `KMeansResult` / `GMMResult` dataclasses — they pass `feature_names`
  through verbatim, so they inherit the corrected values automatically once
  the three producers are fixed. Same applies transitively to
  `outlier_detection.py`'s `detect_outliers_kmeans` / `detect_outliers_gmm` /
  `detect_outliers_hierarchical`, which dict-spread the producer output.
- Add regression coverage in `tests/test_outlier_detection.py` confirming
  `detect_outliers_kmeans` / `_gmm` / `_hierarchical` surface the corrected
  `feature_names`, since these three currently have no `feature_names`
  coverage at all and would otherwise silently carry the bug forward
  undetected.

## Impact

- Affected specs: `serializable-result-types` — new requirements covering
  clustering `feature_names` accuracy and the fully-filtered error path, plus
  a clarifying scenario added to the existing "Non-Breaking Clustering
  Return Shapes" requirement (dict keys are unchanged; only previously-wrong
  `feature_names` values are corrected).
- Affected code:
  - `src/sleap_roots_analyze/clustering.py` — `perform_kmeans_clustering`
    (~L104-111), `perform_gmm_clustering` (~L355-362),
    `perform_hierarchical_clustering` (~L572-579)
  - `src/sleap_roots_analyze/outlier_detection.py` — no code change expected
    (`detect_outliers_kmeans`/`_gmm`/`_hierarchical` inherit the fix via
    dict-spread), but needs new regression tests
  - `tests/fixtures.py` — reuse `pca_constant_feature_data`
    (`tests/fixtures.py:1046`); add a new fixture mixing a constant numeric
    column and a non-numeric (string ID) column in the same frame
  - `tests/test_clustering.py` — parametrized `feature_names`/array-length
    and named-value regression tests across the three producers, for both
    `standardize` values
  - `tests/test_cluster_result.py` — extend `KMeansResult`/`GMMResult`
    adapter coverage so `feature_names` is checked against
    `cluster_centers.shape[1]` / `means.shape[1]` on filtered input
  - `tests/test_outlier_detection.py` — new coverage for
    `detect_outliers_kmeans`/`_gmm`/`_hierarchical` per above
  - `tests/test_step_detect_outliers.py` — guardrail confirming the QC
    pipeline's `DetectOutliersStep` still sees `feature_names` matching the
    cleanup step's surviving trait list (protected today by #177's cleanup
    filter; confirms no regression once this fix lands)
  - `docs/CHANGELOG.md` `[Unreleased]` — new `### Fixed` entry, explicitly
    distinguished from the existing #177 `### Changed` entry (see note below)
- No breaking changes: dict keys, shapes for already-clean input, and typed
  adapter signatures are all unchanged. Only `feature_names` values for
  inputs with constant/non-numeric columns (previously wrong) and
  `standardize=False` + non-numeric-column failures (previously a confusing
  `RuntimeError: ... could not convert string to float`, now either a
  successful cluster over the surviving numeric columns or the same clean
  `RuntimeError: ... No numeric columns with non-zero variance found` the
  `standardize=True` path already raises) are corrected.
- Explicitly out of scope: no new `excluded_features`-style key is added to
  the return dicts (`perform_pca_analysis`, the precedent this ports from,
  doesn't carry one either — see `pca.py:840-844`); pipeline metadata
  propagation (#80) and the QC cleanup filter (#177) are untouched, since
  neither is part of `clustering.py`'s producer-internal contract;
  `HierarchicalResult`/`hierarchical_cluster_labels` (#179/PR #182, still
  open) per the sequencing note above.
- **Documentation note:** `docs/CHANGELOG.md`'s existing `[Unreleased]` #177
  entry says "PCA / UMAP / clustering results are unchanged (those paths
  already dropped constants before fitting)." That claim is true of the
  *fitted array values* (already filtered correctly) but not of the
  `feature_names` *labels* this proposal fixes — the new CHANGELOG entry
  must explicitly disambiguate from that line so a reader doesn't mistake
  #183 as redundant with #177.
- Related: #74/#76 (same bug, already fixed for `perform_pca_analysis`); #80
  (separate — pipeline `trait_names` metadata, not producer-dict internals);
  #177 (QC cleanup shields the pipeline path already; does not fix direct
  callers); #179/PR #182 (open, not yet merged — see sequencing note above).
