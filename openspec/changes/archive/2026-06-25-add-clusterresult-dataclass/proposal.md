# Add ClusterResult Serializable Dataclass Return Types (issue #129)

## Why

`perform_kmeans_clustering` / `perform_gmm_clustering` (`clustering.py`) return
`Dict`s containing `np.ndarray` fields (`cluster_labels`, `cluster_centers` /
`means`, `weights`, …) plus scalar quality metrics. Like PCA and heritability,
this can't cross a JSON boundary (bloom-mcp, caching, an API) without bespoke
conversion. Clustering is also **stochastic**, so the seed must be recorded in
the result to make golden tests reproducible (ties to #118).

The epic review flagged that KMeans and GMM returns differ substantially, so a
single flat type does not fit both. This change models that with a **shared
frozen base `ClusterResult` plus two subclasses** (`KMeansResult`,
`GMMResult`), following the `result_types.py` convention established by
`PCAResult` (#127) and `HeritabilityResult` (#128).

## What Changes

- **Frozen base `ClusterResult`** in `result_types.py` holding the science
  common to both algorithms: `algorithm` (`"kmeans"` | `"gmm"`), `n_clusters`,
  `cluster_labels`, `cluster_sizes`, the three quality metrics
  (`silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`),
  `feature_names`, and the stamped `random_state`.
- **`KMeansResult(ClusterResult)`** adds `cluster_centers` (`(k, n_features)`)
  and `inertia`.
- **`GMMResult(ClusterResult)`** adds `cluster_centers` (the GMM `means`,
  `(k, n_features)`), `weights` (`(k,)`), `bic`, `aic`, `converged`, `n_iter`,
  and `covariance_type`.
- **Adapters** `ClusterResult.from_kmeans_dict(d, *, random_state)` →
  `KMeansResult` and `ClusterResult.from_gmm_dict(d, *, random_state)` →
  `GMMResult`. Each maps the legacy dict's arrays to JSON-native lists with
  explicit native scalar casts, maps GMM `n_components`→`n_clusters` and
  `means`→`cluster_centers`, and stamps `random_state` (the dicts don't carry
  it). Neither adapter mutates `d`.
- **`to_dict()`** on the base (via `dataclasses.asdict`), so each concrete
  subclass serializes flat with no custom serializer. The `algorithm` field is
  the discriminator a JSON consumer reads.
- **Public exports.** `ClusterResult`, `KMeansResult`, `GMMResult` added to the
  package `__all__` with full type hints + Google-style docstrings (every field
  in an `Attributes:` block), satisfying the `test_public_api_docs.py`
  introspection contract.
- **Additive / non-breaking.** Both clustering functions keep returning their
  dicts unchanged (MINOR bump).
- **Tests.** Native-type JSON round-trips for both subclasses (reproducibility
  CI gate), adapter field mapping (incl. GMM `means`→`cluster_centers`),
  `random_state` stamping, a **determinism test** (same seed → identical
  `cluster_labels` via the typed view, #118), exports, and dict-unchanged /
  non-mutating guards.

## Out of Scope (deferred to the epic)

- The shared `docs/result-types.md` pattern doc remains an epic-close
  deliverable.
- Per-sample soft assignments (GMM `probabilities`, `(N, K)`) and per-sample
  distance arrays (`distances_to_centers`) are intentionally omitted from the
  clean summary view; they remain available via the legacy dict.

## Impact

- Affected specs: the `serializable-result-types` capability gains clustering
  requirements (same capability as #127/#128).
- Affected code: `src/sleap_roots_analyze/result_types.py` (base + two
  subclasses + adapters); `src/sleap_roots_analyze/__init__.py` (`__all__`);
  **new** `tests/test_cluster_result.py`.
- No breaking changes; purely additive public API (MINOR version bump).
- Stacked on #128 (`heritabilityresult-dataclass-128`), which is stacked on #127.
