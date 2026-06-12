# Tasks — Add ClusterResult Serializable Dataclasses (issue #129)

> Single `feat: ... (#129)` commit (CI green at HEAD); exports land with the
> dataclasses. Stacked on #128 → #127. OpenSpec archive post-merge.

## 1. Failing tests first (red) — `tests/test_cluster_result.py`
- [x] 1.1 `test_kmeans_json_roundtrip_native_types`: build
      `ClusterResult.from_kmeans_dict(kmeans_cluster_result, random_state=42)`,
      assert `json.dumps(asdict(...))` succeeds, then `json.loads` and assert
      `algorithm == "kmeans"`, `cluster_labels` are `int`, `cluster_centers`
      nested `float`, `inertia`/metrics `float`, `random_state` `int`.
- [x] 1.2 `test_gmm_json_roundtrip_native_types`: same for
      `from_gmm_dict(gmm_cluster_result, random_state=42)`; assert
      `algorithm == "gmm"`, `cluster_centers` from means, `weights` floats,
      `bic`/`aic` float, `converged` bool, `n_iter` int.
- [x] 1.3 `test_kmeans_adapter_maps_fields`: `n_clusters == int(d["n_clusters"])`;
      `cluster_centers` shape `(k, n_features)`; `cluster_sizes` ints summing to
      n_samples; `random_state == 42`.
- [x] 1.4 `test_gmm_adapter_maps_n_components_and_means`: `n_clusters ==
      int(d["n_components"])`; `cluster_centers` equals `d["means"]`;
      `covariance_type` carried.
- [x] 1.5 `test_determinism_same_seed_identical_labels`: run
      `perform_kmeans_clustering(simple_cluster_data, n_clusters=3,
      random_state=42)` twice, build typed views, assert identical
      `cluster_labels` (#118).
- [x] 1.6 `test_exports_and_all`: import `ClusterResult`, `KMeansResult`,
      `GMMResult` from package root; all in `__all__`, no dupes.
- [x] 1.7 `test_dicts_unchanged_and_nonmutating`: kmeans/gmm dict keys preserved;
      adapters do not mutate input.

## 2. Implement to green
- [x] 2.1 Add frozen base `ClusterResult` (common fields + `algorithm` +
      `to_dict()`), then frozen `KMeansResult` / `GMMResult` subclasses to
      `result_types.py` (Google docstrings with `Attributes:` per field).
- [x] 2.2 Implement `ClusterResult.from_kmeans_dict(d, *, random_state)` and
      `ClusterResult.from_gmm_dict(d, *, random_state)`: native list/scalar
      casts; GMM `n_components`→`n_clusters`, `means`→`cluster_centers`; stamp
      `random_state`; no mutation of `d`.
- [x] 2.3 Export `ClusterResult`, `KMeansResult`, `GMMResult` from `__init__.py`
      (`__all__`).

## 3. Verify non-breaking
- [x] 3.1 Existing `tests/test_clustering.py` pass; return shapes unchanged
      (covered by task 1.7).

## 4. Pre-merge
- [x] 4.1 `black` + `ruff` + full `pytest` green; `openspec validate
      add-clusterresult-dataclass --strict` passes.
