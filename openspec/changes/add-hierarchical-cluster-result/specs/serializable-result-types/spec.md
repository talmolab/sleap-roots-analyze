## ADDED Requirements

### Requirement: Hierarchical Clustering Labeled Entry Point

The package SHALL provide a public `hierarchical_cluster_labels(data, *,
n_clusters=None, method="ward", metric="euclidean", standardize=True,
optimization_method="silhouette", max_clusters=10)` that composes
`perform_hierarchical_clustering`, `calculate_optimal_clusters_hierarchical` (only
when `n_clusters` is `None`), and `cut_dendrogram` into a single labeled dict. The
returned dict SHALL carry `cluster_labels`, `n_clusters`, `cluster_sizes`,
`silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`,
`feature_names`, and the hierarchical provenance keys `linkage_method`,
`distance_metric`, `cophenetic_correlation`, and `cut_height`. When `n_clusters` is
`None`, `optimization_method` SHALL be one of `"silhouette"`, `"calinski"`, or
`"davies_bouldin"` (the values `calculate_optimal_clusters_hierarchical` accepts). It
SHALL NOT change the return shape of `perform_hierarchical_clustering`, and SHALL
propagate the underlying exception rather than returning a partial dict: `ValueError`
for the invalid-argument cases below, and the composed functions' `RuntimeError` for
degenerate metric failures (e.g. `n_clusters == n_samples`, where the silhouette score
is undefined).

#### Scenario: Auto-selects the number of clusters when omitted

- **WHEN** `hierarchical_cluster_labels(data)` is called with `n_clusters=None`
- **THEN** `n_clusters` SHALL equal the `optimal_n_clusters` that
  `calculate_optimal_clusters_hierarchical(..., method=optimization_method)` returns
  for the same input
- **AND** `cluster_labels` SHALL have length equal to the number of clustered samples
  and `n_clusters` SHALL equal the number of distinct labels

#### Scenario: Honors an explicit cluster count

- **WHEN** `hierarchical_cluster_labels(data, n_clusters=3)` is called
- **THEN** the returned dict SHALL have `n_clusters == 3`
- **AND** `cluster_sizes` SHALL have length 3 and sum to the number of clustered
  samples

#### Scenario: Deterministic labels for identical input

- **WHEN** `hierarchical_cluster_labels(data)` is called twice on identical input
- **THEN** the two returned `cluster_labels` SHALL be identical

#### Scenario: Single-cluster request yields finite zero-valued metrics

- **WHEN** `hierarchical_cluster_labels(data, n_clusters=1)` is called
- **THEN** the returned dict SHALL have `n_clusters == 1` and `cluster_sizes` of
  length 1
- **AND** `silhouette_score`, `davies_bouldin_score`, and `calinski_harabasz_score`
  SHALL each be `0.0` (finite, so `from_hierarchical_dict(...).to_json()` succeeds
  under `allow_nan=False`)

#### Scenario: Invalid input propagates ValueError

- **WHEN** `hierarchical_cluster_labels` is called with `method="ward"` and a
  non-euclidean `metric`, with fewer than 2 valid rows, or with all-NaN rows
- **THEN** a `ValueError` SHALL be raised and no partial dict SHALL be returned

#### Scenario: Degenerate cluster count propagates the underlying error

- **WHEN** `hierarchical_cluster_labels(data, n_clusters=n_samples)` is called (a
  cluster-per-sample request, for which the silhouette score is undefined)
- **THEN** the underlying `RuntimeError` SHALL propagate and no partial dict SHALL be
  returned

#### Scenario: perform_hierarchical_clustering return shape is preserved

- **WHEN** `perform_hierarchical_clustering(data)` is called
- **THEN** it SHALL keep returning its dendrogram dict (`linkage_matrix`,
  `cophenetic_correlation`, `data_processed`, ...) with no `cluster_labels` key
- **AND** `hierarchical_cluster_labels` SHALL be the opt-in labeled entry point built
  on top of it

## MODIFIED Requirements

### Requirement: Serializable Clustering Result Types

The package SHALL provide a frozen stdlib `@dataclass` base `ClusterResult` and
three frozen subclasses `KMeansResult`, `GMMResult`, and `HierarchicalResult` in
`result_types.py`, each capturing only the JSON-serializable science of a clustering
run (no sklearn estimator objects). Every scalar field SHALL be a native Python type,
and every array field a list thereof, so that `json.dumps(dataclasses.asdict(result))`
succeeds without a custom serializer. The base SHALL provide a `to_dict()` method
returning `dataclasses.asdict(self)`, and an `algorithm` field (`"kmeans"` | `"gmm"`
| `"hierarchical"`) that discriminates the concrete type. The base `random_state`
field SHALL be `Optional[int]` with default `None`, so a deterministic algorithm with
no seed (hierarchical) can omit it while seeded algorithms (KMeans/GMM) stamp the
`int` seed.

#### Scenario: KMeansResult round-trips through JSON as native types

- **WHEN** a `KMeansResult` built from a `perform_kmeans_clustering()` dict is
  passed to `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing it back SHALL yield `algorithm == "kmeans"`, `cluster_labels`
  as a list of `int`, `cluster_centers` as nested `list[list[float]]`,
  `inertia` and the quality metrics as `float`, and `random_state` as `int`

#### Scenario: GMMResult round-trips through JSON as native types

- **WHEN** a `GMMResult` built from a `perform_gmm_clustering()` dict is passed
  to `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing it back SHALL yield `algorithm == "gmm"`, `cluster_centers`
  (the GMM means) as nested `list[list[float]]`, `weights` as a list of
  `float`, `bic`/`aic` as `float`, `converged` as `bool`, and `n_iter` as `int`

#### Scenario: HierarchicalResult round-trips through JSON as native types

- **WHEN** a `HierarchicalResult` built from a `hierarchical_cluster_labels()` dict
  is passed to `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing it back SHALL yield `algorithm == "hierarchical"`, `cluster_labels`
  and `cluster_sizes` as lists of `int`, the three quality metrics and
  `cophenetic_correlation`/`cut_height` as `float`, and `random_state` as `null`

#### Scenario: ClusterResult accepts a null random_state

- **WHEN** a `ClusterResult` (or any subclass) is constructed with
  `random_state=None`
- **THEN** construction SHALL succeed
- **AND** `dataclasses.asdict(result)["random_state"]` SHALL be `None`, serializing to
  JSON `null`

### Requirement: Clustering Adapters From Legacy Dicts

The package SHALL provide `ClusterResult.from_kmeans_dict(d, *, random_state)`
returning a `KMeansResult`, `ClusterResult.from_gmm_dict(d, *, random_state)`
returning a `GMMResult`, and `ClusterResult.from_hierarchical_dict(d)` returning a
`HierarchicalResult`. The KMeans and GMM adapters SHALL stamp the supplied
`random_state`; the hierarchical adapter takes no `random_state` argument (hierarchical
is deterministic) and SHALL stamp `random_state=None`. Each SHALL map the legacy
clustering dict into the typed view and SHALL NOT mutate `d`.

#### Scenario: KMeans adapter maps the legacy dict

- **WHEN** `ClusterResult.from_kmeans_dict(d, random_state=42)` is called on a
  `perform_kmeans_clustering()` dict
- **THEN** `n_clusters` SHALL equal `int(d["n_clusters"])`
- **AND** `cluster_centers` SHALL be `d["cluster_centers"]` as nested
  `list[list[float]]`
- **AND** `cluster_labels` SHALL be a list of `int` and `random_state` SHALL be
  `42`

#### Scenario: GMM adapter maps n_components and means

- **WHEN** `ClusterResult.from_gmm_dict(d, random_state=42)` is called on a
  `perform_gmm_clustering()` dict
- **THEN** `n_clusters` SHALL equal `int(d["n_components"])`
- **AND** `cluster_centers` SHALL be built from `d["means"]`
- **AND** `weights`, `bic`, `aic`, `converged`, `n_iter`, and `covariance_type`
  SHALL be carried from the dict with native casts

#### Scenario: Hierarchical adapter maps the labeled dict without a seed

- **WHEN** `ClusterResult.from_hierarchical_dict(d)` is called on a
  `hierarchical_cluster_labels()` dict
- **THEN** the result SHALL be a `HierarchicalResult` with `algorithm ==
  "hierarchical"` and populated `cluster_labels`, `cluster_sizes`, `n_clusters`, and
  the three quality scores
- **AND** `linkage_method`, `distance_metric`, `cophenetic_correlation`, and
  `cut_height` SHALL be carried from the dict with native casts
- **AND** `random_state` SHALL be `None` and `d` SHALL be unchanged

#### Scenario: Same seed yields identical cluster labels via the typed view

- **WHEN** `perform_kmeans_clustering` is run twice with the same
  `random_state` and each result is passed through `from_kmeans_dict`
- **THEN** the two results' `cluster_labels` SHALL be identical

### Requirement: Clustering Result Public Export

The package SHALL export `ClusterResult`, `KMeansResult`, `GMMResult`,
`HierarchicalResult`, and the producer `hierarchical_cluster_labels` from the
top-level `sleap_roots_analyze` namespace and list them in `__all__`. The
`ALGORITHM_HIERARCHICAL` discriminator constant SHALL be exported from
`result_types` (listed in `result_types.__all__` alongside `ALGORITHM_KMEANS` and
`ALGORITHM_GMM`) rather than the package root, because the root public-API docstring
audit requires every root `__all__` entry to be a class or callable and a bare `str`
constant would fail it.

#### Scenario: Clustering result types importable from package root

- **WHEN** a consumer runs
  `from sleap_roots_analyze import ClusterResult, KMeansResult, GMMResult,
  HierarchicalResult, hierarchical_cluster_labels`
- **THEN** the import SHALL succeed
- **AND** all five names SHALL appear in `sleap_roots_analyze.__all__` with no
  duplicate entries
- **AND** `ALGORITHM_HIERARCHICAL` SHALL be importable from
  `sleap_roots_analyze.result_types` and listed in `result_types.__all__`
