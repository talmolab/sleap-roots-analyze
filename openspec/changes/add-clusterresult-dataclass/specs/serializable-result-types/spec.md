# serializable-result-types Specification (delta)

## ADDED Requirements

### Requirement: Serializable Clustering Result Types

The package SHALL provide a frozen stdlib `@dataclass` base `ClusterResult` and
two frozen subclasses `KMeansResult` and `GMMResult` in `result_types.py`, each
capturing only the JSON-serializable science of a clustering run (no sklearn
estimator objects). Every scalar field SHALL be a native Python type, and every
array field a list thereof, so that `json.dumps(dataclasses.asdict(result))`
succeeds without a custom serializer. The base SHALL provide a `to_dict()`
method returning `dataclasses.asdict(self)`, and an `algorithm` field
(`"kmeans"` | `"gmm"`) that discriminates the concrete type.

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

### Requirement: Clustering Adapters From Legacy Dicts

The package SHALL provide `ClusterResult.from_kmeans_dict(d, *, random_state)`
returning a `KMeansResult`, and `ClusterResult.from_gmm_dict(d, *,
random_state)` returning a `GMMResult`. Each SHALL map the legacy clustering
dict into the typed view, stamp the supplied `random_state`, and not mutate `d`.

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

#### Scenario: Same seed yields identical cluster labels via the typed view

- **WHEN** `perform_kmeans_clustering` is run twice with the same
  `random_state` and each result is passed through `from_kmeans_dict`
- **THEN** the two results' `cluster_labels` SHALL be identical

### Requirement: Clustering Result Public Export

The package SHALL export `ClusterResult`, `KMeansResult`, and `GMMResult` from
the top-level `sleap_roots_analyze` namespace and list them in `__all__`.

#### Scenario: Clustering result types importable from package root

- **WHEN** a consumer runs
  `from sleap_roots_analyze import ClusterResult, KMeansResult, GMMResult`
- **THEN** the import SHALL succeed
- **AND** all three names SHALL appear in `sleap_roots_analyze.__all__` with no
  duplicate entries

### Requirement: Non-Breaking Clustering Return Shapes

Adding the clustering result types SHALL NOT change the return shapes of
`perform_kmeans_clustering` / `perform_gmm_clustering`; both SHALL keep
returning their existing dicts so all current callers continue to work.

#### Scenario: Existing clustering returns are preserved and not mutated

- **WHEN** `perform_kmeans_clustering()` or `perform_gmm_clustering()` is called
- **THEN** it SHALL return its existing dict (including `cluster_labels`,
  `cluster_centers`/`means`, and the quality-metric keys)
- **AND** calling the corresponding adapter SHALL leave the dict's keys and
  values unchanged
- **AND** the typed view SHALL be opt-in, built only via the adapters
