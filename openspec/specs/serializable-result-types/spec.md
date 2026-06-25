# serializable-result-types Specification

## Purpose
TBD - created by archiving change add-pcaresult-dataclass. Update Purpose after archive.
## Requirements
### Requirement: Serializable PCA Result Type

The package SHALL provide a frozen stdlib `@dataclass` `PCAResult` (with a nested
`FeatureContribution` dataclass) in `result_types.py` that captures only the
JSON-serializable science of a `perform_pca_analysis()` run, excluding sklearn
`PCA`/`StandardScaler` objects. Every scalar field SHALL be a native Python type
(`int`, `float`, `str`, `bool`) — not a numpy scalar — and every array field a
list thereof, so that `json.dumps(dataclasses.asdict(result))` succeeds without a
custom serializer. `PCAResult` SHALL provide a `to_dict()` method returning
`dataclasses.asdict(self)`.

#### Scenario: PCAResult round-trips through JSON as native types

- **WHEN** a `PCAResult` is built from a `perform_pca_analysis()` return dict and
  passed to `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing the string back with `json.loads` SHALL yield `n_components` as
  a Python `int`, `standardized` as a Python `bool`, and every element of
  `explained_variance_ratio`, `eigenvalues`, `cumulative_variance_ratio`,
  `loadings`, and `scores` as a Python `float` (no `np.int64`/`np.float64`)
- **AND** the round-tripped numeric values SHALL equal the input values within
  floating-point tolerance

#### Scenario: No sklearn object is present in the clean view

- **WHEN** `dataclasses.asdict(result)` is inspected
- **THEN** it SHALL contain no sklearn `PCA` or `StandardScaler` object
- **AND** SHALL contain no `pca`, `scaler`, or `feature_metrics_df` key

#### Scenario: cumulative_variance property sums retained components

- **WHEN** `result.cumulative_variance` is accessed
- **THEN** it SHALL return a native Python `float` equal to the sum of
  `explained_variance_ratio` over the retained components

### Requirement: PCAResult Adapter From Legacy Dict

The package SHALL provide
`PCAResult.from_pca_dict(d, *, random_state=None, explained_variance_threshold=None)`
that maps the canonical `perform_pca_analysis()` return dict into a `PCAResult`.
The adapter SHALL assume the canonical key set (`n_components_selected`,
`feature_names`, `loadings`, `eigenvalues`, `explained_variance_ratio`,
`cumulative_variance_ratio`, `transformed_data`, `scaler`,
`feature_contributions`); behavior on a partial dict is unspecified. The adapter
SHALL NOT mutate `d`.

#### Scenario: Adapter maps the core fields from a real run

- **WHEN** `PCAResult.from_pca_dict(d)` is called on the dict returned by
  `perform_pca_analysis()`
- **THEN** `n_components` SHALL equal `int(d["n_components_selected"])`
- **AND** `explained_variance_ratio`, `eigenvalues`, and
  `cumulative_variance_ratio` SHALL each be a `list[float]` of length
  `n_components`
- **AND** `scores` SHALL be built from `d["transformed_data"]` as a nested
  `list[list[float]]` of shape `(n_samples, n_components)`
- **AND** `loadings` SHALL be a nested `list[list[float]]` of shape
  `(n_features, n_components)`

#### Scenario: Nested shapes are preserved at n_components == 1

- **WHEN** a run retains a single component and `from_pca_dict` is applied
- **THEN** `loadings` SHALL be `(n_features, 1)` and `scores` SHALL be
  `(n_samples, 1)` — each inner row a one-element list, not a flattened scalar

#### Scenario: standardized flag reflects whether standardization was applied

- **WHEN** `from_pca_dict` is applied to a run with `standardize=True`
- **THEN** `standardized` SHALL be `True` (the dict carries a fitted `scaler`)
- **WHEN** `from_pca_dict` is applied to a run with `standardize=False`
- **THEN** `standardized` SHALL be `False` (`d["scaler"]` is `None`)

#### Scenario: feature_contributions preserve order and name↔value correspondence

- **WHEN** `from_pca_dict` converts the `feature_contributions` DataFrame
- **THEN** the resulting list SHALL be ordered by `total_contribution`
  descending, matching the source DataFrame row order
- **AND** each `FeatureContribution.feature` SHALL be the DataFrame **index**
  label for its row (not a positional pairing with `feature_names`)
- **AND** its `total_contribution` and `fractional_contribution` SHALL be native
  `float` values equal to that row's values

#### Scenario: Provenance arguments are stamped into the result

- **WHEN** `from_pca_dict(d, random_state=42, explained_variance_threshold=0.95)`
  is called
- **THEN** `result.random_state` SHALL be `42` and
  `result.explained_variance_threshold` SHALL be `0.95`
- **WHEN** the provenance arguments are omitted
- **THEN** both fields SHALL be `None`

### Requirement: PCAResult Public Export

The package SHALL export `PCAResult` and `FeatureContribution` from the top-level
`sleap_roots_analyze` namespace and list them in `__all__`.

#### Scenario: Result types importable from package root

- **WHEN** a consumer runs
  `from sleap_roots_analyze import PCAResult, FeatureContribution`
- **THEN** the import SHALL succeed
- **AND** both names SHALL appear in `sleap_roots_analyze.__all__` with no
  duplicate entries

### Requirement: Non-Breaking PCA Return Shape

Adding `PCAResult` SHALL NOT change the return shape of
`perform_pca_analysis()`; the function SHALL keep returning its existing dict so
all current callers (`result["loadings"]`, the wheat EDPIE paper, the pipeline)
continue to work unchanged.

#### Scenario: Existing dict return is preserved and not mutated

- **WHEN** `perform_pca_analysis()` is called
- **THEN** it SHALL return a `dict` containing the existing keys (including
  `loadings`, `transformed_data`, `explained_variance_ratio`, `eigenvalues`,
  `cumulative_variance_ratio`, `feature_names`, `feature_contributions`,
  `scaler`, `pca`, `n_components_selected`)
- **AND** calling `PCAResult.from_pca_dict(d)` SHALL leave `d`'s keys and values
  unchanged
- **AND** the `PCAResult` view SHALL be opt-in, built only via
  `PCAResult.from_pca_dict()`

### Requirement: Serializable Heritability Result Type

The package SHALL provide a frozen stdlib `@dataclass` `HeritabilityResult`
(with a nested `TraitHeritability` dataclass) in `result_types.py` that captures
only the JSON-serializable science of a `calculate_heritability_estimates()`
run. Every scalar field SHALL be a native Python type (`int`, `float`, `str`,
`bool`), so that `json.dumps(dataclasses.asdict(result))` succeeds without a
custom serializer. `HeritabilityResult` SHALL provide a `to_dict()` method
returning `dataclasses.asdict(self)`.

#### Scenario: HeritabilityResult round-trips through JSON as native types

- **WHEN** a `HeritabilityResult` built from a
  `calculate_heritability_estimates()` dict is passed to
  `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing the string back with `json.loads` SHALL yield `threshold` as a
  Python `float` and, for every entry in `per_trait`, `h2` as a Python `float`,
  `passed_threshold` as a Python `bool`, and the count fields as Python `int`
- **AND** the result SHALL contain no statsmodels model object

#### Scenario: mean_h2 and n_above_threshold derive from successful traits

- **WHEN** `result.mean_h2` and `result.n_above_threshold` are accessed
- **THEN** `mean_h2` SHALL be the mean of `h2` over `per_trait` (a native
  `float`, `0.0` when `per_trait` is empty)
- **AND** `n_above_threshold` SHALL be the count of `per_trait` entries whose
  `passed_threshold` is `True`

### Requirement: HeritabilityResult Adapter From Legacy Dict

The package SHALL provide
`HeritabilityResult.from_heritability_dict(d, threshold)` that maps the
`calculate_heritability_estimates()` return dict (the `remove_low_h2=False`
form, or the first element of the `remove_low_h2=True` tuple) into a
`HeritabilityResult`, without mutating `d`.

#### Scenario: Adapter classifies traits by threshold

- **WHEN** `HeritabilityResult.from_heritability_dict(d, threshold)` is called
- **THEN** `method` SHALL equal
  `d["__calculation_metadata__"]["method_used_for_all_traits"]`
- **AND** the `__calculation_metadata__` entry SHALL NOT appear as a trait
- **AND** each trait carrying a `"heritability"` value SHALL become a
  `TraitHeritability` with `h2` cast to `float` and `passed_threshold` equal to
  `h2 >= threshold`

#### Scenario: Failed traits are separated from successful ones

- **WHEN** the dict contains trait entries carrying an `"error"` (or lacking a
  `"heritability"` value)
- **THEN** those trait names SHALL be collected into `failed_traits`
- **AND** SHALL NOT appear in `per_trait`

### Requirement: HeritabilityResult Public Export

The package SHALL export `HeritabilityResult` and `TraitHeritability` from the
top-level `sleap_roots_analyze` namespace and list them in `__all__`.

#### Scenario: Heritability result types importable from package root

- **WHEN** a consumer runs
  `from sleap_roots_analyze import HeritabilityResult, TraitHeritability`
- **THEN** the import SHALL succeed
- **AND** both names SHALL appear in `sleap_roots_analyze.__all__` with no
  duplicate entries

### Requirement: Non-Breaking Heritability Return Shape

Adding `HeritabilityResult` SHALL NOT change the return shape of
`calculate_heritability_estimates()`; the function SHALL keep returning its
existing dict / tuple so all current callers continue to work unchanged.

#### Scenario: Existing heritability return is preserved and not mutated

- **WHEN** `calculate_heritability_estimates()` is called with
  `remove_low_h2=False`
- **THEN** it SHALL return the existing trait-keyed dict (including the
  `__calculation_metadata__` entry)
- **AND** calling `HeritabilityResult.from_heritability_dict(d, threshold)` SHALL
  leave `d`'s keys and values unchanged
- **AND** the `HeritabilityResult` view SHALL be opt-in, built only via
  `HeritabilityResult.from_heritability_dict()`

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

