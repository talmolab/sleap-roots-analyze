## ADDED Requirements

### Requirement: Accurate Clustering Feature Names After Column Filtering

`perform_kmeans_clustering`, `perform_gmm_clustering`, and
`perform_hierarchical_clustering` SHALL derive `feature_names` from the
columns actually present in the data used for clustering, after any
non-numeric or zero-variance column has been dropped, for both
`standardize=True` and `standardize=False`. `feature_names` SHALL never
include a column that was dropped before fitting, and its order SHALL match
the column order of the corresponding feature-indexed array
(`cluster_centers` for KMeans, `means` for GMM, `data_processed` for
hierarchical).

#### Scenario: Constant column excluded from KMeans feature_names

- **WHEN** `perform_kmeans_clustering(df)` is called on a DataFrame
  containing a constant (zero-variance) column, with `standardize=True`
- **THEN** `feature_names` SHALL NOT include the constant column
- **AND** `len(feature_names) == cluster_centers.shape[1]`
- **AND** `dict(zip(feature_names, cluster_centers[k]))` SHALL map each
  surviving name to the centroid value for that named trait, for every
  cluster `k`

#### Scenario: Constant column excluded from GMM feature_names

- **WHEN** `perform_gmm_clustering(df)` is called on a DataFrame containing
  a constant (zero-variance) column, with `standardize=True`
- **THEN** `feature_names` SHALL NOT include the constant column
- **AND** `len(feature_names) == means.shape[1]`
- **AND** `dict(zip(feature_names, means[k]))` SHALL map each surviving name
  to the correct mean value for that named trait, for every component `k`

#### Scenario: Constant column excluded from hierarchical feature_names

- **WHEN** `perform_hierarchical_clustering(df)` is called on a DataFrame
  containing a constant (zero-variance) column, with `standardize=True`
- **THEN** `feature_names` SHALL NOT include the constant column
- **AND** `len(feature_names) == data_processed.shape[1]`
- **AND** `dict(zip(feature_names, data_processed[i]))` SHALL map each
  surviving name to the correct processed value for that named trait, for
  every sample `i` — a pure length check does not catch a reordering bug,
  only a per-sample named-value check does

#### Scenario: standardize=False still filters non-numeric and constant columns

- **WHEN** any of the three producers is called with `standardize=False` on a
  DataFrame containing a non-numeric column, a constant column, or both
- **THEN** the producer SHALL exclude those columns from `feature_names` and
  from the array used for fitting, the same way `standardize_data` filters
  the `standardize=True` path
- **AND** the producer SHALL NOT raise an uncontrolled sklearn/scipy error
  (e.g. a `TypeError` from an object-dtype array reaching `KMeans.fit()` /
  `GaussianMixture.fit()` / `linkage()`)

#### Scenario: Existing clean-input behavior is unchanged

- **WHEN** any of the three producers is called on a DataFrame with no
  constant and no non-numeric columns (nothing for the filter to drop)
- **THEN** `feature_names` SHALL be identical to the previously-observed
  output for that input — the fix corrects only mislabeled cases, it does
  not reorder or rename features that were already correct

### Requirement: Clean Error on Fully-Filtered Clustering Input

`perform_kmeans_clustering`, `perform_gmm_clustering`, and
`perform_hierarchical_clustering` SHALL raise a `RuntimeError` whose message
contains "No numeric columns with non-zero variance found" when every column
of the input is non-numeric or zero-variance, for both `standardize=True`
and `standardize=False`. `RuntimeError` (not a bare `ValueError`) matches
these three producers' existing convention of wrapping every error raised
inside their fitting pipeline — including `standardize_data`'s own
`ValueError` on the `standardize=True` path today — into a
method-specific `RuntimeError`; the fix keeps `standardize=False` consistent
with that existing, already-tested behavior rather than introducing a new,
differently-typed exception for only one branch.

#### Scenario: All columns filtered out raises a clear error

- **WHEN** every column of the input DataFrame is non-numeric or
  zero-variance, for either value of `standardize`
- **THEN** the producer SHALL raise `RuntimeError` with a message containing
  "No numeric columns with non-zero variance found"
- **AND** it SHALL NOT raise an uncontrolled sklearn/scipy error (e.g. a
  string-conversion `ValueError` from an object-dtype array) instead

## MODIFIED Requirements

### Requirement: Non-Breaking Clustering Return Shapes

Adding the clustering result types SHALL NOT change the return shapes of
`perform_kmeans_clustering` / `perform_gmm_clustering`; both SHALL keep
returning their existing dicts so all current callers continue to work.
Correcting a previously-mislabeled `feature_names` value for inputs with
constant or non-numeric columns is a bug fix, not a shape change: the dict's
keys are unchanged, and values are corrected only for inputs where they were
previously wrong (see "Accurate Clustering Feature Names After Column
Filtering").

#### Scenario: Existing clustering returns are preserved and not mutated

- **WHEN** `perform_kmeans_clustering()` or `perform_gmm_clustering()` is called
- **THEN** it SHALL return its existing dict (including `cluster_labels`,
  `cluster_centers`/`means`, and the quality-metric keys)
- **AND** calling the corresponding adapter SHALL leave the dict's keys and
  values unchanged
- **AND** the typed view SHALL be opt-in, built only via the adapters

#### Scenario: Typed adapters inherit the corrected feature_names

- **WHEN** `ClusterResult.from_kmeans_dict` or `from_gmm_dict` is built from a
  producer dict that excluded a constant or non-numeric column
- **THEN** the resulting `KMeansResult` / `GMMResult` `feature_names` SHALL
  match the producer dict's corrected `feature_names` exactly, with no
  separate correction logic in the adapters
