# serializable-result-types Specification (delta)

## ADDED Requirements

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
