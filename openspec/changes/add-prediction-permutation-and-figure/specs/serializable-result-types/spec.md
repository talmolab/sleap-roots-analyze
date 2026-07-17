## ADDED Requirements

### Requirement: Serializable Cross-Platform Permutation Result Type

The package SHALL provide a frozen stdlib `@dataclass` `CrossPlatformPermutationResult` (with a
nested `PermutationResult` dataclass) in `result_types.py` that captures only the
JSON-serializable science of a `permutation_test()` run for one (platform pair, reduction method)
combination, kept **separate from** (not nested inside) `CrossPlatformPredictionResult`. Every
scalar field SHALL be a native Python type, so that `json.dumps(dataclasses.asdict(result))`
succeeds without a custom serializer. `CrossPlatformPermutationResult` SHALL provide a `to_dict()`
method returning `dataclasses.asdict(self)` and a `to_json(**kwargs)` method defaulting
`allow_nan=False`.

`CrossPlatformPermutationResult` SHALL hold `source_platform: str`, `target_platform: str`,
`reduction_method: str`, and `predictions: list[PermutationResult]` — one `PermutationResult`
entry per prediction target (mirroring `CrossPlatformPredictionResult`'s `predictions`/
`TargetPrediction` shape). Each `PermutationResult` SHALL hold `target_name: str`,
`observed_r2: float`, `observed_rmse: float`, `observed_spearman_rho: float`,
`observed_top_quartile_recovery: float`, `null_r2: list[float]`, `null_rmse: list[float]`,
`null_spearman_rho: list[float]`, `null_top_quartile_recovery: list[float]`,
`p_value_r2: float`, `p_value_rmse: float`, `p_value_spearman_rho: float`, and
`n_permutations: int`.

#### Scenario: CrossPlatformPermutationResult round-trips through JSON as native types

- **WHEN** a `CrossPlatformPermutationResult` built from one or more `permutation_test()` outputs
  is passed to `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing the string back with `json.loads` SHALL yield every numeric field, and every
  element of every null-distribution list, as a Python `float` (no `np.float64`)

#### Scenario: null distribution lists have length n_permutations

- **WHEN** a `PermutationResult` is built from a `permutation_test(..., n_permutations=N)` output
- **THEN** `null_r2`, `null_rmse`, `null_spearman_rho`, and `null_top_quartile_recovery` SHALL
  each have length `N`, and `n_permutations` SHALL equal `N`

#### Scenario: PermutationResult is independent of TargetPrediction

- **WHEN** `dataclasses.fields(CrossPlatformPredictionResult)` is inspected
- **THEN** it SHALL contain no field referencing `PermutationResult` or
  `CrossPlatformPermutationResult` — the two result families remain structurally independent

#### Scenario: No sklearn or numpy object is present in the clean view

- **WHEN** `dataclasses.asdict(result)` is inspected for a `CrossPlatformPermutationResult`
- **THEN** it SHALL contain no sklearn `Pipeline`/`PLSRegression`/`Ridge`/`PCA`/`StandardScaler`
  object, and no raw `numpy.ndarray` — every null-distribution list SHALL be a plain Python `list`
  of `float`, matching `CrossPlatformPredictionResult`'s equivalent scenario

### Requirement: CrossPlatformPermutationResult Adapter From permutation_test Output

The package SHALL provide an adapter that maps one or more `permutation_test()` return values,
plus platform-pair and method metadata, into a `CrossPlatformPermutationResult`. The adapter
SHALL NOT mutate its inputs.

#### Scenario: Adapter maps fields from real permutation_test output

- **WHEN** the adapter is called with `permutation_test()` outputs for each prediction target
  (representative traits + PC1) for one platform pair and reduction method
- **THEN** the resulting `CrossPlatformPermutationResult.predictions` list SHALL contain one
  `PermutationResult` per target, with every field matching the corresponding
  `permutation_test()` output exactly

### Requirement: CrossPlatformPermutationResult Public Export

The package SHALL export `CrossPlatformPermutationResult` and `PermutationResult` from the
top-level `sleap_roots_analyze` namespace and list them in `__all__`.

#### Scenario: Result types importable from package root

- **WHEN** a consumer runs
  `from sleap_roots_analyze import CrossPlatformPermutationResult, PermutationResult`
- **THEN** the import SHALL succeed
- **AND** both names SHALL appear in `sleap_roots_analyze.__all__` with no duplicate entries
