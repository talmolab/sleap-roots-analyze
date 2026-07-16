## ADDED Requirements

### Requirement: Serializable Cross-Platform Prediction Result Type

The package SHALL provide a frozen stdlib `@dataclass` `CrossPlatformPredictionResult` (with a
nested `TargetPrediction` dataclass) in `result_types.py` that captures only the
JSON-serializable science of a `logo_cv_predict()` run for one (platform pair, reduction method)
combination, excluding sklearn `Pipeline`/`PLSRegression`/`Ridge`/`PCA` objects. Every scalar
field SHALL be a native Python type (`int`, `float`, `str`, `bool`), so that
`json.dumps(dataclasses.asdict(result))` succeeds without a custom serializer.
`CrossPlatformPredictionResult` SHALL provide a `to_dict()` method returning
`dataclasses.asdict(self)` and a `to_json(**kwargs)` method defaulting `allow_nan=False`.

`CrossPlatformPredictionResult` SHALL hold `source_platform: str`, `target_platform: str`,
`predictor_source: str`, `reduction_method: str`, and `predictions: list[TargetPrediction]` — one
`TargetPrediction` entry per prediction target (each cluster-representative trait in the target
platform, plus one entry for the first principal component, `target_name="PC1"`). `predictor_source`
SHALL be stored as provenance metadata only (`{"blup", "genotype_means"}`) — this dataclass and its
adapter SHALL NOT validate it or branch behavior on it; the corresponding runtime guard is Tier
3.5's `PredictionConfig` scope, not this type's. Each `TargetPrediction` SHALL hold
`target_name: str`, `r2: float`, `rmse: float`, `spearman_rho: float`, `spearman_p: float`,
`genotype_names: list[str]`, `y_true: list[float]`, and `y_pred: list[float]`. Docstrings SHALL
note that `rmse` is not comparable across `TargetPrediction` entries with different underlying
trait scales, and that `spearman_p` is an asymptotic approximation, imprecise below n≈20-30.

#### Scenario: CrossPlatformPredictionResult round-trips through JSON as native types

- **WHEN** a `CrossPlatformPredictionResult` built from one or more `logo_cv_predict()` outputs
  is passed to `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing the string back with `json.loads` SHALL yield every numeric field
  (`r2`, `rmse`, `spearman_rho`, `spearman_p`, and every element of `y_true`/`y_pred`) as a Python
  `float` (no `np.float64`)

#### Scenario: No sklearn or numpy object is present in the clean view

- **WHEN** `dataclasses.asdict(result)` is inspected
- **THEN** it SHALL contain no sklearn `Pipeline`, `PLSRegression`, `Ridge`, `PCA`, or
  `StandardScaler` object, and no raw `numpy.ndarray`

#### Scenario: PC1 prediction is a distinct entry, never merged with representative-trait results

- **WHEN** a `CrossPlatformPredictionResult` includes both representative-trait
  `TargetPrediction` entries and a `target_name="PC1"` entry
- **THEN** the PC1 entry's `r2`/`rmse`/`spearman_rho` SHALL be independently computed and SHALL
  NOT be averaged, summed, or otherwise combined with any representative trait's metrics

### Requirement: CrossPlatformPredictionResult Adapter From logo_cv_predict Output

The package SHALL provide an adapter that maps one or more `logo_cv_predict()` return values,
plus platform-pair and method metadata, into a `CrossPlatformPredictionResult`. The adapter SHALL
NOT mutate its inputs.

#### Scenario: Adapter maps fields from real logo_cv_predict output

- **WHEN** the adapter is called with `logo_cv_predict()` outputs for each prediction target
  (representative traits + PC1) for one platform pair and reduction method
- **THEN** the resulting `CrossPlatformPredictionResult.predictions` list SHALL contain one
  `TargetPrediction` per target, with `r2`/`rmse`/`spearman_rho`/`spearman_p`/`y_true`/`y_pred`
  matching the corresponding `logo_cv_predict()` output exactly

### Requirement: CrossPlatformPredictionResult Public Export

The package SHALL export `CrossPlatformPredictionResult` and `TargetPrediction` from the
top-level `sleap_roots_analyze` namespace and list them in `__all__`.

#### Scenario: Result types importable from package root

- **WHEN** a consumer runs
  `from sleap_roots_analyze import CrossPlatformPredictionResult, TargetPrediction`
- **THEN** the import SHALL succeed
- **AND** both names SHALL appear in `sleap_roots_analyze.__all__` with no duplicate entries
