## ADDED Requirements

### Requirement: Heritability Fixed-Effects Config Field

`StatisticsConfig` SHALL expose `fixed_effects: Optional[List[str]] = None`,
threaded into the `calculate_heritability_estimates(...)` call made by
`StatisticalAnalysisStep`. This field SHALL live on `StatisticsConfig` — the
dataclass `StatisticalAnalysisStep` already reads to control
`calculate_heritability`/`generate_blup_table` — not on `HeritabilityConfig`,
which gates the separate, later low-H² filtering step
(`FilterHeritabilityStep`) and has no relationship to how the model itself is
fit.

`StatisticalAnalysisStep` SHALL resolve `fixed_effects` via the same
`getattr(config, "statistics", None)` fallback already used for
`calculate_heritability` and `generate_blup_table`, defaulting to `None` when
`config.statistics` is absent (the `QCPipelineConfig` case, which has no
`statistics` field at all) — a QC-pipeline run SHALL NOT raise
`AttributeError` and SHALL behave identically to `fixed_effects=None`.

#### Scenario: Default StatisticsConfig has no fixed effects

- **WHEN** a `StatisticsConfig` is constructed with default values
- **THEN** `fixed_effects` SHALL be `None`
- **AND** `StatisticalAnalysisStep` SHALL call
  `calculate_heritability_estimates` without `fixed_effects` set (or with it
  explicitly `None`), reproducing pre-existing behavior exactly

#### Scenario: Configured fixed_effects are threaded into the heritability call

- **WHEN** a pipeline runs `StatisticalAnalysisStep` with
  `statistics.fixed_effects=["experiment"]` and the input data has an
  `"experiment"` column
- **THEN** `calculate_heritability_estimates` SHALL be called with
  `fixed_effects=["experiment"]`
- **AND** the resulting `08_heritability_results.csv` and
  `08_blup_adjusted_means.csv` (when `generate_blup_table` is also enabled)
  SHALL reflect the fixed-effects-corrected model, differing from a run with
  `fixed_effects=None` on the same data

#### Scenario: QC-pipeline config has no way to configure fixed_effects, resolves to None

- **WHEN** `StatisticalAnalysisStep.execute()` runs with a `QCPipelineConfig`
  (which has no `statistics` field)
- **THEN** `fixed_effects` SHALL resolve to `None` (the same default fallback
  already used for `calculate_heritability`/`generate_blup_table` when
  `config.statistics` is absent)
- **AND** `StatisticalAnalysisStep.execute()` SHALL NOT raise `AttributeError`
  when run with such a config
