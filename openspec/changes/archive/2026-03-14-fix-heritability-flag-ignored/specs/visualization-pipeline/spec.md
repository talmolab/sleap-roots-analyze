## ADDED Requirements

### Requirement: Statistical Analysis Heritability Flag
The `StatisticalAnalysisStep` SHALL check `config.statistics.calculate_heritability`
before calling `calculate_heritability_estimates()`. When the flag is `False`, the step
SHALL skip heritability calculation entirely, set `heritability_results` to an empty dict
in output metadata, omit `08_heritability_results.csv`, and record
`"heritability_summary": {"skipped": true}` in the summary JSON.

#### Scenario: Heritability skipped when flag is disabled
- **WHEN** `config.statistics.calculate_heritability` is `False`
- **THEN** `StatisticalAnalysisStep` SHALL NOT call `calculate_heritability_estimates()`
- **AND** `heritability_results` in output metadata SHALL be an empty dict `{}`
- **AND** `08_heritability_results.csv` SHALL NOT be generated
- **AND** the summary JSON SHALL contain `"heritability_summary": {"skipped": true}`

#### Scenario: Heritability calculated when flag is enabled (default)
- **WHEN** `config.statistics.calculate_heritability` is `True` (default)
- **THEN** `StatisticalAnalysisStep` SHALL call `calculate_heritability_estimates()`
- **AND** `heritability_results` SHALL be populated in output metadata
- **AND** `08_heritability_results.csv` SHALL be generated
- **AND** behavior SHALL be identical to current implementation

#### Scenario: Downstream FilterHeritabilityStep handles skipped heritability
- **WHEN** `heritability_results` is an empty dict from a prior step
- **THEN** `FilterHeritabilityStep` SHALL skip filtering gracefully
- **AND** the pipeline SHALL continue without error
