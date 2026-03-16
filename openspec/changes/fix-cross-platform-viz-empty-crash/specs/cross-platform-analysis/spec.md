## ADDED Requirements

### Requirement: Empty Correlation Graceful Handling
The `VisualizeCrossPlatformStep` SHALL handle empty correlation DataFrames gracefully without crashing the pipeline.

#### Scenario: All correlations filtered out
- **WHEN** `VisualizeCrossPlatformStep` receives an empty `correlation_df` (all trait pairs filtered by upstream step)
- **THEN** the step SHALL skip all plot generation
- **AND** the step SHALL log a warning indicating no correlations to visualize
- **AND** the step SHALL return a successful StepResult with `plots_generated: 0`
- **AND** output metadata SHALL include `"empty_correlations": true`

#### Scenario: Non-empty correlation DataFrame (default behavior)
- **WHEN** `VisualizeCrossPlatformStep` receives a non-empty `correlation_df`
- **THEN** the step SHALL generate all configured plots as before
- **AND** output metadata SHALL NOT include `"empty_correlations"` key
- **AND** behavior SHALL be identical to current implementation

#### Scenario: Pipeline completes when all correlations are filtered out
- **GIVEN** two experiments share common genotypes
- **WHEN** `CalculateCrossPlatformCorrelationsStep` filters out all trait-pair correlations (e.g., due to strict p-value threshold or insufficient replicates)
- **AND** the resulting `correlation_df` is empty
- **THEN** `VisualizeCrossPlatformStep` SHALL complete without error
- **AND** the pipeline summary SHALL report status "success"
