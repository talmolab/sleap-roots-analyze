# Pipeline QC - Heritability Diagnostics

## ADDED Requirements

### Requirement: Optional Diagnostic Mode in FilterHeritabilityStep
The FilterHeritabilityStep SHALL support an optional diagnostic mode that generates detailed variance analysis and visualizations during heritability-based trait filtering.

#### Scenario: Enable diagnostic mode via configuration
- **GIVEN** FilterHeritabilityStep with `generate_diagnostics=True` in config
- **WHEN** step executes
- **THEN** perform standard heritability filtering
- **AND** additionally generate diagnostic comparison CSV
- **AND** create diagnostic visualization plots
- **AND** store diagnostic results in step metadata

#### Scenario: Diagnostic mode exports comparison CSV
- **GIVEN** FilterHeritabilityStep with diagnostics enabled
- **WHEN** step completes filtering
- **THEN** export CSV file with diagnostic comparison (from `compare_trait_heritabilities()`)
- **AND** save to run directory with naming pattern `{step_number}_heritability_diagnostics.csv`
- **AND** include all analyzed traits (both retained and removed)

#### Scenario: Diagnostic mode generates variance decomposition plot
- **GIVEN** FilterHeritabilityStep with diagnostics enabled
- **WHEN** step completes filtering
- **THEN** create variance decomposition plot for removed traits
- **AND** save to run directory with naming pattern `{step_number}_variance_decomposition.png`
- **AND** highlight traits that were filtered out

#### Scenario: Diagnostic mode generates trait boxplots
- **GIVEN** FilterHeritabilityStep with diagnostics enabled
- **AND** more than 5 traits were removed due to low heritability
- **WHEN** step completes filtering
- **THEN** create boxplot visualization for removed traits showing distribution by genotype
- **AND** save to run directory with naming pattern `{step_number}_removed_traits_boxplots.png`
- **AND** limit to top 10 lowest heritability traits if >10 were removed

#### Scenario: Diagnostic mode stores results in metadata
- **GIVEN** FilterHeritabilityStep with diagnostics enabled
- **WHEN** step completes
- **THEN** add `diagnostic_results` to step metadata
- **AND** include comparison DataFrame as dict
- **AND** include paths to generated diagnostic files

#### Scenario: Diagnostic mode disabled by default
- **GIVEN** FilterHeritabilityStep with default configuration
- **WHEN** step executes
- **THEN** perform only standard filtering
- **AND** do not generate diagnostic files
- **AND** do not add overhead from diagnostic calculations

### Requirement: HeritabilityConfig Extension for Diagnostics
The HeritabilityConfig dataclass SHALL include an optional `generate_diagnostics` field to control diagnostic mode.

#### Scenario: Add generate_diagnostics field to config
- **GIVEN** HeritabilityConfig dataclass definition
- **WHEN** new instance is created with `generate_diagnostics=True`
- **THEN** field is accessible and validated as boolean
- **AND** default value is `False` for backward compatibility

#### Scenario: Config validation for diagnostic mode
- **GIVEN** HeritabilityConfig with `enabled=False` and `generate_diagnostics=True`
- **WHEN** FilterHeritabilityStep processes config
- **THEN** issue warning that diagnostics require `enabled=True`
- **AND** automatically set `generate_diagnostics=False`
- **AND** continue with filtering disabled

### Requirement: Diagnostic Output Organization
The diagnostic outputs SHALL be organized and named consistently with existing pipeline output conventions.

#### Scenario: Diagnostic files follow pipeline naming convention
- **GIVEN** FilterHeritabilityStep as step 9 in pipeline
- **WHEN** diagnostics are generated
- **THEN** prefix all diagnostic files with `09_`
- **AND** use descriptive suffixes: `_heritability_diagnostics.csv`, `_variance_decomposition.png`, `_removed_traits_boxplots.png`

#### Scenario: Diagnostic files saved to run directory
- **GIVEN** pipeline with configured run directory
- **WHEN** diagnostic mode is enabled
- **THEN** save all diagnostic outputs to same run directory as other step outputs
- **AND** do not create separate subdirectories for diagnostics

#### Scenario: Diagnostic generation logged appropriately
- **GIVEN** FilterHeritabilityStep with diagnostics enabled
- **WHEN** step executes
- **THEN** log info message indicating diagnostic mode is active
- **AND** log paths to generated diagnostic files
- **AND** log any warnings if diagnostic generation fails (but don't fail the step)

### Requirement: Backward Compatibility
The addition of diagnostic functionality SHALL not break existing pipeline configurations or workflows.

#### Scenario: Existing configs work without modification
- **GIVEN** existing YAML config file without `generate_diagnostics` field
- **WHEN** pipeline loads configuration
- **THEN** default `generate_diagnostics=False`
- **AND** pipeline executes normally without diagnostics

#### Scenario: Diagnostic mode has no performance impact when disabled
- **GIVEN** FilterHeritabilityStep with `generate_diagnostics=False`
- **WHEN** step executes
- **THEN** execution time matches pre-diagnostic implementation
- **AND** memory usage matches pre-diagnostic implementation
- **AND** no diagnostic code paths are executed

#### Scenario: Diagnostic mode failure doesn't break pipeline
- **GIVEN** FilterHeritabilityStep with diagnostics enabled
- **WHEN** diagnostic generation encounters error (e.g., plotting failure)
- **THEN** log warning with error details
- **AND** continue with standard filtering
- **AND** return filtered data successfully
- **AND** mark diagnostic generation as failed in metadata