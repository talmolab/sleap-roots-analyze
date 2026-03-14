# config-management Specification Changes

## ADDED Requirements

### Requirement: Heritability Config Cross-Field Validation

The system SHALL validate that `statistics.calculate_heritability` and
`heritability.enabled` are not set to a contradictory combination that would cause
silent data loss. Heritability filtering requires heritability to be calculated first.
Validation occurs in `validate_viz_config()`, which runs at pipeline startup before any
steps execute.

#### Scenario: Reject contradictory heritability config at startup

- **WHEN** a viz pipeline config has `statistics.calculate_heritability=False` and `heritability.enabled=True`
- **THEN** `validate_viz_config()` SHALL raise a `ValueError` before any pipeline steps execute
- **AND** the error message SHALL name both config fields and suggest how to fix the conflict

#### Scenario: Accept valid config — both enabled

- **WHEN** a viz pipeline config has `statistics.calculate_heritability=True` and `heritability.enabled=True`
- **THEN** `validate_viz_config()` SHALL pass without error

#### Scenario: Accept valid config — calculate but skip filtering

- **WHEN** a viz pipeline config has `statistics.calculate_heritability=True` and `heritability.enabled=False`
- **THEN** `validate_viz_config()` SHALL pass without error

#### Scenario: Accept valid config — both disabled

- **WHEN** a viz pipeline config has `statistics.calculate_heritability=False` and `heritability.enabled=False`
- **THEN** `validate_viz_config()` SHALL pass without error

#### Scenario: Default config values pass validation

- **WHEN** a `VizPipelineConfig` is constructed with default values (`statistics.calculate_heritability=True`, `heritability.enabled=True`)
- **THEN** `validate_viz_config()` SHALL pass without error

---

### Requirement: FilterHeritabilityStep Defense-in-Depth Guard

The `FilterHeritabilityStep` SHALL NOT silently remove all traits when
`heritability_results` is empty. This guard provides defense-in-depth for cases where
config validation is bypassed (e.g., programmatic config construction).

#### Scenario: Guard activates when heritability_results is empty and filtering enabled

- **WHEN** `FilterHeritabilityStep` receives empty `heritability_results` ({}) and `heritability.enabled=True`
- **THEN** it SHALL skip filtering and pass all traits through unchanged
- **AND** it SHALL log a warning explaining the config mismatch
- **AND** the result metadata SHALL include `guard_activated: True` and a string `guard_reason`
- **AND** the result summary JSON SHALL include `guard_activated: True` and `traits_removed: 0`

#### Scenario: Guard preserves previous step metadata

- **WHEN** the guard activates due to empty `heritability_results`
- **THEN** all previous step metadata (trait_names, valid_trait_names, heritability_results, pca_results, etc.) SHALL be preserved via spread operator
- **AND** `trait_names` and `valid_trait_names` SHALL contain all original traits

#### Scenario: Guard generates consistent output files

- **WHEN** the guard activates due to empty `heritability_results`
- **THEN** it SHALL generate the same output files as the disabled path: `09_data_high_heritability.csv` (full DataFrame), `09_removed_traits.json` (empty list `[]`), and `09_heritability_filter_summary.json`
- **AND** downstream steps SHALL see a consistent file structure regardless of guard activation

#### Scenario: Guard does not activate when heritability_results is populated

- **WHEN** `FilterHeritabilityStep` receives populated `heritability_results` and `heritability.enabled=True`
- **THEN** it SHALL proceed with normal heritability filtering
- **AND** the result metadata SHALL NOT include `guard_activated`

#### Scenario: Guard does not activate when filtering is disabled

- **WHEN** `FilterHeritabilityStep` receives empty `heritability_results` and `heritability.enabled=False`
- **THEN** it SHALL follow the existing disabled path (pass all traits through)
- **AND** the result metadata SHALL NOT include `guard_activated`
