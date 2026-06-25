## ADDED Requirements

### Requirement: Optional Contracts Dependency

The system SHALL treat `sleap-roots-contracts` as an optional dependency. Input-contract validation SHALL
degrade to a logged no-op when the package is not installed, and SHALL NOT raise `ImportError` at import or
run time.

#### Scenario: Package not installed

- **GIVEN** `sleap-roots-contracts` is not installed
- **WHEN** the QC pipeline loads the entry input with `validate_input` set to `warn` or `strict`
- **THEN** the pipeline runs to completion, logs that validation was skipped, and produces output identical
  to a run with validation `off`

#### Scenario: Package installed runs the validator

- **GIVEN** `sleap-roots-contracts[pandas]>=0.1.0a1` is installed
- **WHEN** the entry input is loaded with `validate_input` not `off`
- **THEN** the contract validator runs on the entry input, so that a malformed entry input is detected and
  handled per the configured mode

### Requirement: Validate Input Config Flag

The system SHALL provide a `validate_input` config flag on the QC `DataConfig` accepting `off`, `warn`, or
`strict`, defaulting to `warn`. Config validation SHALL reject any other value.

#### Scenario: Default value

- **GIVEN** a QC config that does not set `validate_input`
- **WHEN** the config is loaded
- **THEN** `data.validate_input` is `warn`

#### Scenario: Invalid value rejected

- **GIVEN** a QC config with `data.validate_input: lenient`
- **WHEN** `validate_qc_config` runs
- **THEN** validation raises an error naming the allowed values `off | warn | strict`

### Requirement: Canonicalize-Then-Validate Boundary

The system SHALL validate analysis input by, on a copy of the entry frame: renaming the configured role
columns that are present (`genotype`, `barcode`→`sample_id`, and `replicate` when configured) to canonical
contract names, dropping non-trait metadata via `get_trait_columns`, applying `canonicalize_role_dtypes`,
then calling `validate_analysis_input`. The frame fed to the QC pipeline SHALL NOT be modified by this
process.

#### Scenario: Validation runs on a copy

- **GIVEN** an entry DataFrame with a numeric `replicate` role
- **WHEN** boundary validation runs with `validate_input` not `off`
- **THEN** the DataFrame passed downstream to the pipeline is unchanged (same columns, dtypes, and values)

#### Scenario: Replicate role not leaked as a trait

- **GIVEN** a contract-canonical CSV and a config whose role names map to `genotype`/`sample_id`/`replicate`
- **WHEN** boundary validation builds the validation copy
- **THEN** the `replicate` column is treated as a role and excluded from the trait columns

#### Scenario: Optional replicate role absent

- **GIVEN** a config with `columns.replicate` set to `None` and no replicate column in the data
- **WHEN** boundary validation builds the validation copy
- **THEN** the replicate rename is skipped without error and validation proceeds on the present roles

### Requirement: Validate Only the Entry Input

The system SHALL apply input-contract validation only to the QC entry input at the data-loading boundary, and
SHALL NOT re-validate internal step-to-step intermediates.

#### Scenario: Entry input validated once

- **WHEN** the QC pipeline executes
- **THEN** boundary validation is invoked exactly once on the loaded entry frame and not on any downstream
  step output

### Requirement: Validation Severity Behavior

Under `warn`, the system SHALL log non-fatal warnings and hard-fail only on the universal structural errors
(missing `genotype`, no numeric trait, bad role dtype). Under `strict`, the system SHALL raise at the
boundary on any contract error including boundary issues such as missing `sample_id`. Under `off`, the system
SHALL perform no validation work even when the contracts package is installed.

#### Scenario: Off is a no-op even when installed

- **GIVEN** `sleap-roots-contracts` is installed and `validate_input` is `off`
- **WHEN** the entry input is loaded
- **THEN** the contract validator is not called and no validation logging occurs

#### Scenario: Good input passes

- **GIVEN** a well-formed entry input
- **WHEN** boundary validation runs under `warn` or `strict`
- **THEN** no error is raised and the pipeline proceeds

#### Scenario: Malformed input warns under warn

- **GIVEN** an entry input with a non-fatal contract issue (e.g. missing `sample_id`)
- **WHEN** boundary validation runs under `warn`
- **THEN** the issue is logged as a warning and the pipeline proceeds

#### Scenario: Malformed input raises under strict

- **GIVEN** an entry input with a non-fatal contract issue (e.g. missing `sample_id`)
- **WHEN** boundary validation runs under `strict`
- **THEN** the boundary raises a structured error, naming the offending column, before any analysis step runs

#### Scenario: Structural error fails even under warn

- **GIVEN** an entry input missing the `genotype` role
- **WHEN** boundary validation runs under `warn`
- **THEN** the boundary raises a structured error

### Requirement: Enabling Validation Does Not Change Results

Enabling input-contract validation SHALL NOT change any pipeline output relative to running with
`validate_input` `off`.

#### Scenario: Equivalence against golden

- **GIVEN** the #120/#146 `turface_19` reference input that the contract accepts
- **WHEN** the QC pipeline is run with `validate_input: off` and again with `validate_input: warn`
- **THEN** the two runs produce identical deterministic QC and PCA output, matching the committed golden
  values at `rtol=1e-6` (UMAP coordinates excluded as environment-sensitive)
