## ADDED Requirements

### Requirement: Cross-Platform Validate Input Config Flag

The system SHALL provide a `validate_input` flag on `CrossPlatformConfig` accepting `off`, `warn`, or
`strict`, defaulting to `warn`. Config validation SHALL reject any other value.

#### Scenario: Default value

- **GIVEN** a `CrossPlatformConfig` that does not set `validate_input`
- **WHEN** the config is constructed
- **THEN** `validate_input` is `warn`

#### Scenario: Invalid value rejected

- **GIVEN** a `CrossPlatformConfig` constructed with `validate_input="lenient"`
- **WHEN** `__post_init__` runs
- **THEN** a `ValueError` is raised naming the allowed values `off | warn | strict`

### Requirement: Cross-Platform Boundary Validation

The system SHALL validate each loaded cross-platform experiment frame at the load boundary using the same
canonicalize-then-validate side-check as the QC boundary, on a copy. The aligned experiment frames already
carry canonical `genotype`/`replicate` columns, so the helper validates them with fixed canonical role
names. The frames fed to alignment/correlation SHALL NOT be modified.

#### Scenario: Each experiment frame validated once

- **WHEN** `LoadCrossPlatformDataStep` executes with `validate_input` not `off`
- **THEN** the boundary helper is invoked once per loaded experiment frame (exp1 and exp2), and not on any
  downstream step output

#### Scenario: Validation does not change cross-platform output

- **GIVEN** the #120/#146 cross-platform reference inputs that the contract accepts
- **WHEN** the cross-platform pipeline runs with `validate_input: off` and again with `validate_input: warn`
- **THEN** the two runs produce identical output

#### Scenario: Degrades to a no-op without contracts

- **GIVEN** `sleap-roots-contracts` is not installed
- **WHEN** `LoadCrossPlatformDataStep` executes with `validate_input` not `off`
- **THEN** the step runs to completion with identical output and logs that validation was skipped

#### Scenario: Severity behavior at the cross-platform boundary

- **GIVEN** a malformed experiment frame missing the `genotype` role
- **WHEN** boundary validation runs under `warn`
- **THEN** a structured error is raised before alignment
