# Cross-Platform Analysis Spec Delta

## MODIFIED Requirements

### Requirement: Load and Align Cross-Platform Data

The system SHALL load and align data from two experimental platforms with the following **additional** column validation behavior:

- **Replicate column detection**: When searching for replicate columns, if multiple variants are found (e.g., both "Replicate" and "rep"), the system SHALL issue a UserWarning indicating which column will be used.
- **Default argument safety**: Functions with default list parameters (e.g., `calculate_genotype_statistics`) SHALL use `None` as default with runtime initialization to prevent mutable default argument bugs.

#### Scenario: Multiple replicate column variants

- **WHEN** a DataFrame has both "Replicate" and "rep" columns
- **THEN** load_and_align_experiments issues a UserWarning indicating which column will be used
- **AND** the first matching column variant is used consistently

#### Scenario: Mutable default argument protection

- **WHEN** calculate_genotype_statistics is called without statistics parameter
- **THEN** the default statistics list is created fresh for each call
- **AND** mutations to the returned statistics do not affect future calls

### Requirement: Cross-Platform Pipeline Integration

The system SHALL integrate cross-platform analysis steps into the existing pipeline infrastructure with the following **additional** error handling:

- **Log directory failures**: When the CLI attempts to create a log directory and encounters an OSError, it SHALL catch the error, display a user-friendly warning, and continue with console-only logging.

#### Scenario: Log directory creation failure

- **WHEN** an invalid or inaccessible log file path is configured
- **THEN** an OSError is caught during directory creation
- **AND** a warning message is displayed to the user
- **AND** the pipeline continues with console-only logging
