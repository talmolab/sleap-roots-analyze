# data-sanitization Spec Delta

## ADDED Requirements

### Requirement: NaN Sample Removal Detail Tracking

The cleanup pipeline SHALL propagate per-sample NaN removal details from
`remove_nan_samples()` through the cleanup log to the pipeline output file
`02_removed_samples_detail.csv`.

#### Scenario: Removed sample details populated in cleanup log

- **GIVEN** a dataset with samples containing NaN values above `max_nan_fraction`
- **WHEN** `apply_data_cleanup_filters()` calls `remove_nan_samples()`
- **THEN** `cleanup_log["removed_samples_detail"]` contains one entry per removed
  sample
- **AND** the length of the list equals the number of samples removed

#### Scenario: Each removal entry contains required fields

- **GIVEN** a sample removed due to NaN values
- **WHEN** the removal detail entry is created
- **THEN** it contains exactly these keys: `sample_index`, `barcode`, `genotype`,
  `rep`, `nan_count`, `nan_fraction`, `nan_traits`, `removal_reason`
- **AND** `nan_fraction` is a float between 0.0 and 1.0
- **AND** `nan_traits` is a non-empty string listing the NaN trait names

#### Scenario: Removal details written to CSV with correct content

- **GIVEN** `cleanup_log["removed_samples_detail"]` contains N entries
- **WHEN** `CleanupTraitsStep` writes outputs
- **THEN** `02_removed_samples_detail.csv` contains exactly N data rows
- **AND** the file has columns: `sample_index`, `barcode`, `genotype`, `rep`,
  `nan_count`, `nan_fraction`, `nan_traits`, `removal_reason`

#### Scenario: Empty detail list when no samples removed

- **GIVEN** a dataset with no samples exceeding `max_nan_fraction`
- **WHEN** `apply_data_cleanup_filters()` completes
- **THEN** `cleanup_log["removed_samples_detail"]` is an empty list
- **AND** `02_removed_samples_detail.csv` contains only the header row

#### Scenario: Removal details correct when max_nan_fraction is 0.0

- **GIVEN** `max_nan_fraction=0.0` (any NaN triggers removal)
- **WHEN** a sample has exactly one NaN trait
- **THEN** that sample appears in `cleanup_log["removed_samples_detail"]`
- **AND** `nan_count` is 1 and `nan_fraction` is greater than 0.0

#### Scenario: No removal when max_nan_fraction is 1.0

- **GIVEN** `max_nan_fraction=1.0` (only fully-NaN samples removed)
- **WHEN** a sample has some but not all traits as NaN
- **THEN** `cleanup_log["removed_samples_detail"]` is an empty list

#### Scenario: Missing barcode or replicate column produces empty-string fallback

- **GIVEN** a DataFrame that does not have the configured barcode or replicate column
- **WHEN** `remove_nan_samples()` records a removed sample
- **THEN** `barcode` and/or `rep` in the detail entry is `""` (not a KeyError)
