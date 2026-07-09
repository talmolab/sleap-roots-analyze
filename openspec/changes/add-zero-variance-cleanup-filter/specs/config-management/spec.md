## ADDED Requirements

### Requirement: Zero-Variance Cleanup Threshold Configuration

`CleanupConfig` SHALL expose a `min_variance` parameter (default `0.0`) in the `cleanup`
parameter group. Traits whose population variance `var(ddof=0) <= min_variance` SHALL be
removed by the cleanup step; a value of `0.0` drops exactly-constant traits, and a negative
value disables the removal. The default SHALL match `apply_data_cleanup_filters`'
`min_variance` signature default so the canonical-default drift guard holds.

#### Scenario: min_variance is available in the cleanup section with a safe default

- **WHEN** `CleanupConfig()` is instantiated without overrides
- **THEN** it SHALL expose `min_variance` equal to `0.0`
- **AND** `CleanupTraitsStep` SHALL forward `config.cleanup.min_variance` to
  `apply_data_cleanup_filters`

#### Scenario: Negative min_variance disables constant-trait removal

- **WHEN** a config sets `cleanup.min_variance` to a negative value
- **THEN** the cleanup step SHALL NOT remove any trait for zero variance
