## ADDED Requirements

### Requirement: PCA Feature Selection Config Validation

The system SHALL validate that `pca.n_top_features` is a whole number
`>= 1` (within a small floating-point tolerance, not exact equality)
whenever `pca.feature_selection_strategy` is one of `"top_absolute"` or
`"top_contribution"` — methods that read `n_top_features` directly as an
integer count and would silently truncate (to zero, or dropping a
fractional part) if given a value that isn't a positive whole number. The
same whole-number requirement applies to `"top_variance"` whenever
`n_top_features >= 1` (its count branch), but `"top_variance"`
additionally accepts any value `< 1` as a variance-fraction threshold. No
restriction is added for `feature_selection_strategy == "extreme"` (the
field is ignored entirely by this method, so any value is harmless).

Validation occurs in both `validate_qc_config()` and
`validate_viz_config()`, which run at pipeline startup before any steps
execute. `"vector_length"` is intentionally not one of the strategies this
requirement names: `pca.feature_selection_strategy`'s own pre-existing
validation enum (in the same two functions) has never accepted
`"vector_length"` as a value for this field — that string is a valid value
only for the separate `create_pca_biplot(feature_selection=...)`
parameter. This requirement does not widen that enum.

#### Scenario: Reject a fractional n_top_features below 1 for count-only methods

- **WHEN** a config has `pca.feature_selection_strategy` set to
  `"top_absolute"` or `"top_contribution"` and `pca.n_top_features` set to
  a value `< 1`
- **THEN** `validate_qc_config()` / `validate_viz_config()` SHALL raise a
  `ValueError` before any pipeline steps execute
- **AND** the error message SHALL name both config fields, the offending
  strategy, and state that an integer `>= 1` is required for that strategy

#### Scenario: Reject a non-integer n_top_features at or above 1

- **WHEN** a config has `pca.n_top_features` set to a non-integer value
  `>= 1` (e.g. `5.7`) and `pca.feature_selection_strategy` set to any value
  other than `"extreme"` (including `"top_variance"`, `"top_absolute"`, or
  `"top_contribution"`)
- **THEN** `validate_qc_config()` / `validate_viz_config()` SHALL raise a
  `ValueError` before any pipeline steps execute
- **AND** the error message SHALL state that the fractional part would be
  silently truncated

#### Scenario: Accept a variance-fraction threshold for top_variance

- **WHEN** a config has `pca.feature_selection_strategy="top_variance"` and
  `pca.n_top_features` set to a value `< 1` (e.g. `0.8`)
- **THEN** `validate_qc_config()` / `validate_viz_config()` SHALL pass
  without error

#### Scenario: Accept any n_top_features value for extreme

- **WHEN** a config has `pca.feature_selection_strategy="extreme"` and
  `pca.n_top_features` set to any value, including a fractional value or
  one `< 1`
- **THEN** `validate_qc_config()` / `validate_viz_config()` SHALL pass
  without error, since the field is not read for this method

#### Scenario: Accept a whole-number count for any strategy

- **WHEN** a config has `pca.n_top_features` set to a whole-number value
  `>= 1`, regardless of `pca.feature_selection_strategy`
- **THEN** `validate_qc_config()` / `validate_viz_config()` SHALL pass
  without error

#### Scenario: Default config values pass validation

- **WHEN** a `PCAConfig` is constructed with default values
  (`feature_selection_strategy="top_variance"`, `n_top_features=10.0`)
- **THEN** `validate_qc_config()` / `validate_viz_config()` SHALL pass
  without error
