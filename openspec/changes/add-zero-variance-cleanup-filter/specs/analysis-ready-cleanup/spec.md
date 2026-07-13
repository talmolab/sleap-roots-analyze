## ADDED Requirements

### Requirement: Zero-Variance Trait Removal During Cleanup

The cleanup orchestrator `apply_data_cleanup_filters` SHALL remove constant
(zero-variance) traits as its **final** step, after sample removal, gated by a
`min_variance` threshold parameter (default `0.0`). A trait SHALL be removed when its
population variance `var(ddof=0) <= min_variance` (matching the variance test used by
`standardize_data`). Because a trait can become constant only after NaN-carrying rows are
dropped, variance SHALL be evaluated on the **post-sample-removal** frame. Each removed
trait SHALL be recorded in `cleanup_log["removed_traits"]` with `reason="zero_variance"`
together with its `variance` and the `threshold` used, and a
`{"step": "remove_zero_variance_traits", ...}` entry SHALL be appended to
`cleanup_log["cleanup_steps"]`. Setting `min_variance` to a negative value SHALL disable the
step (variance is always `>= 0`). The orchestrator SHALL NOT raise when this empties the
trait set; emptiness is handled by the entry point / validation.

#### Scenario: Constant trait is dropped and named at cleaning time

- **WHEN** `apply_data_cleanup_filters` runs on a frame containing a trait whose values are
  all identical across the surviving rows
- **THEN** that trait SHALL be absent from the returned cleaned frame
- **AND** `cleanup_log["removed_traits"]` SHALL contain an entry for it with
  `reason="zero_variance"`, a numeric `variance`, and the `threshold` used
- **AND** `cleanup_log["cleanup_steps"]` SHALL contain a `"remove_zero_variance_traits"` step
  entry with the traits-removed and remaining-traits counts

#### Scenario: Variance is evaluated after sample removal

- **WHEN** a trait varies across all input rows but the varying rows are dropped as NaN-heavy
  during sample removal, leaving identical values
- **THEN** the final variance step SHALL remove that trait (it is constant on the
  post-sample-removal frame)

#### Scenario: Negative threshold disables the step

- **WHEN** `apply_data_cleanup_filters` is called with `min_variance` set to a negative value
- **THEN** no trait SHALL be removed for zero variance, even an exactly-constant one

### Requirement: Constant-Free Analysis-Ready Output

The trait columns returned by `clean_traits_for_analysis` SHALL contain no zero-variance
(constant) trait, so `perform_pca_analysis`'s internal `standardize_data` drop removes
nothing on the cleanup path. Because the entry point performs its own residual-NaN row
`dropna` after `apply_data_cleanup_filters` returns — which under a loosened
`max_nans_per_sample > 0` can turn a surviving trait constant — the entry point SHALL
re-apply the zero-variance removal **after** that `dropna`, update the surviving trait set,
and append any newly-removed trait to `cleanup_log["removed_traits"]` (with
`reason="zero_variance"`) and `cleanup_log["cleanup_steps"]`. The `min_variance` threshold
SHALL be forwarded to `apply_data_cleanup_filters` and recorded in
`cleanup_log["effective_thresholds"]`. Existing validation check (4) ("at least one
non-constant numeric trait") SHALL be retained unchanged as a defensive guard; with the new
removal it fires only when every trait was constant.

#### Scenario: Constant trait is dropped from the analysis-ready output

- **WHEN** `clean_traits_for_analysis(df)` is called on a frame containing a constant trait
  alongside at least one varying trait
- **THEN** the constant trait SHALL be absent from the returned surviving traits and cleaned
  frame
- **AND** it SHALL be named in `cleanup_log["removed_traits"]` with `reason="zero_variance"`
- **AND** the call SHALL return successfully (the varying trait satisfies the analysis-ready
  gate)

#### Scenario: Trait made constant by the entry point's own dropna is dropped

- **WHEN** `clean_traits_for_analysis(df, max_nans_per_sample=0.5)` leaves a residual NaN
  whose row-drop turns a surviving trait constant
- **THEN** the entry point's post-`dropna` re-check SHALL remove that trait and name it in
  `cleanup_log["removed_traits"]` with `reason="zero_variance"`

#### Scenario: Effective thresholds record the variance threshold

- **WHEN** `clean_traits_for_analysis(df)` is called without threshold overrides
- **THEN** `cleanup_log["effective_thresholds"]` SHALL contain `min_variance` equal to `0.0`

#### Scenario: All-constant input still raises the non-constant guard

- **WHEN** `clean_traits_for_analysis(df)` is called on a frame whose surviving traits are all
  constant
- **THEN** a `ValueError` stating that no non-constant numeric trait remains SHALL be raised
