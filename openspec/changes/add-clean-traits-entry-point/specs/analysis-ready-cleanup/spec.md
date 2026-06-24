## ADDED Requirements

### Requirement: Public Cleanup & Validation API Surface

The package SHALL export, from the top-level `sleap_roots_analyze` namespace, the cleanup
function used by QC step 02 (`apply_data_cleanup_filters`) and the trait-validation
functions extracted from QC step 03 (`validate_clean_traits`, `build_clean_validation_report`),
so the analysis-ready entry point and downstream consumers import them instead of reaching
into internal modules or re-implementing them.

#### Scenario: Cleanup and validation functions are importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import (apply_data_cleanup_filters, validate_clean_traits, build_clean_validation_report)`
- **THEN** the import SHALL succeed
- **AND** each imported object SHALL be identity-equal (`is`) to the function defined in
  `sleap_roots_analyze.data_cleanup`

#### Scenario: Each function is listed in `__all__`

- **WHEN** `sleap_roots_analyze.__all__` is inspected
- **THEN** it SHALL contain `apply_data_cleanup_filters`, `validate_clean_traits`,
  `build_clean_validation_report`, and `clean_traits_for_analysis`
- **AND** SHALL contain no duplicate entries

#### Scenario: Public functions have resolvable type hints and Google-style docstrings

- **WHEN** `typing.get_type_hints(fn)` is called on each newly-public function
- **THEN** it SHALL return without raising `NameError`
- **AND** each function's docstring SHALL include populated Args, Returns, and Raises
  sections in Google style

### Requirement: Analysis-Ready Cleanup Entry Point

The system SHALL provide a single public function `clean_traits_for_analysis` that turns a
raw wide trait table into a clean, analysis-ready table by **importing and composing** the
exposed step-02 cleanup (`apply_data_cleanup_filters`) and step-03 validation
(`validate_clean_traits`) functions — introducing no new cleanup algorithm. It SHALL return
a tuple `(clean_df, trait_cols, cleanup_log)` where `clean_df` is the cleaned table,
`trait_cols` is the list of surviving trait columns, and `cleanup_log` is the
`apply_data_cleanup_filters` log enriched with the effective thresholds used and a
validation summary.

#### Scenario: Compose cleanup over a raw trait table

- **WHEN** `clean_traits_for_analysis(df)` is called on a valid raw wide trait table
- **THEN** trait columns are resolved via `get_trait_columns` when not explicitly passed
- **AND** `apply_data_cleanup_filters` is applied to remove bad traits then NaN rows
- **AND** the function returns a 3-tuple `(clean_df, trait_cols, cleanup_log)`

#### Scenario: Surviving trait columns are derived from the cleaned frame

- **WHEN** `clean_traits_for_analysis(df)` returns `(clean_df, trait_cols, _)`
- **THEN** `trait_cols` SHALL equal the input trait columns that remain as columns of
  `clean_df` (traits dropped by cleanup SHALL be absent)

#### Scenario: Caller-supplied trait columns are honored

- **WHEN** `clean_traits_for_analysis(df, trait_cols=[...])` is called with an explicit
  trait-column list
- **THEN** `get_trait_columns` is not used to infer columns
- **AND** cleanup is applied over exactly the supplied columns

#### Scenario: Cleanup threshold kwargs pass through to the cleanup function

- **WHEN** `clean_traits_for_analysis(df, max_nans_per_trait=0.1)` is called
- **THEN** the tightened threshold SHALL be forwarded to `apply_data_cleanup_filters`
- **AND** the returned `cleanup_log["effective_thresholds"]` SHALL record the value used

### Requirement: Default Thresholds and Column Names

`clean_traits_for_analysis` SHALL default its cleanup thresholds to the documented defaults
of `apply_data_cleanup_filters` (`max_zeros_per_trait=0.5`, `max_nans_per_trait=0.3`,
`max_nans_per_sample=0.2`, `min_samples_per_trait=10`) and its metadata-column names to
`barcode_col="Barcode"`, `genotype_col="geno"`, `replicate_col="rep"`, and SHALL record the
effective thresholds in the returned `cleanup_log`. These defaults are the cleanup
function's own, which differ from the QC pipeline's config defaults; identical output to the
pipeline requires passing matched thresholds and column names.

#### Scenario: Effective thresholds are recorded for auditability

- **WHEN** `clean_traits_for_analysis(df)` is called without threshold overrides
- **THEN** `cleanup_log["effective_thresholds"]` SHALL contain
  `max_zeros_per_trait=0.5`, `max_nans_per_trait=0.3`, `max_nans_per_sample=0.2`,
  `min_samples_per_trait=10`

#### Scenario: Replicate column may be absent

- **WHEN** `clean_traits_for_analysis(df, replicate_col=None)` is called on a table with no
  replicate column
- **THEN** cleanup SHALL proceed without error and not require a replicate column

### Requirement: No NaN in Analysis-Ready Output

The cleaned table returned by `clean_traits_for_analysis` SHALL contain no NaN values in
the surviving trait columns, so that `perform_pca_analysis`'s internal row `dropna()`
removes nothing. After `apply_data_cleanup_filters` removes NaN-heavy traits and samples,
the entry point SHALL drop any rows that still carry NaN in the surviving traits (residual
NaNs below the cleanup thresholds), delivering the clean frame rather than raising. Because
bad traits are dropped first, this row drop loses far fewer samples than a naive
`df.dropna()`.

#### Scenario: Output has no NaNs and PCA row-dropna is a no-op

- **WHEN** `clean_traits_for_analysis(df)` returns `(clean_df, trait_cols, _)` for a fixture
  with several NaN-heavy traits
- **THEN** `clean_df[trait_cols]` contains zero NaN values
- **AND** `perform_pca_analysis(clean_df[trait_cols])` runs successfully and reports a
  sample count equal to `len(clean_df)` (no rows dropped)

#### Scenario: Ordinary sparse data returns a clean frame on default thresholds

- **WHEN** `clean_traits_for_analysis(df)` is called with default thresholds on a frame
  whose only defect is a residual NaN that the per-sample threshold would retain
- **THEN** the offending row is dropped and a NaN-free frame is returned (no `ValueError`)

### Requirement: Sample-Loss Minimization

`clean_traits_for_analysis` SHALL retain at least as many samples as a naive `df.dropna()`
would, by dropping problematic traits before dropping NaN rows.

#### Scenario: Retains more samples than naive dropna

- **GIVEN** a fixture sized so the good traits clear `min_samples_per_trait` and the
  NaN-heavy traits exceed `max_nans_per_trait` (so they are dropped, saving their rows)
- **WHEN** `clean_traits_for_analysis(df)` returns `(clean_df, _, _)`
- **THEN** `len(clean_df)` is greater than `len(df.dropna())`

### Requirement: Analysis-Readiness Validation

`clean_traits_for_analysis` SHALL validate that the cleaned result is runnable for
PCA/UMAP/clustering and SHALL raise a clear, actionable `ValueError` when it is not. The
checks SHALL run in a fixed order — (1) empty input, (2) no NaN in surviving traits, (3) at
least `MIN_SAMPLES_FOR_ANALYSIS` (2) surviving samples, (4) at least one non-constant numeric
trait — so the raised message is deterministic when multiple conditions fail. Check (2) is a
defensive guard via the shared `validate_clean_traits`: residual NaN rows are dropped before
it (see No NaN in Analysis-Ready Output), so under normal flow it does not raise.
"Non-constant" SHALL be defined as `var(ddof=0) > 0`, matching the variance test
`perform_pca_analysis`/`standardize_data` use, and SHALL be evaluated after the no-NaN step.
Two surviving samples is the runnability floor only; the error/docstring SHALL note that
meaningful multivariate analysis needs many more samples than traits.

#### Scenario: Raises its own error on empty input before delegating

- **WHEN** `clean_traits_for_analysis` is called on a table with no rows or no resolvable
  trait columns
- **THEN** a `ValueError` from `clean_traits_for_analysis` with an actionable message is
  raised before `apply_data_cleanup_filters` or `perform_pca_analysis` is reached

#### Scenario: Raises when fewer than 2 samples survive

- **WHEN** cleanup (including the residual-NaN-row drop) leaves fewer than 2 samples
- **THEN** a `ValueError` naming the surviving sample count is raised

#### Scenario: Raises when only a constant trait survives

- **WHEN** cleanup leaves a single numeric trait whose `var(ddof=0)` is 0
- **THEN** a `ValueError` stating that no non-constant trait remains is raised

#### Scenario: Succeeds when at least one trait varies among several

- **WHEN** cleanup leaves multiple traits, at least one with `var(ddof=0) > 0`
- **THEN** validation passes and the function returns successfully (a constant trait
  alongside a varying one does not fail the gate)

### Requirement: Input Misuse Diagnostics

`clean_traits_for_analysis` SHALL reject malformed input up front with an actionable
`ValueError` rather than a bare pandas error or a later, opaque failure: duplicate column
names in the input, explicit `trait_cols` names absent from the dataframe, and explicit
`trait_cols` that are non-numeric.

#### Scenario: Duplicate column names are rejected

- **WHEN** the input dataframe has duplicate column names
- **THEN** a `ValueError` naming the duplicated columns is raised

#### Scenario: Explicit trait_cols not in the dataframe are rejected

- **WHEN** an explicit `trait_cols` entry is not a column of `df`
- **THEN** a `ValueError` naming the missing columns is raised (not a bare `KeyError`)

#### Scenario: Explicit non-numeric trait_cols are rejected

- **WHEN** an explicit `trait_cols` entry is a non-numeric column
- **THEN** a `ValueError` naming the non-numeric columns is raised

### Requirement: Diagnostics for Programmatic Consumers

`clean_traits_for_analysis` SHALL surface its effective behavior beyond the docstring so
programmatic consumers see it: it SHALL log (at INFO) the effective thresholds used and the
note that they differ from the pipeline config and that name-sanitization is not applied,
and it SHALL emit a `UserWarning` when surviving traits outnumber surviving samples (the
p > n regime).

#### Scenario: Warns in the p > n regime

- **WHEN** the cleaned frame has more surviving traits than samples
- **THEN** a `UserWarning` noting `p > n` and statistical unreliability is emitted, and the
  function still returns the frame

### Requirement: Single Source of Truth With Pipeline

`clean_traits_for_analysis` and the QC pipeline steps SHALL share the same underlying
functions so the cleanup algorithm and validation semantics cannot drift: the entry point
SHALL call the exposed `apply_data_cleanup_filters` (the function `CleanupTraitsStep` uses)
and the exposed `validate_clean_traits` / `build_clean_validation_report` (refactored out of
`ValidateCleanStep`'s inline check). This refactor SHALL NOT change the observable behavior
of `QCPipeline` steps 01–03 — same cleaned data, same `03_validation_report.json`, same
`StepResult.metadata`, and the same byte-for-byte error message on failure.

#### Scenario: Pipeline steps 01–03 behavior is unchanged

- **WHEN** the QC pipeline runs steps 01 (load) → 02 (cleanup) → 03 (validate) after the
  refactor, on the existing integration fixture
- **THEN** the cleaned data, the saved validation report contents, the step metadata, and
  the validation outcome SHALL be identical to before the refactor

#### Scenario: Shared validation function rejects residual NaNs identically

- **WHEN** `validate_clean_traits` is given trait columns still containing NaN
- **THEN** it raises a `ValueError` whose message is byte-for-byte the message
  `ValidateCleanStep` raised before the refactor (identifying the offending traits)

### Requirement: API Reference and Changelog Documentation

The project documentation SHALL list the newly-public functions and record them in the
changelog, keeping the hand-maintained reference in sync.

#### Scenario: API reference lists the new public functions

- **WHEN** `docs/API.md` is viewed
- **THEN** it SHALL include reference entries for `clean_traits_for_analysis`,
  `apply_data_cleanup_filters`, `validate_clean_traits`, and `build_clean_validation_report`
  with signatures matching the code

#### Scenario: Changelog records the newly-public API

- **WHEN** `docs/CHANGELOG.md` `[Unreleased]` section is viewed
- **THEN** it SHALL include an `### Added` entry noting these functions are now importable
  from `sleap_roots_analyze`
