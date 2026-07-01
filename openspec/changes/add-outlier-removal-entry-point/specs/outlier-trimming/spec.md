## ADDED Requirements

### Requirement: Outlier-Removal Entry Point

The system SHALL provide a single public function `remove_outlier_samples` that detects and
removes outlier **samples** from a clean, analysis-ready trait table by **importing and
composing** the existing public detection functions (`detect_outliers_mahalanobis` /
`detect_outliers_isolation_forest`) and the existing public removal function
(`remove_outliers_from_data`) — introducing no new outlier-detection or removal algorithm. It
SHALL return a tuple `(trimmed_df, outlier_report)` where `trimmed_df` is the input frame with
the flagged outlier rows removed (metadata columns preserved) and `outlier_report` is an
auditable, JSON-serializable dict describing which samples were removed and why.

#### Scenario: Detect and remove outliers from a clean trait table

- **WHEN** `remove_outlier_samples(clean_df)` is called on a clean (NaN-free) trait table
  containing injected outlier rows
- **THEN** trait columns are resolved via `get_trait_columns` when not explicitly passed
- **AND** the selected detector and `remove_outliers_from_data` are used to drop the flagged rows
- **AND** the function returns a 2-tuple `(trimmed_df, outlier_report)`
- **AND** `len(trimmed_df)` equals `len(clean_df) - outlier_report["n_outliers"]`

#### Scenario: Trimmed frame preserves metadata columns and trait columns

- **WHEN** `remove_outlier_samples(clean_df, trait_cols=[...])` returns `(trimmed_df, _)`
- **THEN** `trimmed_df` SHALL contain the same columns as `clean_df` (metadata and trait
  columns alike — outlier removal drops rows, never columns)
- **AND** the supplied `trait_cols` remain valid columns of `trimmed_df`

#### Scenario: Caller-supplied trait columns are honored

- **WHEN** `remove_outlier_samples(clean_df, trait_cols=[...])` is called with an explicit
  trait-column list
- **THEN** `get_trait_columns` is not used to infer columns
- **AND** outlier detection operates over exactly the supplied trait columns

### Requirement: Method Selection and Parameters

`remove_outlier_samples` SHALL trim using a single detection `method` selected by the caller,
defaulting to `"mahalanobis"` (Mahalanobis distance on PCA-transformed data with a chi-squared
threshold) and also supporting `"isolation_forest"`. It SHALL forward `random_state` and any
per-method `**detect_kwargs` (e.g. `contamination`, `chi2_percentile`, `variance_threshold`) to
the chosen detector unchanged, and SHALL reject an unknown `method` with an actionable error.

#### Scenario: Default method is Mahalanobis

- **WHEN** `remove_outlier_samples(clean_df)` is called without specifying `method`
- **THEN** detection SHALL use `detect_outliers_mahalanobis`
- **AND** `outlier_report["method"]` SHALL equal `"mahalanobis"`
- **AND** the removed rows SHALL be exactly those flagged by `detect_outliers_mahalanobis`

#### Scenario: Isolation-forest method with a contamination parameter

- **WHEN** `remove_outlier_samples(clean_df, method="isolation_forest", contamination=0.2)` is
  called
- **THEN** detection SHALL use `detect_outliers_isolation_forest` with `contamination=0.2`
- **AND** `outlier_report["method"]` SHALL equal `"isolation_forest"` (the dispatch key, not the
  detector's internal `"IsolationForest"` label)
- **AND** `outlier_report["method_params"]` SHALL record `contamination=0.2`

#### Scenario: Mahalanobis kwargs are forwarded and echoed

- **WHEN** `remove_outlier_samples(clean_df, method="mahalanobis", chi2_percentile=99.0)` is
  called
- **THEN** `chi2_percentile=99.0` SHALL be forwarded to `detect_outliers_mahalanobis`
- **AND** `outlier_report["method_params"]` SHALL record `chi2_percentile=99.0`

#### Scenario: Unknown method is rejected

- **WHEN** `remove_outlier_samples(clean_df, method="not_a_method")` is called
- **THEN** a `ValueError` naming the supported methods (`"mahalanobis"`, `"isolation_forest"`)
  SHALL be raised before any detection runs

#### Scenario: Unknown or cross-method detect_kwargs are rejected

- **WHEN** a `**detect_kwargs` key is not a parameter of the chosen detector (a typo, or a
  cross-method knob such as `contamination` with `method="mahalanobis"` or `chi2_percentile`
  with `method="isolation_forest"`)
- **THEN** a `ValueError` naming the unrecognized key(s) and the supported parameter set for that
  method SHALL be raised before any detection runs (not a bare `TypeError` leaking the internal
  detector name, and not silently dropped from `method_params`)

### Requirement: Single Source of Truth With Detection and Removal Primitives

`remove_outlier_samples` and the QC pipeline's outlier steps SHALL share the same underlying
functions so the detection and removal semantics cannot drift: the entry point SHALL call the
public `detect_outliers_mahalanobis` / `detect_outliers_isolation_forest` (the functions
`DetectOutliersStep` uses) and the public `remove_outliers_from_data` (the row-dropping helper
the removal step uses), rather than re-implementing detection or removal. This change SHALL NOT
modify `DetectOutliersStep`, `RemoveOutliersStep`, any other pipeline step, or
`clean_traits_for_analysis`.

#### Scenario: Entry point composes the already-public primitives

- **WHEN** the source of `remove_outlier_samples` is inspected
- **THEN** it SHALL import and call `detect_outliers_mahalanobis` /
  `detect_outliers_isolation_forest` and `remove_outliers_from_data` from
  `sleap_roots_analyze.outlier_detection`
- **AND** it SHALL define no alternative outlier-scoring or row-removal logic of its own

#### Scenario: No behavior change to existing functions

- **WHEN** the existing pipeline-step and outlier-detection test suites run after this change
- **THEN** their outcomes SHALL be unchanged (the change is additive: a new module plus one
  `__all__` entry)

### Requirement: Clean-Input Precondition

`remove_outlier_samples` SHALL require a NaN-free input in the trait columns and SHALL verify
this via the exposed `validate_clean_traits` before detecting. When the precondition is
violated it SHALL raise an actionable `ValueError` that both names the affected traits and
directs the caller to run `clean_traits_for_analysis` first — wrapping `validate_clean_traits`'s
message (which does not itself mention the entry point) so the pointer is present. This is a
correctness guard: the detectors run PCA which silently drops NaN rows and reports outlier
indices against the post-`dropna` index, so a NaN-carrying input would misalign the indices
handed to `remove_outliers_from_data`. A NaN-free input guarantees the detector's
`outlier_indices` align one-to-one with the input frame's rows.

#### Scenario: NaN-carrying input is rejected with guidance

- **WHEN** `remove_outlier_samples` is called on a frame whose trait columns still contain NaN
- **THEN** a `ValueError` naming the affected traits AND mentioning `clean_traits_for_analysis`
  SHALL be raised before any detector runs

#### Scenario: Detection labels align with input rows on clean input

- **WHEN** `remove_outlier_samples(clean_df)` runs on a NaN-free frame
- **THEN** every label in `outlier_report["outlier_indices"]` SHALL be a member of
  `clean_df.index`
- **AND** the rows removed from `trimmed_df` SHALL be exactly those labels

### Requirement: Unique Sample Index Required

`remove_outlier_samples` SHALL require the input frame to have a **unique** index and SHALL
raise an actionable `ValueError` when it does not. Because `clean_traits_for_analysis` does not
reset the index, the input inherits the caller's index, which may carry duplicate labels (e.g.
after `set_index("Barcode")` on replicated barcodes or after concatenation). Removal is
label-based (`remove_outliers_from_data` uses `df.drop(index=...)` / `df.loc[...]`), so a
duplicate label would silently drop or duplicate inlier rows that share a flagged outlier's
label. Enforcing a unique index makes the one-to-one alignment guarantee sound.

#### Scenario: Non-unique index is rejected

- **WHEN** `remove_outlier_samples` is called on a frame whose index has duplicate labels
- **THEN** a `ValueError` stating the index must be unique SHALL be raised before any detector
  runs

#### Scenario: Default RangeIndex input is accepted

- **WHEN** `remove_outlier_samples` is called on a frame with a unique (e.g. default
  `RangeIndex`) index
- **THEN** the uniqueness check SHALL pass and processing SHALL proceed

### Requirement: Output Analysis-Readiness Preserved

The frame returned by `remove_outlier_samples` SHALL remain safe to hand directly to
`perform_pca_analysis` / UMAP / clustering. After removal the function SHALL re-apply the
analysis-readiness gates — at least `MIN_SAMPLES_FOR_ANALYSIS` (2) surviving samples, and at
least one non-constant numeric trait (`var(ddof=0) > 0`, the basis `standardize_data` uses) —
in a fixed order (samples, then non-constant), raising a distinct, actionable `ValueError`
rather than returning a frame too small or degenerate to analyze. The raised `ValueError` SHALL
carry the `outlier_report` (as an attribute) so a caller can still inspect what would have been
removed. The function SHALL additionally emit a `UserWarning` in the `p > n` regime (surviving
samples fewer than surviving traits), matching `clean_traits_for_analysis`'s guardrail, since
trimming can push an `n > p` frame into `p > n`.

#### Scenario: Raises when trimming leaves fewer than 2 samples

- **WHEN** outlier removal would leave fewer than `MIN_SAMPLES_FOR_ANALYSIS` samples
- **THEN** a `ValueError` naming the surviving sample count SHALL be raised
- **AND** the raised error SHALL carry the `outlier_report`

#### Scenario: Raises when trimming leaves only a constant trait

- **WHEN** outlier removal leaves a single numeric trait whose `var(ddof=0)` is 0
- **THEN** a `ValueError` stating that no non-constant trait remains SHALL be raised

#### Scenario: Passes when a constant trait survives alongside a varying one

- **WHEN** outlier removal leaves multiple traits, at least one with `var(ddof=0) > 0`
- **THEN** the readiness gate SHALL pass and the function SHALL return successfully (a constant
  trait alongside a varying one does not fail the gate)

#### Scenario: Trimmed frame runs PCA without dropping rows

- **WHEN** `remove_outlier_samples(clean_df)` returns `(trimmed_df, _)` on a valid clean frame
- **THEN** `perform_pca_analysis(trimmed_df[trait_cols])` SHALL run successfully and report a
  sample count equal to `len(trimmed_df)` (no rows dropped)

#### Scenario: Warns when trimming enters the p > n regime

- **WHEN** the trimmed frame has more surviving traits than samples
- **THEN** a `UserWarning` noting `p > n` statistical unreliability SHALL be emitted and the
  function SHALL still return `(trimmed_df, outlier_report)`

### Requirement: Over-Removal Safety Rail

`remove_outlier_samples` SHALL emit a `UserWarning` when the removed fraction of samples exceeds
a guard fraction (default `0.5`), surfacing the likely-mis-set-`contamination`/threshold case
without itself failing on genuinely dirty data. This warning SHALL be emitted **before** the
output-readiness gates are evaluated, so it is observable even when removal subsequently fails a
readiness gate.

#### Scenario: Warns when the majority of samples are removed

- **WHEN** the selected method flags more than half of the input samples as outliers and ≥2
  varying samples survive
- **THEN** a `UserWarning` noting the large removal fraction SHALL be emitted
- **AND** the function SHALL still return `(trimmed_df, outlier_report)`

#### Scenario: Over-removal warning precedes a readiness failure

- **WHEN** the selected method flags so many samples that fewer than 2 survive
- **THEN** the over-removal `UserWarning` SHALL be emitted before the readiness `ValueError` is
  raised

### Requirement: Mahalanobis Quality Signals

On the Mahalanobis path the default `chi2_percentile=97.5` trims roughly the top 2.5% of samples
by construction — even on outlier-free data — and that threshold's meaning rests on the squared
distances following a chi-squared distribution. So that a routine default trim of genuinely-clean
data does not pass with no signal, `remove_outlier_samples` SHALL emit a `UserWarning` when the
sample count is small (`n < 30` — fragile chi-squared tail / covariance estimate) and SHALL emit a
`UserWarning` when the detector's chi-squared goodness-of-fit reports
`distributional_assumption_valid` is `False`. Both SHALL be emitted before the output-readiness
gates (so they are observable even when an aggressive trim then fails a gate), and SHALL NOT fire
on the isolation-forest path (which has no chi-squared assumption).

#### Scenario: Warns on a small Mahalanobis sample

- **WHEN** `remove_outlier_samples(clean_df)` runs the Mahalanobis path on fewer than 30 samples
- **THEN** a `UserWarning` noting the small-sample fragility SHALL be emitted and the function
  SHALL still return `(trimmed_df, outlier_report)`

#### Scenario: Warns when the chi-squared assumption is violated

- **WHEN** the detector's `goodness_of_fit` reports `distributional_assumption_valid == False`
- **THEN** a `UserWarning` noting the chi-squared assumption is violated SHALL be emitted and the
  function SHALL still return `(trimmed_df, outlier_report)`

#### Scenario: No quality warnings on a clean, large Mahalanobis sample

- **WHEN** `remove_outlier_samples(clean_df)` runs the Mahalanobis path on a clean frame with
  `n ≥ 30` and a well-fit chi-squared tail
- **THEN** neither the small-sample nor the goodness-of-fit `UserWarning` SHALL be emitted

### Requirement: Auditable Outlier Report

`remove_outlier_samples` SHALL return an `outlier_report` dict that makes the removal auditable
and reproducible. It SHALL contain at least: `method` (the dispatch key), `method_params` (the
effective per-method parameters used), `random_state`, `n_input_samples`, `n_outliers`,
`n_output_samples`, `removal_fraction`, `outlier_indices` (the removed sample labels),
`outlier_barcodes` (the barcodes of removed samples when `barcode_col` is a column of the input,
else `None`), the detector's `threshold_type` and `threshold_value`, and — for the Mahalanobis
method — the PCA basis the distances were computed on (`n_components`, effective
`variance_threshold`) and the chi-squared `goodness_of_fit`. For `method="isolation_forest"`,
`threshold_type`/`threshold_value`/`goodness_of_fit` SHALL be `None` (isolation forest has no
distance threshold; its control is `contamination`, recorded in `method_params`). The report
SHALL be JSON-serializable using only plain Python scalar/list types (no numpy scalars or
arrays) and SHALL NOT embed large per-sample arrays (e.g. full distance/score vectors).

#### Scenario: Report records counts, fraction, and removed sample identities

- **WHEN** `remove_outlier_samples(clean_df)` returns `(_, outlier_report)`
- **THEN** `outlier_report["n_input_samples"]`, `n_outliers`, and `n_output_samples` SHALL be
  consistent (`n_input_samples == n_outliers + n_output_samples`)
- **AND** `outlier_report["removal_fraction"]` SHALL equal `n_outliers / n_input_samples`
- **AND** `outlier_report["outlier_indices"]` SHALL list the removed sample labels

#### Scenario: Barcodes are listed when present and None when absent

- **WHEN** the input has a `barcode_col` column
- **THEN** `outlier_report["outlier_barcodes"]` SHALL list the removed rows' barcodes
- **WHEN** the input has no `barcode_col` column
- **THEN** `outlier_report["outlier_barcodes"]` SHALL be `None`

#### Scenario: Mahalanobis report carries threshold and goodness-of-fit; isolation forest does not

- **WHEN** `remove_outlier_samples(clean_df, method="mahalanobis")` returns `(_, report)`
- **THEN** `report["threshold_type"]`, `threshold_value`, `n_components`, and `goodness_of_fit`
  SHALL be populated from the detector result
- **WHEN** `remove_outlier_samples(clean_df, method="isolation_forest")` returns `(_, report)`
- **THEN** `report["threshold_type"]`, `threshold_value`, and `goodness_of_fit` SHALL be `None`

#### Scenario: Report is JSON-serializable with plain types

- **WHEN** `outlier_report` is passed to `json.dumps`
- **THEN** serialization SHALL succeed without raising
- **AND** every value in `outlier_indices` SHALL be a plain Python `int` or `str` (not a numpy
  scalar), and `threshold_value` (when present) SHALL be a plain Python `float`

### Requirement: Deterministic Outlier Selection

`remove_outlier_samples` SHALL be deterministic given its inputs: the same `clean_df`, `method`,
`random_state`, and per-method parameters SHALL produce the same set of `outlier_indices` and
the same `trimmed_df`. The `random_state` (default `42`) SHALL be threaded into the detector and
recorded in `outlier_report`. The seed is only consequential for paths with stochastic
estimators — `method="isolation_forest"`, `robust_covariance=True` (MinCovDet), or large-n
randomized SVD; for the default exact-SVD Mahalanobis path the result is identical regardless of
seed, and the determinism guarantee holds in either case.

#### Scenario: Repeated calls with the same seed are identical

- **WHEN** `remove_outlier_samples(clean_df, random_state=7)` is called twice on the same frame
- **THEN** both calls SHALL return identical `outlier_report["outlier_indices"]`
- **AND** the two `trimmed_df` results SHALL be equal
- **AND** `outlier_report["random_state"]` SHALL equal `7`

#### Scenario: Seed is load-bearing on the isolation-forest path

- **WHEN** `remove_outlier_samples(clean_df, method="isolation_forest")` is run on a fixture
  near the contamination boundary with two different `random_state` values
- **THEN** the two runs MAY differ, and each run SHALL be reproducible under its own seed
  (re-running with the same seed yields the identical `outlier_indices`)

### Requirement: Detector-Failure Surfacing

`remove_outlier_samples` SHALL surface a detector failure rather than silently treating it as
"zero outliers removed". When the chosen detector returns a result carrying an `error` key or
lacking `outlier_indices` (e.g. degenerate PCA, empty/all-NaN data), the entry point SHALL raise
a `ValueError` surfacing the detector's error, not return the input frame unchanged with
`n_outliers == 0`.

#### Scenario: Detector error is raised, not silently swallowed

- **WHEN** the chosen detector returns a result containing an `error` key or no `outlier_indices`
- **THEN** `remove_outlier_samples` SHALL raise a `ValueError` surfacing that error
- **AND** SHALL NOT return `(clean_df, report-with-n_outliers-0)`

### Requirement: Input Misuse Diagnostics

`remove_outlier_samples` SHALL reject malformed input up front with an actionable `ValueError`
rather than a bare pandas error or an opaque later failure: empty input, duplicate column names,
and explicit `trait_cols` that are missing from the dataframe or non-numeric.

#### Scenario: Empty input is rejected before delegating

- **WHEN** `remove_outlier_samples` is called on a table with no rows
- **THEN** a `ValueError` from `remove_outlier_samples` with an actionable message SHALL be
  raised before any detector runs

#### Scenario: Duplicate column names are rejected

- **WHEN** the input dataframe has duplicate column names
- **THEN** a `ValueError` naming the duplicated columns SHALL be raised

#### Scenario: Explicit trait_cols not in the dataframe are rejected

- **WHEN** an explicit `trait_cols` entry is not a column of `clean_df`
- **THEN** a `ValueError` naming the missing columns SHALL be raised (not a bare `KeyError`)

#### Scenario: Explicit non-numeric trait_cols are rejected

- **WHEN** an explicit `trait_cols` entry is a non-numeric column
- **THEN** a `ValueError` naming the non-numeric columns SHALL be raised

### Requirement: Reproducibility-Gate Registration

Because `remove_outlier_samples` exposes a `random_state` parameter, the package-wide
stochastic-determinism sweep (`tests/test_reproducibility.py`, which walks every module and
fails if any `random_state`-bearing public function is unregistered) SHALL continue to pass.
This change SHALL register `remove_outlier_samples` in the reproducibility registry
(`tests/reproducibility_cases.py` `CASES`, or `EXCLUDED` with a documented justification) and
update the pinned `EXPECTED_QUALNAMES` / case-count anchors in `tests/test_reproducibility.py`
in lockstep.

#### Scenario: Reproducibility sweep covers the new function

- **WHEN** `tests/test_reproducibility.py`'s package-wide coverage sweep runs
- **THEN** `remove_outlier_samples` SHALL be present in `CASES` (or `EXCLUDED` with a reason)
- **AND** the suite (including the `EXPECTED_QUALNAMES` / count anchors) SHALL pass

### Requirement: Public API Surface

The package SHALL export `remove_outlier_samples` from the top-level `sleap_roots_analyze`
namespace and list it in `__all__`, with a Google-style docstring (Args/Returns/Raises) and
type hints resolvable by `typing.get_type_hints()`. The detection and removal functions it
composes (`detect_outliers_mahalanobis`, `detect_outliers_isolation_forest`,
`remove_outliers_from_data`) are already exported and SHALL remain exported.

#### Scenario: Entry point is importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import remove_outlier_samples`
- **THEN** the import SHALL succeed
- **AND** the imported object SHALL be identity-equal (`is`) to the function defined in
  `sleap_roots_analyze.outlier_removal`

#### Scenario: Entry point and composed primitives are listed in `__all__`

- **WHEN** `sleap_roots_analyze.__all__` is inspected
- **THEN** it SHALL contain `remove_outlier_samples`, `detect_outliers_mahalanobis`,
  `detect_outliers_isolation_forest`, and `remove_outliers_from_data`
- **AND** SHALL contain no duplicate entries

#### Scenario: Public function satisfies the package API-docs audit

- **WHEN** `typing.get_type_hints(remove_outlier_samples)` is called and the package
  `test_public_api_docs` audit runs
- **THEN** `get_type_hints` SHALL return without raising `NameError`
- **AND** the docstring SHALL include populated Args, Returns, and Raises sections in Google
  style (the audit gate enforces this for every `__all__` entry)

### Requirement: API Reference and Changelog Documentation

The project documentation SHALL list the new public function and record it in the changelog,
keeping the hand-maintained reference in sync. Because the new API.md entry cross-references the
composed primitives, any of those primitives currently absent from `docs/API.md`
(`detect_outliers_isolation_forest`, `remove_outliers_from_data`) SHALL be added so the
references resolve.

#### Scenario: API reference lists the new public function and resolvable cross-references

- **WHEN** `docs/API.md` is viewed
- **THEN** it SHALL include a reference entry for `remove_outlier_samples` with a signature
  matching the code
- **AND** the `outlier_detection` primitives it composes SHALL each have an API.md entry

#### Scenario: Changelog records the new public API

- **WHEN** `docs/CHANGELOG.md` `[Unreleased]` section is viewed
- **THEN** it SHALL include an `### Added` entry noting `remove_outlier_samples` is importable
  from `sleap_roots_analyze` as the outlier-trimming entry point following
  `clean_traits_for_analysis`, with a `(#165)` issue suffix
