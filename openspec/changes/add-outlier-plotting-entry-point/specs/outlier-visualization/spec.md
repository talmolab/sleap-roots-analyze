## ADDED Requirements

### Requirement: Outlier-Plotting Entry Point

The system SHALL provide a single public function `plot_outlier_analysis` that returns the
method-appropriate outlier-detection **figures** for a clean, analysis-ready trait table by
**importing and composing** the existing public `create_*_outlier` figure functions — introducing
no new plotting logic. It SHALL return a `dict` mapping stable string keys to
`matplotlib.figure.Figure` objects, and SHALL perform no file IO (it returns figures; the caller
writes them).

#### Scenario: Returns a dict of named figures for a clean trait table

- **WHEN** `plot_outlier_analysis(clean_df)` is called on a clean (NaN-free) trait table containing
  injected outlier rows
- **THEN** trait columns are resolved via `get_trait_columns` when not explicitly passed
- **AND** the return value is a `dict` whose values are all `matplotlib.figure.Figure` objects
- **AND** the function writes no files and closes no figures (the caller owns IO)

#### Scenario: Delegates to the public figure functions

- **WHEN** `plot_outlier_analysis(clean_df, method="mahalanobis")` runs with
  `create_mahalanobis_outlier_plots` replaced by a test double (monkeypatch)
- **THEN** the test double SHALL be invoked with the frame and the Mahalanobis detector result
- **AND** the function SHALL define no alternative figure-drawing logic of its own

### Requirement: Method-Appropriate Figure Set

`plot_outlier_analysis` SHALL draw a small, method-appropriate set of figures for a single detection
`method`, defaulting to `"mahalanobis"` and also supporting `"isolation_forest"` — the two methods
`remove_outlier_samples` supports. For `"mahalanobis"` the set SHALL be the figures from
`create_mahalanobis_outlier_plots`. For `"isolation_forest"` the set SHALL be the figure from
`create_isolation_forest_plots`. For either method, the per-genotype figure
(`create_outliers_per_genotype_plot`, drawn for the single selected method) SHALL be included when a
genotype column is present. The entry point SHALL NOT draw `create_pca_outlier_plot` (which consumes
a `detect_outliers_pca` reconstruction result absent from a Mahalanobis result), the clustering-method
figures (`kmeans`/`gmm`/`hierarchical`), or the multi-method comparison figures.

#### Scenario: Mahalanobis method returns its figure set

- **WHEN** `plot_outlier_analysis(clean_df, method="mahalanobis")` is called on a frame with a
  genotype column
- **THEN** the returned dict SHALL include the figure keys produced by
  `create_mahalanobis_outlier_plots` and a per-genotype figure key
- **AND** it SHALL NOT include a `create_pca_outlier_plot` figure

#### Scenario: Isolation-forest method returns its figure set

- **WHEN** `plot_outlier_analysis(clean_df, method="isolation_forest")` is called on a frame with a
  genotype column
- **THEN** the returned dict SHALL include the isolation-forest figure key from
  `create_isolation_forest_plots` and a per-genotype figure key
- **AND** it SHALL NOT include Mahalanobis figure keys

#### Scenario: Genotype figure included only when the column is present

- **WHEN** `plot_outlier_analysis` is called on a frame that has no genotype column
- **THEN** the returned dict SHALL omit the per-genotype figure key
- **AND** all other method-appropriate figure keys SHALL still be present

### Requirement: Method Selection Guard

`plot_outlier_analysis` SHALL reject an unknown `method` with an actionable `ValueError` naming the
supported methods (`"mahalanobis"`, `"isolation_forest"`) before any detection runs, and SHALL
reject unknown or cross-method `**detect_kwargs` (e.g. `contamination` with `method="mahalanobis"`)
with a `ValueError` naming the unrecognized key(s) and the supported set for that method — mirroring
`remove_outlier_samples`.

#### Scenario: Unknown method is rejected

- **WHEN** `plot_outlier_analysis(clean_df, method="not_a_method")` is called
- **THEN** a `ValueError` naming the supported methods SHALL be raised before any detection runs

#### Scenario: Unknown or cross-method detect_kwargs are rejected

- **WHEN** a `**detect_kwargs` key is not a parameter of the chosen detector
- **THEN** a `ValueError` naming the unrecognized key(s) and the supported set SHALL be raised
  before any detection runs

### Requirement: Figure Selection via `which`

`plot_outlier_analysis` SHALL accept an optional `which` selector — a single figure-key string **or**
a list of keys — that narrows the returned dict to exactly those keys; `which=None` (default) SHALL
return the method's full available set. A requested key that is not available for the given method
and frame SHALL be rejected with an actionable `ValueError` naming the keys that are available (not
silently dropped and not an empty dict). The selectable keys SHALL be the stable dict keys the
function otherwise returns, so a consumer's selection maps 1:1 to the returned figures.

#### Scenario: which narrows the returned figures

- **WHEN** `plot_outlier_analysis(clean_df, method="mahalanobis", which=["mahalanobis_outlier_detection"])`
  is called and that key is available
- **THEN** the returned dict SHALL contain exactly that one figure and no other keys

#### Scenario: which accepts a single string key

- **WHEN** `which` is a bare string naming an available key (not a list)
- **THEN** the returned dict SHALL contain exactly that one figure (the string SHALL NOT be iterated
  character-by-character)

#### Scenario: None returns the full method set

- **WHEN** `plot_outlier_analysis(clean_df, method="mahalanobis", which=None)` is called
- **THEN** the returned dict SHALL contain the method's full available figure set

#### Scenario: Unavailable which key is rejected

- **WHEN** `which` requests a key not available for that method/frame (e.g. `outliers_per_genotype`
  on a frame with no genotype column, or a misspelled key)
- **THEN** a `ValueError` naming the keys available for that method and frame SHALL be raised

### Requirement: Deterministic Re-Detection Matching Removal

`plot_outlier_analysis` SHALL re-detect outliers by calling the same public detector
(`detect_outliers_mahalanobis` / `detect_outliers_isolation_forest`) that `remove_outlier_samples`
uses, threading the same `random_state` (default `42`) and per-method `**detect_kwargs`. Given equal
inputs, `method`, `random_state`, and params — **and the shared NaN-free + unique-index
preconditions both functions enforce** — the detected outlier **index set** SHALL be identical to the
set `remove_outlier_samples` removes, so the figures depict the same samples that were trimmed. The
determinism guarantee is scoped to the integer `outlier_indices` (exact), not to float score arrays
(which are reproducible only within tolerance). The function SHALL take plain data in (a trait frame,
no `run_dir` / pipeline `config` / `StepResult`) and SHALL accept `random_state=None` without raising.

#### Scenario: Re-detected outliers match remove_outlier_samples

- **WHEN** `remove_outlier_samples(clean_df, method=m, random_state=s, **kw)` and
  `plot_outlier_analysis(clean_df, method=m, random_state=s, **kw)` run on the same clean,
  unique-indexed frame for `m` in {`"mahalanobis"`, `"isolation_forest"`}
- **THEN** the outlier index set `plot_outlier_analysis` re-detects SHALL equal
  `outlier_report["outlier_indices"]` from the removal call

#### Scenario: Repeated calls with the same seed are identical

- **WHEN** `plot_outlier_analysis(clean_df, random_state=7)` is called twice on the same frame
- **THEN** both calls SHALL re-detect the identical outlier index set and return the same figure keys

#### Scenario: Accepts random_state=None

- **WHEN** `plot_outlier_analysis(clean_df, random_state=None)` is called on a valid clean frame
- **THEN** the call SHALL return a figure dict without raising

### Requirement: Clean-Input Precondition

`plot_outlier_analysis` SHALL require a NaN-free input in the trait columns and SHALL verify this via
`validate_clean_traits` before detecting, raising an actionable `ValueError` that names the affected
traits and directs the caller to run `clean_traits_for_analysis` first. This is a correctness guard:
the detectors run PCA that silently drops NaN rows and reports `outlier_indices` against the
post-`dropna` frame, so a NaN-carrying input would misalign the indices the figures (notably
`create_outliers_per_genotype_plot`'s `df.loc[idx, genotype_col]`) index by, and would diverge from
the set `remove_outlier_samples` — which rejects such input — would remove.

#### Scenario: NaN-carrying input is rejected with guidance

- **WHEN** `plot_outlier_analysis` is called on a frame whose trait columns still contain NaN
- **THEN** a `ValueError` naming the affected traits AND mentioning `clean_traits_for_analysis` SHALL
  be raised before any detector runs

### Requirement: Unique Sample Index Required

`plot_outlier_analysis` SHALL require the input frame to have a unique index and SHALL raise an
actionable `ValueError` when it does not, before any detector runs. The per-sample figures index the
frame by re-detected label (`df.loc[idx, ...]`), so a duplicate label would select multiple rows and
mis-draw the per-genotype figure; a unique index makes the one-to-one alignment sound (matching
`remove_outlier_samples`).

#### Scenario: Non-unique index is rejected

- **WHEN** `plot_outlier_analysis` is called on a frame whose index has duplicate labels
- **THEN** a `ValueError` stating the index must be unique SHALL be raised before any detector runs

#### Scenario: Default RangeIndex input is accepted

- **WHEN** `plot_outlier_analysis` is called on a frame with a unique (e.g. default `RangeIndex`)
  index
- **THEN** the uniqueness check SHALL pass and processing SHALL proceed

### Requirement: Input Misuse Diagnostics

`plot_outlier_analysis` SHALL reject malformed input up front with an actionable `ValueError` rather
than a bare pandas error: empty input, duplicate column names, and explicit `trait_cols` that are
missing from the dataframe or non-numeric.

#### Scenario: Empty input is rejected before delegating

- **WHEN** `plot_outlier_analysis` is called on a table with no rows
- **THEN** a `ValueError` with an actionable message SHALL be raised before any detector runs

#### Scenario: Explicit trait_cols not in the dataframe are rejected

- **WHEN** an explicit `trait_cols` entry is not a column of `clean_df`
- **THEN** a `ValueError` naming the missing columns SHALL be raised (not a bare `KeyError`)

### Requirement: IO-Free Figure Return

`plot_outlier_analysis` SHALL return `matplotlib` `Figure` objects and SHALL NOT write files, choose
a file format or DPI, or close the **returned** figures. All persistence SHALL be the caller's
responsibility (the pipeline step `savefig`s with its `config`; an MCP consumer persists via its own
store). The returned dict keys SHALL be stable identifiers suitable for use as filename stems or
artifact names. When a `which` selection narrows the result, any figure that was built but excluded
SHALL be closed so a narrowed call leaves no orphaned figures in matplotlib's global registry (no
unbounded figure growth in a long-running process).

#### Scenario: No files are written

- **WHEN** `plot_outlier_analysis(clean_df)` is called
- **THEN** no image files SHALL be created by the call
- **AND** every returned value SHALL be an open `matplotlib.figure.Figure`

#### Scenario: A which-narrowed call leaves no orphaned figures

- **WHEN** `plot_outlier_analysis(clean_df, method="mahalanobis", which="mahalanobis_outlier_detection")`
  returns one figure
- **THEN** the number of open figures in matplotlib's registry SHALL equal the number returned
  (the figures built but excluded by `which` SHALL have been closed)

### Requirement: Detector-Failure Surfacing

`plot_outlier_analysis` SHALL surface a detector failure by raising a `ValueError` **on the detector
result, before delegating to the figure functions**. When the chosen detector returns a result
carrying an `error` key or lacking `outlier_indices` (e.g. degenerate PCA, empty/all-NaN data), the
function SHALL raise, rather than pass the result to the `create_*` functions — which silently return
an empty figure dict on such input, masking the failure.

#### Scenario: Detector error is raised, not silently swallowed

- **WHEN** the chosen detector (monkeypatched) returns a result containing an `error` key or no
  `outlier_indices`
- **THEN** `plot_outlier_analysis` SHALL raise a `ValueError` surfacing that error before any
  `create_*` figure function is called

### Requirement: Metadata-Column Parameters Match Removal

`plot_outlier_analysis` SHALL accept the same metadata-column parameters as `remove_outlier_samples`
— `barcode_col` (default `"Barcode"`), `genotype_col` (default `"geno"`), and `replicate_col`
(default `"rep"`) — and SHALL forward them to `get_trait_columns` when inferring traits and use
`genotype_col` for the per-genotype figure. This keeps the re-detected trait set (and therefore the
plotted outlier set) aligned with a `remove_outlier_samples` call made with the same metadata columns,
rather than silently diverging on a hardcoded default.

#### Scenario: Metadata columns are forwarded to trait resolution

- **WHEN** `plot_outlier_analysis` is called with explicit `barcode_col` / `genotype_col` /
  `replicate_col` and no `trait_cols`
- **THEN** those column names SHALL be passed to `get_trait_columns` for the trait inference

#### Scenario: Per-genotype figure follows genotype_col

- **WHEN** `plot_outlier_analysis(clean_df, genotype_col="Genotype")` is called on a frame whose
  genotype column is named `"Genotype"` (not the default `"geno"`)
- **THEN** the per-genotype figure SHALL be produced using the `"Genotype"` column

### Requirement: Public Figure-Selection Layer and Single-Detection Reuse

The no-detection figure-selection layer SHALL be public as `select_outlier_figures(df, results,
method, which=None, genotype_col=None)` and listed in `__all__`, so a consumer that already holds a
detector result can select figures without a redundant re-detection. To supply that result without
re-stitching, `remove_outlier_samples` SHALL accept an additive `return_detector_result: bool = False`
parameter that, when `True`, additionally returns the raw detector result dict (a third tuple
element); the default `False` SHALL preserve the existing compact-report 2-tuple return contract.

#### Scenario: Selection layer is public and detection-free

- **WHEN** `select_outlier_figures(df, {method: detector_result}, method)` is called with a
  pre-computed detector result
- **THEN** it SHALL return the method-appropriate figures without running any detector

#### Scenario: Removal can return the raw detector result for reuse

- **WHEN** `remove_outlier_samples(clean_df, return_detector_result=True)` is called
- **THEN** it SHALL return a 3-tuple `(trimmed_df, outlier_report, detector_result)` whose
  `detector_result` feeds `select_outlier_figures` (via `{method: detector_result}`) to plot the
  same outliers without a second detection
- **WHEN** `return_detector_result` is `False` (default)
- **THEN** it SHALL return the 2-tuple `(trimmed_df, outlier_report)` unchanged

### Requirement: Reproducibility-Gate Registration

This change SHALL register `plot_outlier_analysis` in the package reproducibility registry so the
stochastic-determinism sweep continues to pass. Because `plot_outlier_analysis` exposes a
`random_state` parameter, the package-wide sweep (`tests/test_reproducibility.py`) auto-discovers it
and the coverage guard fails until it is registered — so registration SHALL land in the same commit
as the implementation. Registration SHALL be either a `tests/reproducibility_cases.py` `CASES` entry
whose comparable is the re-detected `outlier_indices` (the `Case` comparator cannot compare
`Figure`s) compared exactly, or an `EXCLUDED` entry with the documented reason that determinism is
delegated to the already-swept `detect_outliers_*`. The pinned `EXPECTED_QUALNAMES` / case-count
anchors in `tests/test_reproducibility.py` SHALL be updated in lockstep.

#### Scenario: Reproducibility sweep covers the new function

- **WHEN** `tests/test_reproducibility.py`'s package-wide coverage sweep runs
- **THEN** `plot_outlier_analysis` SHALL be present in `CASES` (comparable on `outlier_indices`) or
  in `EXCLUDED` with a reason
- **AND** the suite (including the `EXPECTED_QUALNAMES` / count anchors) SHALL pass

### Requirement: Public API Surface

The package SHALL export both `plot_outlier_analysis` and `select_outlier_figures` from the top-level
`sleap_roots_analyze` namespace and list them in `__all__`, with Google-style docstrings
(Args/Returns/Raises) and type hints resolvable by `typing.get_type_hints()`. If `**detect_kwargs` is
annotated, the annotation SHALL be importable (e.g. `Any` imported) so `get_type_hints()` does not
raise under `from __future__ import annotations`. The `create_*_outlier` figure functions they compose
are already exported and SHALL remain exported.

#### Scenario: Entry point is importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import plot_outlier_analysis`
- **THEN** the import SHALL succeed
- **AND** the imported object SHALL be identity-equal (`is`) to the function defined in
  `sleap_roots_analyze.outlier_visualization`

#### Scenario: Entry point and selection layer are listed in `__all__` without duplicates

- **WHEN** `sleap_roots_analyze.__all__` is inspected
- **THEN** it SHALL contain `plot_outlier_analysis` and `select_outlier_figures`
- **AND** SHALL contain no duplicate entries

#### Scenario: Public function satisfies the package API-docs audit

- **WHEN** `typing.get_type_hints(plot_outlier_analysis)` is called and the package
  `test_public_api_docs` audit runs
- **THEN** `get_type_hints` SHALL return without raising `NameError`
- **AND** the docstring SHALL include populated Args, Returns, and Raises sections in Google style

### Requirement: API Reference and Changelog Documentation

The project documentation SHALL list the new public function and record it in the changelog, keeping
the hand-maintained reference in sync. Because the new API.md entry cross-references the composed
figure functions, any of those currently absent from `docs/API.md`
(`create_mahalanobis_outlier_plots`, `create_isolation_forest_plots`,
`create_outliers_per_genotype_plot` — the `outlier_visualization` module has no API.md section
today) SHALL be added so the references resolve.

#### Scenario: API reference lists the new public function and resolvable cross-references

- **WHEN** `docs/API.md` is viewed
- **THEN** it SHALL include a reference entry for `plot_outlier_analysis` with a signature matching
  the code
- **AND** the composed `create_*_outlier` figure functions it references SHALL each have an API.md
  entry

#### Scenario: Changelog records the new public API

- **WHEN** `docs/CHANGELOG.md` `[Unreleased]` section is viewed
- **THEN** it SHALL include an `### Added` entry noting `plot_outlier_analysis` is importable from
  `sleap_roots_analyze` as the outlier-plotting sibling of `remove_outlier_samples`, with a `(#173)`
  issue suffix
