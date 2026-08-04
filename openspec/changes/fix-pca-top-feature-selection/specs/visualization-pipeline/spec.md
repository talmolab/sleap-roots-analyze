## ADDED Requirements

### Requirement: PCA Analysis Step Feature Selection Resolution

`PCAAnalysisStep` SHALL always pass `pc_indices=list(range(n_components))`
(where `n_components` is `pca_results["n_components_selected"]`) to
`select_top_features_from_pca()`, never relying on that function's `[0, 1]`
default, so feature selection is scoped to every retained PC regardless of
how many components `pca.n_components` selected.

For `feature_selection_strategy == "extreme"`, `PCAAnalysisStep` SHALL call
`select_top_features_from_pca()` with `n_features_to_select=1`
unconditionally — `config.pca.n_top_features` SHALL NOT be read for this
method, regardless of its value. Combined with the PC-scoping requirement
above, this always selects exactly one most-positive-loading and one
most-negative-loading trait per retained PC.

For `feature_selection_strategy == "top_variance"`, when
`config.pca.n_top_features < 1`, `PCAAnalysisStep` SHALL resolve a concrete
feature count by calling `select_n_features_by_variance(
pca_results["feature_contributions"], config.pca.n_top_features)` before
calling `select_top_features_from_pca()`, and SHALL pass that resolved
integer as `n_features_to_select`. When `config.pca.n_top_features >= 1`,
`PCAAnalysisStep` SHALL pass `int(config.pca.n_top_features)` directly,
unchanged from current behavior — including when the value is exactly
`1.0`, which SHALL be treated as the count `1`, not as a 100%-variance
threshold.

For `feature_selection_strategy` set to `"top_absolute"` or
`"top_contribution"`, `PCAAnalysisStep` SHALL pass
`int(config.pca.n_top_features)` directly, unchanged from current
behavior.

#### Scenario: PCAAnalysisStep scopes selection to all retained PCs for every pc_indices-respecting method

- **GIVEN** a run configured with `pca.n_components` selecting 3 or more
  PCs (e.g. a variance threshold like `0.75`)
- **WHEN** `PCAAnalysisStep` executes with `feature_selection_strategy` set
  to `"extreme"`, `"top_absolute"`, or `"top_contribution"`
- **THEN** `top_features.csv` and the `top_features` metadata list SHALL
  reflect features selected across every retained PC, not only PC1 and PC2

#### Scenario: PCAAnalysisStep extreme selection ignores n_top_features

- **WHEN** `PCAAnalysisStep` executes with
  `feature_selection_strategy="extreme"` and `pca.n_components` selects `k`
  PCs, regardless of what `pca.n_top_features` is set to (including values
  that would previously have produced a different count)
- **THEN** `top_features.csv` and the `top_features` metadata list SHALL
  contain exactly one most-positive-loading and one most-negative-loading
  trait per retained PC (at most `2 * k` entries, fewer if a trait is
  extreme on more than one PC)

#### Scenario: PCAAnalysisStep logs that n_top_features is ignored under extreme

- **WHEN** `PCAAnalysisStep` executes with
  `feature_selection_strategy="extreme"`, regardless of whether
  `pca.n_top_features` was explicitly set in the config
- **THEN** the step SHALL emit a `logger.info()` message stating that
  `n_top_features` is not read for this method
- **AND** this SHALL NOT raise a warning or exception, and SHALL NOT
  affect `top_features.csv` output

#### Scenario: PCAAnalysisStep extreme selection with a single retained PC

- **GIVEN** `pca.n_components` selects exactly 1 PC
- **WHEN** `PCAAnalysisStep` executes with
  `feature_selection_strategy="extreme"`
- **THEN** `top_features.csv` and the `top_features` metadata list SHALL
  contain at most 2 entries (1 most-positive-loading, 1
  most-negative-loading), deduplicated to 1 entry if only a single feature
  is available

#### Scenario: PCAAnalysisStep resolves a top_variance variance-fraction threshold to a feature count

- **WHEN** `PCAAnalysisStep` executes with
  `feature_selection_strategy="top_variance"` and `pca.n_top_features` set
  to a value `< 1` (e.g. `0.8`)
- **THEN** the step SHALL call `select_n_features_by_variance()` with the
  PCA run's `feature_contributions` DataFrame and that threshold to resolve
  a concrete feature count before calling `select_top_features_from_pca()`
- **AND** the cumulative `fractional_contribution` of the selected features
  SHALL meet or exceed the configured threshold
- **AND** the cumulative `fractional_contribution` of the selected features
  minus the smallest selected feature's own `fractional_contribution` SHALL
  be less than the configured threshold (no unnecessary over-selection)

#### Scenario: PCAAnalysisStep resolves a non-positive top_variance threshold to a single feature

- **WHEN** `PCAAnalysisStep` executes with
  `feature_selection_strategy="top_variance"` and `pca.n_top_features` set
  to `0` or a negative value
- **THEN** the step SHALL select exactly 1 feature (the single highest
  `total_contribution` feature), without raising an exception

#### Scenario: PCAAnalysisStep treats a top_variance n_top_features of exactly 1.0 as a count, not a 100%-variance threshold

- **WHEN** `PCAAnalysisStep` executes with
  `feature_selection_strategy="top_variance"` and `pca.n_top_features` set
  to exactly `1.0`
- **THEN** the step SHALL select exactly 1 feature (the `>= 1` count
  branch), and SHALL NOT interpret `1.0` as "select enough features to
  reach 100% cumulative variance"

#### Scenario: PCAAnalysisStep preserves count-based top_variance behavior for n_top_features >= 1

- **WHEN** `PCAAnalysisStep` executes with
  `feature_selection_strategy="top_variance"` and `pca.n_top_features` set
  to a value `>= 1`
- **THEN** the step SHALL select exactly that many features by total
  variance contribution, identical to current behavior
