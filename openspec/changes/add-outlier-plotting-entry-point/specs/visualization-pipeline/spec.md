## ADDED Requirements

### Requirement: Shared Per-Method Outlier-Figure Selection

`VisualizeOutliersStep` SHALL obtain its `mahalanobis` and `isolation_forest` per-method figures via
the same shared per-method selection helper that the public `plot_outlier_analysis` entry point uses
(`_select_outlier_figures`), so the "which `create_*` figures for this method" mapping has a single
source of truth. The step SHALL pass its **already-computed** `outlier_results[method]` into the
helper — it SHALL NOT re-detect and SHALL NOT call `plot_outlier_analysis` — thereby preserving the
pipeline's configured detector parameters. This change SHALL NOT alter the step's rendered output:
the figures it writes, their filenames, and their count SHALL be unchanged, and the step SHALL keep
drawing its `pca`, `kmeans`, `gmm`, and `hierarchical` method figures, its multi-method comparison
figures (including the `Outlier Method Comparison Summary` bar chart), and its cross-method
per-genotype figure (which passes the full multi-method `outlier_results`, distinct from the entry
point's single-method per-genotype figure and therefore not part of the shared helper).

#### Scenario: Step output is byte-identical after delegating selection

- **WHEN** the QC pipeline runs `VisualizeOutliersStep` after this change on a fixture exercising a
  single method (`mahalanobis`) and on one exercising multiple methods (`mahalanobis` +
  `isolation_forest`), each with and without a genotype column
- **THEN** the set of figure filenames and the figure count SHALL be identical to the pre-change
  behavior for each case
- **AND** the step SHALL obtain the `mahalanobis` / `isolation_forest` figures via
  `_select_outlier_figures` using its pre-computed `outlier_results`, without re-running detection

#### Scenario: Existing step and detection suites are unaffected

- **WHEN** the existing pipeline-step and outlier-detection test suites run after this change
- **THEN** their outcomes SHALL be unchanged
