## ADDED Requirements

### Requirement: Public PC-Level Cross-Platform Correlation Workflow

The package SHALL export a public function `cross_platform_pc_correlations` from
the top-level `sleap_roots_analyze` namespace that runs the complete PC-level
cross-platform correlation DAG in a single call and returns a structured `dict`.
The function SHALL accept `pipeline_run` (path to a pipeline-run directory),
`platform_config` (mapping of platform name → `{pca_dir, final_data,
genotype_col, n_pcs}`), `output_dir`, and the keyword parameters `fdr_methods`
(default `["fdr_by", "fdr_bh", "bonferroni"]` when `None`), `primary_fdr_method`
(default `"fdr_by"`), `alpha` (default `0.05`), `confidence_level` (default
`0.95`), `fdr_scope` (default `"both"`), and `make_figures` (default `True`).
The function SHALL have complete type hints and a Google-style docstring and
SHALL NOT hard-code paper-specific platform paths.

#### Scenario: Function is importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import cross_platform_pc_correlations`
- **THEN** the import SHALL succeed
- **AND** `cross_platform_pc_correlations` SHALL be listed in `sleap_roots_analyze.__all__`

#### Scenario: Runs the DAG from a pipeline-run directory

- **WHEN** the function is called with a pipeline-run directory containing
  per-platform PCA outputs (`pc_scores.csv`) and QC `final_data` plus a
  `platform_config` for three platforms
- **THEN** it SHALL load per-platform sample PC scores and genotype labels,
  aggregate sample PC scores to genotype means, align on common genotypes,
  compute every-PC-by-every-PC correlations per platform pair, apply FDR
  correction, and return a `dict` exposing the correlation table, the summary,
  and the genotype-mean PC scores

### Requirement: Sample-Level PCA Before Genotype Aggregation

The workflow SHALL aggregate the pipeline's **sample-level** PC scores to
genotype means and SHALL correlate on those genotype-mean PC scores; it SHALL
NOT average traits to genotype means before PCA. Because the pipeline computes
sample-level PCA upstream, the workflow consumes `pc_scores.csv` as-is and only
averages per genotype after loading.

#### Scenario: Genotype-mean PC score equals the mean of sample PC scores

- **WHEN** a platform has multiple samples per genotype
- **THEN** the genotype-mean PC score for a genotype SHALL equal the mean of
  that genotype's sample-level PC scores

### Requirement: Cross-Platform PC Correlations With Multi-Scope FDR

For every unordered pair of platforms the workflow SHALL compute the correlation
between each retained PC of one platform and each retained PC of the other, over
the genotypes common to all platforms. The number of tests for a pair SHALL
equal the product of the two platforms' retained PC counts. Multiple-testing
correction SHALL be available at two scopes: `combined` (one correction across
all pairs' tests) and `per_pair` (correction within each platform pair), with
`both` emitting both column families.

#### Scenario: Test count matches the retained-PC configuration

- **WHEN** the workflow runs on three platforms retaining 3, 4, and 5 PCs
- **THEN** the total number of cross-platform PC tests SHALL be 47 (3×4 + 3×5 + 4×5)

#### Scenario: Combined and per-pair FDR produce distinct columns

- **WHEN** `fdr_scope="both"` and `fdr_methods` includes `fdr_by`
- **THEN** the correlation table SHALL contain both `significant_combined_fdr_by`
  and `significant_per_pair_fdr_by` columns
- **AND** the combined correction SHALL be computed from the full pooled set of
  tests, not per pair

### Requirement: Confidence Intervals and Power Reuse Existing Helpers

For each cross-platform PC test the workflow SHALL include a Fisher
z-transformed confidence interval and an achieved-power value, computed using
the existing `calculate_correlation_ci` and `achieved_power` functions from the
cross-experiment analysis module rather than re-implementing them. The summary
SHALL include the minimum detectable correlation at 80% power via the existing
`minimum_detectable_correlation` function.

#### Scenario: Every test carries a CI and power, reusing shared statistics

- **WHEN** the result is produced
- **THEN** every cross-platform PC test row SHALL include confidence-interval
  bounds and an achieved-power value
- **AND** the CI/power values SHALL come from the shared cross-experiment
  statistics functions

### Requirement: Reproducible Workflow Artifacts

The workflow SHALL write a reproducible set of artifacts under `output_dir`:
`correlations.csv` (all tests with both FDR scopes), `significant_combined.csv`,
`significant_per_pair.csv`, per-platform `genotype_means_*.csv`, a
`metadata.json` recording parameters and headline results, and — when
`make_figures=True` — combined/per-pair correlation figures and a sensitivity
figure. The function SHALL also return the in-memory equivalents so the workflow
is testable without reading files back.

#### Scenario: Artifacts and return value are both produced

- **WHEN** the workflow completes with `make_figures=False`
- **THEN** `correlations.csv` and `metadata.json` SHALL exist under `output_dir`
- **AND** the returned `dict` SHALL expose the correlation table and summary
  whose test count matches `correlations.csv`

### Requirement: Wheat EDPIE PC-Correlation Golden Reproduction

The workflow SHALL reproduce the wheat EDPIE paper's headline numbers given the
paper's post-QC inputs and PC configuration (Turface 3 PCs, Cylinder 4 PCs,
Field 5 PCs): 47 cross-platform PC tests over 19 common genotypes, with 0 tests
surviving combined `fdr_by` correction. The regression test asserting these
numbers SHALL be skipped when the post-QC fixture is absent.

#### Scenario: Golden numbers reproduced when the fixture is available

- **WHEN** the wheat EDPIE post-QC fixture is present and the workflow runs with
  the paper configuration
- **THEN** the result SHALL report 47 PC tests, 19 common genotypes, and 0
  combined-`fdr_by`-significant tests

#### Scenario: Regression is skipped without the fixture

- **WHEN** the fixture is absent
- **THEN** the golden regression test SHALL be skipped rather than fail
