## ADDED Requirements

### Requirement: Public Cross-Platform PC-Correlation Workflow Function

The package SHALL export a public function `cross_platform_pc_correlations` from the top-level
`sleap_roots_analyze` namespace that runs the complete per-platform PCA → cross-platform
PC-correlation workflow in a single call and returns a typed `CrossPlatformPCResult`. The function
SHALL accept a mapping of platform name → sample-level trait DataFrame, a mapping of platform name →
trait columns, a mapping of platform name → number of principal components, and the keyword
parameters `genotype_col` (default `"genotype"`), `alpha` (default `0.05`), `correction_method`
(default `"fdr_bh"`), and `random_state`. The function SHALL have complete type hints and a
Google-style docstring.

#### Scenario: Function and result type are importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import cross_platform_pc_correlations, CrossPlatformPCResult`
- **THEN** the import SHALL succeed
- **AND** `cross_platform_pc_correlations` SHALL be listed in `sleap_roots_analyze.__all__`
- **AND** `CrossPlatformPCResult` SHALL be listed in `sleap_roots_analyze.__all__`

#### Scenario: Runs on synthetic three-platform data

- **WHEN** the function is called with three synthetic platforms, their trait columns, and a
  per-platform component count
- **THEN** it SHALL return a `CrossPlatformPCResult` with one PCA result per platform, one
  correlation table per unordered platform pair, and a significant-correlations list

### Requirement: Sample-Level PCA Before Genotype Aggregation

For each platform the workflow SHALL fit PCA on the sample-level trait matrix and SHALL THEN
aggregate the resulting sample-level PC scores to genotype means; it SHALL NOT average traits to
genotype means before fitting PCA. Cross-platform correlations SHALL be computed on the
genotype-mean PC scores.

#### Scenario: Ordering is sample-PCA-then-aggregate, not average-then-PCA

- **WHEN** a platform contains multiple samples per genotype
- **THEN** PCA SHALL be fit on the sample-level rows
- **AND** the genotype-mean PC score for a genotype SHALL equal the mean of that genotype's
  sample-level PC scores

### Requirement: Cross-Platform PC Correlations With Pooled FDR Correction

For every unordered pair of platforms the workflow SHALL compute the correlation between each PC of
one platform and each PC of the other, over the genotypes common to all platforms (a single shared
panel). The number of tests for a pair SHALL equal the product of the two platforms' component
counts, and the total test family SHALL be the sum over all unordered pairs. Multiple-testing
correction SHALL be applied once across the entire pooled family using `correction_method`, and each
test SHALL carry its FDR-corrected p-value.

#### Scenario: Test count matches the component configuration

- **WHEN** the workflow runs on three platforms with component counts 3, 4, and 5
- **THEN** the total number of cross-platform PC tests SHALL be 47 (3×4 + 3×5 + 4×5)

#### Scenario: FDR correction is pooled across all pairs

- **WHEN** correction is applied
- **THEN** the corrected p-values SHALL be computed from the full pooled set of cross-platform PC
  tests, not per pair
- **AND** a test SHALL be flagged significant only if its corrected p-value is below `alpha`

### Requirement: Confidence Intervals and Power in the Result

The `CrossPlatformPCResult` SHALL include, for each cross-platform PC test, a Fisher z-transformed
confidence interval for the correlation and an achieved-power value, computed from the number of
common genotypes used for that test.

#### Scenario: Each tested pair carries a CI and power

- **WHEN** the result is produced
- **THEN** every cross-platform PC test row SHALL include a confidence-interval lower and upper bound
  and an achieved-power value

### Requirement: Genotype Alignment Across Platforms

The workflow SHALL align all platforms to a single shared panel of genotypes — those present in
every platform — and SHALL use that panel for all pairwise correlations, recording the panel size on
every result row. This yields a consistent genotype set (and a uniform `n`) across all platform
pairs.

#### Scenario: Disjoint genotypes yield no usable correlation

- **WHEN** the platforms share fewer than the minimum genotypes needed for a correlation
- **THEN** the workflow SHALL not raise
- **AND** the tests SHALL be reported with their common-genotype count so the caller can see why
  they were not evaluated

### Requirement: Wheat EDPIE Golden Reproduction

The workflow SHALL reproduce the wheat EDPIE paper's headline numbers given the post-QC per-platform
input tables and the paper's component configuration (Turface 19 → 3 PCs, Cylinder → 4 PCs, Field →
5 PCs): 47 cross-platform PC tests over 19 common genotypes, with 0 tests surviving FDR correction.
The regression test asserting these numbers SHALL be skipped when the post-QC fixture is not present
in the repository.

#### Scenario: Golden numbers reproduced when the fixture is available

- **WHEN** the wheat EDPIE post-QC fixture is present and the workflow is run with the paper's
  component configuration
- **THEN** the result SHALL report 47 cross-platform PC tests and 19 common genotypes
- **AND** the number of tests surviving FDR correction SHALL be 0

#### Scenario: Regression test is skipped without the fixture

- **WHEN** the wheat EDPIE post-QC fixture is absent
- **THEN** the golden regression test SHALL be skipped rather than fail
