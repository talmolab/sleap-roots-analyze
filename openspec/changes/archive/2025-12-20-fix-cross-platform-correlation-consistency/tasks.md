## 1. Write Failing Tests (TDD)

- [x] 1.1 Create test that verifies correlation values in joint plot match pre-computed CSV values
- [x] 1.2 Create test that verifies n_genotypes in joint plot matches pre-computed CSV values
- [x] 1.3 Create test with `min_samples_per_genotype` filter that would expose the bug
- [x] 1.4 Run tests to confirm they fail with current implementation

## 2. Implementation

- [x] 2.1 Add optional `correlation`, `p_value`, `n_genotypes` parameters to `create_joint_plot`
- [x] 2.2 Update `create_joint_plot` to use pre-computed values when provided
- [x] 2.3 Add optional `correlation_df` parameter to `create_scatter_plot_grid` (N/A - not used in pipeline)
- [x] 2.4 Update `create_scatter_plot_grid` to look up values from correlation_df when provided (N/A - not used in pipeline)
- [x] 2.5 Update `VisualizeCrossPlatformStep` to pass correlation values to `create_joint_plot`
- [x] 2.6 Update `VisualizeCrossPlatformStep` to pass correlation_df to `create_scatter_plot_grid` (N/A - not used in pipeline)

## 3. Verification

- [x] 3.1 Run failing tests to confirm they now pass
- [x] 3.2 Run full test suite to ensure no regressions (43 tests passed)
- [x] 3.3 Run debug script to verify CSV and image values match
- [x] 3.4 Run cross-platform pipeline and visually verify image annotations
      - CSV: correlation=-0.7420, p=0.000423, n=18
      - Image: Spearman rho=-0.742, p=0.000423, n=18 genotypes
      - VERIFIED: Values match exactly!

## 4. Documentation

- [x] 4.1 Update docstrings for modified functions
- [x] 4.2 Add inline comments explaining DRY principle for correlation values
