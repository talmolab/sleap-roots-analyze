## 1. TDD - Write Tests First

- [x] 1.1 Add test for CSV containing both spearman_r/spearman_p and pearson_r/pearson_p columns
- [x] 1.2 Add test verifying `correlation_method` config still determines sort order
- [x] 1.3 Add test for create_joint_plot accepting both pearson and spearman pre-computed values
- [x] 1.4 Add regression test: visualization values exactly match CSV values for both metrics

## 2. Update Correlation Calculation Step

- [x] 2.1 Modify `CalculateCrossPlatformCorrelationsStep.execute()` to calculate both Pearson and Spearman
- [x] 2.2 Update CSV output columns: `spearman_r`, `spearman_p`, `pearson_r`, `pearson_p`
- [x] 2.3 Sort by `abs(primary_correlation)` based on `correlation_method` config
- [x] 2.4 Update metadata to include both correlation types

## 3. Update Visualization Functions

- [x] 3.1 Add `pearson_r` and `pearson_p` parameters to `create_joint_plot()`
- [x] 3.2 Update docstring to document all pre-computed parameters
- [x] 3.3 Remove inline Pearson recalculation when pre-computed values provided
- [x] 3.4 Update `VisualizeCrossPlatformStep` to pass both metrics from CSV

## 4. Integration and Verification

- [x] 4.1 Run cross-platform pipeline with real data
- [x] 4.2 Verify CSV contains all four correlation columns
- [x] 4.3 Verify joint plot annotations match CSV values exactly
- [x] 4.4 Verify sorting still works correctly based on correlation_method
- [x] 4.5 Run full test suite - all tests pass (1422 passed)
