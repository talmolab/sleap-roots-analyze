# Proposal: Add Regression Plotting for Publication Figures

## Problem Statement

Users need to create publication-quality linear regression plots for analyzing relationships between root traits (e.g., root biomass vs. surface area, root vs. shoot biomass). Currently, the package provides scatter plots, PCA biplots, and interactive visualizations, but lacks a dedicated function for simple bivariate regression analysis with statistical annotations.

While seaborn is already a dependency and users could manually create regression plots in notebooks, providing a standardized function will:
1. Ensure consistent styling across publications
2. Reduce code duplication in analysis notebooks
3. Provide proper statistical annotations (R², p-value, regression equation)
4. Follow the package's existing visualization patterns

## Proposed Solution

Add a `create_regression_plot()` function to `visualization.py` that creates publication-ready linear regression plots with:
- Scatter points with optional genotype/group coloring
- Linear regression line with confidence interval
- Statistical annotations (Pearson R, R², p-value, regression equation)
- Publication-ready styling consistent with existing visualization functions
- Optional marginal distributions (via seaborn jointplot)

The implementation will be adapted from the existing `create_joint_plot()` function in the EDPIE_wheat_analysis codebase, simplified for single-experiment use cases.

## Success Criteria

- Function creates regression plots with R², p-value, and regression equation annotations
- Works with DataFrame input (consistent with existing visualization functions)
- Supports optional color-by grouping (e.g., by genotype)
- Returns matplotlib Figure for further customization and saving
- Includes comprehensive tests (edge cases, statistical validation, plot generation)
- Documentation includes examples for common use cases
- Integrates seamlessly into existing notebook workflows

## Impact

**Users Affected:**
- Plant biologists creating publication figures from root trait data
- Users performing correlation analysis between traits
- Anyone needing simple bivariate regression visualization

**Benefits:**
- Reduces notebook code complexity for common regression analysis
- Ensures statistical annotations are correct and consistent
- Publication-ready output without manual styling
- Easier reproducibility across experiments

**Risks:**
- Low risk: Purely additive feature, no breaking changes
- Function is optional and doesn't affect existing workflows

## Implementation Scope

**In Scope:**
- Add `create_regression_plot()` to `visualization.py`
- Support simple linear regression with Pearson correlation
- Statistical annotations (R, R², p-value, equation)
- Optional color-by grouping
- Comprehensive unit tests
- Documentation with examples

**Out of Scope:**
- Non-linear regression models (polynomial, exponential, etc.)
- Multiple regression (>2 variables)
- Robust regression methods (Theil-Sen, RANSAC)
- Interactive plotly version (can be added later if needed)
- Automated outlier removal in regression

## Dependencies

**Required:**
- scipy.stats (already a dependency) for `pearsonr()` and `linregress()`
- seaborn (already a dependency) for regression line with confidence intervals
- matplotlib (already a dependency) for plotting
- pandas (already a dependency) for DataFrame handling

**No new dependencies required.**

## Alternatives Considered

1. **Use seaborn directly in notebooks**
   - Pro: No new code needed
   - Con: Code duplication, inconsistent styling, manual statistical annotations
   - Decision: Rejected - function provides value through standardization

2. **Create interactive plotly version only**
   - Pro: Modern, interactive plots
   - Con: Doesn't integrate with existing static publication workflow
   - Decision: Deferred - start with static version, add interactive later if needed

3. **Add to cross_experiment_analysis.py instead**
   - Pro: Groups correlation functions together
   - Con: This is for single-experiment use cases, not cross-experiment
   - Decision: Rejected - belongs in visualization.py

## Open Questions

None - implementation is straightforward and well-defined.