# Add Heritability Diagnostics Proposal

## Why

Currently, when traits have low or zero heritability (H² = 0.00), users have no built-in tools to understand **why**. This is problematic because:

1. **Lack of explanatory power**: Users see H² = 0.00 but don't know if it's due to high within-genotype variance, low between-genotype variance, insufficient data, or model failure
2. **Adjacent depth paradox**: Traits at similar depths (e.g., `c_50_60_1` vs `c_50_60_2`) can have vastly different heritabilities (0.00 vs 0.698), suggesting data quality or biological issues that need investigation
3. **No variance visualization**: Users can see heritability values but cannot visualize the underlying variance components (σ²_G vs σ²_E) that determine heritability
4. **Limited debugging**: When heritability filtering removes traits, users cannot easily diagnose whether removal was appropriate or if data cleaning could help

This proposal adds diagnostic functions and visualizations to help users understand the variance structure underlying heritability estimates, identify problematic traits, and make informed decisions about trait filtering.

## What Changes

### New Diagnostic Analysis Functions (statistics.py)
- `analyze_trait_variance()` - Decompose variance into between-genotype and within-genotype components
- `diagnose_heritability_issues()` - Identify specific causes of low/zero heritability with explanations
- `compare_trait_heritabilities()` - Side-by-side comparison of multiple traits with variance metrics

### New Diagnostic Visualization Functions (visualization.py)
- `create_variance_decomposition_plot()` - Multi-panel plot showing genetic vs environmental variance across traits
- `create_trait_by_genotype_boxplots()` - Boxplots showing trait distribution by genotype with H² annotations
- `create_heritability_diagnostic_dashboard()` - Comprehensive 4-panel diagnostic figure

### Enhanced Pipeline Integration (optional)
- Add `generate_diagnostics: bool` flag to `FilterHeritabilityStep`
- Export diagnostic plots and CSV summaries when enabled
- Store diagnostic results in pipeline metadata

### Comprehensive Test Coverage
- Test fixtures for diagnostic scenarios (zero variance, high within-variance, etc.)
- Edge case tests (single genotype, missing data, model failures)
- Visualization output validation tests
- Integration tests with existing heritability pipeline

## Impact

### Affected Specs
- **statistics-analysis** (new capability) - Add diagnostic functions for variance decomposition and heritability explanation
- **visualization** (modified) - Add diagnostic plotting functions for heritability analysis
- **pipeline-qc** (modified) - Enhance FilterHeritabilityStep with optional diagnostic output

### Affected Code
- `src/sleap_roots_analyze/statistics.py` - Add 3 new diagnostic functions (~150 lines)
- `src/sleap_roots_analyze/visualization.py` - Add 3 new plotting functions (~200 lines)
- `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py` - Add optional diagnostic mode (~50 lines)
- `tests/test_statistics.py` - Add diagnostic function tests (~200 lines)
- `tests/test_visualization.py` - Add diagnostic plot tests (~150 lines)
- `tests/fixtures.py` - Add diagnostic test fixtures (~100 lines)

### User Benefits
1. **Transparency**: Understand why traits have low heritability
2. **Data quality**: Identify measurement issues vs true biological variation
3. **Informed decisions**: Better trait filtering based on variance structure
4. **Debugging**: Quickly diagnose heritability calculation issues
5. **Publication**: Diagnostic plots suitable for supplementary materials

### Breaking Changes
None - all functionality is additive and opt-in.

### Migration Required
None - existing code continues to work unchanged.

## Technical Approach

### Design Principles
1. **Reuse existing infrastructure**: Use existing heritability calculation results, don't recalculate
2. **Consistent API**: Match existing function signatures and naming conventions from statistics.py
3. **Test-Driven Development**: Write tests before implementation for each function
4. **Modular design**: Each function has single responsibility and can be used independently
5. **Minimal dependencies**: Use existing matplotlib/seaborn patterns, no new visualization libraries

### Key Design Decisions
1. **Variance analysis separate from calculation**: Diagnostic functions accept pre-calculated heritability results to avoid redundant computation
2. **Return structured data**: Functions return dicts/DataFrames that can be easily serialized or further analyzed
3. **Composable visualizations**: Individual plot functions that can be combined into dashboards
4. **Optional pipeline integration**: Diagnostics are opt-in to avoid slowing down standard QC workflows
