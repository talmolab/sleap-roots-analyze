# Cross-Platform Analysis Pipeline - Technical Design

## Context

### Background

Researchers conducting root phenotyping experiments across multiple platforms (e.g., cylinder growth systems, turface pots, field trials) need to validate trait consistency and understand platform-specific effects. Currently, this analysis is performed ad-hoc using Jupyter notebooks with hardcoded paths and manual parameter tuning.

### Constraints

- Must integrate with existing pipeline infrastructure (NetworkX DAG, OmegaConf configuration)
- Must leverage existing `cross_experiment_analysis.py` functions to avoid duplication
- Must support reproducible outputs with timestamped directories
- Must handle edge cases: missing data, no common genotypes, insufficient samples
- Must provide publication-quality visualizations matching notebook standards

### Stakeholders

- **Plant biologists**: Need automated, reproducible cross-platform validation
- **Data analysts**: Need flexible correlation methods and comprehensive output
- **Pipeline maintainers**: Need consistent step patterns and clear error messages

## Goals / Non-Goals

### Goals

1. **Automate cross-platform correlation analysis** - Replace manual notebook workflows with configurable pipeline
2. **Support multiple correlation methods** - Spearman, Pearson, Kendall via configuration
3. **Generate publication-ready visualizations** - 4-panel summaries, joint plots, boxplots
4. **Ensure reproducibility** - Configuration-driven, structured outputs, comprehensive logging
5. **Maintain code quality** - TDD approach, >90% test coverage, consistent with project patterns

### Non-Goals

1. **Advanced statistical methods** - Not implementing partial correlations, regression, or causal inference
2. **Multi-platform comparisons** - Not supporting >2 platforms in single analysis (future work)
3. **Interactive web interface** - Pipeline is CLI/config driven, not building web UI
4. **Real-time analysis** - Batch processing only, no streaming or incremental updates
5. **Automated platform selection** - User must specify which platforms to compare

## Decisions

### Decision 1: Three Separate Pipeline Steps

**Choice**: Implement as three distinct steps (Load → Calculate → Visualize) rather than single monolithic step

**Rationale**:
- **Modularity**: Users can run only data loading to check genotype alignment before expensive correlation computation
- **Debugging**: Easier to isolate failures (e.g., visualization bug doesn't require re-running correlations)
- **Extensibility**: Future work can add additional analysis steps between calculation and visualization
- **Consistency**: Matches existing QC and Viz pipeline patterns (multiple focused steps vs monoliths)

**Alternatives Considered**:
- Single combined step: Simpler but less flexible, harder to debug, violates single-responsibility principle
- Two steps (Load+Calculate, Visualize): Middle ground but arbitrary boundary, doesn't provide enough granularity

### Decision 2: Leverage Existing `cross_experiment_analysis.py` Functions

**Choice**: Use existing functions (`load_and_align_experiments`, `calculate_genotype_means`, `create_joint_plot`, `create_genotype_boxplots`) rather than reimplementing

**Rationale**:
- **DRY principle**: These functions already tested and working in notebooks
- **Consistency**: Ensures pipeline produces identical results to manual notebook analysis
- **Maintenance**: Single source of truth, bug fixes benefit both notebooks and pipeline
- **Time efficiency**: Focus effort on pipeline integration, not re-solving solved problems

**Alternatives Considered**:
- Reimplement from scratch: More control but high duplication risk, inconsistent results
- Refactor existing functions: Unnecessary, current API already suitable for pipeline use

### Decision 3: Support Three Correlation Methods

**Choice**: Implement Spearman, Pearson, and Kendall correlation methods via configuration parameter

**Rationale**:
- **Statistical flexibility**: Different methods appropriate for different data characteristics
  - Spearman: Non-linear monotonic relationships, most commonly used in plant biology
  - Pearson: Linear relationships with normal distributions, traditional choice
  - Kendall: More robust for small samples, handles ties better
- **Low implementation cost**: All three methods available in scipy.stats with identical API
- **Research needs**: Existing notebooks demonstrate users already using multiple methods

**Alternatives Considered**:
- Spearman only: Simpler but limits statistical options unnecessarily
- Add more methods (partial correlation, etc.): Over-engineering, no demonstrated need

### Decision 4: Configuration-Driven Rather Than Programmatic API

**Choice**: Primary interface is YAML configuration files, not Python API

**Rationale**:
- **Reproducibility**: Configuration files are version-controlled artifacts documenting analysis parameters
- **Consistency**: Matches existing QC and Viz pipeline patterns
- **Accessibility**: Non-programmers can modify configs without touching code
- **Batch processing**: Easy to create multiple configs for different platform comparisons

**Alternatives Considered**:
- Python API first: More flexible but harder to track analysis provenance
- Both equally supported: Maintenance burden, potential API inconsistencies

### Decision 5: Export Intermediate Results as CSV

**Choice**: Export full correlation results DataFrame to CSV, not just summary statistics

**Rationale**:
- **Transparency**: Users can inspect all correlations, not just top N
- **Post-hoc analysis**: Users can filter/sort differently without re-running pipeline
- **Validation**: Easy to compare pipeline output against notebook results
- **Standard format**: CSV readable by Excel, R, Python for downstream analysis

**Alternatives Considered**:
- JSON only: Less accessible to non-programmers
- Database: Over-engineering for typical dataset sizes (<10K correlations)
- Summary stats only: Loses information, users would request full export anyway

### Decision 6: TDD Implementation Approach

**Choice**: Write tests before implementation for all components (test-first)

**Rationale**:
- **Specification clarity**: Tests document expected behavior unambiguously
- **Regression prevention**: Comprehensive test suite prevents future breakage
- **Design feedback**: Writing tests first reveals API issues early
- **Project standards**: Consistent with project's 95% coverage goals and quality standards

**Alternatives Considered**:
- Test after implementation: Faster initially but risks incomplete coverage and poor API design
- No comprehensive tests: Unacceptable for production pipeline code

### Decision 7: Unified Helper Function for Correlation Methods

**Choice**: Implement `calculate_correlations(x, y, method)` helper function wrapping scipy methods

**Rationale**:
- **Consistency**: Single interface ensures uniform handling of edge cases across methods
- **Error handling**: Centralized NaN handling, insufficient data detection, etc.
- **Testability**: One function to test thoroughly rather than scattered logic
- **Maintainability**: Future correlation methods added in one place

**Alternatives Considered**:
- Direct scipy calls in step: More duplicated code, harder to test, inconsistent error handling
- Class-based strategy pattern: Over-engineering for three simple function calls

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cross-Platform Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 1: LoadCrossPlatformDataStep                         │  │
│  │ ┌────────────────────────────────────────────────────┐   │  │
│  │ │ Input: exp1_path, exp2_path, genotype columns      │   │  │
│  │ │ Uses: load_and_align_experiments()                 │   │  │
│  │ │       get_trait_columns()                          │   │  │
│  │ │ Output: aligned DataFrames, common genotypes       │   │  │
│  │ └────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 2: CalculateCrossPlatformCorrelationsStep           │  │
│  │ ┌────────────────────────────────────────────────────┐   │  │
│  │ │ Input: aligned DataFrames, correlation method      │   │  │
│  │ │ Uses: calculate_genotype_means()                   │   │  │
│  │ │       calculate_correlations()                     │   │  │
│  │ │ Output: correlation DataFrame → CSV                │   │  │
│  │ └────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 3: VisualizeCrossPlatformStep                       │  │
│  │ ┌────────────────────────────────────────────────────┐   │  │
│  │ │ Input: correlations, aligned DataFrames            │   │  │
│  │ │ Uses: create_correlation_summary_plot()            │   │  │
│  │ │       create_joint_plot()                          │   │  │
│  │ │       create_genotype_boxplots()                   │   │  │
│  │ │ Output: PNG figures → figures/                     │   │  │
│  │ └────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

Supporting Modules:
┌─────────────────────────────────────────────────────────────────┐
│ cross_experiment_analysis.py                                     │
│ - load_and_align_experiments()                                   │
│ - calculate_genotype_means()                                     │
│ - calculate_correlations()          [NEW]                        │
│ - create_correlation_summary_plot() [NEW]                        │
│ - create_joint_plot()                                            │
│ - create_genotype_boxplots()                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ data_cleanup.py                                                  │
│ - get_trait_columns()                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Configuration (YAML)
     │
     ├─► CrossPlatformConfig
     │        │
     │        ▼
     │   LoadCrossPlatformDataStep
     │        │
     │        ├─► Load exp1.csv
     │        ├─► Load exp2.csv
     │        ├─► Align by genotypes
     │        ├─► Extract trait columns
     │        │
     │        ▼
     │   Metadata: {
     │     "exp1_df": DataFrame,
     │     "exp2_df": DataFrame,
     │     "common_genotypes": List[str],
     │     "exp1_traits": List[str],
     │     "exp2_traits": List[str]
     │   }
     │        │
     │        ▼
     │   CalculateCrossPlatformCorrelationsStep
     │        │
     │        ├─► Calculate genotype means (exp1)
     │        ├─► Calculate genotype means (exp2)
     │        ├─► For each trait pair:
     │        │     ├─► Remove NaN pairs
     │        │     ├─► Calculate correlation
     │        │     └─► Calculate p-value
     │        │
     │        ├─► Sort by absolute correlation
     │        ├─► Export to CSV
     │        │
     │        ▼
     │   Metadata: {
     │     ...(previous),
     │     "correlation_df": DataFrame,
     │     "exp1_means": DataFrame,
     │     "exp2_means": DataFrame
     │   }
     │        │
     │        ▼
     │   VisualizeCrossPlatformStep
     │        │
     │        ├─► Generate 4-panel summary
     │        ├─► Generate joint plots (top N)
     │        ├─► Generate boxplots (top N)
     │        └─► Save all figures
     │
     ▼
Output Directory:
  - cross_platform_correlations.csv
  - summary.json
  - figures/
      - correlation_summary.png
      - joint_01_*.png
      - boxplot_01.png
      - ...
```

## Risks / Trade-offs

### Risk 1: Large Memory Usage with Many Traits

**Risk**: Calculating correlations for 100+ traits per experiment = 10,000+ trait pairs = large memory footprint

**Likelihood**: Medium (some experiments have 50-100 traits)

**Impact**: High (pipeline crashes or slows significantly)

**Mitigation**:
- Process correlations in chunks rather than all at once
- Add memory usage logging/warnings
- Document recommended trait counts in configuration template
- Consider adding max_trait_pairs parameter for safety

### Risk 2: Long Execution Time for Visualization

**Risk**: Generating 20+ joint plots and boxplots can take minutes, frustrating users

**Likelihood**: High (existing notebooks show this pattern)

**Impact**: Medium (annoying but not blocking)

**Mitigation**:
- Add progress logging ("Generating joint plot 3/6...")
- Make top_n parameters configurable so users can reduce if needed
- Consider parallel figure generation in future (not in v1)

### Risk 3: No Common Genotypes Between Platforms

**Risk**: Users may select datasets with no overlapping genotypes, causing pipeline failure

**Likelihood**: Medium (happens when comparing different studies)

**Impact**: High (complete pipeline failure)

**Mitigation**:
- Clear error message listing available genotypes from each experiment
- LoadCrossPlatformDataStep fails fast before expensive operations
- Consider adding genotype validation to configuration schema

### Risk 4: Correlation Method Misunderstanding

**Risk**: Users may not understand when to use Spearman vs Pearson vs Kendall

**Likelihood**: High (statistical expertise varies)

**Impact**: Low (wrong method selected but pipeline runs)

**Mitigation**:
- Comprehensive inline documentation in template config
- Add guidance in CLAUDE.md about method selection
- Default to Spearman (most robust, commonly used in biology)

### Risk 5: Inconsistency with Notebook Results

**Risk**: Pipeline may produce different results than manual notebooks due to implementation differences

**Likelihood**: Medium (happens during development)

**Impact**: High (undermines trust in pipeline)

**Mitigation**:
- TDD approach with test fixtures derived from notebooks
- Manual validation against notebook results in task 11.2
- Use exact same functions from cross_experiment_analysis.py

## Migration Plan

### For Existing Notebook Users

1. **Phase 1: Parallel Running** (Week 1-2)
   - Run both notebook and pipeline on same data
   - Verify identical results
   - Document any differences

2. **Phase 2: Template Creation** (Week 2)
   - Create configs matching existing notebook analyses
   - Test with historical data

3. **Phase 3: Migration** (Week 3+)
   - Replace notebook workflows with pipeline configs
   - Archive notebooks as reference
   - Update lab protocols

### Rollback Plan

If critical bugs discovered:
- Notebooks remain functional, users can revert
- Pipeline changes isolated to new files (no breaking changes to existing modules)
- Easy to disable new pipeline steps via configuration

## Open Questions

1. **Should we support asymmetric analysis?** (e.g., correlate 10 traits from exp1 with ALL traits from exp2 rather than all-vs-all)
   - **Status**: Deferred to future work
   - **Decision**: Start with all-vs-all, add filtering if needed

2. **Should we implement multiple testing correction?** (e.g., Bonferroni, FDR)
   - **Status**: Deferred to future work
   - **Rationale**: Users typically interested in top correlations regardless of significance, can add p-value adjustment post-hoc

3. **Should correlation results include trait metadata?** (e.g., trait descriptions, units)
   - **Status**: Deferred to future work
   - **Rationale**: Trait metadata not consistently available in current datasets, would require schema extension

4. **Should we support weighted correlations by replicate count?**
   - **Status**: Deferred to future work
   - **Rationale**: Current implementation uses genotype means (already accounts for replicates), adding weights would complicate interpretation

## Testing Strategy

### Unit Tests
- Configuration validation (invalid inputs, edge cases)
- Helper functions (all three correlation methods, NaN handling)
- Individual step execution (mocked dependencies)

### Integration Tests
- Complete pipeline execution (load → calculate → visualize)
- Metadata passing between steps
- Output directory structure validation

### Validation Tests
- Results match notebook outputs on same data
- Edge cases: no common genotypes, insufficient samples, missing data
- Performance tests: large dataset handling (1000+ samples, 100+ traits)

### Coverage Goals
- >95% for configuration and helper functions
- >90% for pipeline steps
- >85% overall for new code
