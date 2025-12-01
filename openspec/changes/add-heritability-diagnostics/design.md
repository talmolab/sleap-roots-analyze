# Heritability Diagnostics Design Document

## Context

Users analyzing root trait heritability frequently encounter traits with zero or unexpectedly low heritability values. Without diagnostic tools, they cannot determine whether this is due to:
- True lack of genetic variation
- High measurement/environmental noise
- Data quality issues
- Insufficient sample sizes
- Model convergence problems

This creates a "black box" problem where users must either accept filtering decisions blindly or manually investigate using ad-hoc notebooks. The diagnostic scripts created for troubleshooting revealed the need for integrated diagnostic capabilities.

**Constraints:**
- Must integrate with existing heritability calculation infrastructure
- Cannot break existing pipeline workflows
- Must follow TDD approach with >95% test coverage
- Should reuse existing visualization patterns

**Stakeholders:**
- Plant biologists analyzing root phenotyping data
- Researchers investigating genetic vs environmental contributions
- Pipeline users needing reproducible QC diagnostics

## Goals / Non-Goals

**Goals:**
1. Provide transparent diagnostic functions for heritability analysis
2. Enable users to understand variance structure (genetic vs environmental)
3. Identify data quality issues affecting heritability estimates
4. Integrate diagnostics into existing QC pipeline as opt-in feature
5. Generate publication-quality diagnostic visualizations

**Non-Goals:**
1. Modify existing heritability calculation algorithm
2. Add new statistical methods beyond existing mixed model/ANOVA approaches
3. Create interactive web-based diagnostic tools (keep to static matplotlib)
4. Automatically fix data quality issues (diagnostics only identify problems)
5. Support real-time or streaming heritability analysis

## Decisions

### Decision 1: Diagnostic Functions Accept Pre-Calculated Results
**What:** Diagnostic functions take heritability results as input rather than raw data and recalculate.

**Why:**
- Avoids redundant computation (heritability calculation is expensive with mixed models)
- Maintains single source of truth for heritability values
- Allows diagnostic analysis of previously calculated results without re-running full analysis
- Consistent with existing pattern where pipeline steps pass metadata forward

**Alternatives Considered:**
- **Calculate heritability within diagnostic functions:** Rejected due to redundancy and inconsistency risk
- **Require raw data and always recalculate:** Rejected due to performance impact

### Decision 2: Three-Level API (Functions → Visualizations → Dashboard)
**What:** Provide three composable layers:
1. **Analysis functions** (`analyze_trait_variance`, `diagnose_heritability_issues`, `compare_trait_heritabilities`)
2. **Individual plot functions** (`create_variance_decomposition_plot`, `create_trait_by_genotype_boxplots`)
3. **Dashboard function** (`create_heritability_diagnostic_dashboard`)

**Why:**
- Modularity: Users can call individual functions for specific needs
- Composability: Advanced users can combine functions in custom ways
- Testability: Each level can be tested independently
- Progressive complexity: Simple needs use simple functions, complex needs use dashboard

**Alternatives Considered:**
- **Single monolithic diagnostic function:** Rejected as inflexible and hard to test
- **Only provide dashboard:** Rejected as too limiting for programmatic use

### Decision 3: Pipeline Integration via Optional Flag
**What:** Add `generate_diagnostics: bool = False` to HeritabilityConfig rather than creating separate diagnostic step.

**Why:**
- Simpler configuration (single boolean vs additional step configuration)
- Diagnostics logically belong with filtering decision
- Reduces pipeline complexity (no additional edges in DAG)
- Backward compatible (defaults to False)

**Alternatives Considered:**
- **Separate DiagnosticStep after FilterHeritabilityStep:** Rejected as adds complexity, diagnostics only useful when filtering occurs
- **Always generate diagnostics:** Rejected as adds overhead and file clutter for users who don't need them

### Decision 4: Return Structured Dictionaries (Not Dataclasses)
**What:** Analysis functions return `Dict[str, Any]` rather than custom dataclasses.

**Why:**
- Consistency with existing `calculate_heritability_estimates()` return format
- Easy serialization to JSON/YAML for pipeline metadata
- No new types to document and maintain
- Flexible for future additions without breaking changes

**Alternatives Considered:**
- **Create VarianceAnalysisResult dataclass:** Rejected due to added complexity and learning curve
- **Return tuples:** Rejected as less self-documenting than dictionaries

### Decision 5: Percentage Variance Between Genotypes as Key Metric
**What:** Prominently report `pct_variance_between_geno` (what percentage of total trait variance is explained by genotype differences).

**Why:**
- Intuitive metric: 80% between-genotype = strong genetic control, 10% = weak genetic control
- Complements heritability estimate which includes replicate structure
- Helps identify scenarios where low H² is due to high within-replicate noise vs truly low genetic variation
- Easy to visualize and compare across traits

**Alternatives Considered:**
- **Only report raw variance values:** Rejected as harder to interpret magnitude
- **Use intraclass correlation:** Rejected as less intuitive for non-statisticians

### Decision 6: Issue Diagnosis with Severity Levels
**What:** `diagnose_heritability_issues()` categorizes issues as "critical", "warning", or "info".

**Why:**
- Helps users prioritize which traits need investigation
- "Critical" issues (e.g., zero variance) indicate clear data problems
- "Warning" issues (e.g., low sample size) indicate caution needed but not necessarily problematic
- "Info" issues provide context without alarm

**Severity Levels:**
- **Critical:** Zero variance, model failure, no data
- **Warning:** Low sample size (<30), imbalanced design (>2x variation in replicate counts), high within-variance (>10x between)
- **Info:** Moderate within-variance (3-10x between), moderate sample size (30-50)

## Risks / Trade-offs

### Risk 1: Diagnostic Overhead in Pipeline Mode
**Risk:** Generating diagnostics (especially plots) could slow down pipelines significantly.

**Mitigation:**
- Default to `generate_diagnostics=False`
- Limit boxplots to top 10 traits if >10 removed
- Generate diagnostics only for removed traits (not all traits)
- Allow diagnostic generation to fail without breaking pipeline

**Trade-off:** Users wanting diagnostics pay performance cost, but opt-in nature keeps default pipeline fast.

### Risk 2: Over-Interpretation of Diagnostics
**Risk:** Users might misinterpret diagnostic output and make poor filtering decisions.

**Mitigation:**
- Clear documentation emphasizing diagnostics are exploratory
- Include recommendations in diagnosis but note they're suggestions not rules
- Warn that low sample sizes reduce diagnostic reliability
- Provide examples of common scenarios and interpretations

**Trade-off:** More explanation text needed in docstrings and documentation.

### Risk 3: Matplotlib Compatibility Issues
**Risk:** Different matplotlib versions or backends could cause plot generation failures.

**Mitigation:**
- Use only stable matplotlib APIs (no experimental features)
- Test with matplotlib version range specified in pyproject.toml
- Wrap plot generation in try-except in pipeline mode
- Follow existing visualization.py patterns proven to work

**Trade-off:** Cannot use cutting-edge matplotlib features.

### Risk 4: Test Maintenance Burden
**Risk:** 60+ new tests (fixtures, analysis, visualization) add maintenance overhead.

**Mitigation:**
- Reuse existing test fixtures where possible
- Use parametrized tests to reduce code duplication
- Clear test names that explain what's being tested
- Group related tests in classes

**Trade-off:** More tests to run and maintain, but better confidence in functionality.

## Migration Plan

### Phase 1: Core Analysis Functions (Days 1-2)
1. Add test fixtures for diagnostic scenarios
2. Implement and test `analyze_trait_variance()`
3. Implement and test `diagnose_heritability_issues()`
4. Implement and test `compare_trait_heritabilities()`

**Validation:** All `test_statistics.py` tests pass, coverage >95%

### Phase 2: Visualization Functions (Days 3-4)
1. Implement and test `create_variance_decomposition_plot()`
2. Implement and test `create_trait_by_genotype_boxplots()`
3. Implement and test `create_heritability_diagnostic_dashboard()`

**Validation:** All `test_visualization.py` tests pass, plots render correctly

### Phase 3: Pipeline Integration (Day 5)
1. Extend HeritabilityConfig with `generate_diagnostics` field
2. Modify FilterHeritabilityStep to support diagnostic mode
3. Add integration tests for pipeline diagnostic mode

**Validation:** Existing pipeline tests pass unchanged, new diagnostic tests pass

### Phase 4: Documentation and Cleanup (Day 6)
1. Add docstring examples to all new functions
2. Create example notebook demonstrating diagnostic workflow
3. Update CLAUDE.md with diagnostic function overview
4. Remove temporary diagnostic scripts

**Validation:** All docstrings pass pydocstyle, example notebook runs successfully

### Rollback Plan
If critical issues discovered:
1. Diagnostic functions are isolated - can be removed without affecting core functionality
2. Pipeline flag defaults to False - can be disabled in configs
3. Git revert possible since changes are additive (no modifications to existing functions)

## Open Questions

### Q1: Should diagnostics include trait-trait correlation analysis?
**Status:** Deferred to future work

**Rationale:** Correlation analysis useful but scope creep. Can add later if needed without breaking changes.

### Q2: Should we provide text-based diagnostic reports?
**Status:** No, only plots and CSV

**Rationale:** Plots and CSV provide flexibility for users to interpret. Text reports risk being too prescriptive.

### Q3: Should diagnostic mode analyze retained traits or only removed traits?
**Status:** Analyze all traits in comparison, but visualize only problematic ones

**Rationale:** Full comparison provides context, but visualizing 50+ boxplots is overwhelming. Focus plots on traits of concern.

### Q4: Should we add confidence intervals to heritability estimates?
**Status:** Deferred to future work

**Rationale:** Mixed models can provide confidence intervals but adds complexity. Bootstrap confidence intervals computationally expensive. Can add in future enhancement.