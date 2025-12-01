# Statistics Analysis - Heritability Diagnostics

## ADDED Requirements

### Requirement: Trait Variance Decomposition Analysis
The system SHALL provide a function `analyze_trait_variance()` that decomposes trait variance into between-genotype and within-genotype components to support heritability diagnostics.

#### Scenario: Successful variance analysis for normal trait
- **GIVEN** a DataFrame with trait data, genotypes, and replicates
- **AND** a valid trait column name
- **WHEN** `analyze_trait_variance()` is called
- **THEN** return a dictionary containing:
  - `n_observations`: Total number of valid observations
  - `n_genotypes`: Number of unique genotypes
  - `mean_reps_per_geno`: Average replicates per genotype
  - `overall_variance`: Total variance of trait values
  - `between_genotype_variance`: Variance of genotype means
  - `within_genotype_variance`: Mean variance within genotypes
  - `pct_variance_between_geno`: Percentage of variance between genotypes
  - `trait_mean`, `trait_std`, `trait_cv`: Basic trait statistics

#### Scenario: Variance analysis with zero variance trait
- **GIVEN** a DataFrame where all trait values are identical
- **WHEN** `analyze_trait_variance()` is called
- **THEN** return variance components with `overall_variance = 0.0`
- **AND** `between_genotype_variance = 0.0`
- **AND** `within_genotype_variance = 0.0`

#### Scenario: Variance analysis with missing data
- **GIVEN** a DataFrame with NaN values in trait column
- **WHEN** `analyze_trait_variance()` is called
- **THEN** automatically exclude NaN values before analysis
- **AND** return `n_observations` reflecting only valid data points

#### Scenario: Variance analysis with insufficient data
- **GIVEN** a DataFrame with fewer than 3 observations after removing NaNs
- **WHEN** `analyze_trait_variance()` is called
- **THEN** return error indicator in result dictionary
- **AND** include message explaining insufficient data

### Requirement: Heritability Issue Diagnosis
The system SHALL provide a function `diagnose_heritability_issues()` that identifies specific causes of low or zero heritability with actionable explanations.

#### Scenario: Diagnose trait with zero heritability due to no between-genotype variance
- **GIVEN** a trait with heritability = 0.0
- **AND** variance analysis showing `between_genotype_variance = 0.0`
- **WHEN** `diagnose_heritability_issues()` is called
- **THEN** return diagnosis dictionary containing:
  - `has_issues: True`
  - `issues`: List including "No variation between genotype means"
  - `severity`: "critical" for this issue
  - `recommendations`: Suggestions like "Check if trait is constant across genotypes"

#### Scenario: Diagnose trait with zero heritability due to high within-genotype variance
- **GIVEN** a trait with heritability = 0.0
- **AND** variance analysis showing `within_genotype_variance >> between_genotype_variance`
- **WHEN** `diagnose_heritability_issues()` is called
- **THEN** include issue "Within-genotype variation >> between-genotype variation"
- **AND** provide ratio of within/between variance
- **AND** recommend checking for measurement noise or environmental factors

#### Scenario: Diagnose trait with low sample size
- **GIVEN** a trait with `n_observations < 30`
- **WHEN** `diagnose_heritability_issues()` is called
- **THEN** include issue with severity "warning"
- **AND** note "Low sample size may reduce heritability estimate reliability"

#### Scenario: Diagnose trait with good heritability
- **GIVEN** a trait with heritability > 0.5
- **AND** sufficient sample size and balanced design
- **WHEN** `diagnose_heritability_issues()` is called
- **THEN** return `has_issues: False`
- **AND** empty issues list

#### Scenario: Diagnose trait with mixed model failure
- **GIVEN** heritability result containing `"error"` key
- **WHEN** `diagnose_heritability_issues()` is called
- **THEN** identify model failure as critical issue
- **AND** include error message from heritability result
- **AND** recommend checking data quality or trying ANOVA-based method

### Requirement: Multi-Trait Heritability Comparison
The system SHALL provide a function `compare_trait_heritabilities()` that generates side-by-side comparison of variance components and heritability metrics for multiple traits.

#### Scenario: Compare multiple traits with varying heritability
- **GIVEN** a DataFrame with multiple trait columns
- **AND** heritability results for those traits
- **WHEN** `compare_trait_heritabilities()` is called with trait list
- **THEN** return pandas DataFrame with one row per trait
- **AND** columns including: `trait`, `heritability`, `var_genetic`, `var_residual`, `between_geno_var`, `within_geno_var`, `pct_var_between`, `n_observations`, `n_genotypes`, `mean_reps_per_geno`

#### Scenario: Compare traits with sorting by heritability
- **GIVEN** comparison DataFrame for multiple traits
- **WHEN** `compare_trait_heritabilities()` is called with `sort_by="heritability"`
- **THEN** return DataFrame sorted in ascending order by H² values
- **AND** traits with lowest heritability appear first

#### Scenario: Compare traits with error handling
- **GIVEN** a trait with error in heritability results
- **WHEN** `compare_trait_heritabilities()` is called
- **THEN** include trait in comparison DataFrame
- **AND** fill numeric columns with NaN for that trait
- **AND** note error in a separate column or metadata

#### Scenario: Export comparison to CSV
- **GIVEN** comparison DataFrame from `compare_trait_heritabilities()`
- **WHEN** optional `output_path` parameter is provided
- **THEN** save DataFrame to CSV at specified path
- **AND** return the DataFrame for further analysis

### Requirement: Diagnostic Function Integration with Existing Infrastructure
The diagnostic functions SHALL integrate seamlessly with existing heritability calculation results and data structures.

#### Scenario: Use existing heritability results without recalculation
- **GIVEN** pre-calculated heritability results from `calculate_heritability_estimates()`
- **WHEN** diagnostic functions are called
- **THEN** extract variance components from existing results
- **AND** do not recalculate heritability values

#### Scenario: Compatible with pipeline metadata structure
- **GIVEN** pipeline metadata containing `heritability_results` dictionary
- **WHEN** diagnostic functions are called with this metadata
- **THEN** successfully extract and analyze heritability results
- **AND** return diagnostic results in compatible format for pipeline storage

#### Scenario: Handle both mixed model and ANOVA-based results
- **GIVEN** heritability results from mixed model approach
- **OR** heritability results from ANOVA-based approach
- **WHEN** diagnostic functions are called
- **THEN** correctly process variance components from either method
- **AND** note which method was used in diagnostic output
