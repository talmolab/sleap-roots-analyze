## MODIFIED Requirements

### Requirement: Calculate Cross-Platform Correlations

The system SHALL calculate pairwise trait correlations between experiments through `CalculateCrossPlatformCorrelationsStep` with the following behavior:

- Calculate genotype means for each trait in both experiments using `calculate_genotype_means()`
- Compute **both Pearson and Spearman** correlations for all trait pairs, regardless of which method is configured as primary
- Remove NaN pairs before correlation calculation
- Calculate p-values for each correlation (both Pearson and Spearman)
- Apply FDR correction to p-values based on `fdr_correction_method` config:
  - `fdr_bh`: Benjamini-Hochberg correction (assumes test independence)
  - `fdr_by`: Benjamini-Yekutieli correction (valid under arbitrary dependence)
  - `none`: No correction applied (adjusted p-values equal raw p-values)
- **Handle NaN p-values gracefully**: If any correlation produces NaN p-values (e.g., from constant-valued traits or insufficient data), the FDR correction SHALL:
  - Filter out NaN p-values before applying correction
  - Apply correction only to valid p-values
  - Preserve NaN in adjusted p-value columns for invalid correlations
  - Set `significant_fdr` to False for rows with NaN adjusted p-values
  - Log a warning indicating the count of NaN p-values encountered
- Store results with columns: `exp1_trait`, `exp2_trait`, `spearman_r`, `spearman_p`, `pearson_r`, `pearson_p`, `n_genotypes`, `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
- Sort results by absolute value of the **primary** correlation (determined by `correlation_method` config), descending
- Export results to `cross_platform_correlations.csv` in output directory

#### Scenario: NaN p-values from constant trait do not corrupt FDR correction

- **WHEN** one trait pair produces NaN p-values (e.g., constant values in one trait)
- **AND** other trait pairs have valid p-values
- **THEN** FDR correction is applied only to valid p-values
- **AND** NaN p-values remain NaN in the adjusted columns
- **AND** `significant_fdr` is False for rows with NaN adjusted p-values
- **AND** valid correlations receive correct FDR-adjusted p-values

#### Scenario: Single correlation (m=1)

- **WHEN** only one trait pair is tested (m=1)
- **THEN** no FDR correction is applied (single test, no multiple testing)
- **AND** adjusted p-values equal raw p-values
- **AND** `significant_fdr` is based on raw p-value comparison

#### Scenario: Fewer than 3 genotypes for a trait pair

- **WHEN** after removing NaN pairs, fewer than 3 valid genotype pairs remain
- **THEN** step sets all correlation values to NaN for that trait pair
- **AND** adjusted p-values are NaN for that trait pair
- **AND** `significant_fdr` is False for that trait pair
- **AND** logs warning about insufficient data

#### Scenario: Dual correlation calculation with Spearman primary

- **WHEN** correlation method is "spearman" with 18 valid genotypes and 50 trait pairs
- **THEN** step calculates BOTH Spearman and Pearson correlations for all 50 pairs
- **AND** exports CSV with columns: exp1_trait, exp2_trait, spearman_r, spearman_p, pearson_r, pearson_p, n_genotypes, spearman_p_adjusted, pearson_p_adjusted, significant_fdr
- **AND** sorts results by absolute Spearman correlation (descending)

#### Scenario: FDR correction with Benjamini-Hochberg

- **WHEN** fdr_correction_method is "fdr_bh" and 1000 trait pairs are tested
- **THEN** spearman_p_adjusted and pearson_p_adjusted contain BH-corrected p-values
- **AND** adjusted p-values are >= raw p-values
- **AND** significant_fdr is True when primary adjusted p < significance_level

#### Scenario: FDR correction with Benjamini-Yekutieli (default)

- **WHEN** fdr_correction_method is "fdr_by" (default) and traits are correlated
- **THEN** BY correction is applied (valid under arbitrary dependence)
- **AND** BY produces more conservative (larger) adjusted p-values than BH

#### Scenario: No FDR correction

- **WHEN** fdr_correction_method is "none"
- **THEN** spearman_p_adjusted equals spearman_p
- **AND** pearson_p_adjusted equals pearson_p
- **AND** significant_fdr uses raw p-values for threshold comparison

## ADDED Requirements

### Requirement: Edge Case Documentation

The system SHALL document edge case behavior in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Explanation of what produces NaN p-values:
  - Constant-valued traits (zero variance)
  - Fewer than 3 valid genotype pairs after NaN removal
- How NaN p-values are handled during FDR correction
- Why `significant_fdr` is False for NaN adjusted p-values
- Minimum sample size requirements (n >= 3) for correlation testing

#### Scenario: User understands NaN p-value behavior

- **WHEN** user sees NaN in adjusted p-value columns
- **THEN** documentation explains this occurs when:
  - A trait has constant values across all genotypes (zero variance)
  - Fewer than 3 genotypes have valid data for both traits
- **AND** documentation confirms this is expected behavior, not a bug
- **AND** documentation explains that `significant_fdr` is False for these rows
