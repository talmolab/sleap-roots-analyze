# QC Pipeline: Root Core Processing

## ADDED Requirements

### Requirement: Per-Core Value Outlier Detection

The QC pipeline MUST detect and remove individual cores with anomalous values before aggregation to prevent measurement errors from contaminating plot-level statistics and destroying heritability.

#### Scenario: Detect core with extreme deviation from median

**GIVEN** a plot with 3 cores for GH_7371 Rep 1 at depth 0-30cm:
- Core 0: 0.7636 g
- Core 1: 0.7071 g
- Core 2: 0.3132 g (56% below median)

**AND** config parameter `max_deviation_from_median: 0.30` (30% threshold)

**WHEN** Step 00c (QCCoreLevelStep) executes with `detect_value_outliers: true`

**THEN** the system must:
1. Calculate median = 0.7071 g
2. Calculate percent deviations:
   - Core 0: 8% (NOT flagged)
   - Core 1: 0% (NOT flagged)
   - Core 2: 56% (FLAGGED as outlier)
3. Remove core 2 from the group
4. Record metadata: `{"core_id": "5_1_2", "reason": "value_deviation_0.56", "value": 0.3132, "median": 0.7071}`

**AND** downstream aggregation (Step 00d) must compute mean([0.7636, 0.7071]) = 0.735 g

#### Scenario: Normal variation not flagged

**GIVEN** a plot with 3 cores showing natural biological variation:
- Core 0: 0.72 g
- Core 1: 0.68 g
- Core 2: 0.75 g

**AND** config parameter `max_deviation_from_median: 0.30`

**WHEN** Step 00c executes value outlier detection

**THEN** the system must:
1. Calculate median = 0.72 g
2. Calculate percent deviations: [6%, 6%, 4%]
3. NOT flag any cores (all < 30%)
4. Keep all 3 cores for aggregation

#### Scenario: Safety keeps minimum cores when all flagged

**GIVEN** a plot with 3 cores all differing significantly:
- Core 0: 0.1 g
- Core 1: 0.5 g
- Core 2: 0.9 g

**AND** config parameters:
- `max_deviation_from_median: 0.20` (strict)
- `min_cores_after_qc: 1`

**WHEN** Step 00c detects that 2+ cores would be flagged

**THEN** the system must:
1. Calculate median = 0.5 g
2. Identify that cores 0 and 2 have >20% deviation
3. Apply safety: Keep at least 1 core (the one closest to median)
4. Keep only core 1 (0% deviation from median)
5. NOT remove all cores (prevents empty groups)

#### Scenario: Skip detection when median is zero

**GIVEN** a group with median value = 0 (e.g., root counts at deep depth)

**WHEN** Step 00c attempts percent deviation calculation

**THEN** the system must:
1. Detect median == 0
2. Skip percent deviation (would cause division by zero)
3. NOT flag any cores in this group
4. Log: "Skipped value QC for group (median=0)"

#### Scenario: Skip detection with insufficient cores

**GIVEN** a group with only 1 core remaining after missing data QC

**WHEN** Step 00c attempts value outlier detection

**THEN** the system must:
1. Detect len(values) < 2
2. Skip value outlier detection (need ≥2 for median comparison)
3. Keep the single core
4. Proceed to aggregation

### Requirement: Configurable Detection Parameters

Users MUST be able to configure detection sensitivity via threshold parameters to balance false positive vs false negative rates.

#### Scenario: Conservative threshold (default)

**GIVEN** config parameter `max_deviation_from_median: 0.30`

**WHEN** Step 00c executes

**THEN** the system must only flag cores with >30% deviation from median

**AND** must NOT flag cores with 20-29% deviation

#### Scenario: Aggressive threshold

**GIVEN** config parameter `max_deviation_from_median: 0.15`

**WHEN** Step 00c executes

**THEN** the system must flag cores with >15% deviation

**AND** must catch more outliers but may remove natural variation

#### Scenario: Disable value outlier detection

**GIVEN** config parameter `detect_value_outliers: false`

**WHEN** Step 00c executes

**THEN** the system must:
1. Skip value outlier detection entirely
2. Only perform missing data detection (existing behavior)
3. Preserve backward compatibility with old configs

### Requirement: Metadata Tracking

The system MUST record detailed metadata about flagged cores for diagnostic purposes and threshold tuning.

#### Scenario: Metadata includes flagged core details

**GIVEN** Step 00c flags 2 cores as value outliers

**WHEN** metadata is written to `00c_core_qc_metadata.json`

**THEN** the file must contain:
```json
{
  "sources": [
    {
      "data_type": "biomass",
      "total_cores": 180,
      "flagged_cores": 5,
      "flagged_by_method": {
        "missing_data": 2,
        "value_outlier": 3
      },
      "flagged_cores_list": [
        {
          "core_id": "5_1_2",
          "plot": 5,
          "rep": 1,
          "geno": "GH_7371",
          "depth_cm": 15.0,
          "value": 0.3132,
          "median": 0.7071,
          "deviation_pct": 0.56,
          "reason": "value_deviation",
          "threshold": 0.30
        }
      ]
    }
  ]
}
```

#### Scenario: Metadata enables threshold tuning

**GIVEN** user wants to assess if 30% threshold is appropriate

**WHEN** user inspects `00c_core_qc_metadata.json`

**THEN** the user must be able to:
1. See total number of cores flagged
2. See deviation percentages for each flagged core
3. Determine if threshold should be adjusted (increase if too many false positives, decrease if outliers remain)

### Requirement: Per-Group Detection

Value outlier detection MUST be performed independently for each group (Plot, Rep, Genotype, Depth) to avoid cross-contamination.

#### Scenario: Independent detection across plots

**GIVEN** two plots with different value ranges:
- Plot 1: cores = [0.3, 0.32, 0.35] (low values)
- Plot 2: cores = [0.7, 0.72, 0.75] (high values)

**WHEN** Step 00c performs value outlier detection

**THEN** the system must:
1. Analyze Plot 1 independently (median=0.32, deviations ~6-9%)
2. Analyze Plot 2 independently (median=0.72, deviations ~3-4%)
3. NOT flag any cores (both plots show normal within-plot variation)
4. NOT compare values across plots (scale differences are expected)

#### Scenario: Independent detection across depths

**GIVEN** same genotype at two depths:
- Depth 15cm: cores = [0.7, 0.71, 0.72] (high biomass)
- Depth 45cm: cores = [0.09, 0.095, 0.1] (low biomass)

**WHEN** Step 00c performs value outlier detection

**THEN** the system must:
1. Analyze each depth independently
2. NOT compare 15cm values to 45cm values
3. Apply percent deviation within each depth group

### Requirement: Integration with Existing QC

Value outlier detection MUST integrate seamlessly with existing missing data detection without breaking current functionality.

#### Scenario: Combined detection (missing data + value outliers)

**GIVEN** a group with 3 cores:
- Core 0: 0.75 g (normal)
- Core 1: NaN (missing value)
- Core 2: 0.2 g (extreme value outlier)

**AND** config:
```yaml
core_qc:
  enabled: true
  max_missing_proportion: 0.5  # Missing data detection
  detect_value_outliers: true  # Value detection
  max_deviation_from_median: 0.30
```

**WHEN** Step 00c executes

**THEN** the system must:
1. Flag core 1 as "missing_data" (has NaN)
2. Calculate median from remaining cores [0.75, 0.2] = 0.475
3. Calculate deviations and flag core 2 as "value_outlier" (58% deviation)
4. Remove both cores
5. Keep only core 0 for aggregation

#### Scenario: Backward compatibility with disabled core QC

**GIVEN** config `core_qc.enabled: false`

**WHEN** pipeline executes

**THEN** the system must:
1. Skip Step 00c entirely (existing behavior)
2. NOT perform missing data or value outlier detection
3. Use all cores for aggregation
4. Match old pipeline behavior exactly

## MODIFIED Requirements

None (this is a new capability for root core processing)

## REMOVED Requirements

None
