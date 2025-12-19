# Data Sanitization Specification

## ADDED Requirements

### Requirement: Depth Range Detection
The system SHALL detect depth suffixes in column names and extract numeric depth values.

#### Scenario: Detect standard depth suffix
- **GIVEN** a column name `"RootDW_15cm"`
- **WHEN** depth detection is performed
- **THEN** the system extracts depth value `15.0`

#### Scenario: Detect fractional depth
- **GIVEN** a column name `"RootCount_7.5cm"`
- **WHEN** depth detection is performed
- **THEN** the system extracts depth value `7.5`

#### Scenario: No depth suffix present
- **GIVEN** a column name `"Median.Number.of.Roots"`
- **WHEN** depth detection is performed
- **THEN** the system returns `None` (no depth detected)

#### Scenario: Depth with different unit is not detected
- **GIVEN** a column name `"RootDW_15mm"`
- **WHEN** depth detection using cm pattern is performed
- **THEN** the system returns `None` (only cm supported)

---

### Requirement: Depth Range Formatting
The system SHALL map numeric depth values to depth range strings using provided mappings.

#### Scenario: Map midpoint to depth range
- **GIVEN** depth value `15.0` and mapping `{15.0: "0-30", 45.0: "30-60"}`
- **WHEN** depth range formatting is applied
- **THEN** the system returns `"0-30cm"`

#### Scenario: Fallback to original depth when no mapping provided
- **GIVEN** depth value `15.0` and no mapping (`None`)
- **WHEN** depth range formatting is applied
- **THEN** the system returns `"15cm"` (original notation)

#### Scenario: Fallback for unmapped depth value
- **GIVEN** depth value `25.0` and mapping `{15.0: "0-30", 45.0: "30-60"}`
- **WHEN** depth range formatting is applied
- **THEN** the system returns `"25cm"` (depth not in mapping)

#### Scenario: Integer formatting for whole numbers
- **GIVEN** depth value `15.0` (whole number) and no mapping
- **WHEN** depth range formatting is applied
- **THEN** the system returns `"15cm"` (not `"15.0cm"`)

#### Scenario: Preserve fractional depths
- **GIVEN** depth value `7.5` and no mapping
- **WHEN** depth range formatting is applied
- **THEN** the system returns `"7.5cm"` (fraction preserved)

---

### Requirement: Biomass Column Sanitization
The system SHALL sanitize biomass column names with depth ranges to produce scientifically clear labels.

#### Scenario: Biomass column with depth range mapping
- **GIVEN** column name `"RootDW_15cm"` and depth mapping `{15.0: "0-30"}`
- **WHEN** sanitization is performed with `depth_range_mapping` parameter
- **THEN** the output column name is `"Root Biomass DW (g) 0-30cm"`

#### Scenario: Biomass column without depth range mapping
- **GIVEN** column name `"RootDW_15cm"` and no depth mapping
- **WHEN** sanitization is performed
- **THEN** the output column name is `"Rootdw 15Cm"` (standard sanitization)

#### Scenario: Second depth range
- **GIVEN** column name `"RootDW_45cm"` and depth mapping `{15.0: "0-30", 45.0: "30-60"}`
- **WHEN** sanitization is performed with `depth_range_mapping` parameter
- **THEN** the output column name is `"Root Biomass DW (g) 30-60cm"`

#### Scenario: Preserve unit notation for biomass
- **GIVEN** column name `"RootDW_15cm"` with depth mapping
- **WHEN** sanitization is performed
- **THEN** the output includes unit `"(g)"` for grams

---

### Requirement: Root Count Column Sanitization
The system SHALL sanitize root counting column names to preserve single-depth clarity.

#### Scenario: Root count at single depth
- **GIVEN** column name `"RootCount_5cm"`
- **WHEN** sanitization is performed (no depth mapping needed)
- **THEN** the output column name is `"Root Count 5cm"`

#### Scenario: Root count at zero depth
- **GIVEN** column name `"RootCount_0cm"`
- **WHEN** sanitization is performed
- **THEN** the output column name is `"Root Count 0cm"`

#### Scenario: Root count with depth range (optional)
- **GIVEN** column name `"RootCount_15cm"` and depth mapping `{15.0: "0-30"}`
- **WHEN** sanitization is performed with `depth_range_mapping` parameter
- **THEN** the output column name is `"Root Count 0-30cm"` (if ranges applicable to counting)

---

### Requirement: Depth Range Mapping Parameter
The system SHALL accept an optional `depth_range_mapping` parameter in `sanitize_trait_names()` function.

#### Scenario: Function signature includes depth mapping
- **GIVEN** the `sanitize_trait_names()` function
- **WHEN** function is called
- **THEN** it accepts parameter `depth_range_mapping: Optional[Dict[float, str]] = None`

#### Scenario: Depth mapping format validation
- **GIVEN** depth mapping `{15.0: "0-30", 45.0: "30-60"}`
- **WHEN** mapping is used
- **THEN** keys are numeric depths (float) and values are range strings (str)

#### Scenario: Backward compatibility without depth mapping
- **GIVEN** existing code calling `sanitize_trait_names(df, trait_cols)`
- **WHEN** function is called without `depth_range_mapping` parameter
- **THEN** the function works as before (no errors, default behavior)

---

### Requirement: Non-Biomass Column Preservation
The system SHALL NOT modify non-biomass columns when applying depth range sanitization.

#### Scenario: Standard trait column unchanged
- **GIVEN** column name `"Median.Number.of.Roots"` and depth mapping provided
- **WHEN** sanitization is performed
- **THEN** the output column name is `"Med Num Roots"` (standard processing, depth logic not applied)

#### Scenario: Metadata column unchanged
- **GIVEN** column name `"geno"` and depth mapping provided
- **WHEN** sanitization is performed with `genotype_col="geno"`
- **THEN** the output column name is `"Genotype"` (metadata sanitization only)

#### Scenario: Column with cm in name but not depth suffix
- **GIVEN** column name `"Total.Root.Length.cm"` (unit, not depth)
- **WHEN** sanitization is performed
- **THEN** the output is `"Total Root Length (cm)"` (unit conversion, not depth range)

---

### Requirement: Return Mapping with Depth Ranges
The system SHALL include depth range transformations in the returned name mapping dictionary.

#### Scenario: Mapping includes biomass depth range transformation
- **GIVEN** column `"RootDW_15cm"` with depth mapping `{15.0: "0-30"}`
- **WHEN** sanitization is performed with `return_mapping=True`
- **THEN** mapping contains `{"RootDW_15cm": "Root Biomass DW (g) 0-30cm"}`

#### Scenario: Mapping tracks original to sanitized names
- **GIVEN** multiple biomass columns with depth ranges
- **WHEN** sanitization is performed with `return_mapping=True`
- **THEN** mapping includes all transformations for reproducibility

---

### Requirement: Test Coverage for Depth Range Logic
The system SHALL have comprehensive test coverage (>95%) for depth range detection and formatting.

#### Scenario: Unit tests for depth detection helper
- **GIVEN** the `_detect_depth_suffix()` helper function
- **WHEN** tests are run
- **THEN** all edge cases are covered (standard, fractional, missing, invalid)

#### Scenario: Unit tests for depth range formatting
- **GIVEN** the `_format_depth_range()` helper function
- **WHEN** tests are run
- **THEN** all mapping scenarios are covered (mapped, unmapped, no mapping, fractional)

#### Scenario: Integration tests for biomass sanitization
- **GIVEN** the full `sanitize_trait_names()` function with depth mapping
- **WHEN** integration tests are run
- **THEN** end-to-end sanitization produces correct labels

#### Scenario: Regression tests for backward compatibility
- **GIVEN** all existing tests for `sanitize_trait_names()`
- **WHEN** tests are run after depth range changes
- **THEN** zero test failures (no regressions)