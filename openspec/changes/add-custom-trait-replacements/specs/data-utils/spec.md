# Data Utils Specification

## ADDED Requirements

### Requirement: Custom Trait Name Replacements
The `sanitize_trait_names()` function SHALL support custom user-defined term replacements that are applied before standard sanitization transformations.

Custom replacements enable domain-specific terminology changes (e.g., "crown" → "seminal" for wheat roots) while preserving all standard sanitization features (unit conversion, abbreviations, title case).

#### Scenario: Basic custom replacement
- **GIVEN** a DataFrame with trait column "crown_length_mm"
- **WHEN** `sanitize_trait_names()` is called with `custom_replacements={"crown": "seminal"}`
- **THEN** the column is renamed to "Seminal Length (mm)"
- **AND** standard sanitization (units, title case) is still applied

#### Scenario: Case-insensitive matching
- **GIVEN** a DataFrame with trait columns "Crown.Length", "CROWN.Width", "crown_angle"
- **WHEN** `sanitize_trait_names()` is called with `custom_replacements={"crown": "seminal"}`
- **THEN** all three columns have "crown" replaced with "seminal" regardless of case
- **AND** results are "Seminal Length", "Seminal Width", "Seminal Angle (°)"

#### Scenario: Multiple custom replacements
- **GIVEN** a DataFrame with traits "crown.length", "primary.root.count", "lateral.number"
- **WHEN** `sanitize_trait_names()` is called with `custom_replacements={"crown": "seminal", "primary": "main", "lateral": "branch"}`
- **THEN** all three replacements are applied: "Seminal Length", "Main Root Count", "Branch Num"

#### Scenario: Custom replacements with abbreviations
- **GIVEN** a DataFrame with trait "crown.maximum.length.mm"
- **WHEN** `sanitize_trait_names()` is called with `custom_replacements={"crown": "seminal"}` and `abbreviate=True`
- **THEN** the result is "Seminal Max Length (mm)"
- **AND** both custom replacement AND abbreviation are applied

#### Scenario: Empty custom replacements preserves existing behavior
- **GIVEN** a DataFrame with trait "crown.length.mm"
- **WHEN** `sanitize_trait_names()` is called with `custom_replacements=None` or `custom_replacements={}`
- **THEN** the column is renamed to "Crown Length (mm)" using only standard sanitization
- **AND** no custom replacements are applied

#### Scenario: Custom replacements do not affect metadata columns
- **GIVEN** a DataFrame with genotype column "geno" and trait column "crown.length"
- **WHEN** `sanitize_trait_names()` is called with `custom_replacements={"crown": "seminal"}`, `genotype_col="geno"`
- **THEN** "geno" is renamed to "Genotype" (metadata sanitization)
- **AND** "crown.length" is renamed to "Seminal Length" (custom + standard sanitization)
- **AND** custom replacements only apply to trait columns, not metadata

### Requirement: Trait Name Sanitization
The `sanitize_trait_names()` function SHALL transform trait column names from technical formats to human-readable visualization names.

The function processes trait names through a pipeline:
1. Apply custom user-defined replacements (if provided)
2. Convert unit suffixes to parenthetical format with proper symbols
3. Split by dots, hyphens, underscores
4. Remove filler words ("of", "the")
5. Apply abbreviations (if enabled)
6. Apply title case
7. Fix unit capitalization

Metadata columns can optionally be sanitized separately for consistency across plots.

#### Scenario: Unit conversion with symbols
- **GIVEN** a DataFrame with trait columns "root_length_mm", "root_area_mm2", "root_volume_mm3", "root_angle_deg"
- **WHEN** `sanitize_trait_names()` is called with these trait columns
- **THEN** columns are renamed to "Root Length (mm)", "Root Area (mm²)", "Root Volume (mm³)", "Root Angle (°)"

#### Scenario: Abbreviation mode enabled
- **GIVEN** a DataFrame with trait "Median.Number.of.Roots"
- **WHEN** `sanitize_trait_names()` is called with `abbreviate=True`
- **THEN** the result is "Med Num Roots"
- **AND** "Median" → "Med", "Number" → "Num", and "of" is removed

#### Scenario: Abbreviation mode disabled
- **GIVEN** a DataFrame with trait "Median.Number.of.Roots"
- **WHEN** `sanitize_trait_names()` is called with `abbreviate=False`
- **THEN** the result is "Median Number Roots"
- **AND** words are not abbreviated but filler words are still removed

#### Scenario: Title case applied to all names
- **GIVEN** a DataFrame with trait "root_LENGTH_mm"
- **WHEN** `sanitize_trait_names()` is called
- **THEN** the result is "Root Length (mm)"
- **AND** inconsistent capitalization is normalized to title case

#### Scenario: Metadata column sanitization
- **GIVEN** a DataFrame with columns "geno", "rep", "barcode"
- **WHEN** `sanitize_trait_names()` is called with `sanitize_metadata=True`, `genotype_col="geno"`, `replicate_col="rep"`, `barcode_col="barcode"`
- **THEN** columns are renamed to "Genotype", "Replicate", "Barcode"

#### Scenario: Return mapping dictionary
- **GIVEN** a DataFrame with traits to sanitize
- **WHEN** `sanitize_trait_names()` is called with `return_mapping=True`
- **THEN** the function returns a tuple (DataFrame, dict)
- **AND** the dict maps old names to new names for all changed columns

#### Scenario: Handle gram and milligram units
- **GIVEN** a DataFrame with trait columns "root_weight_g", "seed_mass_mg"
- **WHEN** `sanitize_trait_names()` is called
- **THEN** columns are renamed to "Root Weight (g)", "Seed Mass (mg)"
