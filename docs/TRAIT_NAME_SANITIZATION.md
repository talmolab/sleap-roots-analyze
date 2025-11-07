# Trait Name Sanitization Guide

This guide explains how to use the `sanitize_trait_names()` function to improve trait name readability in visualizations.

## Overview

The `sanitize_trait_names()` function transforms trait names like `Median.Number.of.Roots` into more readable formats like `Med Num Roots` for better visualization.

**New in this version:** Also automatically sanitizes metadata column names for consistency:
- `geno` → `Genotype`
- `rep` → `Replicate`  
- `barcode` → `Barcode`

## Usage in Notebooks

Add this code immediately after loading data and getting trait columns:

```python
from sleap_roots_analyze.data_utils import sanitize_trait_names

# Load the data
df_traits = load_trait_data(
    csv_path=trait_csv_path.as_posix(),
    barcode_col=BARCODE_COL,
    genotype_col=GENOTYPE_COL,
    replicate_col=REPLICATE_COL,
)

# Get initial trait columns
trait_cols = get_trait_columns(
    df=df_traits,
    barcode_col=BARCODE_COL,
    genotype_col=GENOTYPE_COL,
    replicate_col=REPLICATE_COL,
    additional_exclude=ADDITIONAL_EXCLUDE_COLS,
)

# **SANITIZE TRAIT NAMES AND METADATA COLUMNS FOR BETTER VISUALIZATION**
df_traits, trait_name_mapping = sanitize_trait_names(
    df=df_traits,
    trait_cols=trait_cols,
    abbreviate=True,  # Use abbreviations (Med, Avg, Max, etc.)
    return_mapping=True,  # Get mapping of old -> new names
    sanitize_metadata=True,  # Also sanitize metadata columns (default)
    genotype_col=GENOTYPE_COL,  # Rename geno -> Genotype
    replicate_col=REPLICATE_COL,  # Rename rep -> Replicate
    barcode_col=BARCODE_COL  # Ensure Barcode is title case
)

# **UPDATE COLUMN REFERENCES TO USE SANITIZED NAMES**
GENOTYPE_COL = "Genotype"  # Was "geno"
REPLICATE_COL = "Replicate"  # Was "rep"
BARCODE_COL = "Barcode"

# Update trait columns with new sanitized names
trait_cols = get_trait_columns(
    df=df_traits,
    barcode_col=BARCODE_COL,
    genotype_col=GENOTYPE_COL,
    replicate_col=REPLICATE_COL,
    additional_exclude=ADDITIONAL_EXCLUDE_COLS,
)

# Save trait name mapping for reference
mapping_path = RUN_DIR / "00_trait_name_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(trait_name_mapping, f, indent=2)
print(f"✅ Saved trait name mapping to: {mapping_path.name}")

# Store original data (with sanitized names)
df_traits_original = df_traits.copy()

print(f"✅ Data loaded and trait names sanitized!")
print(f"  - Samples: {len(df_traits):,}")
print(f"  - Trait columns: {len(trait_cols)}")
print(f"  - Genotypes: {df_traits[GENOTYPE_COL].nunique()}")
```

## Example Transformations

### With Abbreviations (`abbreviate=True`)

| Original | Sanitized |
|----------|-----------|
| `Median.Number.of.Roots` | `Med Num Roots` |
| `Maximum.Number.of.Roots` | `Max Num Roots` |
| `Total.Root.Length.mm` | `Total Root Length (mm)` |
| `Average.Hole.Size.mm2` | `Avg Hole Size (mm²)` |
| `Volume.mm3` | `Volume (mm³)` |
| `Average.Root.Orientation.deg` | `Avg Root Orient (°)` |
| `Width-to-Depth.Ratio` | `Width To Depth Ratio` |
| `root_g` | `Root (g)` |
| `shoot_g` | `Shoot (g)` |
| `biomass_mg` | `Biomass (mg)` |
| `root_shoot_ratio` | `Root Shoot Ratio` |

### Without Abbreviations (`abbreviate=False`)

| Original | Sanitized |
|----------|-----------|
| `Median.Number.of.Roots` | `Median Number Roots` |
| `Maximum.Number.of.Roots` | `Maximum Number Roots` |
| `Total.Root.Length.mm` | `Total Root Length (mm)` |
| `Average.Hole.Size.mm2` | `Average Hole Size (mm²)` |

## Transformations Applied

1. **Replace dots and hyphens with spaces**
   - `Median.Number.of.Roots` → `Median Number of Roots`
   - `Width-to-Depth.Ratio` → `Width to Depth Ratio`

2. **Remove filler words** (`of`, `the`)
   - `Median Number of Roots` → `Median Number Roots`

3. **Convert units to parenthetical format**
   - `.mm` → ` (mm)`
   - `.mm2` → ` (mm²)`
   - `.mm3` → ` (mm³)`
   - `.deg` → ` (°)`
   - `_g` → ` (g)`
   - `_mg` → ` (mg)`

4. **Apply title case** for consistent capitalization
   - `root_shoot_ratio` → `Root Shoot Ratio`
   - `root_g` → `Root (g)`
   - Units remain lowercase: `(g)`, `(mm)`, not `(G)`, `(Mm)`

5. **Optional abbreviations** (when `abbreviate=True`)
   - `Number` → `Num`
   - `Average` → `Avg`
   - `Maximum` → `Max`
   - `Minimum` → `Min`
   - `Median` → `Med`
   - `Frequency` → `Freq`
   - `Orientation` → `Orient`

6. **Custom term replacements** (when `custom_replacements` provided)
   - Domain-specific terminology changes applied BEFORE standard sanitization
   - Case-insensitive matching: `crown`, `Crown`, `CROWN` all match
   - Example: `crown` → `seminal` for wheat root terminology

7. **Metadata column sanitization** (when `sanitize_metadata=True`, default)
   - `geno` → `Genotype`
   - `genotype` → `Genotype`
   - `rep` → `Replicate`
   - `replicate` → `Replicate`
   - `barcode` → `Barcode`

## Custom Term Replacements

For domain-specific terminology changes, use the `custom_replacements` parameter. This is particularly useful for:
- Standardizing terminology across experiments (e.g., "crown" → "seminal" for wheat roots)
- Changing technical terms to more common names
- Harmonizing trait names from different data sources

**Example: Wheat crown → seminal root conversion**
```python
df_traits, mapping = sanitize_trait_names(
    df=df_traits,
    trait_cols=trait_cols,
    custom_replacements={"crown": "seminal"},  # Replace "crown" with "seminal"
    abbreviate=True,
    return_mapping=True
)
# "crown_length_mm" → "Seminal Length (mm)"
# "crown.angle.deg" → "Seminal Angle (°)"
# "Crown.Root.Count" → "Seminal Root Count"
```

**Example: Multiple custom replacements**
```python
df_traits = sanitize_trait_names(
    df=df_traits,
    trait_cols=trait_cols,
    custom_replacements={
        "crown": "seminal",
        "primary": "main",
        "lateral": "branch"
    },
    abbreviate=True
)
# "crown.length.mm" → "Seminal Length (mm)"
# "primary.root.count" → "Main Root Count"
# "lateral.number.of.roots" → "Branch Num Roots"
```

**Key Features:**
- **Case-insensitive**: Matches `crown`, `Crown`, `CROWN`, etc.
- **Applied first**: Custom replacements happen before abbreviations and unit conversions
- **Works with all features**: Combines with abbreviations, units, and metadata sanitization
- **Metadata safe**: Only affects trait columns, not metadata columns

## Metadata Column Sanitization

By default, the function also sanitizes common metadata column names for consistent display across all plots:

| Original | Sanitized |
|----------|-----------|
| `geno` | `Genotype` |
| `genotype` | `Genotype` |
| `rep` | `Replicate` |
| `replicate` | `Replicate` |
| `barcode` | `Barcode` |

This ensures that plots display "Genotype" instead of "geno", creating a more professional appearance.

**To disable metadata sanitization:**
```python
df_traits = sanitize_trait_names(
    df=df_traits,
    trait_cols=trait_cols,
    sanitize_metadata=False  # Keep metadata columns unchanged
)
```

**After sanitization, update your column references:**
```python
# Update column name constants
GENOTYPE_COL = "Genotype"  # Was "geno"
REPLICATE_COL = "Replicate"  # Was "rep"  
BARCODE_COL = "Barcode"
```

## Function Signature

```python
def sanitize_trait_names(
    df: pd.DataFrame,
    trait_cols: List[str],
    abbreviate: bool = True,
    return_mapping: bool = False,
    sanitize_metadata: bool = True,
    genotype_col: Optional[str] = None,
    replicate_col: Optional[str] = None,
    barcode_col: Optional[str] = None,
    custom_replacements: Optional[Dict[str, str]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, str]]]:
    """Sanitize trait column names and metadata columns for better visualization.

    Args:
        df: DataFrame containing trait data
        trait_cols: List of trait column names to sanitize
        abbreviate: If True, abbreviate common words (Default: True)
        return_mapping: If True, return (df, mapping_dict) tuple (Default: False)
        sanitize_metadata: If True, also sanitize metadata columns (Default: True)
        genotype_col: Current genotype column name to rename to "Genotype"
        replicate_col: Current replicate column name to rename to "Replicate"
        barcode_col: Current barcode column name to ensure title case as "Barcode"
        custom_replacements: Optional dict mapping old terms to new terms
            (e.g., {"crown": "seminal"}). Matching is case-insensitive.

    Returns:
        If return_mapping=False: DataFrame with sanitized column names
        If return_mapping=True: Tuple of (DataFrame, mapping dict)
    """
```

## Benefits

1. **Improved Readability**: Shorter, cleaner names in all plots
2. **Professional Appearance**: Consistent formatting across visualizations
3. **Proper Units**: Units displayed with proper symbols (mm², °, etc.)
4. **Traceability**: Mapping file preserves original→sanitized relationship
5. **Data Integrity**: All data values and metadata columns remain unchanged

## Tips

- **Use early in pipeline**: Apply sanitization right after loading data, before any analysis
- **Save mapping**: Always save the mapping file for traceability
- **Choose abbreviation level**: Use `abbreviate=True` for very long trait names, `False` for more descriptive names
- **Non-trait columns**: Metadata columns (barcode, geno, rep) are automatically preserved unchanged

## Complete Example

See the updated `trait_qc_150_genotypes_turface_20251105.ipynb` for a complete working example integrating trait name sanitization into the full QC pipeline.
