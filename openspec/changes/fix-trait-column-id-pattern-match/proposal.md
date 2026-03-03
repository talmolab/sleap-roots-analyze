# Fix: `get_trait_columns()` incorrectly classifies width and solidity traits as metadata

**Change ID:** fix-trait-column-id-pattern-match
**Type:** Bug fix
**GitHub Issue:** #75
**Affects:** `data_cleanup.py` — `get_trait_columns()` function

## Problem

The `get_trait_columns()` function (data_cleanup.py:74-156) uses substring matching to
identify metadata columns. The `common_metadata` list includes `"id"` (line 118), which
is matched via:

```python
if any(meta in col_lower for meta in common_metadata):
    exclude_cols.append(col)
```

This causes false-positive matches against legitimate trait columns whose names happen
to contain the substring `"id"`:

| Pattern | Affected columns | Count | Example |
|---------|-----------------|-------|---------|
| `root_widths` → `w**id**ths` | root_widths_* | 0 in test fixtures (but reported in production) |
| `network_width` → `w**id**th` | network_width_depth_ratio_* | 9 | `network_width_depth_ratio_min` |
| `network_solidity` → `sol**id**ity` | network_solidity_* | 9 | `network_solidity_min` |
| `chull_max_width` → `w**id**th` | chull_max_width_* | 9 | `chull_max_width_mean` |

**Total traits silently dropped: 27 columns** in `traits_11DAG` fixture, 27 in
`traits_summary`, and potentially ~90 in production datasets with additional
`root_widths_*` columns.

### Verified against real fixture data

Fixture: `tests/data/traits_11DAG_cleaned_qc_scanner_independent.csv` (880 columns)
- **Legitimate ID metadata columns** that SHOULD be excluded: `scan_id`, `accession_id`,
  `experiment_id`, `phenotyper_id`, `plant_id`, `scanner_id`, `species_id`, `wave_id`
  (all end with `_id`)
- **Trait columns falsely excluded**: all 18 width columns + 9 solidity columns = 27 traits

Fixture: `tests/data/traits_summary.csv` (924 columns)
- Same 8 legitimate `_id` columns, same 27 falsely excluded traits

## Root Cause

The `"id"` entry in `common_metadata` is too broad. All actual ID columns in SLEAP Roots
data follow the pattern `*_id` (suffix). No legitimate metadata column is named simply
`"id"` or has `"id"` as a prefix.

## Proposed Fix

Replace the bare `"id"` entry in `common_metadata` with the specific suffix `"_id"` and
change the matching logic for `"_id"` to use `str.endswith()` instead of substring
containment.

### Implementation Strategy

**Option chosen: Targeted suffix match for ID patterns**

In the `common_metadata` list:
1. Remove `"id"` (line 118)
2. Remove `"plant_id"` (line 137) — redundant once `"_id"` suffix matching is added
3. Add `"_id"` as a **suffix-matched** pattern

In the matching loop (lines 139-142), introduce a distinction between:
- **Suffix patterns** (prefixed with convention marker): matched with `str.endswith()`
- **Substring patterns** (default): matched with `in` operator as before

Concrete implementation — split `common_metadata` into two lists:

```python
# Patterns matched as substrings (existing behavior, safe)
metadata_substring_patterns = [
    "date", "time", "sterilization", "experiment", "batch",
    "operator", "notes", "comments", "index", "qc_", "outlier",
    "wave_name", "wave_number", "germ_day", "plant_age", "age_days",
    "day_", "_color", "dot", "scan_", "scanner", "phenotyper",
    "uploaded", "accession", "species_", "plant_name",
]

# Patterns matched as suffixes only
metadata_suffix_patterns = [
    "_id",
]
```

Matching logic:
```python
for col in df.columns:
    col_lower = col.lower()
    if any(meta in col_lower for meta in metadata_substring_patterns):
        exclude_cols.append(col)
    elif any(col_lower.endswith(suffix) for suffix in metadata_suffix_patterns):
        exclude_cols.append(col)
```

### Why this approach

- **Minimal change**: Only the `"id"` pattern changes; all other patterns keep existing
  substring behavior (which is correct for them)
- **Matches real data**: Every ID column in both fixture files ends with `_id`
- **No false negatives**: `plant_id`, `scan_id`, `experiment_id`, etc. are all still
  caught by `_id` suffix matching
- **No false positives**: `network_width`, `network_solidity`, `chull_max_width`,
  `root_widths` do NOT end with `_id`

## TDD Plan

### Phase 1: Failing Unit Tests (Red)

Add to `TestGetTraitColumns` in `tests/test_data_cleanup.py`:

#### Test 1: `test_width_columns_not_excluded_by_id_pattern`
**Purpose**: Directly reproduce the bug — width traits must not be classified as metadata.
```
GIVEN a DataFrame with columns: Barcode, geno, rep, network_width_depth_ratio_min,
      network_width_depth_ratio_max, chull_max_width_mean, root_widths_min_min
WHEN get_trait_columns() is called
THEN all 4 width columns MUST be in the returned trait list
AND Barcode, geno, rep MUST NOT be in the returned list
```

#### Test 2: `test_solidity_columns_not_excluded_by_id_pattern`
**Purpose**: Catch the secondary false-positive pattern (solidity contains "id").
```
GIVEN a DataFrame with columns: Barcode, geno, rep, network_solidity_min,
      network_solidity_max, network_solidity_mean
WHEN get_trait_columns() is called
THEN all 3 solidity columns MUST be in the returned trait list
```

#### Test 3: `test_actual_id_columns_still_excluded`
**Purpose**: Ensure the fix doesn't break exclusion of real ID metadata.
```
GIVEN a DataFrame with columns: Barcode, geno, rep, scan_id, plant_id,
      accession_id, experiment_id, species_id, trait1
WHEN get_trait_columns() is called
THEN scan_id, plant_id, accession_id, experiment_id, species_id MUST NOT be
     in the returned trait list
AND trait1 MUST be in the returned trait list
```

#### Test 4: `test_mixed_id_and_width_columns`
**Purpose**: Combined scenario ensuring both exclusions and inclusions work together.
```
GIVEN a DataFrame with columns: Barcode, geno, rep, scan_id, plant_id,
      network_width_depth_ratio_min, network_solidity_max, chull_max_width_mean,
      primary_length_max, trait1
WHEN get_trait_columns() is called
THEN excluded: Barcode, geno, rep, scan_id, plant_id
AND included: network_width_depth_ratio_min, network_solidity_max,
     chull_max_width_mean, primary_length_max, trait1
```

### Phase 2: Failing Integration Tests (Red)

Integration tests run `get_trait_columns()` against **real fixture CSVs** with the
correct per-dataset column names (as the pipeline would configure them), then verify
that the returned trait list and the implied metadata classification are both correct.

Each dataset has different column naming conventions, so these tests exercise different
code paths and ensure the fix generalizes across real-world data formats.

#### Test 5: `test_pipeline_trait_classification_11dag` (integration)
**Purpose**: Validate trait vs metadata classification on SLEAP Roots summary data
(snake_case columns, `_id` suffix metadata, width/solidity traits).
```
GIVEN the traits_11dag_df fixture loaded from
      tests/data/traits_11DAG_cleaned_qc_scanner_independent.csv
      (880 columns, barcode_col="plant_qr_code", genotype_col="Geno",
       replicate_col="Rep")
WHEN get_trait_columns() is called with those column mappings
THEN:
  -- Width traits MUST be classified as traits (not metadata):
     network_width_depth_ratio_min, network_width_depth_ratio_max,
     chull_max_width_min, chull_max_width_max IN result
  -- Solidity traits MUST be classified as traits (not metadata):
     network_solidity_min, network_solidity_max IN result
  -- Actual ID metadata MUST be excluded:
     scan_id, plant_id, accession_id, experiment_id, species_id,
     scanner_id, phenotyper_id, wave_id NOT IN result
  -- Other metadata MUST be excluded:
     plant_qr_code, Geno, Rep, Sterilization, DOT, QC_SLEAP,
     Date_QC, germ_day, plant_name, species_name NOT IN result
  -- Exact width trait count: 18 columns containing "width"
  -- Exact solidity trait count: 9 columns containing "solidity"
  -- Total trait count > 800 (880 total minus ~30 metadata/non-numeric)
```

#### Test 6: `test_pipeline_trait_classification_traits_summary` (integration)
**Purpose**: Validate on the traits_summary dataset which has a different column order
and includes `uploaded_at`, `wave_number`, `wave_name` metadata columns.
```
GIVEN the traits_summary_df fixture loaded from tests/data/traits_summary.csv
      (924 columns, barcode_col="plant_qr_code", genotype_col="Geno",
       replicate_col="Rep")
WHEN get_trait_columns() is called with those column mappings
THEN:
  -- Width traits MUST be classified as traits:
     network_width_depth_ratio_min, chull_max_width_min IN result
  -- Solidity traits MUST be classified as traits:
     network_solidity_min IN result
  -- Actual ID metadata MUST be excluded:
     scan_id, plant_id, accession_id NOT IN result
  -- Additional metadata MUST be excluded:
     uploaded_at, wave_number, wave_name NOT IN result
  -- Exact width trait count: 18 columns containing "width"
  -- Exact solidity trait count: 9 columns containing "solidity"
```

#### Test 7: `test_pipeline_trait_classification_traits_summary_lateral` (integration)
**Purpose**: Validate on the lateral-root-specific summary which has a different trait
prefix set (lateral_* instead of crown_*) but same metadata structure.
```
GIVEN the traits_summary_lateral_df fixture loaded from
      tests/data/traits_summary_lateral.csv
      (barcode_col="plant_qr_code", genotype_col="Geno", replicate_col="Rep")
WHEN get_trait_columns() is called with those column mappings
THEN:
  -- Width traits MUST be classified as traits (not metadata)
  -- Solidity traits MUST be classified as traits (not metadata)
  -- Actual ID metadata MUST be excluded
  -- Width trait count matches expected (18 columns containing "width")
  -- Solidity trait count matches expected (9 columns containing "solidity")
```

#### Test 8: `test_pipeline_trait_classification_turface` (integration)
**Purpose**: Validate on a completely different dataset format — agronomic field trial
data with different column naming conventions (lowercase `geno`, no `_id` columns,
`plant_identifier` which contains "id" but should NOT be excluded as a trait).
```
GIVEN the turface_traits_df fixture loaded from
      tests/data/Turface_all_traits_2024.csv
      (41 columns, barcode_col="Barcode", genotype_col="geno",
       replicate_col="Rep")
WHEN get_trait_columns() is called with those column mappings
THEN:
  -- All agronomic trait columns MUST be in result:
     GY_Calc_gm2, BM_Calc_gm2, PH_M_cm, GW_M_g1000grn, etc.
  -- Root core depth columns MUST be in result:
     c_0_30, c_30_60, c_0_10_1, c_0_10_2, etc.
  -- Metadata columns MUST be excluded:
     geno, Rep NOT IN result
  -- No false exclusions: total trait count equals expected
     (all numeric columns minus geno/Rep/Barcode metadata)
```

#### Test 9: `test_pipeline_trait_classification_features` (integration)
**Purpose**: Validate on RhizoVision output format — dotted PascalCase column names
(`Maximum.Width.mm`, `Width-to-Depth.Ratio`, `Solidity`). Ensures the fix doesn't
break anything for datasets that use completely different naming conventions.
```
GIVEN the features_df fixture loaded from tests/data/features.csv
      (38 columns, barcode_col="File.Name", genotype_col="geno",
       replicate_col="rep")
      Note: geno/rep columns don't exist in this CSV; this tests graceful handling
WHEN get_trait_columns() is called
THEN:
  -- Width trait MUST be in result: Maximum.Width.mm, Width-to-Depth.Ratio
  -- Solidity trait MUST be in result: Solidity
  -- All numeric measurement columns MUST be in result
  -- Computation.Time.s SHOULD match "time" metadata pattern and be excluded
```

#### Test 10: `test_metadata_columns_are_complement_of_traits` (integration, regression guard)
**Purpose**: For each fixture dataset, verify that the set of trait columns + the set of
metadata columns = the full column set (no columns lost or double-counted). This is the
key regression guard.
```
FOR EACH of: traits_11dag_df, traits_summary_df, traits_summary_lateral_df,
             turface_traits_df
  GIVEN the fixture DataFrame
  WHEN get_trait_columns() returns trait_cols
  AND metadata_cols = [col for col in df.columns if col not in trait_cols]
  THEN set(trait_cols) | set(metadata_cols) == set(df.columns)
  AND set(trait_cols) & set(metadata_cols) == empty set
  AND every column in trait_cols is numeric
  AND no column in trait_cols matches any known metadata pattern
```

### Phase 3: Implement Fix (Green)

Apply the changes described in the Implementation Strategy section above.

### Phase 4: Verify & Refactor

1. All 10 new tests pass (4 unit + 6 integration)
2. All 5 existing `TestGetTraitColumns` tests still pass
3. Full test suite passes (`uv run pytest`)
4. No other `common_metadata` patterns produce false positives on fixture data

## Files Changed

| File | Change |
|------|--------|
| `src/sleap_roots_analyze/data_cleanup.py` | Split `common_metadata` into substring and suffix lists; update matching logic |
| `tests/test_data_cleanup.py` | Add 10 new tests (4 unit, 6 integration) |

## Risk Assessment

- **Low risk**: Change is isolated to one function's pattern-matching logic
- **No API change**: Function signature unchanged, return type unchanged
- **Backward compatible**: All previously-correct exclusions remain excluded
- **Only improvement**: Previously-incorrect exclusions (width/solidity traits) now correctly included
- **Multi-dataset validation**: Integration tests cover 5 different real fixture files with different column naming conventions, providing high confidence the fix generalizes
