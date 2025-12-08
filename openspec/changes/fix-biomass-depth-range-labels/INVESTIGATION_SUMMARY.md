# Biomass Depth Range Labeling Investigation Summary

## Problem Statement

Root core biomass columns are being sanitized with confusing and scientifically inaccurate labels that obscure the actual depth range of measurements.

**Issue**: `c_0_30` (0-30cm biomass) → `RootDW_15cm` → `"Rootdw 15Cm"` ❌
**Expected**: `c_0_30` (0-30cm biomass) → `RootDW_15cm` → `"Root Biomass DW (g) 0-30cm"` ✅

## Root Cause Analysis

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA (CSV Input)                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Columns: c_0_30 = 2.5g, c_30_60 = 1.2g                                 │
│ Meaning: Biomass from 0-30cm depth, biomass from 30-60cm depth         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. PIPELINE CONFIG (configs/qc_root_core_edpie.yaml)                   │
├─────────────────────────────────────────────────────────────────────────┤
│ depth_mapping:                                                          │
│   "0-30": 15.0   # Map 0-30cm range → 15cm midpoint                    │
│   "30-60": 45.0  # Map 30-60cm range → 45cm midpoint                   │
│                                                                         │
│ depth_column_prefix: "RootDW"  # Prefix for output columns             │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. TRANSFORM DEPTH DATA STEP (Step 0b)                                 │
│    File: pipeline/steps/transform_depth_data.py                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Input: Wide format (c_0_30, c_30_60)                                   │
│ Process: Pivot to long format with Depth_cm column                     │
│ Output: Depth_cm = [15.0, 45.0], Root_DW_g = [2.5, 1.2]               │
│                                                                         │
│ NOTE: At this point, depth range (0-30) is LOST - only midpoint (15)  │
│       remains. This is the first point where information is discarded. │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. RESHAPE FOR TRAIT QC STEP (Step 0e)                                 │
│    File: pipeline/steps/reshape_for_trait_qc.py:171-175                │
├─────────────────────────────────────────────────────────────────────────┤
│ Code:                                                                   │
│   new_cols[col] = f"{prefix}_{depth_int}cm"                            │
│                                                                         │
│ Creates: RootDW_15cm, RootDW_45cm                                      │
│                                                                         │
│ Problem: Column name uses midpoint (15cm) but represents range (0-30cm)│
│          This is semantically misleading but computationally convenient│
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. CLEANUP TRAITS STEP (Step 1)                                        │
│    File: pipeline/steps/cleanup_traits.py:69                           │
├─────────────────────────────────────────────────────────────────────────┤
│ Calls: sanitize_trait_names(df, trait_cols, ...)                       │
│                                                                         │
│ Currently NO depth_range_mapping passed (parameter doesn't exist yet)  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. SANITIZE TRAIT NAMES FUNCTION                                       │
│    File: src/sleap_roots_analyze/data_utils.py:52-235                  │
├─────────────────────────────────────────────────────────────────────────┤
│ Input: "RootDW_15cm"                                                    │
│                                                                         │
│ Process:                                                                │
│ 1. Handle unit suffixes (lines 149-175):                               │
│    - Checks for .mm3, .mm2, .mm, .deg, etc.                            │
│    - "RootDW_15cm" → no unit suffix match (underscore, not dot)        │
│    - Unchanged: "RootDW_15cm"                                           │
│                                                                         │
│ 2. Split by delimiters (line 177):                                     │
│    - Split on dots, hyphens, underscores                               │
│    - "RootDW_15cm" → ["RootDW", "15cm"]                                │
│                                                                         │
│ 3. Apply abbreviations (lines 193-206):                                │
│    - No abbreviations apply to "RootDW" or "15cm"                      │
│    - Parts: ["RootDW", "15cm"]                                          │
│                                                                         │
│ 4. Title case (line 209):                                              │
│    - "Rootdw 15Cm"                                                      │
│                                                                         │
│ Problem: No special handling for depth suffixes!                       │
│          The "15cm" is treated as a regular word, not a depth marker   │
│          No awareness that 15cm represents 0-30cm range                 │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. FINAL OUTPUT                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Column name: "Rootdw 15Cm"                                              │
│                                                                         │
│ USER SEES: "Rootdw 15Cm" in visualizations and CSV outputs             │
│                                                                         │
│ CONFUSION: Looks like biomass at 15cm depth, but actually represents   │
│            total biomass from 0-30cm depth range!                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why This Matters

### Scientific Accuracy
- **Misleading labels**: "15cm" suggests a point measurement, not a 30cm integrated section
- **Data interpretation errors**: Users seeing "15cm" will misunderstand the measurement protocol
- **Publication clarity**: Figure labels must accurately communicate measurement depth

### Evidence from Codebase

1. **Script workaround required** ([analyze_biomass_depth_correlations.py:71-84](scripts/analyze_biomass_depth_correlations.py#L71-L84)):
   ```python
   # Script has to detect TWO naming conventions:
   notebook_style = {"shallow_biomass": "c_0_30", "deep_biomass": "c_30_60"}
   pipeline_style = {"shallow_biomass": "Rootdw 15Cm", "deep_biomass": "Rootdw 45Cm"}
   ```
   This proves the pipeline output is confusing enough that analysis code needs special handling.

2. **Lost semantic information**:
   - Original: `c_0_30` → Clear that this is 0-30cm range
   - Pipeline: `Rootdw 15Cm` → Unclear, looks like 15cm point measurement

## The Gap: Missing Context Preservation

The fundamental issue is **metadata loss**:

```
Config knows:        depth_mapping: {"0-30": 15.0}
                     ↓
Reshape step knows:  Create column RootDW_15cm
                     ↓
Sanitization knows:  Column name is "RootDW_15cm"
                     ↓
Sanitization does NOT know: 15cm actually means 0-30cm range!
```

**The context is never passed forward!**

## Solution Design

### Three-Part Fix

#### 1. **Preserve Metadata** (Config Schema)
Add `depth_range_mapping` to config:
```yaml
depth_mapping:
  "0-30": 15.0   # Technical: create RootDW_15cm column
depth_range_mapping:
  15.0: "0-30"   # Display: show as 0-30cm in outputs
```

#### 2. **Enhance Sanitization** (data_utils.py)
Add depth-aware logic:
```python
def sanitize_trait_names(
    df,
    trait_cols,
    depth_range_mapping: Optional[Dict[float, str]] = None,  # NEW
    ...
):
    # Detect depth suffix: RootDW_15cm → depth=15.0
    depth = _detect_depth_suffix(new_name)

    if depth and depth_range_mapping:
        # Map to range: 15.0 → "0-30"
        range_str = _format_depth_range(depth, depth_range_mapping)
        # Output: "Root Biomass DW (g) 0-30cm"
```

#### 3. **Pass Context** (Pipeline Steps)
Update cleanup step to pass mapping:
```python
df, mapping = sanitize_trait_names(
    df=df,
    trait_cols=trait_cols,
    depth_range_mapping=config.root_core.depth_range_mapping,  # NEW
)
```

## Test-Driven Development Approach

Following TDD principles:

### Phase 1: Write Failing Tests
```python
def test_biomass_depth_range_sanitization():
    """Test RootDW_15cm → Root Biomass DW (g) 0-30cm"""
    df = pd.DataFrame({"RootDW_15cm": [2.5], "RootDW_45cm": [1.2]})
    mapping = {15.0: "0-30", 45.0: "30-60"}

    result = sanitize_trait_names(
        df,
        ["RootDW_15cm", "RootDW_45cm"],
        depth_range_mapping=mapping
    )

    assert "Root Biomass DW (g) 0-30cm" in result.columns  # FAILS initially
    assert "Root Biomass DW (g) 30-60cm" in result.columns  # FAILS initially
```

### Phase 2: Implement Minimal Code
- Add `depth_range_mapping` parameter
- Implement helper functions (`_detect_depth_suffix`, `_format_depth_range`)
- Integrate into sanitization loop

### Phase 3: Verify Tests Pass
- All new tests pass
- All existing tests still pass (backward compatibility)
- Coverage >95% for new code

## Files Changed

### Core Implementation
- `src/sleap_roots_analyze/data_utils.py` - Add depth range logic to `sanitize_trait_names()`
- `src/sleap_roots_analyze/pipeline/config/components.py` - Add `depth_range_mapping` field
- `src/sleap_roots_analyze/pipeline/steps/cleanup_traits.py` - Pass mapping to sanitization

### Tests (TDD)
- `tests/test_data_utils.py` - Add depth range sanitization tests
- `tests/test_step_cleanup_traits.py` - Integration tests with mapping

### Configuration
- `configs/qc_root_core_edpie.yaml` - Example with `depth_range_mapping`

### Documentation
- `CLAUDE.md` - Document depth range labeling conventions
- Function docstrings - Update with new parameter

## Backward Compatibility

**ZERO BREAKING CHANGES:**
- ✅ Existing configs work without modification
- ✅ `depth_range_mapping` is optional (defaults to `None`)
- ✅ Without mapping, behavior is identical to current
- ✅ All existing tests pass without changes

## Success Criteria

1. ✅ **Test Coverage**: >95% coverage for depth range logic
2. ✅ **Output Quality**: Pipeline outputs show "Root Biomass DW (g) 0-30cm" not "Rootdw 15Cm"
3. ✅ **Backward Compatibility**: Zero test failures in existing suite
4. ✅ **Documentation**: Clear examples in configs and docstrings
5. ✅ **User Clarity**: Scientists immediately understand depth from column names

## Next Steps

1. **Review this proposal** - Ensure approach is sound
2. **Get approval** - Confirm this solves the problem correctly
3. **Implement with TDD** - Follow tasks.md checklist
4. **Test integration** - Run full pipeline and verify outputs
5. **Update documentation** - Ensure users understand new feature

---

**Created**: 2025-12-05
**Change ID**: `fix-biomass-depth-range-labels`
**OpenSpec Location**: `openspec/changes/fix-biomass-depth-range-labels/`
