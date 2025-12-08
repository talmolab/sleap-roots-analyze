# Fix Biomass Depth Range Labels

## Why

Root core biomass columns are being sanitized with confusing and misleading labels that hide critical depth range information.

**Current Problem:**
- Raw biomass data from 0-30cm depth range → stored with midpoint label `RootDW_15cm`
- Raw biomass data from 30-60cm depth range → stored with midpoint label `RootDW_45cm`
- After `sanitize_trait_names()` → becomes `"Rootdw 15Cm"` and `"Rootdw 45Cm"`

**Why This Is Confusing:**
1. **Loss of semantic meaning**: "Rootdw 15Cm" suggests biomass measured at only 15cm depth, when it's actually the **total biomass from 0-30cm range**
2. **Scientific accuracy**: The midpoint representation (15cm, 45cm) is an internal computational convenience for depth mapping, but should **never** be shown to users in visualizations or outputs
3. **Data interpretation errors**: Users seeing "15cm" will misinterpret this as a single-depth measurement rather than an integrated 30cm section
4. **Inconsistency with raw data**: Original data uses descriptive ranges like `c_0_30` which clearly communicates the measurement domain

**Root Cause Analysis:**

The issue spans multiple components:

1. **Pipeline Design** ([reshape_for_trait_qc.py:171-175](c:\repos\sleap-roots-analyze\src\sleap_roots_analyze\pipeline\steps\reshape_for_trait_qc.py#L171-L175)):
   - Uses `depth_column_prefix` + midpoint depth: `f"{prefix}_{depth_int}cm"`
   - Creates technical column names like `RootDW_15cm`, `RootDW_45cm`
   - Midpoint values (15, 45) come from config `depth_mapping: {"0-30": 15.0, "30-60": 45.0}`

2. **Sanitization Function** ([data_utils.py:52-235](c:\repos\sleap-roots-analyze\src\sleap_roots_analyze\data_utils.py#L52-L235)):
   - Processes `RootDW_15cm` → splits on `_` → `["RootDW", "15cm"]`
   - Converts to title case → `"Rootdw 15Cm"`
   - **No special handling for depth range notation**
   - **No awareness that "15cm" represents a 0-30cm range**

3. **Missing Context Preservation**:
   - Depth mapping config (`0-30 → 15.0`) is only used during reshape
   - By the time sanitization runs, the original depth range (0-30cm) is lost
   - No metadata tracks "this 15cm is actually 0-30cm range"

**Impact on Users:**

This affects multiple downstream uses:
- **Visualization plots**: Axis labels show misleading "Rootdw 15Cm" instead of "Root Biomass DW (g) 0-30cm"
- **Analysis scripts**: Must manually handle two naming conventions (notebook: `c_0_30`, pipeline: `Rootdw 15Cm`)
- **Scientific communication**: Figure labels are unclear and require manual explanation
- **Reproducibility**: Users cannot easily map output columns back to measurement protocol

## What Changes

This proposal fixes the biomass depth range labeling issue through three coordinated changes:

### 1. **Preserve Depth Range Metadata** (NEW)
- Add `depth_range_mapping` to pipeline config schema
- Store actual depth ranges alongside midpoint values
- Pass metadata through pipeline steps to sanitization

### 2. **Enhance Sanitization with Depth-Aware Logic** (MODIFIED)
- Detect biomass/root count columns with depth suffixes
- Recognize patterns: `RootDW_15cm` → depth range context needed
- Apply depth-range-aware formatting: `Root Biomass DW (g) 0-30cm`
- Support both midpoint notation (15cm) and range notation (0-30cm)

### 3. **Update Tests with TDD Approach** (NEW)
- Add test fixtures for biomass columns with various depth formats
- Test depth range detection and labeling
- Verify backward compatibility with non-biomass columns
- Validate both config-driven and auto-detection modes

### Breaking Changes
**NONE** - This is backward compatible:
- Existing configs continue to work (midpoints valid, but improved labels)
- Non-biomass columns unchanged
- Can opt-in to new depth range labels via config

## Impact

**Affected Specs:**
- `data-sanitization` (NEW SPEC) - Documents sanitization rules and depth handling
- `qc-pipeline` (if exists) - Update to include depth range metadata

**Affected Code:**
- `src/sleap_roots_analyze/data_utils.py:52-235` - `sanitize_trait_names()` function
- `src/sleap_roots_analyze/pipeline/steps/reshape_for_trait_qc.py:134-197` - `_pivot_to_wide()` method
- `src/sleap_roots_analyze/pipeline/config/components.py` - Add `depth_range_mapping` to config schema
- `tests/test_data_utils.py` - Add depth range sanitization tests
- `configs/qc_root_core_edpie.yaml` - Example config with new depth_range_mapping

**User-Facing Changes:**
- **Visualization labels**: Clear depth ranges in plots (0-30cm, 30-60cm)
- **Column names**: Semantically meaningful names in output CSVs
- **Documentation**: Updated examples showing proper depth range handling

**Migration Path:**
- Existing configs work as-is (no breaking changes)
- Recommended: Add `depth_range_mapping` to configs for improved labels
- Scripts using old column names continue working (we preserve compatibility)

## Success Criteria

1. ✅ Test coverage: Biomass depth range sanitization has >95% coverage
2. ✅ Output validation: Pipeline outputs show "Root Biomass DW (g) 0-30cm" not "Rootdw 15Cm"
3. ✅ Backward compatibility: All existing tests pass
4. ✅ Documentation: Config examples show both midpoint and range notation
5. ✅ User clarity: Scientists can immediately understand measurement depth from column names