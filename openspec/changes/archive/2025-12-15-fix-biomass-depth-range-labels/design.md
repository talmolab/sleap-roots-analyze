# Design Document: Biomass Depth Range Labels

## Context

The current pipeline creates confusing column labels for root biomass measurements at different depths. The core issue is that depth ranges (0-30cm, 30-60cm) are converted to midpoint values (15cm, 45cm) for technical convenience, but these midpoints are then displayed to users without context.

**Current Data Flow:**
```
Raw Data          Pipeline Config         Reshape Step            Sanitization         User Sees
─────────         ────────────────         ────────────            ────────────         ─────────
c_0_30 = 2.5g  →  depth_mapping:        →  RootDW_15cm = 2.5g  →  Rootdw 15Cm      →  "Rootdw 15Cm" 😕
c_30_60 = 1.2g     "0-30": 15.0             RootDW_45cm = 1.2g     Rootdw 45Cm         "Rootdw 45Cm" 😕
                   "30-60": 45.0
```

**Desired Data Flow:**
```
Raw Data          Pipeline Config              Reshape Step            Sanitization                    User Sees
─────────         ────────────────              ────────────            ────────────                    ─────────
c_0_30 = 2.5g  →  depth_mapping:             →  RootDW_15cm = 2.5g  →  Root Biomass DW (g) 0-30cm  →  Clear! ✅
c_30_60 = 1.2g     "0-30": 15.0                  RootDW_45cm = 1.2g     Root Biomass DW (g) 30-60cm     Clear! ✅
                   "30-60": 45.0
                  depth_range_mapping:
                    15.0: "0-30"
                    45.0: "30-60"
```

**Stakeholders:**
- **Scientists/Users**: Need clear, scientifically accurate labels in visualizations and outputs
- **Pipeline Developers**: Need maintainable, backward-compatible code
- **Config Authors**: Need simple, intuitive configuration format

**Constraints:**
- Must maintain backward compatibility (existing configs must work)
- Cannot change internal depth representation (15, 45 used in calculations)
- Must support both biomass (ranges) and counting (single depths) measurements
- Must work with auto-detected depths from counting data

## Goals / Non-Goals

**Goals:**
1. **Primary**: Display scientifically accurate depth ranges in user-facing outputs
2. Enable clear, unambiguous column labels in visualizations and CSVs
3. Maintain backward compatibility with existing configs and code
4. Support both manual depth mappings (biomass) and auto-detected depths (counting)
5. Follow TDD approach with comprehensive test coverage

**Non-Goals:**
1. Change internal depth representation (keep midpoints for calculations)
2. Modify reshape step logic (column creation unchanged)
3. Auto-detect depth ranges from midpoints (requires config)
4. Support fractional depth ranges (e.g., 0-15.5cm) - use integer ranges only
5. Internationalization of depth units (keep cm only)

## Decisions

### Decision 1: Optional Config Parameter Approach
**What**: Add optional `depth_range_mapping` to config, don't auto-detect ranges from midpoints.

**Why**:
- **Explicit over implicit**: Scientific data should have explicit metadata, not guessed
- **Accuracy**: Only users know true measurement protocol (0-30cm vs 0-40cm)
- **Simplicity**: No complex heuristics to maintain
- **Backward compatible**: Existing configs work as-is

**Alternatives Considered:**
1. **Auto-detect ranges from midpoints** (15 → guess 0-30)
   - ❌ Fragile: What if midpoint is 20cm? Is that 0-40 or 10-30?
   - ❌ Assumes fixed range sizes (not always 30cm)
   - ❌ Requires magic numbers and heuristics

2. **Change reshape step to use ranges directly** (e.g., `RootDW_0-30cm`)
   - ❌ Breaking change to column naming
   - ❌ Complicates downstream calculations expecting numeric depths
   - ❌ Would require changing depth parsing logic in multiple places

3. **Store metadata in DataFrame.attrs**
   - ❌ Lost when saving/loading CSVs
   - ❌ Not accessible in sanitization step (runs on fresh loaded data)
   - ❌ Fragile and hard to debug

### Decision 2: Two-Stage Formatting (Detection + Formatting)
**What**: Separate depth detection from range formatting via helper functions.

**Why**:
- **Testability**: Each stage can be unit tested independently
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new depth patterns (e.g., `_Dmm` for mm)

**Implementation:**
```python
def _detect_depth_suffix(col_name: str) -> Optional[float]:
    """Extract numeric depth from column name like 'RootDW_15cm' → 15.0"""
    match = re.search(r'_(\d+\.?\d*)cm$', col_name)
    return float(match.group(1)) if match else None

def _format_depth_range(depth: float, mapping: Dict[float, str]) -> str:
    """Map depth to range: 15.0 → '0-30cm' or fallback to '15cm'"""
    if mapping and depth in mapping:
        return f"{mapping[depth]}cm"
    return f"{int(depth) if depth == int(depth) else depth}cm"
```

**Alternatives Considered:**
1. **Single monolithic function**
   - ❌ Hard to test edge cases
   - ❌ Mixing detection and formatting concerns

2. **Regex-based replacement in place**
   - ❌ Can't distinguish biomass ranges from counting depths
   - ❌ No validation of depth values

### Decision 3: Integrate at Sanitization Time
**What**: Apply depth range formatting inside `sanitize_trait_names()`, not in reshape step.

**Why**:
- **Single source of truth**: All column name formatting in one place
- **Consistent with other transformations**: Units, abbreviations handled here
- **User control**: Can choose to sanitize or not
- **Backward compatible**: Doesn't change reshape output

**Integration Point** ([data_utils.py:149-175](src/sleap_roots_analyze/data_utils.py#L149-L175)):
```python
# After unit conversion but before splitting/processing
if depth_range_mapping:
    depth = _detect_depth_suffix(new_name)
    if depth is not None:
        range_str = _format_depth_range(depth, depth_range_mapping)
        # Replace depth suffix with range
        new_name = re.sub(r'_\d+\.?\d*cm$', f'_{range_str}', new_name)
```

### Decision 4: Config Schema Update
**What**: Add `depth_range_mapping` as sibling to existing `depth_mapping` in source config.

**Why**:
- **Clear distinction**: `depth_mapping` = technical (for calcs), `depth_range_mapping` = display
- **Optional**: Doesn't break existing configs
- **Co-located**: Both mappings in same source definition for clarity

**Config Structure:**
```yaml
root_core:
  sources:
    - csv_path: "..."
      data_type: "biomass"
      depth_column_prefix: "RootDW"

      # Existing: technical mapping for reshape (midpoints)
      depth_mapping:
        "0-30": 15.0   # Create column RootDW_15cm
        "30-60": 45.0  # Create column RootDW_45cm

      # NEW: display mapping for sanitization (ranges)
      depth_range_mapping:
        15.0: "0-30"   # Display as "Root Biomass DW (g) 0-30cm"
        45.0: "30-60"  # Display as "Root Biomass DW (g) 30-60cm"
```

## Risks / Trade-offs

### Risk: Config Complexity
**Risk**: Two mappings (`depth_mapping` + `depth_range_mapping`) might confuse users.

**Mitigation**:
- Document clearly in config comments and examples
- Show both mappings in template configs
- Add validation: warn if `depth_range_mapping` keys don't match `depth_mapping` values

### Risk: Mapping Mismatch
**Risk**: User provides wrong mapping (e.g., 15.0 → "0-40" when actual is 0-30).

**Mitigation**:
- Document that this is user responsibility (explicit metadata)
- Validation warning if mapping seems inconsistent
- Fallback to midpoint if no mapping provided (graceful degradation)

### Risk: Breaking Changes to Scripts
**Risk**: External scripts expecting "Rootdw 15Cm" column names will break.

**Mitigation**:
- **Low risk**: Sanitization is optional, can be disabled
- Most scripts use raw pipeline outputs (before sanitization)
- Analysis script already handles both naming conventions ([analyze_biomass_depth_correlations.py:71-84](scripts/analyze_biomass_depth_correlations.py#L71-L84))

### Trade-off: Config Verbosity vs Auto-Magic
**Trade-off**: Requiring explicit config is more verbose than auto-detection.

**Accepted Because**:
- Scientific accuracy > convenience
- Explicit metadata prevents errors
- Config is one-time setup, used repeatedly
- Auto-detection would be fragile and error-prone

## Migration Plan

### Phase 1: Add New Functionality (This Proposal)
1. Add `depth_range_mapping` parameter to `sanitize_trait_names()`
2. Implement depth detection and formatting helpers
3. Update tests with TDD approach
4. **No breaking changes**: All existing code works as-is

### Phase 2: Config Updates (Recommended, Not Required)
1. Update `configs/qc_root_core_edpie.yaml` with example `depth_range_mapping`
2. Document in CLAUDE.md and config comments
3. Users can gradually adopt when they update configs

### Phase 3: Deprecation (Future, Optional)
1. If all users adopt range notation, could deprecate midpoint display
2. Not planned for this proposal - would require user survey first

### Rollback Plan
If issues discovered:
1. Remove `depth_range_mapping` from config (backward compatible)
2. Revert `sanitize_trait_names()` changes (helper functions isolated)
3. No data loss - internal representation unchanged

## Open Questions

1. **Should we validate that `depth_range_mapping` keys match `depth_mapping` values?**
   - Proposed: Yes, add warning but don't fail (user might have good reason for mismatch)
   - Implementation: Check in config validation step

2. **How to handle counting data with many depths (0, 5, 10, 15, 20cm)?**
   - Proposed: Don't require mapping for single-depth measurements
   - Auto-format as "Root Count 5cm" (already clear)
   - Only biomass ranges need special handling

3. **Should we support custom range formats (e.g., "0-30" vs "0 to 30")?**
   - Proposed: Start with simple "0-30" format only
   - Can extend later if users request

4. **How to handle fractional depths (e.g., 15.5cm)?**
   - Proposed: Use current rounding logic (int if whole number)
   - Document that depth ranges should use integer cm values

## Success Metrics

1. **Test Coverage**: >95% coverage for depth range logic
2. **User Feedback**: Scientists confirm labels are clear and accurate
3. **Backward Compatibility**: Zero regressions in existing test suite
4. **Adoption**: At least one production config uses new mapping within 2 weeks
5. **Documentation**: All config examples updated with depth range mapping