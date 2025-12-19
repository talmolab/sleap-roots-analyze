# Proposal: Integrate Custom Replacements into Pipeline Config

**Change ID**: `integrate-custom-replacements-config`
**Status**: Draft
**Created**: 2025-12-02
**Owner**: elizabeth

## Summary

Integrate the existing `custom_replacements` parameter from `sanitize_trait_names()` into the QC pipeline configuration system, enabling users to specify domain-specific trait name replacements (e.g., "crown" → "seminal" for wheat) via config files instead of hardcoding them in notebooks.

## Motivation

**Current State:**
- `sanitize_trait_names()` function supports `custom_replacements` parameter ✅
- Function is well-tested with custom replacements ✅
- Parameter is used in Jupyter notebooks (e.g., `trait_qc_cylinders_20251105.ipynb`) ✅
- **Pipeline config does NOT support `custom_replacements`** ❌
- Users cannot specify custom replacements in YAML config files ❌

**Problem:**
The Cylinder platform QC config needs `custom_replacements: {"crown": "seminal"}` to handle wheat-specific terminology, but there's no way to specify this in the config file. The functionality exists at the function level but isn't exposed through the pipeline config system.

**Desired State:**
```yaml
cleanup:
  max_nan_fraction: 0.0
  custom_replacements:
    crown: "seminal"  # Wheat: crown roots → seminal roots
    primary: "main"   # Optional: additional replacements
```

## Goals

1. **Add `custom_replacements` to `CleanupConfig`** dataclass
2. **Update `CleanupTraitsStep`** to call `sanitize_trait_names()` with custom_replacements
3. **Update config schema** to support optional dict parameter
4. **Add tests** for config-based custom replacements
5. **Update example configs** to demonstrate usage

## Non-Goals

- Modifying `sanitize_trait_names()` function behavior (already works correctly)
- Changing trait sanitization logic (only exposing existing functionality)
- Adding new sanitization features beyond custom replacements

## Design Rationale

### Why CleanupConfig?

Trait name sanitization happens during the cleanup phase (Step 2: CleanupTraitsStep), so `custom_replacements` logically belongs in `CleanupConfig` alongside other cleanup parameters.

### Custom Replacements Format

The `custom_replacements` parameter accepts a dictionary where:
- **Keys**: Words to find in trait names (case-insensitive matching)
- **Values**: Replacement words

**Examples:**
```yaml
custom_replacements:
  crown: "seminal"      # "crown_length" → "Seminal Length"
  primary: "main"       # "primary_root" → "Main Root"
  lateral: "branch"     # "lateral_count" → "Branch Count"
```

**Matching behavior:**
- Case-insensitive: "crown", "Crown", "CROWN" all match
- Whole-word matching: Only replaces complete words after splitting by `.`, `-`, `_`
- Applied during sanitization: After splitting, before abbreviation/title-casing

### Integration Point

The `CleanupTraitsStep` currently does NOT call `sanitize_trait_names()` - this happens ad-hoc in notebooks. We need to add this call.

**Decision:** Add sanitization to `CleanupTraitsStep.execute()`:
1. After getting trait columns from previous step
2. Before applying cleanup filters
3. Pass custom_replacements from config

### Config Structure

```yaml
cleanup:
  # Existing parameters
  max_nan_fraction: 0.0
  max_zeros_per_trait: 0.5
  max_nans_per_trait: 0.2
  min_samples_per_trait: 10

  # NEW: Custom trait name replacements (optional)
  custom_replacements:
    crown: "seminal"      # Wheat terminology
    primary: "main"       # Optional: additional mappings
```

Python dataclass:
```python
@dataclass
class CleanupConfig:
    max_nan_fraction: float = 0.0
    max_zeros_per_trait: float = 0.5
    max_nans_per_trait: float = 0.2
    min_samples_per_trait: int = 10
    custom_replacements: Optional[Dict[str, str]] = None  # NEW
```

## Dependencies

- ✅ `sanitize_trait_names()` function exists in `data_utils.py`
- ✅ Function supports `custom_replacements` parameter
- ✅ Tests exist for custom replacements in `test_data_utils.py`
- ✅ `CleanupConfig` exists in `pipeline/config/components.py`
- ✅ `CleanupTraitsStep` exists in `pipeline/steps/cleanup_traits.py`

## Impact Assessment

**Files to Modify:**
1. `src/sleap_roots_analyze/pipeline/config/components.py` - Add field to CleanupConfig
2. `src/sleap_roots_analyze/pipeline/steps/cleanup_traits.py` - Call sanitize_trait_names() with custom_replacements
3. `configs/qc_cylinder_edpie.yaml` - Add custom_replacements example (NEW FILE - part of add-multi-platform-configs)
4. `tests/test_step_cleanup.py` - Add test for custom_replacements in pipeline

**Backward Compatibility:**
- ✅ `custom_replacements` defaults to `None` (optional parameter)
- ✅ Existing configs without custom_replacements continue to work unchanged
- ✅ No breaking changes to existing behavior
- ✅ When None, `sanitize_trait_names()` behaves exactly as before

## Open Questions

1. **Should we track the trait name mapping in step metadata?**
   - **Answer**: Yes - add `trait_name_mapping` to CleanupTraitsStep metadata
   - Rationale: Useful for debugging and understanding transformations
   - Only include mappings where names actually changed

2. **Should sanitize_trait_names() always run, or only when custom_replacements is set?**
   - **Answer**: Always run with abbreviate=True (current notebook behavior)
   - Rationale: Provides consistent, readable trait names across all pipelines
   - custom_replacements is optional, sanitization itself is standard

3. **Should we update existing QC configs to show custom_replacements: null?**
   - **Answer**: No - omit it entirely when not used
   - Rationale: Optional parameters with None default don't clutter configs

## Success Criteria

- [ ] `CleanupConfig` has `custom_replacements: Optional[Dict[str, str]]` field with docstring
- [ ] `CleanupTraitsStep` calls `sanitize_trait_names()` with config.cleanup.custom_replacements
- [ ] Trait name mapping saved in step metadata when names change
- [ ] Sanitized trait names used throughout rest of pipeline
- [ ] New test in `test_step_cleanup.py` validates custom_replacements work through pipeline
- [ ] All existing tests pass (1109+ tests)
- [ ] Config validation accepts custom_replacements parameter
- [ ] Example usage documented in Cylinder config (separate proposal)

## Related Proposals

- **Depends on**: None (uses existing functionality)
- **Blocks**: `add-multi-platform-configs` (Cylinder config needs this feature)
- **Related**: `add-custom-trait-replacements` (archived - added function-level support in 2024)