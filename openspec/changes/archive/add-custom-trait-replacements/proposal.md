# Proposal: Add Custom Trait Name Replacements

## Why

Users need to perform domain-specific trait name transformations that go beyond the standard sanitization (e.g., "crown" → "seminal" for wheat root terminology). The current `sanitize_trait_names()` function provides fixed transformations (units, abbreviations, title case) but doesn't support custom user-defined replacements.

This is a common need in plant phenotyping where:
- Different experiments use different naming conventions
- Trait names need to be standardized across datasets for comparison
- Domain-specific terminology changes (e.g., "crown root" vs "seminal root")

## What Changes

- Add `custom_replacements` parameter to `sanitize_trait_names()` function
  - Type: `Optional[Dict[str, str]]` mapping old terms to new terms
  - Applied BEFORE standard sanitization pipeline
  - Case-insensitive matching for flexibility
  - Preserves standard sanitization behavior when no custom replacements provided

Example usage:
```python
df_sanitized = sanitize_trait_names(
    df=df,
    trait_cols=trait_cols,
    custom_replacements={"crown": "seminal", "primary": "main"},
    abbreviate=True
)
# "crown.length.mm" → "Seminal Length (mm)"
# "primary.root.angle.deg" → "Main Root Angle (°)"
```

## Impact

**Affected specs:**
- `data-utils` (NEW) - Creating first spec for data utility functions

**Affected code:**
- `src/sleap_roots_analyze/data_utils.py` - Add `custom_replacements` parameter
- `tests/test_data_utils.py` - Add tests for custom replacement functionality
- `docs/TRAIT_NAME_SANITIZATION.md` - Update documentation with examples

**Backward compatibility:**
- ✅ Fully backward compatible (parameter is optional with default `None`)
- No changes required to existing code
