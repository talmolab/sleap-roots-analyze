# Implementation Tasks

## 1. Implementation
- [ ] 1.1 Add `custom_replacements: Optional[Dict[str, str]] = None` parameter to `sanitize_trait_names()` signature
- [ ] 1.2 Implement case-insensitive custom replacement logic before standard sanitization
- [ ] 1.3 Update function docstring with custom_replacements parameter documentation and examples

## 2. Testing
- [ ] 2.1 Add test for basic custom replacement (e.g., "crown" → "seminal")
- [ ] 2.2 Add test for case-insensitive matching (e.g., "Crown", "CROWN", "crown" all match)
- [ ] 2.3 Add test for multiple custom replacements in one call
- [ ] 2.4 Add test that custom replacements work with standard sanitization (units, abbreviations)
- [ ] 2.5 Add test that None/empty dict preserves existing behavior
- [ ] 2.6 Add test for custom replacement in trait name with dots/underscores

## 3. Documentation
- [ ] 3.1 Update `docs/TRAIT_NAME_SANITIZATION.md` with custom replacements section
- [ ] 3.2 Add example showing crown→seminal replacement
- [ ] 3.3 Add note about case-insensitive matching behavior
- [ ] 3.4 Add example combining custom replacements with metadata sanitization

## 4. Validation
- [ ] 4.1 Run all existing tests to ensure backward compatibility
- [ ] 4.2 Run new tests for custom replacements
- [ ] 4.3 Verify coverage remains >95% for data_utils.py
