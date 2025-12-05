# Implementation Tasks

## Phase 1: Update Config Schema

- [ ] 1.1 Add `custom_replacements: Optional[Dict[str, str]] = None` to `CleanupConfig` in `src/sleap_roots_analyze/pipeline/config/components.py`
  - Add to dataclass attributes
  - Add docstring explaining format and usage
  - Include example: `{"crown": "seminal"}` for wheat terminology

## Phase 2: Integrate into Pipeline Step

- [ ] 2.1 Update `CleanupTraitsStep.execute()` in `src/sleap_roots_analyze/pipeline/steps/cleanup_traits.py`
  - Call `sanitize_trait_names()` after getting trait columns from previous step
  - Pass parameters: `abbreviate=True`, `return_mapping=True`, `custom_replacements=config.cleanup.custom_replacements`
  - Also pass metadata column names: `genotype_col`, `replicate_col`, `barcode_col` from config
  - Update trait_cols list with sanitized names
  - Store trait name mapping in step metadata (only if names changed)

- [ ] 2.2 Update column references after sanitization
  - Update `GENOTYPE_COL`, `REPLICATE_COL`, `BARCODE_COL` to standardized names ("Genotype", "Replicate", "Barcode")
  - Ensure downstream code uses sanitized column names

## Phase 3: Testing

- [ ] 3.1 Add test to `tests/test_step_cleanup.py`: `test_cleanup_step_with_custom_replacements`
  - Create test config with custom_replacements: {"test": "replaced"}
  - Create test data with trait names containing "test" word
  - Run CleanupTraitsStep
  - Verify trait names were transformed
  - Verify metadata contains trait_name_mapping

- [ ] 3.2 Add test: `test_cleanup_step_without_custom_replacements`
  - Create test config with custom_replacements=None
  - Verify step still works (backward compatibility)
  - Verify standard sanitization still happens (abbreviation, etc.)

- [ ] 3.3 Run full test suite to ensure no regressions
  - `uv run pytest tests/` - should pass all 1109+ tests

## Phase 4: Validation

- [ ] 4.1 Test config validation accepts custom_replacements
  - Create temporary test config with custom_replacements
  - Run `sleap-roots-analyze config validate <test-config>`
  - Verify no validation errors

- [ ] 4.2 Test invalid custom_replacements formats
  - Test with non-dict value (should fail gracefully)
  - Test with non-string keys/values (should fail gracefully)

## Dependencies

- All tasks in Phase 1 must complete before Phase 2
- Phase 2 must complete before Phase 3
- Phase 3 must complete before Phase 4
- This entire proposal must complete before `add-multi-platform-configs` can create Cylinder config

## Notes

- This change is purely additive - no breaking changes
- Existing configs without custom_replacements will work unchanged
- The Cylinder config (separate proposal) will be the first to use this feature
