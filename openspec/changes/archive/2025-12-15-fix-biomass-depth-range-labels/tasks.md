# Implementation Tasks

## 1. Test-Driven Development Setup
- [x] 1.1 Create test fixtures for biomass columns with depth ranges
  - [x] 1.1.1 Add fixture: `RootDW_15cm`, `RootDW_45cm` (midpoint notation)
  - [x] 1.1.2 Add fixture: `c_0_30`, `c_30_60` (range notation - notebook style)
  - [x] 1.1.3 Add fixture: Mixed depth formats in same DataFrame
- [x] 1.2 Write failing tests for depth range detection
  - [x] 1.2.1 Test `_detect_depth_pattern()` helper function
  - [x] 1.2.2 Test pattern recognition: `_Ncm` suffix detection
  - [x] 1.2.3 Test extraction of numeric depth from column names
- [x] 1.3 Write failing tests for depth range sanitization
  - [x] 1.3.1 Test `RootDW_15cm` → `Root Biomass DW (g) 0-30cm`
  - [x] 1.3.2 Test `RootDW_45cm` → `Root Biomass DW (g) 30-60cm`
  - [x] 1.3.3 Test `RootCount_0cm` → `Root Count 0cm` (single depth)
  - [x] 1.3.4 Test `RootCount_5cm` → `Root Count 5cm`
- [x] 1.4 Write tests for depth range mapping parameter
  - [x] 1.4.1 Test passing `depth_range_mapping` dict to sanitize function
  - [x] 1.4.2 Test lookup: 15 → "0-30cm", 45 → "30-60cm"
  - [x] 1.4.3 Test fallback: unknown depths use original notation
- [x] 1.5 Write backward compatibility tests
  - [x] 1.5.1 Test: Non-biomass columns unchanged (e.g., `Median.Number.of.Roots`)
  - [x] 1.5.2 Test: Function works without `depth_range_mapping` parameter
  - [x] 1.5.3 Test: All existing test cases still pass

## 2. Core Implementation - Data Sanitization
- [x] 2.1 Add depth range mapping parameter to `sanitize_trait_names()`
  - [x] 2.1.1 Add `depth_range_mapping: Optional[Dict[float, str]] = None` parameter
  - [x] 2.1.2 Update function signature and docstring
  - [x] 2.1.3 Document parameter format: `{15.0: "0-30", 45.0: "30-60"}`
- [x] 2.2 Implement depth pattern detection helper
  - [x] 2.2.1 Create `_detect_depth_suffix(col_name: str) -> Optional[float]`
  - [x] 2.2.2 Regex pattern: `r"_(\d+)cm$"` to extract depth
  - [x] 2.2.3 Return None if no depth pattern found
- [x] 2.3 Implement depth range formatting logic
  - [x] 2.3.1 Create `_format_depth_range(depth: float, mapping: Dict) -> str`
  - [x] 2.3.2 Lookup depth in mapping → return range string (e.g., "0-30cm")
  - [x] 2.3.3 If not in mapping, return original (e.g., "15cm")
- [x] 2.4 Integrate depth-aware logic into sanitization loop
  - [x] 2.4.1 After unit conversion, check for depth suffix
  - [x] 2.4.2 If depth found and mapping provided, apply range formatting
  - [x] 2.4.3 Handle both biomass (`RootDW`) and counting (`RootCount`) prefixes
- [x] 2.5 Update trait name construction
  - [x] 2.5.1 For depth ranges: `Root Biomass DW (g) 0-30cm` format
  - [x] 2.5.2 For single depths: `Root Count 5cm` format
  - [x] 2.5.3 Preserve unit notation: `(g)` for biomass

## 3. Pipeline Integration
- [x] 3.1 Update config schema to include depth range mapping
  - [x] 3.1.1 Add `depth_range_mapping` field to `RootCoreSource` class
  - [x] 3.1.2 Make it optional: `depth_range_mapping: Optional[Dict[str, float]]`
  - [x] 3.1.3 Document format in config components docstring
- [x] 3.2 Pass depth mapping through pipeline steps
  - [x] 3.2.1 Store depth mapping in `ReshapeForTraitQCStep` metadata
  - [x] 3.2.2 Pass to `CleanupTraitsStep` via step result
  - [x] 3.2.3 Ensure mapping available when calling `sanitize_trait_names()`
- [x] 3.3 Update reshape step to include range metadata
  - [x] 3.3.1 Build reverse mapping: midpoint → range (15.0 → "0-30")
  - [x] 3.3.2 Include in output metadata JSON
  - [x] 3.3.3 Document metadata structure

## 4. Configuration Updates
- [x] 4.1 Update example config with depth range mapping
  - [x] 4.1.1 Edit `configs/qc_root_core_edpie.yaml`
  - [x] 4.1.2 Add `depth_range_mapping` alongside existing `depth_mapping`
  - [x] 4.1.3 Document both fields in comments
- [x] 4.2 Create config template for depth ranges
  - [x] 4.2.1 Add to `configs/templates/` if templates exist
  - [x] 4.2.2 Show best practices for biomass depth notation

## 5. Testing and Validation
- [x] 5.1 Run all new tests and verify they pass
  - [x] 5.1.1 `uv run pytest tests/test_data_utils.py -k depth_range -v`
  - [x] 5.1.2 Ensure 100% coverage for new depth range code
- [x] 5.2 Run existing test suite for regressions
  - [x] 5.2.1 `uv run pytest tests/test_data_utils.py -v`
  - [x] 5.2.2 `uv run pytest tests/test_step_cleanup_traits.py -v`
  - [x] 5.2.3 Confirm all 779+ tests still pass
- [x] 5.3 Integration test with full QC pipeline
  - [x] 5.3.1 Run pipeline with `configs/qc_root_core_edpie.yaml`
  - [x] 5.3.2 Verify output CSV has proper depth range labels
  - [x] 5.3.3 Check visualization plots use correct labels

## 6. Documentation
- [x] 6.1 Update function docstrings
  - [x] 6.1.1 Document `depth_range_mapping` parameter with examples
  - [x] 6.1.2 Add usage examples to `sanitize_trait_names()` docstring
- [x] 6.2 Update CLAUDE.md project guidelines
  - [x] 6.2.1 Document depth range labeling conventions
  - [x] 6.2.2 Add examples of proper biomass column naming
- [x] 6.3 Update config documentation
  - [x] 6.3.1 Explain difference between `depth_mapping` and `depth_range_mapping`
  - [x] 6.3.2 Show when to use each

## 7. Code Review and Cleanup
- [x] 7.1 Run linting and formatting
  - [x] 7.1.1 `uv run black src/sleap_roots_analyze/data_utils.py`
  - [x] 7.1.2 `uv run ruff check src/sleap_roots_analyze/`
- [x] 7.2 Review test coverage
  - [x] 7.2.1 `uv run pytest --cov=sleap_roots_analyze.data_utils --cov-branch`
  - [x] 7.2.2 Ensure >95% coverage for modified code
- [x] 7.3 Manual verification
  - [x] 7.3.1 Generate sample visualization with new labels
  - [x] 7.3.2 Confirm labels are scientifically accurate and clear