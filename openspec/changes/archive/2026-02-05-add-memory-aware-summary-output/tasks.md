# Implementation Tasks

## 1. TDD: Write Tests First (Red Phase)

### 1.1 Test Image Size Calculation
- [x] 1.1.1 Test: `test_calculate_total_image_size_empty` - returns 0 for no images
- [x] 1.1.2 Test: `test_calculate_total_image_size_single_image` - accurate size for one image
- [x] 1.1.3 Test: `test_calculate_total_image_size_multiple_images` - sums all image sizes
- [x] 1.1.4 Test: `test_calculate_total_image_size_missing_file` - handles missing files gracefully

### 1.2 Test Image Mode Selection
- [x] 1.2.1 Test: `test_file_path_mode_uses_relative_paths` - markdown contains relative paths
- [x] 1.2.2 Test: `test_embed_mode_under_threshold_embeds` - images embedded when under 10MB
- [x] 1.2.3 Test: `test_embed_mode_over_threshold_warns_and_fallback` - warning + file_path when > 10MB
- [x] 1.2.4 Test: `test_auto_mode_under_threshold_embeds` - auto embeds when small
- [x] 1.2.5 Test: `test_auto_mode_over_threshold_uses_file_path` - auto uses paths when large
- [x] 1.2.6 Test: `test_threshold_configurable` - custom threshold respected

### 1.3 Test Output Format Options
- [x] 1.3.1 Test: `test_to_html_generates_valid_html` - .html output generated
- [x] 1.3.2 Test: `test_html_output_renders_tables` - HTML contains properly formatted tables
- [x] 1.3.3 Test: `test_html_output_renders_images` - HTML img tags point to correct paths
- [x] 1.3.4 Test: `test_html_includes_styling` - HTML has embedded CSS for presentation

### 1.4 Test File Path Reference Correctness
- [x] 1.4.1 Test: `test_file_path_references_are_relative` - paths work from SUMMARY.md location
- [x] 1.4.2 Test: `test_file_path_handles_special_characters` - works for traits with spaces/slashes

### 1.5 Test Summary File Size
- [x] 1.5.1 Test: `test_file_path_mode_produces_small_file` - no embedded images = small file
- [x] 1.5.2 Test: `test_embed_mode_can_produce_large_file` - embedded images increase file size

### 1.6 Run Tests (Red Phase)
- [x] 1.6.1 Run `uv run pytest tests/test_cross_platform_summary.py -v -k image` - confirmed 18 failures
- [x] 1.6.2 All tests failed due to missing functions (expected)

## 2. Implementation (Green Phase)

### 2.1 Add Size Calculation Utilities
- [x] 2.1.1 Add `_calculate_total_image_size(paths: List[Path]) -> int` to cross_platform_summary.py
- [x] 2.1.2 Add `DEFAULT_EMBED_THRESHOLD_BYTES = 10 * 1024 * 1024` constant

### 2.2 Update CrossPlatformSummary.to_markdown()
- [x] 2.2.1 Add `image_mode: Literal["file_path", "embed", "auto"]` parameter
- [x] 2.2.2 Add `embed_threshold_bytes: int` parameter (default 10MB)
- [x] 2.2.3 Implement auto mode logic: calculate total size, choose mode
- [x] 2.2.4 Add warning logging when embed mode exceeds threshold
- [x] 2.2.5 Update all image references to use `should_embed` variable

### 2.3 Add HTML Output Generation
- [x] 2.3.1 Add `to_html()` method to CrossPlatformSummary
- [x] 2.3.2 Include embedded CSS for styling (tables, code blocks, images)
- [x] 2.3.3 Add `_markdown_to_html()` helper method
- [x] 2.3.4 Add `_table_to_html()` helper method
- [x] 2.3.5 Add `_process_inline_markdown()` helper method

### 2.4 Update Pipeline Runner
- [x] 2.4.1 Change hardcoded `embed_images=True` to `image_mode="file_path"`

### 2.5 Run Tests (Green Phase)
- [x] 2.5.1 Run `uv run pytest tests/test_cross_platform_summary.py -v` - 56 passed
- [x] 2.5.2 All new tests pass, no regressions

## 3. Update Scripts and Commands

### 3.1 Update convert_summary_to_html.py Script
- [x] 3.1.1 Add command-line argument support
- [x] 3.1.2 Add `--latest` flag to find most recent run
- [x] 3.1.3 Use `CrossPlatformSummaryGenerator.to_html()` when available
- [x] 3.1.4 Add fallback markdown conversion for compatibility

## 4. Documentation

- [x] 4.1 Update `docs/CROSS_PLATFORM_ANALYSIS.md` with output format options
- [x] 4.2 Document memory limits (10MB threshold)
- [x] 4.3 Add HTML conversion section with usage examples

## 5. Final Validation

- [x] 5.1 Run `uv run pytest tests/test_cross_platform_summary.py -v` - 56 passed
- [x] 5.2 Run integration test: `uv run sleap-roots-analyze run-all --cross-only` - 4/4 succeeded
- [x] 5.3 Verify SUMMARY.md is < 1MB with file_path mode - 28KB (was 77MB)
- [x] 5.4 Verify HTML output works - 24.8KB HTML generated
- [x] 5.5 Run linting: `uv run ruff check` and `uv run black` - passed
- [x] 5.6 Run `/pre-merge-check` to ensure CI passes
