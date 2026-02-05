# Implementation Tasks

## 1. TDD: Write Tests First (Red Phase)

### 1.1 FDR Method Tests
- [x] 1.1.1 Test: `test_methods_section_uses_config_fdr_method` - verify FDR method from config
- [x] 1.1.2 Test: `test_fdr_method_mapping_bh` - "fdr_bh" → "Benjamini-Hochberg"
- [x] 1.1.3 Test: `test_fdr_method_mapping_by` - "fdr_by" → "Benjamini-Yekutieli"

### 1.2 Image Embedding Tests
- [x] 1.2.1 Test: `test_images_embedded_as_base64` - verify PNG converted to data URI
- [x] 1.2.2 Test: `test_missing_image_handled_gracefully` - no crash if image missing
- [x] 1.2.3 Test: `test_base64_image_valid_format` - verify `data:image/png;base64,` prefix

### 1.3 Variable Definition Tests
- [x] 1.3.1 Test: `test_table_headers_include_definitions` - verify inline definitions
- [x] 1.3.2 Test: `test_legend_includes_all_abbreviations` - verify legend text

### 1.4 Confidence Interval Tests
- [x] 1.4.1 Test: `test_correlation_table_includes_ci` - verify CI in output
- [x] 1.4.2 Test: `test_ci_format_brackets` - verify format like "[-0.85, -0.26]"
- [x] 1.4.3 Test: `test_missing_ci_handled` - graceful handling if CI columns missing

### 1.5 Power Analysis Tests
- [x] 1.5.1 Test: `test_power_section_includes_parameters` - α, n displayed
- [x] 1.5.2 Test: `test_power_section_includes_mde` - minimum detectable effect shown
- [x] 1.5.3 Test: `test_power_warning_when_underpowered` - warning at >50% below 80%
- [x] 1.5.4 Test: `test_power_section_includes_sample_size_recommendation` - n for target power
- [x] 1.5.5 Test: `test_no_power_warning_when_adequately_powered` - no warning if mostly powered
- [x] 1.5.6 Test: `test_power_stats_uses_config_significance_level` - alpha from config
- [x] 1.5.7 Test: `test_power_stats_defaults_alpha_when_not_in_config` - default alpha

### 1.6 FDR=0 Interpretation Tests
- [x] 1.6.1 Test: `test_fdr_zero_shows_interpretation` - interpretation section appears
- [x] 1.6.2 Test: `test_fdr_zero_shows_nominal_count` - shows raw p<0.05 count
- [x] 1.6.3 Test: `test_fdr_zero_shows_recommendations` - sample size recommendation
- [x] 1.6.4 Test: `test_fdr_nonzero_no_interpretation` - no extra section when FDR>0

### 1.7 Run tests, confirm failures (Red phase complete)
- [x] 1.7.1 Run `uv run pytest tests/test_cross_platform_summary.py -v` - all new tests fail

## 2. Implementation (Green Phase)

### 2.1 Data Classes Updates
- [x] 2.1.1 Add `ci_low: Optional[float]` and `ci_high: Optional[float]` to `TopCorrelation`
- [x] 2.1.2 Add `alpha: float` and `n_genotypes_modal: int` to `PowerStats`
- [x] 2.1.3 Add `minimum_detectable_r: float` to `PowerStats`
- [x] 2.1.4 Add `recommended_n_for_r40: int` to `PowerStats`

### 2.2 Image Embedding
- [x] 2.2.1 Add `_embed_image_base64(self, image_path: Path) -> str` method
- [x] 2.2.2 Update `to_markdown()` to use embedded images instead of relative paths
- [x] 2.2.3 Handle missing images gracefully (skip embedding)

### 2.3 Table Formatting
- [x] 2.3.1 Update correlation table headers with inline definitions
- [x] 2.3.2 Add legend line below table
- [x] 2.3.3 Format CI as `r [ci_low, ci_high]` in single column

### 2.4 Power Analysis Section
- [x] 2.4.1 Add `_calculate_minimum_detectable_r(n: int, alpha: float, power: float) -> float`
- [x] 2.4.2 Add `_calculate_required_n(r: float, alpha: float, power: float) -> int`
- [x] 2.4.3 Update `_calculate_power_stats` to include MDE and required n (reads alpha from config)
- [x] 2.4.4 Update `to_markdown()` to show power parameters table
- [x] 2.4.5 Add warning logic when pct_above_80 < 50
- [x] 2.4.6 Add TARGET_POWER and DEFAULT_ALPHA constants (no hardcoded values)

### 2.5 FDR Interpretation Section
- [x] 2.5.1 Add `_format_fdr_interpretation(stats: CorrelationStats) -> List[str]`
- [x] 2.5.2 Include nominal significant count in interpretation
- [x] 2.5.3 Include sample size recommendation
- [x] 2.5.4 Update `to_markdown()` to include interpretation when fdr_significant=0

### 2.6 FDR Method Fix in pipeline_runner.py
- [x] 2.6.1 Add `_get_cross_platform_fdr_method() -> str` method
- [x] 2.6.2 Add FDR method mapping dict: {"fdr_bh": "Benjamini-Hochberg", "fdr_by": "Benjamini-Yekutieli"}
- [x] 2.6.3 Update `_format_methods_section()` to use dynamic FDR method

### 2.7 Run tests, confirm pass (Green phase complete)
- [x] 2.7.1 Run `uv run pytest tests/test_cross_platform_summary.py -v` - all tests pass

## 3. Integration Testing

- [x] 3.1 Run full pipeline: `uv run sleap-roots-analyze run-all --cross-only`
- [x] 3.2 Verify SUMMARY.md images render (base64 embedded, 599KB file)
- [x] 3.3 Verify power warnings appear for underpowered analyses
- [x] 3.4 Verify FDR interpretation appears when no FDR-significant correlations
- [x] 3.5 Verify Methods section shows correct FDR method name (Benjamini-Hochberg)

## 4. Code Quality

- [x] 4.1 Run `uv run ruff check src/sleap_roots_analyze/summary/`
- [x] 4.2 Run `uv run black src/sleap_roots_analyze/summary/`
- [x] 4.3 Ensure all new functions have docstrings
- [x] 4.4 Run full test suite: 87 tests pass

## 5. Final Validation

- [x] 5.1 Verify all acceptance criteria met
- [x] 5.2 Run integration test with pipeline output
- [x] 5.3 Manual review of generated SUMMARY.md for scientific accuracy
