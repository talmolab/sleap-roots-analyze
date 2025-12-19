# Eliminate Dangerous Configuration Defaults

**Status**: PROPOSED
**Created**: 2025-12-01
**Type**: Configuration Enhancement (Validation with Warnings)

## Why

Configuration defaults can cause silent failures and incorrect results without user awareness. Analysis of the codebase revealed multiple critical parameters that should be explicitly configured:

### Critical Issues Identified:

1. **Cleanup thresholds affect data entry** - Defaults like `max_nan_fraction: 0.25` or `max_zeros_per_trait: 0.5` silently remove data without user awareness
2. **Heritability threshold inconsistency** - Default 0.60 may not match user intent; filtering can remove traits silently
3. **Root core aggregation method** - Default "mean" instead of recommended "median" affects trait calculation from cores
4. **PCA variance threshold** - Default 0.95 affects dimensionality reduction and downstream outlier detection
5. **Column mappings are dataset-specific** - Column names like `genotype` and `replicate` vary across experiments

### User Impact:

- **Silent data removal**: Cleanup thresholds remove samples/traits without explicit user choice
- **Unexpected trait filtering**: Heritability defaults may not match analysis goals
- **Statistical bias**: Mean vs median aggregation affects robustness to outliers
- **Reproducibility issues**: Different defaults in different contexts lead to inconsistent results
- **Missing important steps**: Users may not realize outlier detection is disabled

### What's Acceptable:

**Outlier detection is OPTIONAL** - Users may only want data cleanup (NaN/zero removal) without outlier detection. Empty `traditional_methods` and `clustering_methods` is valid.

However, users should be **warned** when important-but-optional features are disabled so they make conscious choices.

## What Changes

### Phase 1: Add Validation with Warnings (Non-Breaking)

Add validation that checks for both **required** parameters (errors) and **important-but-optional** parameters (warnings).

#### 1.1 Add Validation to Pipeline Execution

**File**: `src/sleap_roots_analyze/pipeline/qc_pipeline.py`

**Add validation function**:
```python
def validate_explicit_config(config: QCPipelineConfig) -> None:
    """Validate configuration and warn about important missing options.

    Raises:
        ValueError: If required parameters are missing

    Warns:
        UserWarning: If important-but-optional parameters are not set
    """
    import warnings

    errors = []
    warnings_list = []

    # REQUIRED: Cleanup thresholds (affect data entry)
    if config.cleanup.max_nan_fraction is None:
        errors.append(
            "cleanup.max_nan_fraction must be explicitly set\n"
            "  Recommended: 0.25 (removes samples with >25% missing data)\n"
            "  Range: 0.0-1.0 (lower = stricter)"
        )
    if config.cleanup.max_zeros_per_trait is None:
        errors.append(
            "cleanup.max_zeros_per_trait must be explicitly set\n"
            "  Recommended: 0.5 (removes traits with >50% zeros)\n"
            "  Range: 0.0-1.0 (lower = stricter)"
        )
    if config.cleanup.low_variance_threshold is None:
        errors.append(
            "cleanup.low_variance_threshold must be explicitly set\n"
            "  Recommended: 1e-10 (removes near-constant traits)\n"
            "  Typical: 1e-10 to 1e-6"
        )

    # REQUIRED: Column mappings (dataset-specific)
    if config.columns.genotype is None:
        errors.append(
            "columns.genotype must be explicitly set (dataset-specific)\n"
            "  Examples: 'geno', 'genotype', 'accession', 'salk_geno'"
        )
    if config.columns.replicate is None:
        errors.append(
            "columns.replicate must be explicitly set (dataset-specific)\n"
            "  Examples: 'rep', 'replicate', 'Rep', 'block'"
        )

    # REQUIRED: PCA variance threshold (affects dimensionality)
    if config.pca.variance_threshold is None:
        errors.append(
            "pca.variance_threshold must be explicitly set\n"
            "  Recommended: 0.95 (retain components explaining 95% variance)\n"
            "  Range: 0.90-0.99 (higher = more components retained)"
        )

    # REQUIRED: Heritability threshold (if filtering enabled)
    if config.heritability.enabled and config.heritability.threshold is None:
        errors.append(
            "heritability.threshold must be explicitly set when filtering enabled\n"
            "  Typical range: 0.3-0.6\n"
            "  Higher = more stringent (fewer traits retained)"
        )

    # REQUIRED: Root core aggregation method (if root_core processing enabled)
    if config.root_core is not None:
        for i, source in enumerate(config.root_core.sources):
            if source.aggregation_method is None:
                errors.append(
                    f"root_core.sources[{i}].aggregation_method must be explicitly set\n"
                    f"  Recommended: 'median' (robust to outliers and typos)\n"
                    f"  Alternative: 'mean' (if no outliers expected)"
                )

    # REQUIRED: Outlier removal strategy (if outlier detection configured)
    has_outlier_detection = (
        len(config.outlier_detection.traditional_methods) > 0 or
        len(config.outlier_detection.clustering_methods) > 0
    )
    if has_outlier_detection and config.outlier_removal.strategy is None:
        errors.append(
            "outlier_removal.strategy must be set when outlier detection enabled\n"
            "  Options:\n"
            "    - 'remove': Delete outlier samples from dataset\n"
            "    - 'flag': Add outlier_* columns but keep samples\n"
            "    - 'none': Detect but don't remove or flag"
        )

    # IMPORTANT BUT OPTIONAL: Outlier detection
    if not has_outlier_detection:
        warnings_list.append(
            "No outlier detection methods configured (traditional_methods and clustering_methods are empty).\n"
            "  This is valid if you only want data cleanup (NaN/zero removal).\n"
            "  Consider adding outlier detection for robust QC:\n"
            "    - traditional_methods: ['mahalanobis_pca', 'isolation_forest']\n"
            "    - clustering_methods: ['dbscan']\n"
            "  See configs/templates/ for examples."
        )

    # Raise errors if any required parameters missing
    if errors:
        error_msg = (
            "Configuration Validation Failed\n"
            "================================================================\n"
            "Critical parameters must be explicitly set to avoid silent failures.\n\n"
        )
        error_msg += "\n\n".join(errors)
        error_msg += (
            "\n\n================================================================\n"
            "See configuration templates in configs/templates/:\n"
            "   - qc_cleanup_only_template.yaml (data cleanup only)\n"
            "   - qc_full_pipeline_template.yaml (cleanup + outlier detection)\n"
        )
        raise ValueError(error_msg)

    # Issue warnings for important-but-optional parameters
    for warning_msg in warnings_list:
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
```

**Call validation at pipeline start** (in `run_qc_pipeline()`):
```python
def run_qc_pipeline(config: QCPipelineConfig, output_dir: Path) -> Dict[str, Any]:
    """Run complete QC pipeline.

    Args:
        config: QC pipeline configuration
        output_dir: Directory for pipeline outputs

    Returns:
        Dictionary with pipeline results and metadata

    Raises:
        ValueError: If configuration is invalid or incomplete
    """
    # Validate explicit configuration (errors for required, warnings for important-but-optional)
    validate_explicit_config(config)

    # Rest of pipeline execution...
```

#### 1.2 Change Root Core Aggregation Default to Median

**File**: `src/sleap_roots_analyze/pipeline/config/components.py`

**Current code** (line 565):
```python
aggregation_method: str = "mean"
```

**Fix** - Change default to recommended value:
```python
aggregation_method: str = "median"  # Robust to outliers and measurement errors
```

**Rationale**: Median is statistically robust to outliers, typos, and miscounts. The existing config file (`qc_root_core_edpie.yaml`) already documents this and uses median. Changing the default aligns with best practices.

#### 1.3 Update Configuration Dataclass Docstrings

Add clear guidance about which parameters need explicit setting:

**File**: `src/sleap_roots_analyze/pipeline/config/components.py`

```python
@dataclass
class CleanupConfig:
    """Data cleanup configuration.

    IMPORTANT: All threshold parameters should be explicitly set in your config
    to avoid silent data removal. Default values are provided for convenience but
    may not match your analysis requirements.

    Attributes:
        max_nan_fraction: Max fraction of NaN values per sample (0.0-1.0).
            Samples exceeding this will be removed. Recommended: 0.25
        max_zeros_per_trait: Max fraction of zero values per trait (0.0-1.0).
            Traits exceeding this will be removed. Recommended: 0.5
        low_variance_threshold: Minimum variance for trait inclusion.
            Near-constant traits will be removed. Recommended: 1e-10
    """
    # Defaults kept for convenience, but validation warns if not explicitly set
    max_nan_fraction: float = 0.25
    max_zeros_per_trait: float = 0.5
    low_variance_threshold: float = 1e-10
    # ... rest of config

@dataclass
class HeritabilityConfig:
    """Heritability filtering configuration.

    IMPORTANT: If filtering is enabled, threshold should be explicitly set
    to match your scientific objectives.

    Attributes:
        enabled: Whether to filter traits by heritability.
        threshold: Minimum heritability (H²) for trait retention (0.0-1.0).
            Typical range: 0.3 (permissive) to 0.6 (stringent). Must be set if enabled.
    """
    enabled: bool = False
    threshold: float = 0.60  # Default kept but validation requires explicit setting if enabled
    # ... rest of config

@dataclass
class RootCoreSourceConfig:
    """Configuration for a single root core data source.

    IMPORTANT: aggregation_method should be explicitly set to match your
    statistical requirements.

    Attributes:
        aggregation_method: Method for aggregating cores ("mean" or "median").
            Recommended: "median" (robust to outliers, typos, measurement errors).
            Use "mean" only if you're confident data has no outliers.
    """
    # ... other fields
    aggregation_method: str = "median"  # CHANGED: Now defaults to recommended value
    # ... rest of config
```

### Phase 2: Document Existing Configs

Review all existing configs to ensure they follow best practices.

**File**: `docs/configuration_review.md` (NEW)

```markdown
# Configuration Review

Review of all existing configuration files for compliance with explicit configuration requirements.

## Status: 2025-12-01

### QC Pipeline Configs

#### qc_turface_150genotypes.yaml - COMPLIANT
- **Status**: Ready for use, exemplary configuration
- All required parameters explicitly set
- Column names: `genotype: "geno"`, `replicate: "rep"`
- Cleanup thresholds: `max_nan_fraction: 0.0`, `max_zeros_per_trait: 0.5`
- PCA variance: `variance_threshold: 0.95`
- Heritability: enabled with `threshold: 0.40`
- Outlier detection: Mahalanobis configured
- Outlier removal: `strategy: "single"`, `method: "mahalanobis"`
- Adaptive sizing enabled with customized parameters

#### qc_mahalanobis.yaml - NEEDS DATA PATH
- **Status**: Template - requires user to set data path
- **Issue**: `csv_path: ???` - User must set before running
- All other required parameters set correctly
- Fast single-method outlier detection (Mahalanobis only)
- **Use case**: Quick QC runs, prototyping

#### qc_root_core_edpie.yaml - COMPLIANT (EXEMPLARY)
- **Status**: Exemplary config with excellent documentation
- **Highlights**:
  - Documents rationale for aggregation method choice (lines 24, 35-36)
  - Explains why median is preferred over mean
  - Shows best practice for core-level QC (disable it, use median instead)
- Root core aggregation: `aggregation_method: "median"` (both sources)
- Demonstrates proper documentation of parameter choices
- **Recommendation**: Use as example for documenting config decisions

#### qc_consensus_6method.yaml - COMPLIANT
- **Status**: Ready for use
- Uses 6 outlier detection methods for maximum robustness
- Consensus strategy: minimum 3/6 methods must agree
- **Use case**: Production analyses requiring high confidence

#### qc_clustering_strict.yaml - COMPLIANT
- **Status**: Ready for use
- Uses only clustering methods (DBSCAN, HDBSCAN)
- Stringent cleanup thresholds
- **Use case**: Datasets with complex multimodal distributions

#### qc_permissive.yaml - COMPLIANT
- **Status**: Ready for use
- Permissive thresholds for less strict QC
- **Use case**: Exploratory analysis, preserving borderline samples

### Summary

- **6/6 QC configs are compliant** with explicit configuration requirements
- **1 config is template-style**: `qc_mahalanobis.yaml` requires user to set `csv_path`
- **1 exemplary config**: `qc_root_core_edpie.yaml` demonstrates documentation best practices
- **No changes needed** to existing configs

### Recommendations

1. **Keep existing configs unchanged** - They follow best practices
2. **Use as examples** for new users:
   - `qc_turface_150genotypes.yaml` - Full pipeline example
   - `qc_mahalanobis.yaml` - Fast single-method template
   - `qc_root_core_edpie.yaml` - Root core processing with documentation
3. **Create templates directory** with minimal examples for new users
```

### Phase 3: Create Configuration Templates

Create minimal templates for common use cases.

**File**: `configs/templates/README.md` (NEW)

```markdown
# QC Pipeline Configuration Templates

Template configuration files to help you get started with the QC pipeline.

## Quick Start

1. **Choose a template**:
   - `qc_cleanup_only_template.yaml` - Data cleanup without outlier detection
   - `qc_full_pipeline_template.yaml` - Full QC with outlier detection

2. **Copy and customize**:
   ```bash
   cp configs/templates/qc_full_pipeline_template.yaml configs/my_analysis.yaml
   # Edit my_analysis.yaml with your dataset-specific values
   ```

3. **Run the pipeline**:
   ```bash
   sleap-roots-analyze qc configs/my_analysis.yaml
   ```

## Required Parameters

You MUST set these parameters in your config:

- `columns.genotype` - Your genotype column name (e.g., "geno", "accession")
- `columns.replicate` - Your replicate column name (e.g., "rep", "block")
- `data.csv_path` - Path to your trait CSV file
- `cleanup.max_nan_fraction` - Max NaN per sample (typical: 0.25)
- `cleanup.max_zeros_per_trait` - Max zeros per trait (typical: 0.5)
- `cleanup.low_variance_threshold` - Min trait variance (typical: 1e-10)
- `pca.variance_threshold` - PCA variance threshold (typical: 0.95)

## Conditionally Required

- `heritability.threshold` - Required if `heritability.enabled: true` (typical: 0.3-0.6)
- `outlier_removal.strategy` - Required if outlier detection enabled
- `root_core.sources[].aggregation_method` - Required if processing cores (use "median")

## Optional But Important

- `outlier_detection.traditional_methods` - Can be empty for cleanup-only pipeline
  - **WARNING**: You will be warned if this is empty (to ensure conscious choice)

## Examples

The `configs/` directory contains real-world examples:
- `qc_turface_150genotypes.yaml` - Full QC for 150 genotype dataset
- `qc_mahalanobis.yaml` - Fast single-method outlier detection
- `qc_root_core_edpie.yaml` - Root core processing (excellent documentation)
```

**Files**: `configs/templates/qc_cleanup_only_template.yaml` and `qc_full_pipeline_template.yaml`

(Similar to original proposal, but more concise - focus on required fields with comments)

### Phase 4: Update Documentation

**File**: `CLAUDE.md` - Add configuration philosophy section

```markdown
## Configuration Philosophy

### Explicit Over Implicit

The QC pipeline requires explicit configuration for critical parameters to prevent silent failures.

**Approach**: Defaults are kept for convenience, but validation ensures you're aware of values being used.

### Required Parameters

These parameters MUST be explicitly set (validation error):

- `cleanup.max_nan_fraction` - Controls sample retention
- `cleanup.max_zeros_per_trait` - Controls trait retention  
- `cleanup.low_variance_threshold` - Removes near-constant traits
- `pca.variance_threshold` - Affects dimensionality reduction
- `columns.genotype`, `columns.replicate` - Dataset-specific
- `heritability.threshold` - If heritability filtering enabled
- `root_core.sources[].aggregation_method` - If processing cores (use "median")
- `outlier_removal.strategy` - If outlier detection configured

### Important-But-Optional Parameters

These parameters trigger warnings if not set (to ensure conscious choice):

- `outlier_detection.traditional_methods` - Can be empty for cleanup-only pipeline
- `outlier_detection.clustering_methods` - Can be empty for cleanup-only pipeline

**Why warn?** Ensures users make conscious decisions about important features.

### Root Core Aggregation Default Changed

**Default changed from "mean" to "median"** because:

- Robust to outlier cores from typos, miscounts, measurement errors
- Recommended for small sample sizes (typically 3 cores per plot)
- Existing configs already use median (e.g., `qc_root_core_edpie.yaml`)
- Statistical best practice when core-level N is too small for outlier detection

### Configuration Templates

Use templates as starting points:
- `configs/templates/qc_cleanup_only_template.yaml`
- `configs/templates/qc_full_pipeline_template.yaml`

See `configs/templates/README.md` for detailed guidance.
```

## Impact

### Files Modified/Created: 8-9 files

**Modified**:
1. `src/sleap_roots_analyze/pipeline/qc_pipeline.py` - Add validation with warnings
2. `src/sleap_roots_analyze/pipeline/config/components.py` - Update docstrings, change aggregation default
3. `CLAUDE.md` - Add configuration philosophy section

**Created**:
4. `configs/templates/qc_cleanup_only_template.yaml`
5. `configs/templates/qc_full_pipeline_template.yaml`
6. `configs/templates/README.md`
7. `docs/configuration_review.md`
8. `tests/test_config_validation.py`

**Unchanged**:
- Existing configs (`qc_*.yaml`) - Already compliant

### Backwards Compatibility

**YES** - Fully backwards compatible:
- Defaults kept for convenience
- Validation warnings (not errors) for important-but-optional parameters
- Existing configs work without modification
- Only addition is validation layer and templates

### Test Coverage

New tests needed:
- Test validation with missing required parameters (should error)
- Test warnings for empty outlier detection
- Test validation with full config (no errors/warnings)
- Test that validation runs at pipeline start
- Test new median default for root core aggregation

### Benefits

1. **Prevents silent failures** - Validation catches missing critical parameters
2. **Conscious choices** - Warnings ensure users know when features are disabled
3. **Better defaults** - Median aggregation is statistically robust
4. **Guided configuration** - Templates and documentation
5. **Backwards compatible** - Existing configs work without modification
6. **Flexible** - Outlier detection remains optional (warnings inform)

### Risks

- **Low risk** - Non-breaking change, only adds validation
- **User education** - Need to communicate why warnings appear
- **Test coverage** - Must test all validation paths

## Implementation Checklist

- [ ] Add `validate_explicit_config()` to qc_pipeline.py
- [ ] Call validation at pipeline start
- [ ] Change `aggregation_method` default to "median" in components.py
- [ ] Update dataclass docstrings (CleanupConfig, HeritabilityConfig, RootCoreSourceConfig)
- [ ] Create `configs/templates/qc_cleanup_only_template.yaml`
- [ ] Create `configs/templates/qc_full_pipeline_template.yaml`
- [ ] Create `configs/templates/README.md`
- [ ] Create `docs/configuration_review.md`
- [ ] Update CLAUDE.md with configuration philosophy
- [ ] Add `tests/test_config_validation.py`
- [ ] Test cleanup-only config (should warn about empty outlier detection)
- [ ] Test full pipeline config (no warnings)
- [ ] Test missing required params (should error)
- [ ] Run full test suite
- [ ] Verify existing configs still work

## Timeline

**Estimated**: 8-10 hours total

- Validation function: 2 hours
- Change aggregation default and test: 1 hour
- Configuration templates: 2 hours
- Documentation: 3 hours
- Testing: 2 hours

## Related

- User request: "Defaults are helpful, just don't want confusion"
- User feedback: "Outlier detection can be optional"
- User feedback: "Warn when important things are not included"
- Related to: fix-qc-visualization-issues (completed)
- Non-breaking: Keeps defaults, adds validation layer
