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
3. **Point users to templates** - New templates in `configs/templates/` provide starting points
