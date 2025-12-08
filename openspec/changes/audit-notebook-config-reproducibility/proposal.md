# Audit and Fix Notebook-to-Config Reproducibility

## Why

For scientific publication, pipeline configurations must exactly replicate the analyses performed in Jupyter notebooks. Currently, there are 4 main datasets (Turface 150 genotypes, Turface 19 genotypes, Cylinder EDPIE, Root Coring EDPIE) with QC, visualization, and cross-platform analysis notebooks. While many parameters match their corresponding configs, systematic auditing is needed to identify and fix any inconsistencies that would prevent exact replication of published results.

**Problem:** Without a comprehensive audit, we risk:
- Publishing results that cannot be exactly reproduced via the pipeline
- Missing critical parameter differences that affect scientific conclusions
- Undocumented parameter choices that reviewers may question
- Config files that don't reflect the actual analysis performed

**Impact on publication:**
- Ensures Methods section accurately describes analysis parameters
- Enables peer review verification of results
- Supports open science and reproducibility standards
- Prevents retractions due to irreproducible results

## What Changes

### 1. Systematic Parameter Audit
- **Extract all parameters** from latest notebooks for each dataset:
  - QC notebooks: cleanup, outlier detection, heritability thresholds, PCA settings
  - Viz notebooks: PCA variance, genotype highlighting, UMAP settings
  - Cross-platform notebooks: correlation methods, data sources
- **Compare with configs** in `configs/` directory
- **Document actual inconsistencies** (not intentional dataset-specific variations)

### 2. Config File Updates
- **Fix mismatched parameters** where configs don't match notebooks:
  - Cylinder viz: PCA variance threshold (config: 0.95, notebook: 0.75)
  - Cylinder viz: Missing genotype highlighting lists
  - Any other parameter mismatches discovered
- **Add missing parameters** that exist in notebooks but not configs
- **Update data paths** to point to correct QC pipeline outputs

### 3. Documentation
- **Create parameter reference table** in OpenSpec specs showing all parameters per dataset
- **Document intentional variations** (e.g., heritability thresholds) with scientific rationale
- **Add validation notes** to each config file header

### 4. Validation Infrastructure (Optional Future Work)
- **Add config validation tests** that compare against notebook parameter exports
- **Create notebook parameter extraction utility**
- **Add CI checks** to catch config drift

## Impact

### Affected Specs
- **NEW:** `config-management` - Configuration validation and reproducibility tracking

### Affected Configs
- `configs/qc_cylinder_edpie.yaml` - Parameter fixes
- `configs/viz_cylinder_edpie.yaml` - PCA variance, genotype highlighting
- `configs/viz_turface_150genotypes.yaml` - Verification only
- `configs/viz_turface_19genotypes.yaml` - Needs creation or verification
- `configs/viz_root_coring.yaml` - Verification only
- `configs/cross_platform_turface19_vs_cylinder.yaml` - Verification only
- `configs/cross_platform_turface_150vs19_genotypes.yaml` - Needs verification

### Affected Code
- None (config-only changes for initial audit)

### Affected Documentation
- Config file headers (add parameter source documentation)
- `openspec/specs/config-management/` - New spec for reproducibility tracking

## Breaking Changes
None - this is a config audit and correction effort that improves reproducibility without changing code behavior.

## Migration Path
None required - configs are backward compatible, just more accurate.

## Risks
- **Time intensive:** Requires careful comparison of 7+ notebooks against configs
- **Notebook size:** Some notebooks are large and may require chunked reading
- **Parameter location:** Parameters may be scattered across multiple cells
- **Implicit parameters:** Some parameters may use function defaults rather than explicit values

## Success Criteria
1. ✅ All configs verified against latest notebooks for each dataset
2. ✅ Inconsistencies documented with specific notebook cells and line numbers
3. ✅ Critical mismatches fixed (cylinder viz config)
4. ✅ Parameter reference table created in OpenSpec specs
5. ✅ Config headers updated with notebook source references
6. ✅ Intentional parameter variations documented with rationale

## Timeline
- **Audit phase:** Systematic notebook parameter extraction
- **Fix phase:** Update configs to match notebooks
- **Documentation phase:** Create reference table and update config headers
- **Validation phase:** Verify all configs produce expected results

## Related Work
- REPLICATION_GUIDE.md shows example of config parameter impact on results
- QC_CONFIG_COMPARISON.md documents parameter analysis
- Issue #37 - QC pipeline test coverage
