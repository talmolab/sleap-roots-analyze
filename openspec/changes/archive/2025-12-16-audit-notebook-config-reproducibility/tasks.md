# Implementation Tasks: Notebook-to-Config Audit

**Status**: ✅ COMPLETED (2025-12-08)
**Commit**: 91c9c0d - "feat: Complete notebook-config reproducibility audit for publication"

## 1. Systematic Parameter Extraction

### 1.1 Dataset: Turface 150 Genotypes
- [x] 1.1.1 Extract ALL parameters from `trait_qc_150_genotypes_turface_20251105.ipynb`
- [x] 1.1.2 Extract ALL parameters from `trait_viz_150_genotypes_turface_20251105.ipynb`
- [x] 1.1.3 Compare against `configs/qc_turface_150genotypes.yaml`
- [x] 1.1.4 Compare against `configs/viz_turface_150genotypes.yaml`
- [x] 1.1.5 Document any mismatches with cell references

### 1.2 Dataset: Turface 19 Genotypes
- [x] 1.2.1 Extract ALL parameters from `trait_qc_19_genotypes_turface_20251105.ipynb`
- [x] 1.2.2 Extract ALL parameters from `trait_viz_turface_20251105.ipynb`
- [x] 1.2.3 Compare against `configs/qc_turface_19genotypes.yaml`
- [x] 1.2.4 Verify `configs/viz_turface_19genotypes.yaml` exists or create it
- [x] 1.2.5 Document any mismatches with cell references

### 1.3 Dataset: Cylinder EDPIE
- [x] 1.3.1 Extract ALL parameters from `trait_qc_cylinders_20251105.ipynb`
- [x] 1.3.2 Extract ALL parameters from `trait_viz_cylinder_20251105.ipynb`
- [x] 1.3.3 Compare against `configs/qc_cylinder_edpie.yaml`
- [x] 1.3.4 Compare against `configs/viz_cylinder_edpie.yaml`
- [x] 1.3.5 Document mismatches (FIXED: PCA variance 0.95 → 0.75, added genotype lists)

### 1.4 Dataset: Root Coring EDPIE
- [x] 1.4.1 Extract ALL parameters from `trait_qc_root_coring_20251126.ipynb`
- [x] 1.4.2 Extract ALL parameters from `trait_viz_root_coring_20251130.ipynb`
- [x] 1.4.3 Compare against `configs/qc_root_core_edpie.yaml`
- [x] 1.4.4 Compare against `configs/viz_root_coring.yaml`
- [x] 1.4.5 Verify config matches REPLICATION_GUIDE.md fixes

### 1.5 Cross-Platform Analysis
- [x] 1.5.1 Extract parameters from `cross_experiment_spearman_turface_cylinder_20250919.ipynb`
- [x] 1.5.2 Check for other cross-platform notebooks
- [x] 1.5.3 Compare against `configs/cross_platform_turface19_vs_cylinder.yaml`
- [x] 1.5.4 Compare against `configs/cross_platform_turface_150vs19_genotypes.yaml`
- [x] 1.5.5 Document any mismatches

## 2. Create Parameter Reference Documentation

- [x] 2.1 Create spec file with requirements for config-notebook consistency
- [x] 2.2 Create parameter comparison table for each dataset
- [x] 2.3 Document intentional parameter variations (heritability thresholds by panel size)
- [x] 2.4 Add notebook source references to all tables

## 3. Fix Config Inconsistencies

### 3.1 Critical Fixes (Must Fix Before Publication)
- [x] 3.1.1 Fix `configs/viz_cylinder_edpie.yaml` (PCA 0.95→0.75, added genotype highlighting)
- [x] 3.1.2 Verify `configs/qc_root_core_edpie.yaml` matches Nov 30 notebook
- [x] 3.1.3 Check for any other critical mismatches found in audit

### 3.2 Minor Fixes (Recommended)
- [x] 3.2.1 Update data paths in viz configs to point to latest QC outputs
- [x] 3.2.2 Ensure all image_dir paths are correct and accessible
- [x] 3.2.3 Add missing parameters from notebooks if any

### 3.3 Config Header Documentation
- [x] 3.3.1 Add config file headers with notebook source references
- [x] 3.3.2 Update header format template for future configs

## 4. Validation

- [x] 4.1 Run QC pipeline with each updated config (all 4 datasets verified)
- [x] 4.2 Run viz pipeline with each updated config
- [x] 4.3 Run cross-platform analysis with updated configs
- [x] 4.4 Document any remaining discrepancies and their causes

## 5. Final Documentation

- [x] 5.1 Create summary document in OpenSpec specs
- [x] 5.2 Update config file comments with verification dates
- [x] 5.3 Create reproducibility checklist for future datasets

## 6. Future Enhancements (Optional, Not Required for Publication)

- [ ] 6.1 Create notebook parameter extraction utility (DEFERRED)
- [ ] 6.2 Add pytest tests for config validation (DEFERRED)
- [ ] 6.3 Add CI/CD checks (DEFERRED)

## Verification Results

| Dataset | Samples | Traits | H² Threshold | Status |
|---------|---------|--------|--------------|--------|
| Turface 150 Genotypes | 890 | 13 | ≥ 0.40 | ✅ VERIFIED |
| Turface 19 Genotypes | 153 | 8 | ≥ 0.60 | ✅ VERIFIED |
| Cylinder EDPIE | 123 | 588 | ≥ 0.60 | ✅ VERIFIED |
| Root Coring EDPIE | 57 | 23 | ≥ 0.50 | ✅ VERIFIED |
