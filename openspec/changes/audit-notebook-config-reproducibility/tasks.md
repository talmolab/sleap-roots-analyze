# Implementation Tasks: Notebook-to-Config Audit

## 1. Systematic Parameter Extraction

### 1.1 Dataset: Turface 150 Genotypes
- [ ] 1.1.1 Extract ALL parameters from `trait_qc_150_genotypes_turface_20251105.ipynb`
  - [ ] Input data path and column mappings
  - [ ] Cleanup thresholds (max_nan_fraction, max_zeros_per_trait, max_nans_per_trait)
  - [ ] Outlier detection settings (methods, Mahalanobis params)
  - [ ] Outlier removal strategy
  - [ ] PCA variance threshold
  - [ ] Heritability threshold and filtering settings
  - [ ] Custom exclusions and replacements
- [ ] 1.1.2 Extract ALL parameters from `trait_viz_150_genotypes_turface_20251105.ipynb`
  - [ ] Input data path (QC output reference)
  - [ ] Image directory path
  - [ ] PCA variance threshold for visualization
  - [ ] PCA feature selection strategy and top N features
  - [ ] UMAP parameters (n_neighbors, min_dist)
  - [ ] Genotypes to color list
  - [ ] Genotypes to highlight list
  - [ ] Figure DPI and format settings
- [ ] 1.1.3 Compare against `configs/qc_turface_150genotypes.yaml`
- [ ] 1.1.4 Compare against `configs/viz_turface_150genotypes.yaml`
- [ ] 1.1.5 Document any mismatches with cell references

### 1.2 Dataset: Turface 19 Genotypes
- [ ] 1.2.1 Extract ALL parameters from `trait_qc_19_genotypes_turface_20251105.ipynb`
  - [ ] Same parameter categories as 1.1.1
- [ ] 1.2.2 Extract ALL parameters from `trait_viz_turface_20251105.ipynb`
  - [ ] Same parameter categories as 1.1.2
- [ ] 1.2.3 Compare against `configs/qc_turface_19genotypes.yaml`
- [ ] 1.2.4 Verify `configs/viz_turface_19genotypes.yaml` exists or create it
- [ ] 1.2.5 Document any mismatches with cell references

### 1.3 Dataset: Cylinder EDPIE
- [ ] 1.3.1 Extract ALL parameters from `trait_qc_cylinders_20251105.ipynb`
  - [ ] Same parameter categories as 1.1.1
  - [ ] Custom trait name replacements (crown → seminal)
- [ ] 1.3.2 Extract ALL parameters from `trait_viz_cylinder_20251105.ipynb`
  - [ ] Same parameter categories as 1.1.2
  - [ ] Cylinder-specific image settings
  - [ ] Biplot arrow scale
- [ ] 1.3.3 Compare against `configs/qc_cylinder_edpie.yaml`
- [ ] 1.3.4 Compare against `configs/viz_cylinder_edpie.yaml`
- [ ] 1.3.5 Document mismatches (KNOWN: PCA variance 0.75 vs 0.95, missing genotype lists)

### 1.4 Dataset: Root Coring EDPIE
- [ ] 1.4.1 Extract ALL parameters from `trait_qc_root_coring_20251126.ipynb`
  - [ ] Same parameter categories as 1.1.1
  - [ ] Root core specific: aggregation methods, depth mappings
  - [ ] Core-level QC settings
  - [ ] Merge settings for above-ground traits
- [ ] 1.4.2 Extract ALL parameters from `trait_viz_root_coring_20251130.ipynb`
  - [ ] Same parameter categories as 1.1.2
- [ ] 1.4.3 Compare against `configs/qc_root_core_edpie.yaml`
- [ ] 1.4.4 Compare against `configs/viz_root_coring.yaml`
- [ ] 1.4.5 Verify config matches REPLICATION_GUIDE.md fixes

### 1.5 Cross-Platform Analysis
- [ ] 1.5.1 Extract parameters from `cross_experiment_spearman_turface_cylinder_20250919.ipynb`
  - [ ] Data source paths (cylinder and turface QC outputs)
  - [ ] Correlation method (Spearman)
  - [ ] Min samples per genotype threshold
  - [ ] Top N correlations/plots settings
- [ ] 1.5.2 Check for other cross-platform notebooks
- [ ] 1.5.3 Compare against `configs/cross_platform_turface19_vs_cylinder.yaml`
- [ ] 1.5.4 Compare against `configs/cross_platform_turface_150vs19_genotypes.yaml`
- [ ] 1.5.5 Document any mismatches

## 2. Create Parameter Reference Documentation

- [ ] 2.1 Create spec file: `openspec/changes/audit-notebook-config-reproducibility/specs/config-management/spec.md`
  - [ ] Define requirements for config-notebook consistency
  - [ ] Add scenarios for parameter validation
  - [ ] Document parameter categories (cleanup, outlier, PCA, viz, etc.)
- [ ] 2.2 Create parameter comparison table for each dataset
  - [ ] Turface 150: Notebook vs Config parameters
  - [ ] Turface 19: Notebook vs Config parameters
  - [ ] Cylinder: Notebook vs Config parameters
  - [ ] Root Coring: Notebook vs Config parameters
  - [ ] Cross-Platform: Notebook vs Config parameters
- [ ] 2.3 Document intentional parameter variations
  - [ ] Heritability thresholds: Why 0.40 vs 0.50 vs 0.60
  - [ ] PCA variance thresholds: Why 0.75 vs 0.80 vs 0.95
  - [ ] Dataset-specific settings and their rationale
- [ ] 2.4 Add notebook source references to all tables
  - [ ] Include cell numbers where parameters are defined
  - [ ] Include line numbers for key parameter assignments

## 3. Fix Config Inconsistencies

### 3.1 Critical Fixes (Must Fix Before Publication)
- [ ] 3.1.1 Fix `configs/viz_cylinder_edpie.yaml`
  - [ ] Change `pca.n_components: 0.95` → `0.75` (matches notebook)
  - [ ] Add `static_viz.genotypes_to_color: ["GH_7293", "GH_7378", "GH_7327"]`
  - [ ] Add `static_viz.highlight_genotypes: ["GH_7293", "GH_7378", "GH_7327"]`
  - [ ] Verify all other parameters match notebook
- [ ] 3.1.2 Verify `configs/qc_root_core_edpie.yaml` matches Nov 30 notebook
  - [ ] Confirm cleanup parameters: max_nan_fraction=0.0, max_zeros=0.5, max_nans=0.2
  - [ ] Confirm these match REPLICATION_GUIDE.md
- [ ] 3.1.3 Check for any other critical mismatches found in audit

### 3.2 Minor Fixes (Recommended)
- [ ] 3.2.1 Update data paths in viz configs to point to latest QC outputs
- [ ] 3.2.2 Ensure all image_dir paths are correct and accessible
- [ ] 3.2.3 Add missing parameters from notebooks if any

### 3.3 Config Header Documentation
- [ ] 3.3.1 Add to each config file header:
  - [ ] Source notebook filename and date
  - [ ] Cell numbers where critical parameters are defined
  - [ ] Last verification date
  - [ ] Any intentional deviations from notebook with rationale
- [ ] 3.3.2 Update header format template for future configs

## 4. Validation

- [ ] 4.1 Run QC pipeline with each updated config
  - [ ] Turface 150: Verify output matches notebook results
  - [ ] Turface 19: Verify output matches notebook results
  - [ ] Cylinder: Verify output matches notebook results
  - [ ] Root Coring: Verify output matches REPLICATION_GUIDE expectations
- [ ] 4.2 Run viz pipeline with each updated config
  - [ ] Verify PCA plots match notebook visualizations
  - [ ] Verify genotype highlighting matches notebook
  - [ ] Verify statistical outputs match notebook
- [ ] 4.3 Run cross-platform analysis with updated configs
  - [ ] Verify correlation counts match notebook
  - [ ] Verify top correlations match notebook rankings
- [ ] 4.4 Document any remaining discrepancies and their causes

## 5. Final Documentation

- [ ] 5.1 Create summary document in OpenSpec specs
  - [ ] List all verified configs with verification dates
  - [ ] Document all fixes applied
  - [ ] List any acceptable differences with rationale
- [ ] 5.2 Update config file comments
  - [ ] Add "Verified against notebook: [filename] on [date]"
  - [ ] Add parameter source references for key settings
- [ ] 5.3 Create reproducibility checklist for future datasets
  - [ ] Template for extracting notebook parameters
  - [ ] Validation steps for new configs
  - [ ] Documentation requirements

## 6. Future Enhancements (Optional, Not Required for Publication)

- [ ] 6.1 Create notebook parameter extraction utility
  - [ ] Script to parse notebooks and extract parameter dictionaries
  - [ ] Compare extracted params against config files
  - [ ] Generate validation report
- [ ] 6.2 Add pytest tests for config validation
  - [ ] Test that configs match reference parameter sets
  - [ ] Warning if configs deviate from documented values
- [ ] 6.3 Add CI/CD checks
  - [ ] Validate configs on PR
  - [ ] Alert if config changes without notebook reference update
