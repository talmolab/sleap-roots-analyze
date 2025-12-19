# Proposal: Add Multi-Platform Pipeline Configurations

**Change ID**: `add-multi-platform-configs`
**Status**: Draft
**Created**: 2025-12-02
**Owner**: elizabeth

## Summary

Create missing QC and visualization pipeline configurations for all experimental platforms to enable reproducible CLI-driven analysis. Specifically: add QC config for Cylinder platform and Viz configs for all 4 platforms (Turface 150, Turface 19, Cylinder, Root Coring).

## Motivation

We have QC configs for 3 out of 4 platforms, but **no Viz configs** for any platform. Visualization is currently done via notebooks, but should be reproducible via CLI using the cleaned data output from QC pipelines.

**Current State:**
- ✅ QC: Turface 150 genotypes
- ✅ QC: Turface 19 genotypes
- ❌ QC: Cylinder (819 traits, needs special config)
- ✅ QC: Root Coring/Field (has config)
- ❌ Viz: ALL platforms (no viz configs exist)

**Desired State:**
- Complete QC and Viz config coverage for all platforms
- CLI-driven workflows replace manual notebook execution
- Reproducible analysis with version-controlled configs

## Goals

1. **Create Cylinder QC config** with platform-specific requirements (crown→seminal, H²=0.60)
2. **Create Viz configs** for all 4 platforms that consume QC pipeline outputs
3. **Document** platform-specific configuration patterns in config comments

## Non-Goals

- Cross-experiment analysis (separate proposal/pipeline)
- Batch execution scripts (separate proposal)
- Modifying existing QC configs (only creating new ones)
- Implementing new pipeline features

## Platform Requirements

### Cylinder (EDPIE) - Needs QC Config
- **Traits**: 819 depth-profile measurements
- **Custom Replacements**: `crown → seminal` for wheat terminology
- **H² Threshold**: 0.60 (stringent due to large trait count)
- **Column Names**: `plant_qr_code`, `Geno`, `Rep`
- **Notebook**: `trait_qc_cylinders_20251105.ipynb`
- **Output**: `configs/qc_cylinder_edpie.yaml`

### Root Coring/Field (EDPIE) - Has QC, Needs Viz
- **Nature**: Field trials with root coring + above-ground traits
- **QC Pipeline**: Merges root core depth data with above-ground traits
- **Existing**: `configs/qc_root_core_edpie.yaml` ✅
- **Viz Needs**: Depth profile plots, aggregated trait visualizations
- **Notebook**: `trait_viz_root_coring_20251130.ipynb`
- **Output**: `configs/viz_root_coring.yaml`

### Turface Platforms - Have QC, Need Viz
Both platforms have QC configs, need corresponding Viz configs:

**150 Genotypes:**
- **Existing QC**: `configs/qc_turface_150genotypes.yaml` ✅
- **Notebook**: `trait_viz_150_genotypes_turface_20251105.ipynb`
- **Output**: `configs/viz_turface_150genotypes.yaml`

**19 Genotypes:**
- **Existing QC**: `configs/qc_turface_19genotypes.yaml` ✅
- **Notebook**: `trait_viz_19_genotypes_turface_*.ipynb` (need to find latest)
- **Output**: `configs/viz_turface_19genotypes.yaml`

## Viz Config Pattern

All viz configs follow this pattern:
```yaml
data:
  csv_path: "runs/qc_<platform>/cleaned_traits.csv"  # Output from QC pipeline

columns:
  genotype: "Genotype"  # Post-sanitization column names
  replicate: "Replicate"
  barcode: "Barcode"

# Standard viz sections: pca, clustering, static_viz, etc.
```

## Dependencies

- ✅ QC and Viz pipeline infrastructure (complete)
- ✅ Custom trait name replacement (`sanitize_trait_names`)
- ✅ Root core merge functionality
- ❓ Depth profile visualization in viz pipeline (need to verify)

## Open Questions

1. Does viz pipeline support depth profile plots for root coring data?
2. Is there a `trait_viz_19_genotypes_turface` notebook or do we extrapolate from 150 genotype viz?

## Success Criteria

- [ ] `configs/qc_cylinder_edpie.yaml` created and validates
- [ ] `configs/viz_turface_150genotypes.yaml` created and validates
- [ ] `configs/viz_turface_19genotypes.yaml` created and validates
- [ ] `configs/viz_cylinder_edpie.yaml` created and validates
- [ ] `configs/viz_root_coring.yaml` created and validates
- [ ] All configs have clear comments explaining platform-specific choices
- [ ] `configs/README.md` updated with new config descriptions
- [ ] At least one full QC→Viz pipeline run per platform validates configs work

## Related Proposals

- **Future**: Cross-experiment analysis pipeline
- **Future**: Batch pipeline execution system