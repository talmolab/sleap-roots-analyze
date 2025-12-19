# Implementation Tasks

## Phase 1: Investigation
- [ ] 1.1 Check if trait_viz notebook exists for 19 genotypes Turface
- [ ] 1.2 Verify depth profile visualization support in viz pipeline

## Phase 2: Create Cylinder QC Config
- [ ] 2.1 Create `configs/qc_cylinder_edpie.yaml` based on `trait_qc_cylinders_20251105.ipynb`
  - Base on existing `qc_turface_150genotypes.yaml` template
  - Set `custom_replacements: {"crown": "seminal"}`
  - Set `heritability.threshold: 0.60`
  - Set columns: `barcode: plant_qr_code`, `genotype: Geno`, `replicate: Rep`
  - Set data path to cylinder CSV
  - Add explanatory comments about platform-specific choices
- [ ] 2.2 Validate config: `sleap-roots-analyze config validate configs/qc_cylinder_edpie.yaml`

## Phase 3: Create Visualization Configs
- [ ] 3.1 Create `configs/viz_turface_150genotypes.yaml`
  - Base on viz pipeline template
  - Set `data.csv_path` to QC output: `runs/qc_turface_150geno/cleaned_traits.csv`
  - Set standard viz options (PCA, clustering, static_viz)
- [ ] 3.2 Create `configs/viz_turface_19genotypes.yaml`
  - Similar to 3.1 but for 19 genotype dataset
  - Set `data.csv_path` to QC output: `runs/qc_turface_19geno/cleaned_traits.csv`
- [ ] 3.3 Create `configs/viz_cylinder_edpie.yaml`
  - Set `data.csv_path` to QC output: `runs/qc_cylinder/cleaned_traits.csv`
  - Include any cylinder-specific viz options
- [ ] 3.4 Create `configs/viz_root_coring.yaml`
  - Set `data.csv_path` to QC output: `runs/qc_root_coring/cleaned_traits.csv`
  - Include depth profile visualizations (if supported)
- [ ] 3.5 Validate all viz configs: `sleap-roots-analyze config validate configs/viz_*.yaml`

## Phase 4: Documentation
- [ ] 4.1 Add platform descriptions to `configs/README.md`
  - Document Cylinder platform
  - Document Root Coring/Field platform
  - Update Turface platform descriptions
- [ ] 4.2 Add inline comments to all new configs explaining choices
- [ ] 4.3 Update config list in README with new configs

## Phase 5: Validation (Optional - smoke test)
- [ ] 5.1 Run Cylinder QC (dry-run): `sleap-roots-analyze qc configs/qc_cylinder_edpie.yaml --dry-run`
- [ ] 5.2 Run sample Viz (dry-run): `sleap-roots-analyze viz configs/viz_turface_150genotypes.yaml --dry-run`
- [ ] 5.3 If dry-run passes, run full Cylinder QC to validate config works end-to-end

## Dependencies
- Task 2.1 is independent
- Tasks 3.1-3.4 can be done in parallel
- Task 3.5 depends on 3.1-3.4
- Phase 4 can be done in parallel with Phase 3
- Phase 5 depends on all previous phases

## Notes
- This proposal does NOT include cross-experiment analysis (separate future proposal)
- This proposal does NOT include batch execution scripts (separate future proposal)
- Focus is on creating configs, not running full pipelines (that's validation step)
