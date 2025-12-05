# Configuration Files

This directory contains configuration files for `sleap-roots-analyze` pipelines.

## Directory Structure

```
configs/
├── README.md                          # This file
├── templates/                         # Template configs with placeholders
│   ├── README.md
│   ├── qc_full_pipeline_template.yaml
│   └── qc_cleanup_only_template.yaml
├── qc_turface_150genotypes.yaml      # QC config for Turface 150 genotype analysis
├── qc_turface_19genotypes.yaml       # QC config for Turface 19 genotype panel
├── qc_cylinder_edpie.yaml            # QC config for Cylinder platform (wheat)
├── qc_root_core_edpie.yaml           # QC with root core data merging
├── qc_consensus_6method.yaml         # QC with consensus outlier detection
├── qc_clustering_strict.yaml         # QC with strict clustering methods
├── qc_permissive.yaml                # QC with permissive outlier detection
├── qc_mahalanobis.yaml               # QC with Mahalanobis distance only
├── viz_turface_150genotypes.yaml     # Viz config for Turface 150 genotypes
├── viz_turface_19genotypes.yaml      # Viz config for Turface 19 genotypes
├── viz_cylinder_edpie.yaml           # Viz config for Cylinder platform
├── viz_root_coring.yaml              # Viz config for root coring/field data
├── viz_standard.yaml                 # Standard visualization pipeline (template)
├── viz_minimal.yaml                  # Minimal visualization (template)
├── viz_comprehensive.yaml            # Comprehensive visualization (template)
└── viz_publication.yaml              # Publication-ready figures (template)
```

## Quick Start

### For New Users

Start with a template:

```bash
# Copy a template to create your own config
cp configs/templates/qc_full_pipeline_template.yaml configs/my_experiment.yaml

# Edit the config with your data paths
vim configs/my_experiment.yaml

# Run the pipeline
sleap-roots-analyze qc configs/my_experiment.yaml
```

### Using Existing Configs

Browse the configs in this directory for real-world examples:

```bash
# List all available configs
sleap-roots-analyze config list

# Validate a config before running
sleap-roots-analyze config validate configs/qc_turface_150genotypes.yaml

# Run an existing config
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml
```

## Configuration Types

### QC Pipeline Configs

Quality control pipeline configurations for trait data analysis:

**Platform-Specific Configs:**

- **qc_turface_150genotypes.yaml** - Full QC pipeline for 150 genotype Turface experiment
  - Includes outlier detection, heritability filtering (H² > 0.40), comprehensive visualization
  - Good starting point for most analyses

- **qc_turface_19genotypes.yaml** - QC pipeline for 19 genotype Turface panel
  - Similar to 150 genotype config but for smaller panel
  - Heritability threshold: H² > 0.40

- **qc_cylinder_edpie.yaml** - QC pipeline for Cylinder platform (wheat EDPIE)
  - 819 depth-profiled root traits from cylinder greenhouse system
  - Custom trait name replacements: "crown" → "seminal" for wheat terminology
  - Higher heritability threshold (H² > 0.60) due to large trait count
  - Scanner-independent QC applied upstream

- **qc_root_core_edpie.yaml** - QC with root core data merging
  - Merges biomass/counting data from field root cores with above-ground traits
  - Includes core-level QC and aggregation

**Method-Specific Configs (Alternative Approaches):**

- **qc_consensus_6method.yaml** - Consensus outlier detection
  - Uses 6 outlier detection methods
  - Requires consensus across methods

- **qc_clustering_strict.yaml** - Strict clustering-based detection
  - Uses DBSCAN, Isolation Forest, Local Outlier Factor
  - More conservative outlier identification

- **qc_permissive.yaml** - Permissive outlier settings
  - Looser thresholds for outlier detection
  - Keeps more borderline samples

- **qc_mahalanobis.yaml** - Mahalanobis distance only
  - Single method approach
  - Fast and interpretable

### Viz Pipeline Configs

Visualization pipeline configurations for publication-quality figures:

**Platform-Specific Configs:**

These configs consume cleaned trait data from the corresponding QC pipelines:

- **viz_turface_150genotypes.yaml** - Visualization for Turface 150 genotype panel
  - Input: `runs/qc_turface_150geno/cleaned_traits.csv`
  - PCA analysis, statistical summaries, interesting genotype identification
  - Heritability threshold: H² > 0.40

- **viz_turface_19genotypes.yaml** - Visualization for Turface 19 genotype panel
  - Input: `runs/qc_turface_19geno/cleaned_traits.csv`
  - Similar to 150 genotype viz but for smaller panel
  - Adjusted max_genotypes for smaller dataset

- **viz_cylinder_edpie.yaml** - Visualization for Cylinder platform (wheat)
  - Input: `runs/qc_cylinder/cleaned_traits.csv`
  - Optimized for large trait count (~588 traits after filtering)
  - Larger figure sizes, more top features displayed (n=15)
  - Heritability threshold: H² > 0.60

- **viz_root_coring.yaml** - Visualization for root coring/field data
  - Input: `runs/qc_root_coring/cleaned_traits.csv`
  - Includes merged root core and above-ground trait visualizations
  - Depth profile plots (if supported)

**Template Configs (General-Purpose):**

- **viz_standard.yaml** - Balanced visualization set (template)
  - PCA analysis, trait distributions, correlations
  - Good for exploratory analysis

- **viz_minimal.yaml** - Essential visualizations only (template)
  - Minimal set for quick inspection
  - Faster execution

- **viz_comprehensive.yaml** - All available visualizations (template)
  - Includes UMAP, interactive plots, image galleries
  - Comprehensive analysis

- **viz_publication.yaml** - Publication-ready figures (template)
  - High DPI, vector formats (PDF, SVG)
  - Customized fonts and styling

## Contributing Your Configs

If you create a config for your experiment, consider committing it to the repository for reuse:

1. **Name it descriptively**: `qc_<experiment>_<detail>.yaml`
2. **Add comments**: Document key parameters and choices
3. **Test it**: Run `sleap-roots-analyze config validate` first
4. **Commit it**: Add and commit to the repository
5. **Document it**: Add a brief description to this README

This helps others learn from your configuration choices and enables reproducibility.

## Documentation

For detailed information on configuration parameters and pipeline usage:

- **QC Pipeline Guide**: See [docs/QC_PIPELINE_GUIDE.md](../docs/QC_PIPELINE_GUIDE.md)
- **Templates**: See [templates/README.md](templates/README.md)
- **CLI Help**: Run `sleap-roots-analyze qc --help`

## Validation

Always validate your config before running long pipelines:

```bash
sleap-roots-analyze config validate configs/my_config.yaml
```

This checks:
- Valid YAML syntax
- Required fields present
- Data files exist
- Parameter types correct