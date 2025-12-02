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
├── qc_root_core_edpie.yaml           # QC with root core data merging
├── qc_consensus_6method.yaml         # QC with consensus outlier detection
├── qc_clustering_strict.yaml         # QC with strict clustering methods
├── qc_permissive.yaml                # QC with permissive outlier detection
├── qc_mahalanobis.yaml               # QC with Mahalanobis distance only
├── viz_standard.yaml                 # Standard visualization pipeline
├── viz_minimal.yaml                  # Minimal visualization
├── viz_comprehensive.yaml            # Comprehensive visualization
└── viz_publication.yaml              # Publication-ready figures
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

- **qc_turface_150genotypes.yaml** - Full QC pipeline for 150 genotype Turface experiment
  - Includes outlier detection, heritability filtering, comprehensive visualization
  - Good starting point for most analyses

- **qc_root_core_edpie.yaml** - QC with root core data merging
  - Merges biomass/counting data from multiple cores
  - Includes core-level QC and aggregation

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

- **viz_standard.yaml** - Balanced visualization set
  - PCA analysis, trait distributions, correlations
  - Good for exploratory analysis

- **viz_minimal.yaml** - Essential visualizations only
  - Minimal set for quick inspection
  - Faster execution

- **viz_comprehensive.yaml** - All available visualizations
  - Includes UMAP, interactive plots, image galleries
  - Comprehensive analysis

- **viz_publication.yaml** - Publication-ready figures
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