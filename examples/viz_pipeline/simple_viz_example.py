"""Simple example of using the VizPipeline for trait visualization.

This example demonstrates the minimal usage of the visualization pipeline
with the viz_minimal.yaml preset configuration.
"""

from pathlib import Path

from omegaconf import OmegaConf

from sleap_roots_analyze.viz_pipeline import VizPipeline, VizPipelineConfig

# Define paths
DATA_CSV = "path/to/your/traits.csv"  # Replace with your data path
OUTPUT_DIR = "./viz_runs"  # Where pipeline outputs will be saved

# Load the minimal preset configuration
config_path = Path("configs/viz_minimal.yaml")
omega_conf = OmegaConf.load(config_path)

# Configure required settings
# The csv_path is marked as MISSING in the preset, so we must set it
OmegaConf.set_struct(omega_conf, False)  # Allow adding fields
omega_conf.data.csv_path = DATA_CSV

# Optional: Configure column names if they differ from defaults
# omega_conf.columns.barcode = "Barcode"     # Default
# omega_conf.columns.genotype = "geno"       # Default
# omega_conf.columns.replicate = "rep"       # Default

# Optional: Add image directory for interactive visualizations
# omega_conf.data.image_dir = "path/to/images"

# Optional: Enable additional analysis
# omega_conf.statistics.calculate_anova = True
# omega_conf.statistics.calculate_heritability = True

OmegaConf.set_struct(omega_conf, True)  # Re-enable struct mode

# Convert to structured config object
structured = OmegaConf.structured(VizPipelineConfig)
merged = OmegaConf.merge(structured, omega_conf)
config = OmegaConf.to_object(merged)

# Create and run the pipeline
print(f"Running visualization pipeline: {config.pipeline_name}")
print(f"Input CSV: {config.data.csv_path}")
print(f"Output directory: {OUTPUT_DIR}")

pipeline = VizPipeline(config, output_dir=OUTPUT_DIR)
result = pipeline.run()

# Check results
if result.success:
    print(f"\nPipeline completed successfully!")
    print(f"Analyzed {result.metadata.get('n_samples')} samples")
    print(f"Used {result.metadata.get('n_traits_final')} traits")

    # Find the run directory
    run_name = f"{config.pipeline_name}_*"
    run_dirs = list(Path(OUTPUT_DIR).glob(run_name))
    if run_dirs:
        run_dir = sorted(run_dirs)[-1]  # Get most recent
        print(f"\nOutputs saved to: {run_dir}")
        print(f"\nGenerated files:")
        print(f"  - {run_dir / 'summary' / 'SUMMARY.md'}")
        print(f"  - {run_dir / 'pca' / 'pc_scores.csv'}")
        print(f"  - {run_dir / 'pca' / 'loadings.csv'}")
        print(f"  - {run_dir / 'pca' / 'explained_variance.csv'}")
else:
    print(f"\nPipeline failed!")
    print(f"Check the logs for details.")

# Example: Accessing specific results
print(f"\nPCA Analysis:")
print(f"  - Components: {result.metadata.get('pca_n_components', 'N/A')}")
print(f"  - Variance explained: {result.metadata.get('pca_variance_explained', 'N/A')}")
print(f"  - Top features: {len(result.metadata.get('pca_top_features', []))}")
