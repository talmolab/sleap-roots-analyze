#!/usr/bin/env python
"""DEPRECATED: Use 'sleap-roots-analyze qc' instead.

This script is deprecated and will be removed in v0.1.0.
Please use the new CLI:

    sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml -o ./qc_runs

For more info: sleap-roots-analyze qc --help

---

This script replicates the analysis from trait_qc_150_genotypes_turface_20251024.ipynb
using the tested QC pipeline infrastructure.

Usage:
    python run_turface_qc.py

Outputs will be saved to: ./qc_runs/turface_150genotypes_qc_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import warnings

warnings.warn(
    "run_turface_qc.py is deprecated. Use 'sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml'",
    DeprecationWarning,
    stacklevel=2,
)

import logging
from pathlib import Path

from sleap_roots_analyze.pipeline import QCPipeline, load_qc_config


def main():
    """Run the Turface 150 genotypes QC pipeline."""
    # Set up logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    config_path = Path("configs/qc_turface_150genotypes.yaml")
    logger.info(f"Loading configuration from: {config_path}")

    try:
        config = load_qc_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    # Display configuration summary
    logger.info("=" * 60)
    logger.info("QC Pipeline Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"Pipeline Name: {config.pipeline_name}")
    logger.info(f"Data Path: {config.data.csv_path}")
    logger.info(f"Image Directory: {config.data.image_dir}")
    logger.info("")
    logger.info("Data Cleanup:")
    logger.info(f"  - Max NaN fraction per sample: {config.cleanup.max_nan_fraction}")
    logger.info(f"  - Max zeros per trait: {config.cleanup.max_zeros_per_trait}")
    logger.info(f"  - Max NaNs per trait: {config.cleanup.max_nans_per_trait}")
    logger.info(f"  - Min samples per trait: {config.cleanup.min_samples_per_trait}")
    logger.info("")
    logger.info("Outlier Detection:")
    logger.info(f"  - Methods: {config.outlier_detection.traditional_methods}")
    logger.info(f"  - Removal strategy: {config.outlier_removal.strategy}")
    logger.info(f"  - Removal method: {config.outlier_removal.method}")
    logger.info("")
    logger.info("Heritability Filtering:")
    logger.info(f"  - Enabled: {config.heritability.enabled}")
    logger.info(f"  - Threshold: {config.heritability.threshold}")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path("./qc_runs")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"\nOutputs will be saved to: {output_dir.absolute()}")

    # Initialize and run pipeline
    logger.info("\nInitializing QC pipeline...")
    pipeline = QCPipeline(config=config, output_dir=output_dir)

    logger.info("Starting pipeline execution...")
    logger.info("=" * 60)

    try:
        results = pipeline.run()

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

        # Display summary
        final_result = results.get("10_generate_summary")
        if final_result and hasattr(final_result, "data"):
            step_result = final_result.data
            if step_result and hasattr(step_result, "metadata"):
                summary = step_result.metadata.get("summary", {})

                logger.info("\nPipeline Summary:")
                logger.info(f"  - Run directory: {pipeline.run_dir}")
                logger.info(f"  - Total steps completed: {len(results)}")

                if "pipeline_summary" in summary:
                    ps = summary["pipeline_summary"]
                    logger.info(f"\nData Processing:")
                    logger.info(f"  - Original samples: {ps.get('original_samples', 'N/A')}")
                    logger.info(f"  - Final samples: {ps.get('final_samples', 'N/A')}")
                    logger.info(f"  - Samples removed: {ps.get('total_samples_removed', 'N/A')}")
                    logger.info(f"\nTrait Processing:")
                    logger.info(f"  - Original traits: {ps.get('original_traits', 'N/A')}")
                    logger.info(f"  - Final traits: {ps.get('final_traits', 'N/A')}")
                    logger.info(f"  - Traits removed: {ps.get('total_traits_removed', 'N/A')}")

        logger.info("\n" + "=" * 60)
        logger.info(f"All outputs saved to: {pipeline.run_dir}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
