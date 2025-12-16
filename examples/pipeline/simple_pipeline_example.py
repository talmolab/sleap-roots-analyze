"""Example of a simple data analysis pipeline.

This example demonstrates how to create a basic pipeline that:
1. Loads data from a CSV file
2. Cleans the data by removing missing values
3. Performs basic statistical analysis
4. Generates a summary report
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sleap_roots_analyze.pipeline import (
    BasePipeline,
    PipelineConfig,
    Task,
    TaskResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)


class SimpleAnalysisPipeline(BasePipeline):
    """A simple data analysis pipeline."""

    def create_tasks(self):
        """Create the pipeline tasks.

        The tasks form a linear dependency chain:
        load_data -> clean_data -> analyze_data -> create_report
        """
        return [
            Task(
                func=self.load_data,
                name="load_data",
                description="Load data from CSV file",
            ),
            Task(
                func=self.clean_data,
                name="clean_data",
                depends_on=["load_data"],
                description="Remove missing values and filter data",
            ),
            Task(
                func=self.analyze_data,
                name="analyze_data",
                depends_on=["clean_data"],
                description="Perform statistical analysis",
            ),
            Task(
                func=self.create_report,
                name="create_report",
                depends_on=["analyze_data"],
                description="Generate summary report",
            ),
        ]

    def load_data(self, config, run_dir, logger):
        """Load data from CSV file.

        Args:
            config: Pipeline configuration.
            run_dir: Run directory for outputs.
            logger: Logger instance.

        Returns:
            TaskResult with loaded DataFrame.
        """
        logger.info(f"Loading data from {config.data.input_path}")

        df = pd.read_csv(config.data.input_path)

        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        return TaskResult(
            data=df,
            metadata={
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": list(df.columns),
            },
        )

    def clean_data(self, config, run_dir, logger, load_data):
        """Clean the data.

        Args:
            config: Pipeline configuration.
            run_dir: Run directory for outputs.
            logger: Logger instance.
            load_data: TaskResult from load_data task.

        Returns:
            TaskResult with cleaned DataFrame.
        """
        df = load_data.data
        logger.info(f"Cleaning data - original shape: {df.shape}")

        # Remove missing values
        n_missing = df.isna().sum().sum()
        df_clean = df.dropna()

        logger.info(f"Removed {n_missing} missing values, new shape: {df_clean.shape}")

        # Save cleaned data
        output_file = run_dir / "cleaned_data.csv"
        df_clean.to_csv(output_file, index=False)

        return TaskResult(
            data=df_clean,
            metadata={
                "n_rows_original": len(df),
                "n_rows_clean": len(df_clean),
                "n_missing_removed": n_missing,
            },
            files_generated=[output_file],
        )

    def analyze_data(self, config, run_dir, logger, clean_data):
        """Perform statistical analysis.

        Args:
            config: Pipeline configuration.
            run_dir: Run directory for outputs.
            logger: Logger instance.
            clean_data: TaskResult from clean_data task.

        Returns:
            TaskResult with analysis results.
        """
        df = clean_data.data
        logger.info("Performing statistical analysis")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        logger.info(f"Analyzing {len(numeric_cols)} numeric columns")

        # Calculate descriptive statistics
        stats = df[numeric_cols].describe()

        # Calculate correlations
        correlations = df[numeric_cols].corr()

        # Save results
        stats_file = run_dir / "descriptive_statistics.csv"
        corr_file = run_dir / "correlations.csv"

        stats.to_csv(stats_file)
        correlations.to_csv(corr_file)

        logger.info(f"Statistics saved to {stats_file}")
        logger.info(f"Correlations saved to {corr_file}")

        return TaskResult(
            data={"statistics": stats, "correlations": correlations},
            metadata={
                "n_numeric_columns": len(numeric_cols),
                "numeric_columns": list(numeric_cols),
            },
            files_generated=[stats_file, corr_file],
        )

    def create_report(self, config, run_dir, logger, analyze_data):
        """Create a summary report.

        Args:
            config: Pipeline configuration.
            run_dir: Run directory for outputs.
            logger: Logger instance.
            analyze_data: TaskResult from analyze_data task.

        Returns:
            TaskResult with report file.
        """
        logger.info("Creating summary report")

        results = analyze_data.data
        stats = results["statistics"]
        corr = results["correlations"]

        # Create markdown report
        report = []
        report.append(f"# Analysis Report: {config.pipeline_name}")
        report.append(f"\nVersion: {config.version}\n")

        report.append("## Summary Statistics\n")
        report.append(stats.to_markdown())

        report.append("\n## Correlation Matrix\n")
        report.append(corr.to_markdown())

        # Save report
        report_file = run_dir / "report.md"
        report_file.write_text("\n".join(report))

        logger.info(f"Report saved to {report_file}")

        return TaskResult(
            data={"report_path": report_file},
            files_generated=[report_file],
        )


def main():
    """Run the example pipeline."""
    # Create configuration
    config = PipelineConfig(
        pipeline_name="simple_analysis",
        version="1.0",
    )

    # Set input data path (adjust this to your data file)
    config.data.input_path = "data.csv"

    # Create and run pipeline
    pipeline = SimpleAnalysisPipeline(
        config=config,
        output_dir="./pipeline_outputs",
    )

    print("Starting pipeline...")
    results = pipeline.run()

    print("\nPipeline completed successfully!")
    print(f"Results saved to: {pipeline.run_dir}")

    # Access specific results
    report_path = results["create_report"].data["report_path"]
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
