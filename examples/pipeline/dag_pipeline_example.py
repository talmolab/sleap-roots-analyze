r"""Example of a pipeline with parallel DAG branches.

This example demonstrates a more complex pipeline with parallel processing:

        load_data
       /    |    \
  filter1 filter2 filter3
       \    |    /
         merge
           |
        analyze
"""

from __future__ import annotations

import logging

import pandas as pd

from sleap_roots_analyze.pipeline import (
    BasePipeline,
    PipelineConfig,
    Task,
    TaskResult,
)

logging.basicConfig(level=logging.INFO)


class ParallelFilterPipeline(BasePipeline):
    """Pipeline with parallel data filtering branches."""

    def create_tasks(self):
        r"""Create tasks with parallel branches.

        DAG structure:
                load_data
               /    |    \
          filter1 filter2 filter3
               \    |    /
                 merge
                   |
                analyze
        """
        return [
            # Load data (no dependencies)
            Task(
                func=self.load_data,
                name="load_data",
                description="Load raw data",
            ),
            # Parallel filtering (all depend on load_data)
            Task(
                func=self.filter_high_values,
                name="filter_high",
                depends_on=["load_data"],
                description="Filter high values",
            ),
            Task(
                func=self.filter_low_values,
                name="filter_low",
                depends_on=["load_data"],
                description="Filter low values",
            ),
            Task(
                func=self.filter_outliers,
                name="filter_outliers",
                depends_on=["load_data"],
                description="Filter outliers",
            ),
            # Merge results (depends on all filters)
            Task(
                func=self.merge_results,
                name="merge",
                depends_on=["filter_high", "filter_low", "filter_outliers"],
                description="Merge filtered datasets",
            ),
            # Final analysis (depends on merge)
            Task(
                func=self.analyze,
                name="analyze",
                depends_on=["merge"],
                description="Analyze merged data",
            ),
        ]

    def load_data(self, config, run_dir, logger):
        """Load data from CSV."""
        logger.info(f"Loading data from {config.data.input_path}")
        df = pd.read_csv(config.data.input_path)

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        return TaskResult(
            data={"df": df, "numeric_cols": numeric_cols},
            metadata={"n_rows": len(df), "n_numeric_cols": len(numeric_cols)},
        )

    def filter_high_values(self, config, run_dir, logger, load_data):
        """Filter rows with high values (> 75th percentile)."""
        df = load_data.data["df"]
        numeric_cols = load_data.data["numeric_cols"]

        logger.info("Filtering high values (> 75th percentile)")

        # Calculate 75th percentile for each numeric column
        high_threshold = df[numeric_cols].quantile(0.75)

        # Keep rows where at least one value is above threshold
        mask = (df[numeric_cols] > high_threshold).any(axis=1)
        df_filtered = df[mask]

        logger.info(f"Kept {len(df_filtered)}/{len(df)} rows with high values")

        output_file = run_dir / "high_values.csv"
        df_filtered.to_csv(output_file, index=False)

        return TaskResult(
            data=df_filtered,
            metadata={"n_rows": len(df_filtered), "filter": "high"},
            files_generated=[output_file],
        )

    def filter_low_values(self, config, run_dir, logger, load_data):
        """Filter rows with low values (< 25th percentile)."""
        df = load_data.data["df"]
        numeric_cols = load_data.data["numeric_cols"]

        logger.info("Filtering low values (< 25th percentile)")

        # Calculate 25th percentile for each numeric column
        low_threshold = df[numeric_cols].quantile(0.25)

        # Keep rows where at least one value is below threshold
        mask = (df[numeric_cols] < low_threshold).any(axis=1)
        df_filtered = df[mask]

        logger.info(f"Kept {len(df_filtered)}/{len(df)} rows with low values")

        output_file = run_dir / "low_values.csv"
        df_filtered.to_csv(output_file, index=False)

        return TaskResult(
            data=df_filtered,
            metadata={"n_rows": len(df_filtered), "filter": "low"},
            files_generated=[output_file],
        )

    def filter_outliers(self, config, run_dir, logger, load_data):
        """Filter rows with outlier values (> 3 std from mean)."""
        df = load_data.data["df"]
        numeric_cols = load_data.data["numeric_cols"]

        logger.info("Filtering outliers (> 3 std from mean)")

        # Calculate z-scores
        z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

        # Keep rows where at least one value is an outlier
        mask = (z_scores.abs() > 3).any(axis=1)
        df_filtered = df[mask]

        logger.info(f"Found {len(df_filtered)} outlier rows")

        output_file = run_dir / "outliers.csv"
        df_filtered.to_csv(output_file, index=False)

        return TaskResult(
            data=df_filtered,
            metadata={"n_rows": len(df_filtered), "filter": "outliers"},
            files_generated=[output_file],
        )

    def merge_results(
        self, config, run_dir, logger, filter_high, filter_low, filter_outliers
    ):
        """Merge the filtered datasets."""
        logger.info("Merging filtered datasets")

        # Create summary of each filter's results
        summary = {
            "high_values": len(filter_high.data),
            "low_values": len(filter_low.data),
            "outliers": len(filter_outliers.data),
        }

        logger.info(f"Filter summary: {summary}")

        # Save summary
        import json

        summary_file = run_dir / "filter_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))

        return TaskResult(
            data=summary,
            metadata=summary,
            files_generated=[summary_file],
        )

    def analyze(self, config, run_dir, logger, merge):
        """Analyze the merged results."""
        summary = merge.data

        logger.info("Creating final analysis report")

        # Create report
        report = []
        report.append("# Parallel Filter Analysis Report\\n")
        report.append(f"Pipeline: {config.pipeline_name}\\n")
        report.append("## Filter Results\\n")

        for filter_name, count in summary.items():
            report.append(f"- **{filter_name}**: {count} rows\\n")

        total = sum(summary.values())
        report.append(f"\\n**Total unique rows across filters**: {total}\\n")

        # Save report
        report_file = run_dir / "analysis_report.md"
        report_file.write_text("".join(report))

        logger.info(f"Analysis complete. Report saved to {report_file}")

        return TaskResult(
            data={"total_rows": total},
            files_generated=[report_file],
        )


def main():
    """Run the parallel DAG pipeline example."""
    # Create configuration
    config = PipelineConfig(
        pipeline_name="parallel_filter_analysis",
        version="1.0",
    )

    # Set input data path
    config.data.input_path = "data.csv"

    # Create and run pipeline
    pipeline = ParallelFilterPipeline(
        config=config,
        output_dir="./pipeline_outputs",
    )

    print("Starting parallel pipeline...")
    print("\\nDAG Structure:")
    print("        load_data")
    print("       /    |    \\\\")
    print("  filter1 filter2 filter3")
    print("       \\\\    |    /")
    print("         merge")
    print("           |")
    print("        analyze\\n")

    results = pipeline.run()

    print("\\nPipeline completed successfully!")
    print(f"Results saved to: {pipeline.run_dir}")

    # Show execution order
    print("\\nExecution order was:")
    summary = pipeline.get_summary()
    for i, step in enumerate(summary.steps, 1):
        print(f"{i}. {step.name} ({step.elapsed_time:.2f}s)")


if __name__ == "__main__":
    main()
