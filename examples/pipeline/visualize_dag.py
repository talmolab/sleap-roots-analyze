"""Example showing how to visualize pipeline DAG structure.

This example demonstrates creating a pipeline with complex dependencies
and visualizing the DAG structure.
"""

from sleap_roots_analyze.pipeline import DAGExecutor, Task, TaskResult


def create_example_pipeline():
    """Create an example pipeline with branching dependencies."""

    # Define simple placeholder functions
    def load_data(config, run_dir, logger):
        logger.info("Loading data...")
        return TaskResult(data="loaded_data")

    def clean_data(config, run_dir, logger, load_data):
        logger.info("Cleaning data...")
        return TaskResult(data="cleaned_data")

    def feature_engineering(config, run_dir, logger, clean_data):
        logger.info("Engineering features...")
        return TaskResult(data="features")

    def pca_analysis(config, run_dir, logger, feature_engineering):
        logger.info("Running PCA...")
        return TaskResult(data="pca_results")

    def clustering(config, run_dir, logger, feature_engineering):
        logger.info("Running clustering...")
        return TaskResult(data="clusters")

    def outlier_detection(config, run_dir, logger, pca_analysis):
        logger.info("Detecting outliers...")
        return TaskResult(data="outliers")

    def create_report(
        config, run_dir, logger, pca_analysis, clustering, outlier_detection
    ):
        logger.info("Creating report...")
        return TaskResult(data="report")

    # Create tasks with dependencies
    tasks = [
        Task(
            func=load_data,
            name="load_data",
            description="Load raw data from file",
        ),
        Task(
            func=clean_data,
            name="clean_data",
            depends_on=["load_data"],
            description="Clean and preprocess data",
        ),
        Task(
            func=feature_engineering,
            name="feature_engineering",
            depends_on=["clean_data"],
            description="Engineer features for analysis",
        ),
        Task(
            func=pca_analysis,
            name="pca_analysis",
            depends_on=["feature_engineering"],
            description="Perform PCA dimensionality reduction",
        ),
        Task(
            func=clustering,
            name="clustering",
            depends_on=["feature_engineering"],
            description="Perform clustering analysis",
        ),
        Task(
            func=outlier_detection,
            name="outlier_detection",
            depends_on=["pca_analysis"],
            description="Detect outliers using Mahalanobis distance",
        ),
        Task(
            func=create_report,
            name="create_report",
            depends_on=["pca_analysis", "clustering", "outlier_detection"],
            description="Generate analysis report",
        ),
    ]

    return tasks


def main():
    """Create and visualize the pipeline DAG."""
    print("Creating example pipeline...")

    # Create tasks
    tasks = create_example_pipeline()

    # Create DAG executor
    executor = DAGExecutor(tasks)

    # Validate the DAG
    print("Validating DAG structure...")
    executor.validate()
    print("✓ DAG is valid (no cycles detected)")

    # Get execution order
    execution_order = executor.get_execution_order()
    print(f"\nExecution order ({len(execution_order)} tasks):")
    for i, task_name in enumerate(execution_order, 1):
        print(f"  {i}. {task_name}")

    # Visualize the DAG
    print("\nGenerating DAG visualization...")
    output_file = "pipeline_dag.png"
    executor.visualize(output_file, figsize=(14, 10))
    print(f"✓ DAG visualization saved to: {output_file}")

    # Show DAG structure as text
    print("\nDAG Structure:")
    print("              load_data")
    print("                  |")
    print("              clean_data")
    print("                  |")
    print("          feature_engineering")
    print("            /         \\")
    print("     pca_analysis   clustering")
    print("           |")
    print("   outlier_detection")
    print("            \\         /")
    print("           create_report")

    # Show dependencies
    print("\nTask Dependencies:")
    task_graph = executor.get_task_graph()
    for task_name, deps in task_graph.items():
        if deps:
            print(f"  {task_name} depends on: {', '.join(deps)}")
        else:
            print(f"  {task_name} (no dependencies)")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if "matplotlib" in str(e):
            print("\nError: matplotlib is required for visualization")
            print("Install with: pip install matplotlib")
        else:
            raise
