# Pipeline Infrastructure

A lightweight DAG-based pipeline framework for executing data analysis workflows with explicit dependency management.

Built using NetworkX for graph management, consistent with the SLEAP Roots ecosystem.

## Features

- **DAG-based execution**: Tasks are executed in topological order based on their dependencies
- **NetworkX-powered**: Uses NetworkX for graph operations, aligning with SLEAP Roots' approach
- **DAG visualization**: Visualize pipeline structure with matplotlib
- **Structured configuration**: OmegaConf-based configuration management
- **Run tracking**: Automatic generation of JSON summaries for each pipeline run
- **Flexible task interface**: Tasks can be any callable with explicit dependencies

## Quick Start

### Creating a Simple Pipeline

```python
from sleap_roots_analyze.pipeline import (
    BasePipeline,
    Task,
    TaskResult,
    PipelineConfig,
)
import pandas as pd

class MyAnalysisPipeline(BasePipeline):
    """A simple analysis pipeline."""

    def create_tasks(self):
        """Define the pipeline tasks and their dependencies."""
        return [
            Task(
                func=self.load_data,
                name="load_data",
                description="Load input data from CSV",
            ),
            Task(
                func=self.clean_data,
                name="clean_data",
                depends_on=["load_data"],
                description="Remove missing values",
            ),
            Task(
                func=self.analyze_data,
                name="analyze_data",
                depends_on=["clean_data"],
                description="Perform statistical analysis",
            ),
        ]

    def load_data(self, config, run_dir, logger):
        """Load data from file."""
        logger.info(f"Loading data from {config.data.input_path}")
        df = pd.read_csv(config.data.input_path)
        return TaskResult(data=df)

    def clean_data(self, config, run_dir, logger, load_data):
        """Clean the loaded data."""
        df = load_data.data
        logger.info(f"Original shape: {df.shape}")
        df_clean = df.dropna()
        logger.info(f"After cleaning: {df_clean.shape}")
        return TaskResult(data=df_clean)

    def analyze_data(self, config, run_dir, logger, clean_data):
        """Analyze the cleaned data."""
        df = clean_data.data

        # Perform analysis
        results = df.describe()

        # Save results
        output_file = run_dir / "analysis_results.csv"
        results.to_csv(output_file)

        return TaskResult(
            data=results,
            files_generated=[output_file],
        )

# Create configuration
config = PipelineConfig(
    pipeline_name="my_analysis",
    version="1.0",
)
config.data.input_path = "data.csv"

# Run the pipeline
pipeline = MyAnalysisPipeline(
    config=config,
    output_dir="./outputs",
)
results = pipeline.run()
```

### Using Configuration Files

Create a YAML configuration file:

```yaml
# config.yaml
pipeline_name: my_analysis
version: 1.0

data:
  input_path: data.csv
  output_dir: ./outputs
  min_heritability: 0.3

outlier_detection:
  method: mahalanobis
  threshold: 0.01
  use_pca: true
  n_components: 0.95

pca:
  n_components: 0.95
  standardize: true
  feature_selection_strategy: top_variance
  n_top_features: 10

visualization:
  create_pca_plots: true
  create_outlier_plots: true
  interactive: false
  dpi: 300
```

Load and use the configuration:

```python
from sleap_roots_analyze.pipeline import load_config

# Load configuration
config = load_config("config.yaml")

# Create and run pipeline
pipeline = MyAnalysisPipeline(config=config, output_dir=config.data.output_dir)
results = pipeline.run()
```

## Core Components

### Task

A `Task` wraps a callable function as a node in the DAG:

```python
from sleap_roots_analyze.pipeline import Task, TaskResult

def my_function(config, run_dir, logger, dependency_name):
    # Access dependency results
    previous_data = dependency_name.data

    # Do work
    result = process(previous_data)

    # Return result
    return TaskResult(
        data=result,
        metadata={"rows_processed": len(result)},
        files_generated=["output.csv"],
    )

task = Task(
    func=my_function,
    name="my_task",
    depends_on=["dependency_name"],
    description="Process the data",
)
```

### DAGExecutor

The `DAGExecutor` validates and executes tasks in topological order using NetworkX:

```python
from sleap_roots_analyze.pipeline import DAGExecutor

executor = DAGExecutor([task1, task2, task3])

# Validate DAG structure
executor.validate()  # Raises DAGValidationError if cycles exist

# Get execution order (uses NetworkX topological sort)
order = executor.get_execution_order()  # ['task1', 'task2', 'task3']

# Visualize the DAG structure
executor.visualize("pipeline_dag.png")  # Requires matplotlib

# Execute
results = executor.execute(config, run_dir, logger)
```

### BasePipeline

Extend `BasePipeline` to create custom pipelines:

```python
from sleap_roots_analyze.pipeline import BasePipeline

class MyPipeline(BasePipeline):
    def create_tasks(self):
        """Return list of Task objects."""
        return [
            Task(func=self.task1, name="task1"),
            Task(func=self.task2, name="task2", depends_on=["task1"]),
        ]

    def task1(self, config, run_dir, logger):
        return TaskResult(data="result1")

    def task2(self, config, run_dir, logger, task1):
        return TaskResult(data=f"processed_{task1.data}")
```

### PipelineSummary

Every pipeline run automatically generates a summary:

```json
{
  "pipeline_name": "my_analysis",
  "version": "1.0",
  "start_time": "2024-10-21T14:30:00",
  "end_time": "2024-10-21T14:32:15",
  "total_elapsed_time": 135.2,
  "status": "success",
  "steps": [
    {
      "name": "load_data",
      "description": "Load input data",
      "status": "success",
      "elapsed_time": 2.5,
      "files_generated": [],
      "metadata": {"rows": 1000}
    }
  ],
  "output_directory": "./outputs/my_analysis_20241021_143000"
}
```

## Configuration

### Configuration Structure

The configuration system uses OmegaConf for structured configs:

- `DataConfig`: Input/output paths and data filtering
- `OutlierDetectionConfig`: Outlier detection parameters
- `PCAConfig`: PCA analysis parameters
- `ClusteringConfig`: Clustering parameters
- `VisualizationConfig`: Visualization options
- `LoggingConfig`: Logging settings

### Merging Configurations

```python
from sleap_roots_analyze.pipeline import get_default_config, merge_configs

# Get default configuration
base_config = get_default_config("my_pipeline")

# Override specific values
overrides = {
    "data": {"input_path": "new_data.csv"},
    "pca": {"n_components": 0.9},
}

config = merge_configs(base_config, overrides)
```

### Validating Configuration

```python
from sleap_roots_analyze.pipeline import validate_config

config.data.input_path = "data.csv"
config.pipeline_name = "test"

# Raises ValueError if configuration is invalid
validate_config(config)
```

## Utilities

### Environment Information

```python
from sleap_roots_analyze.pipeline import get_environment_info

env_info = get_environment_info()
# Returns: {
#     'git_commit': 'abc123...',
#     'git_branch': 'main',
#     'pandas': '2.0.0',
#     'numpy': '1.24.0',
#     ...
# }
```

### Run Directories

```python
from sleap_roots_analyze.pipeline import create_run_directory

run_dir = create_run_directory("./outputs", "my_pipeline")
# Creates: ./outputs/my_pipeline_20241021_143052
```

## Advanced Usage

### Complex DAG Example

```python
# Create a DAG with parallel branches:
#       load
#      /    \
#   clean1  clean2
#      \    /
#      merge
#        |
#      analyze

tasks = [
    Task(func=load, name="load"),
    Task(func=clean1, name="clean1", depends_on=["load"]),
    Task(func=clean2, name="clean2", depends_on=["load"]),
    Task(func=merge, name="merge", depends_on=["clean1", "clean2"]),
    Task(func=analyze, name="analyze", depends_on=["merge"]),
]

pipeline = MyPipeline(config, output_dir="./outputs")
```

### Custom TaskResults

Tasks can return custom metadata and file lists:

```python
def my_task(config, run_dir, logger):
    # Generate output file
    output_file = run_dir / "results.csv"
    df.to_csv(output_file)

    return TaskResult(
        data=df,
        metadata={
            "rows": len(df),
            "columns": list(df.columns),
            "processing_time": elapsed,
        },
        files_generated=[output_file, run_dir / "plot.png"],
    )
```

## DAG Visualization

The pipeline uses NetworkX for DAG management, allowing you to visualize the task dependency graph. This requires matplotlib to be installed.

### Basic Visualization

```python
from sleap_roots_analyze.pipeline import DAGExecutor, Task

# Create tasks
tasks = [
    Task(func=load_data, name="load_data"),
    Task(func=clean_data, name="clean_data", depends_on=["load_data"]),
    Task(func=analyze_data, name="analyze_data", depends_on=["clean_data"]),
]

# Create executor and visualize
executor = DAGExecutor(tasks)
executor.visualize("pipeline_structure.png")
```

### Custom Styling

```python
executor.visualize(
    "pipeline_structure.png",
    figsize=(14, 10),
    node_color="lightgreen",
    node_size=3000,
    font_size=12,
    edge_color="gray",
)
```

### Alignment with SLEAP Roots

This approach is consistent with [SLEAP Roots' trait pipeline implementation](https://github.com/talmolab/sleap-roots/blob/main/sleap_roots/trait_pipelines.py), which also uses NetworkX for DAG management.

## Error Handling

Pipelines automatically handle errors and save summaries even on failure:

```python
try:
    results = pipeline.run()
except Exception as e:
    # Summary is still saved with status="failed"
    summary = pipeline.get_summary()
    print(f"Pipeline failed: {e}")
    print(f"Summary saved to: {pipeline.run_dir / 'pipeline_summary.json'}")
```

## Future Enhancements

- **Parallel execution**: The DAG framework supports parallel execution of independent tasks (currently sequential)
- **Caching**: Cache task results to avoid re-running unchanged tasks
- **Pipeline composition**: Combine multiple pipelines into larger workflows