# CLI Interface Specification

**Component**: CLI Interface  
**Version**: 1.0  
**Status**: Draft

## Overview

The CLI interface provides a command-line interface for running QC and Viz pipelines, validating configurations, and accessing package utilities. It replaces hardcoded dataset-specific scripts with a generalizable, testable CLI using Typer.

## ADDED Requirements

### Requirement: Config Path as Argument
The CLI SHALL accept configuration file paths as command-line arguments, not hardcode them. All commands that operate on configs SHALL require an explicit config path argument.

#### Scenario: Run QC with Any Config

**Given**: User has multiple QC config files for different datasets

```bash
ls configs/
  qc_turface_150genotypes.yaml
  qc_wheat_cylinder.yaml
  qc_my_experiment.yaml
```

**When**: User runs CLI with different configs

```bash
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml
sleap-roots-analyze qc configs/qc_wheat_cylinder.yaml
sleap-roots-analyze qc /absolute/path/to/qc_my_experiment.yaml
```

**Then**:
- Each command runs with the specified config
- No config paths are hardcoded
- Pipelines can process any dataset via config

#### Scenario: Missing Config File

**Given**: User provides non-existent config path

```bash
sleap-roots-analyze qc configs/nonexistent.yaml
```

**When**: CLI attempts to load config

**Then**:
- Clear error message: "Config file does not exist: configs/nonexistent.yaml"
- Exit code 1
- Suggests checking path and using `config list` to see examples

### Requirement: Output Directory Control
The CLI SHALL allow users to specify output directories via command-line options. Default output directories SHALL be `./qc_runs` for QC and `./viz_runs` for Viz pipelines.

#### Scenario: Custom Output Directory

**Given**: User wants results in specific location

```bash
sleap-roots-analyze qc config.yaml -o /data/experiment_2025/qc_results
```

**When**: Pipeline executes

**Then**:
- All outputs saved to `/data/experiment_2025/qc_results/<pipeline_name>_<timestamp>/`
- Directory created if it doesn't exist
- Pipeline prints final output location

#### Scenario: Default Output Directory

**Given**: User does not specify output directory

```bash
sleap-roots-analyze qc config.yaml
```

**When**: Pipeline executes

**Then**:
- Outputs saved to `./qc_runs/<pipeline_name>_<timestamp>/`
- Current working directory used as base

### Requirement: Logging Configuration
The CLI SHALL provide command-line flags to control logging verbosity and output. Users SHALL be able to set log levels, enable file logging, and control console output.

#### Scenario: Verbose Logging

**Given**: User wants detailed debugging information

```bash
sleap-roots-analyze qc config.yaml --verbose
```

**When**: Pipeline runs

**Then**:
- Logging level set to DEBUG
- All debug messages printed to console
- Includes detailed step information and timing

#### Scenario: Quiet Mode

**Given**: User wants minimal console output

```bash
sleap-roots-analyze qc config.yaml --quiet
```

**When**: Pipeline runs

**Then**:
- Logging level set to WARNING
- Only warnings and errors printed
- Progress information suppressed

#### Scenario: Log to File

**Given**: User wants to save logs

```bash
sleap-roots-analyze qc config.yaml --log-file pipeline.log
```

**When**: Pipeline runs

**Then**:
- Logs written to `pipeline.log`
- Console output remains normal
- File contains timestamped log entries

### Requirement: Config Validation
The CLI SHALL provide commands to validate configuration files before running pipelines. Validation SHALL check required fields, data types, file existence, and logical constraints.

#### Scenario: Validate Valid Config

**Given**: User has a valid config file

```bash
sleap-roots-analyze config validate configs/qc_turface_150genotypes.yaml
```

**When**: Validation runs

**Then**:
- Prints: "✓ Configuration is valid"
- Exit code 0
- Shows config summary (pipeline name, data path, key settings)

#### Scenario: Validate Invalid Config

**Given**: Config missing required field

```yaml
# config.yaml - missing pipeline_name
data:
  csv_path: "data.csv"
```

**When**: User validates

```bash
sleap-roots-analyze config validate config.yaml
```

**Then**:
- Prints: "✗ Configuration is invalid: Missing required field 'pipeline_name'"
- Exit code 1
- Shows which validation check failed

### Requirement: Dry Run Mode
The CLI SHALL support a dry-run mode that validates configuration and displays execution plan without running the pipeline. This allows users to verify settings before expensive computations.

#### Scenario: Dry Run Before Execution

**Given**: User wants to verify config before running

```bash
sleap-roots-analyze qc config.yaml --dry-run
```

**When**: Command executes

**Then**:
- Config loaded and validated
- Execution plan displayed (steps, expected outputs)
- Pipeline does NOT execute
- Exit code 0
- Message: "Dry run mode - would execute QC pipeline with 10 steps"

### Requirement: Help Text Generation
The CLI SHALL provide comprehensive help text for all commands and options. Help text SHALL be auto-generated from function signatures, type hints, and docstrings.

#### Scenario: Global Help

**Given**: User needs to see available commands

```bash
sleap-roots-analyze --help
```

**When**: Help displayed

**Then**:
- Shows package description
- Lists all commands: qc, viz, config, version
- Shows global options
- Includes usage examples

#### Scenario: Command-Specific Help

**Given**: User needs details on qc command

```bash
sleap-roots-analyze qc --help
```

**When**: Help displayed

**Then**:
- Shows command description
- Lists all arguments (config path)
- Lists all options (-o, --verbose, --dry-run, etc.)
- Shows examples with typical usage
- Includes default values

### Requirement: Entry Point Configuration
The CLI SHALL be accessible via the `sleap-roots-analyze` command installed by the package. The entry point SHALL be defined in `pyproject.toml` and point to a `main()` function.

#### Scenario: Package Installation

**Given**: User installs package

```bash
uv pip install sleap-roots-analyze
```

**When**: Installation completes

**Then**:
- `sleap-roots-analyze` command available in PATH
- Command works from any directory
- `python -m sleap_roots_analyze` also works

#### Scenario: Command Invocation

**Given**: Package is installed

**When**: User runs command

```bash
sleap-roots-analyze qc config.yaml
```

**Then**:
- Entry point `sleap_roots_analyze.cli:main` is invoked
- CLI app runs with provided arguments
- Equivalent to running `python -m sleap_roots_analyze qc config.yaml`

### Requirement: Error Handling
The CLI SHALL provide clear, actionable error messages for all failure modes. Exit codes SHALL indicate success (0) or failure (non-zero). Error messages SHALL guide users toward resolution.

#### Scenario: Invalid Config Format

**Given**: Config file has syntax errors

```yaml
# bad.yaml - invalid YAML
data:
  csv_path: "data.csv
```

**When**: User attempts to run

```bash
sleap-roots-analyze qc bad.yaml
```

**Then**:
- Clear error: "Failed to parse config: Invalid YAML syntax at line 3"
- Suggests fixing YAML format
- Exit code 1

#### Scenario: Data File Not Found

**Given**: Config references non-existent data file

```yaml
data:
  csv_path: "/nonexistent/data.csv"
```

**When**: Pipeline starts

**Then**:
- Error: "Data file not found: /nonexistent/data.csv"
- Suggests checking path in config
- Exit code 1

## References

- Design documentation: [../design.md](../design.md)
- Typer documentation: https://typer.tiangolo.com/
- Entry point configuration: `pyproject.toml:63-64`
- Current hardcoded script: `run_turface_qc.py`
