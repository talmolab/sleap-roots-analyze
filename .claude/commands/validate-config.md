# Validate QC Pipeline Configuration

Validate a QC pipeline configuration file and optionally show the execution plan with a dry-run.

## Task

You are helping the user validate their QC pipeline configuration. This command checks:
1. Configuration loads correctly
2. All required fields are present
3. Data files exist (if paths specified)
4. Root core configuration is valid (if present)
5. Pipeline can be initialized (dry-run)

## Instructions

### 1. Get the Config File Path

If the user hasn't provided a path, ask for it:
- "Which config file would you like to validate?"

### 2. Load and Validate Configuration

Use Python to load and validate:

```python
from pathlib import Path
from sleap_roots_analyze.pipeline import load_qc_config, QCPipeline

config_path = "path/to/config.yaml"  # Replace with actual path

try:
    # Load config (this validates structure)
    config = load_qc_config(config_path)

    # Try to initialize pipeline (dry-run check)
    pipeline = QCPipeline(config=config, output_dir="./temp_validation")

    # Get task list to show execution plan
    tasks = pipeline.create_tasks()

    print(f"✓ Configuration is VALID!")
    print(f"\nPipeline: {config.pipeline_name}")
    print(f"Version: {config.version}")

    # Check if root core enabled
    if config.root_core is not None:
        print(f"\n✓ Root Core Processing: ENABLED")
        print(f"  Sources: {len(config.root_core.sources)}")
        for i, src in enumerate(config.root_core.sources, 1):
            print(f"    {i}. {src.data_type}: {src.csv_path}")
        print(f"  Core QC: {'Enabled' if config.root_core.core_qc.enabled else 'Disabled'}")
    else:
        print(f"\n  Root Core Processing: Disabled")
        print(f"  Data: {config.data.csv_path}")

    # Show execution plan
    print(f"\n✓ Pipeline Execution Plan: {len(tasks)} steps")
    for task in tasks:
        print(f"  - {task.name}: {task.description}")

    # Show key settings
    print(f"\n✓ Key Settings:")
    outlier_methods = config.outlier_detection.traditional_methods + config.outlier_detection.clustering_methods
    print(f"  Outlier Detection: {', '.join(outlier_methods) if outlier_methods else 'None'}")
    print(f"  Heritability Filter: {'Enabled' if config.heritability.enabled else 'Disabled'}")
    if config.heritability.enabled:
        print(f"    Threshold: {config.heritability.threshold}")
    print(f"  PCA: {config.pca.n_components} components")

    print(f"\n✓ Configuration is ready to use!")
    print(f"\nRun with:")
    print(f"  sleap-roots-analyze qc {config_path}")

except FileNotFoundError as e:
    print(f"✗ File Not Found: {e}")
    print(f"\nFix: Check that the config file path is correct")

except ValueError as e:
    print(f"✗ Invalid Configuration: {e}")
    print(f"\nFix: Review the error above and correct the config file")

except Exception as e:
    print(f"✗ Validation Failed: {e}")
    print(f"\nThis could be due to:")
    print(f"  - Missing required fields")
    print(f"  - Invalid data types")
    print(f"  - Incorrect YAML syntax")
```

### 3. Check for Common Issues

If validation fails, help diagnose:

**Common Issues:**

1. **Missing root_core fields:**
   - Each source needs: `csv_path`, `data_type`, `depth_column_prefix`
   - Biomass sources need: `depth_mapping`

2. **Wrong field names:**
   - Use `heritability` not `filter_heritability`
   - Use `outlier_detection` not `outlier_methods`

3. **Invalid YAML:**
   - Check indentation (use spaces, not tabs)
   - Check quotes around strings with special chars

4. **Data files don't exist:**
   - Update paths in config to actual file locations

### 4. Show Helpful Next Steps

After successful validation:
- Suggest running with `--dry-run` flag
- Explain how to update data paths if needed
- Show example run command

## Example Interactions

**User:** "validate my config"
**Assistant:** "Which config file would you like to validate?"

**User:** "configs/qc_root_core_edpie.yaml"
**Assistant:** *Runs validation code above and shows results*

**User:** "check if configs/my_qc.yaml is correct"
**Assistant:** *Validates immediately since path was provided*

## Special Validation for Root Core Configs

When root_core is present, also check:
- At least one source is configured
- Each source has valid `data_type` ("biomass" or "counting")
- Biomass sources have `depth_mapping`
- Column prefixes are unique between sources

## Remember

- Always show the full execution plan (15 steps for root core, 10 for standard)
- Explain what each validation check does
- Provide actionable fixes for errors
- Be encouraging when config is valid!