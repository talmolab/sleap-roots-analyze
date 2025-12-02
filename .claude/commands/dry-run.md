# Dry-Run Pipeline Command

Validate a QC or Viz pipeline configuration and show what would be executed without actually running it.

## Usage

```bash
sleap-roots-analyze qc <config_file> --dry-run
sleap-roots-analyze viz <config_file> --dry-run
```

## What It Does

1. **Load and validate** the configuration file
2. **Check** that all input files exist
3. **Display** the pipeline execution plan:
   - Number of steps
   - Step names and order
   - Key configuration parameters
   - Output directory location
4. **Exit** without running any analysis

## Examples

```bash
# Dry-run QC pipeline
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml --dry-run

# Dry-run Viz pipeline
sleap-roots-analyze viz configs/viz_example.yaml --dry-run -o ./custom_output
```

## Output Format

```
Loading configuration: configs/qc_turface_150genotypes.yaml
Pipeline: turface_150genotypes_qc
Data: C:/path/to/data.csv
Output: C:\repos\sleap-roots-analyze\qc_runs

Dry run mode - validation complete
Would execute QC pipeline with 10 steps:
  1. LoadData - Load and validate CSV data
  2. CleanupTraits - Remove problematic traits and samples
  3. ValidateClean - Validate no NaN values remain
  4. ExploratoryAnalysis - Generate EDA visualizations
  5. DetectOutliers - Detect outliers using configured methods
  6. VisualizeOutliers - Create outlier visualizations
  7. RemoveOutliers - Remove outliers based on strategy
  8. StatisticalAnalysis - Calculate ANOVA and heritability
  9. FilterHeritability - Filter low heritability traits
  10. GenerateSummary - Generate complete pipeline summary

Configuration is valid ✓
All input files exist ✓
```

## Benefits

- **Fast validation** - Check config without running full pipeline
- **Catch errors early** - Find configuration issues before analysis starts
- **Preview execution** - See what will happen before committing resources
- **CI/CD friendly** - Test configs in automated workflows

## Implementation Status

✅ Already implemented! The `--dry-run` flag is available in the CLI:
- Validates configuration
- Checks file existence
- Shows execution plan
- Exits without running analysis