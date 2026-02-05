---
description: Generate detailed cross-platform analysis summary with visualizations and statistics
---

# Cross-Platform Summary

Generate a comprehensive summary report for cross-platform correlation analysis results.

## Purpose

This command produces detailed markdown summaries for cross-platform analysis runs, including:

1. Trait reduction statistics (clustering results per experiment)
2. Correlation statistics (total, significant, by direction)
3. Top correlations table with effect sizes and p-values
4. Power analysis statistics (achieved power, minimum detectable r)
5. Configuration information and validation status
6. Links to generated visualizations (dendrograms, heatmaps, joint plots)

## Arguments

$ARGUMENTS

Provide the cross-platform output directory path. If not provided, ask for it or use the latest run.

## Step 1: Locate Cross-Platform Output

Find the cross-platform analysis output directory:

```python
from pathlib import Path

# If argument provided, use it directly
output_dir = Path("$ARGUMENTS") if "$ARGUMENTS" else None

# Otherwise, find the latest cross-platform run
if not output_dir or not output_dir.exists():
    pipeline_runs = Path("pipeline_runs")
    if pipeline_runs.exists():
        # Look for cross_platform subdirectories
        cross_dirs = list(pipeline_runs.rglob("cross_platform/*"))
        if cross_dirs:
            # Sort by modification time
            latest = max(cross_dirs, key=lambda p: p.stat().st_mtime)
            output_dir = latest
            print(f"Using latest cross-platform run: {output_dir}")
```

## Step 2: Generate Summary Using CrossPlatformSummaryGenerator

Use the tested Python implementation to generate the summary:

```python
from sleap_roots_analyze.summary.cross_platform_summary import (
    CrossPlatformSummaryGenerator,
)

# Initialize generator with output directory
generator = CrossPlatformSummaryGenerator(output_dir)

# Generate summary (parses all run directories)
summary = generator.generate()

# Convert to markdown
markdown_report = summary.to_markdown()

print(markdown_report)
```

## Step 3: Display Summary Report

The generated markdown includes:

### Trait Reduction Section
- Original vs reduced trait counts per experiment
- Number of clusters formed
- Clustering parameters (threshold, linkage method)
- Links to dendrograms and cluster heatmaps

### Correlation Statistics Section
- Total correlations tested
- Significant correlations (before/after FDR correction)
- Positive vs negative correlations
- Distribution of effect sizes

### Top Correlations Table
- Trait pairs ranked by absolute correlation
- Spearman r values with confidence intervals
- Raw and FDR-adjusted p-values
- Achieved power for each correlation

### Power Analysis Section
- Alpha level used
- Minimum detectable effect size
- Sample sizes (modal n genotypes)
- Well-powered vs underpowered correlation counts

### Validation Guardrails
- Green checks for passing validations
- Yellow warnings for potential issues
- Red alerts for failures requiring attention

## Step 4: Check Visualizations

List generated visualization files:

```python
# Find all visualization files
viz_patterns = [
    "*_dendrogram.png",
    "*_heatmap.png",
    "*joint_plot*.png",
    "*boxplot*.png",
    "*correlation_heatmap*.png",
]

for pattern in viz_patterns:
    files = list(output_dir.rglob(pattern))
    if files:
        print(f"\n{pattern}:")
        for f in files:
            print(f"  {f.relative_to(output_dir)}")
```

## Example Output

```markdown
# Cross-Platform Analysis Summary

## Run: cross_platform_turface_150vs19

### Trait Reduction
| Experiment | Original | Clusters | Representatives |
|------------|----------|----------|-----------------|
| exp1       | 45       | 12       | 12              |
| exp2       | 38       | 10       | 10              |

Clustering: threshold=0.80, linkage=complete

### Correlation Statistics
- Total correlations: 120
- Significant (raw p < 0.05): 45 (37.5%)
- Significant (FDR q < 0.05): 18 (15.0%)
- Positive: 95 | Negative: 25

### Top 10 Correlations
| Exp1 Trait | Exp2 Trait | r | p_adj | Power |
|------------|------------|---|-------|-------|
| TotalLength | RootLength | 0.89 | 0.001 | 0.95 |
| ...        | ...        | ... | ...   | ...   |

### Power Analysis
- Alpha: 0.05
- Min detectable r: 0.45
- Modal n: 19 genotypes
- Well-powered (>0.8): 85/120 (70.8%)

### Validation Status
- [PASS] Correlation values in valid range [-1, 1]
- [PASS] P-values in valid range [0, 1]
- [WARN] 15 correlations have power < 0.5
```

## Integration

- Run after `/run-pipelines` completes cross-platform analysis
- Use with `/verify-results` for comprehensive validation
- Pair with `/validate-config` before running pipelines
- Results can be copied to reports or documentation

## Troubleshooting

### No cross-platform runs found
Ensure cross-platform pipelines ran successfully:
```bash
ls pipeline_runs/*/cross_platform/
```

### Missing correlation data
Check that the correlation step completed:
```bash
ls <output_dir>/*correlations*.csv
```

### No trait clusters found
Trait reduction may be disabled in config. Check:
```yaml
trait_reduction_method: "clustering"  # Must be "clustering" not "none"
trait_reduction_target: "both"        # Required when clustering enabled
```
