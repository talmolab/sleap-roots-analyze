---
description: Verify scientific accuracy and reproducibility of pipeline outputs
---

# Verify Pipeline Results

Verify the scientific accuracy and reproducibility of QC pipeline outputs.

## Purpose

This command validates pipeline outputs for:

1. CSV output schema correctness (expected columns present)
2. Metadata JSON completeness (required fields for reproducibility)
3. Statistical value sanity checks (power ranges, FDR, CIs)
4. Cross-run reproducibility (compare outputs between runs)
5. Anomaly detection (all-zero values, NaN prevalence, missing columns)

## Arguments

$ARGUMENTS

Provide the output directory path to verify. If not provided, ask for it.

## Step 1: Verify Output Structure

Check that the expected output files exist:

```python
from pathlib import Path

output_dir = Path("<output_dir>")

# Expected files for a QC pipeline run
expected_patterns = [
    "*.csv",           # Data outputs
    "*_metadata.json", # Metadata files
    "pipeline_summary.json",  # Pipeline summary
]

for pattern in expected_patterns:
    files = list(output_dir.rglob(pattern))
    if files:
        print(f"Found {len(files)} files matching {pattern}")
        for f in files:
            print(f"  {f.name}")
    else:
        print(f"WARNING: No files matching {pattern}")
```

## Step 2: Verify CSV Schema

For each CSV output, check that expected columns are present:

```python
import pandas as pd

for csv_file in output_dir.rglob("*.csv"):
    df = pd.read_csv(csv_file)
    print(f"\n{csv_file.name}: {len(df)} rows, {len(df.columns)} columns")

    # Check for common required columns
    for col in ["genotype", "trait"]:
        if col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique values")

    # Check for NaN prevalence
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"  Columns with NaN: {nan_cols}")
        for col in nan_cols:
            pct = df[col].isna().mean() * 100
            print(f"    {col}: {pct:.1f}% NaN")
```

## Step 3: Verify Metadata JSON

Check metadata files for required fields:

```python
import json

for meta_file in output_dir.rglob("*_metadata.json"):
    with open(meta_file) as f:
        meta = json.load(f)

    print(f"\n{meta_file.name}:")

    # Check for statistical metadata
    required_fields = [
        "fdr_correction_method",
        "confidence_level",
        "n_comparisons",
    ]

    for field in required_fields:
        if field in meta:
            print(f"  {field}: {meta[field]}")
        else:
            print(f"  WARNING: Missing {field}")

    # Check for power analysis fields (cross-platform only)
    power_fields = ["power_analysis_alpha", "minimum_detectable_r", "modal_n"]
    has_power = any(f in meta for f in power_fields)
    if has_power:
        for field in power_fields:
            if field in meta:
                print(f"  {field}: {meta[field]}")
            else:
                print(f"  WARNING: Missing power analysis field: {field}")
```

## Step 4: Statistical Sanity Checks

Verify that statistical values are within expected ranges:

```python
# For cross-platform correlation outputs
for csv_file in output_dir.rglob("*correlation*.csv"):
    df = pd.read_csv(csv_file)
    print(f"\n{csv_file.name} - Statistical Sanity Checks:")

    # Correlation coefficients should be in [-1, 1]
    for col in ["r_value", "spearman_r"]:
        if col in df.columns:
            r_min, r_max = df[col].min(), df[col].max()
            ok = -1 <= r_min and r_max <= 1
            print(f"  {col} range: [{r_min:.4f}, {r_max:.4f}] {'OK' if ok else 'OUT OF RANGE'}")

    # P-values should be in [0, 1]
    for col in ["p_value", "p_value_fdr"]:
        if col in df.columns:
            p_min, p_max = df[col].min(), df[col].max()
            ok = 0 <= p_min and p_max <= 1
            print(f"  {col} range: [{p_min:.4f}, {p_max:.4f}] {'OK' if ok else 'OUT OF RANGE'}")

    # Power should be in [0, 1]
    if "achieved_power" in df.columns:
        pow_min, pow_max = df["achieved_power"].min(), df["achieved_power"].max()
        ok = 0 <= pow_min and pow_max <= 1
        print(f"  achieved_power range: [{pow_min:.4f}, {pow_max:.4f}] {'OK' if ok else 'OUT OF RANGE'}")

        # Flag if all power is 0 (likely a bug)
        if pow_max == 0:
            print(f"  ANOMALY: All achieved_power values are 0.0")

    # Check FDR significance
    if "significant_fdr" in df.columns:
        n_sig = df["significant_fdr"].sum()
        n_total = len(df)
        print(f"  FDR significant: {n_sig}/{n_total} ({n_sig/n_total*100:.1f}%)")
```

## Step 5: Reproducibility Check (Optional)

Compare outputs between two runs:

```python
# Compare pipeline_summary.json between runs
run1 = Path("<output_dir_1>/pipeline_summary.json")
run2 = Path("<output_dir_2>/pipeline_summary.json")

if run1.exists() and run2.exists():
    with open(run1) as f:
        summary1 = json.load(f)
    with open(run2) as f:
        summary2 = json.load(f)

    # Compare key fields
    for key in ["pipeline_name", "version", "total_tasks", "completed_tasks"]:
        v1 = summary1.get(key)
        v2 = summary2.get(key)
        match = v1 == v2
        print(f"  {key}: {'MATCH' if match else 'MISMATCH'} ({v1} vs {v2})")
```

## Step 6: Report Summary

Summarize findings:

```
## Verification Report

### Output Structure
- CSV files: X found
- Metadata files: X found
- Pipeline summary: Present/Missing

### Data Quality
- Total rows across all CSVs: X
- Columns with NaN: [list]
- Anomalies detected: [list]

### Statistical Validity
- Correlation range: OK/OUT OF RANGE
- P-value range: OK/OUT OF RANGE
- Power range: OK/OUT OF RANGE
- FDR significant: X/Y (Z%)

### Reproducibility
- Cross-run comparison: MATCH/MISMATCH

### Verdict: PASS / FAIL / NEEDS REVIEW
```

## Common Anomalies

1. **All power = 0**: Sample size too small for any detectable effect
2. **All p_value_fdr = 1.0**: FDR correction eliminated all significance (expected with many tests)
3. **NaN in r_value**: Insufficient data for correlation (check min_genotypes filter)
4. **Missing columns**: Pipeline configuration may have changed between runs

## Integration

- Run after `/run-pipelines` to verify outputs
- Use before publishing results or sharing data
- Run `/coverage` to ensure test coverage for statistical functions
- Use `/validate-config` to check pipeline configuration before running
