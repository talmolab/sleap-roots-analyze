# Proposal: Fix Critical Bugs in Grouped Pipeline Execution

## Why

GitHub Copilot identified 3 critical bugs in the grouped pipeline implementation (PR #63) that compromise scientific reproducibility, correctness, and data integrity:

### Bug #1: Saved Configs Reference Non-Existent CSV Files (REPRODUCIBILITY BUG)
**File:** `src/sleap_roots_analyze/pipeline/utils.py:584`
**Severity:** HIGH - Breaks reproducibility guarantee

Grouped pipelines write each group's data to a temporary CSV in `/tmp`, update `config.data.csv_path` to point to it, run the pipeline, then delete the temp file. The pipeline saves the modified config to `<run_dir>/config.yaml` for reproducibility. **This saved config references a CSV path that no longer exists**, making it impossible to re-run.

**Impact:**
- Saved configs cannot be reused
- Violates reproducibility requirement
- Metadata tracing is broken
- Users cannot verify what data was actually processed

### Bug #2: CLI --group-by Flag Doesn't Trigger Viz Fan-Out (FUNCTIONAL BUG)
**File:** `src/sleap_roots_analyze/pipeline_runner.py:214`
**Severity:** CRITICAL - Core feature doesn't work

When `run-all --group-by plant_age_days` is used (CLI flag), the QC pipeline runs grouped correctly via the CLI flag, but viz fan-out detection only checks `config.data.group_by` (the config file value). If the config has no `group_by` or a different one, `qc_grouped_outputs` won't be populated and viz fan-out won't happen.

**Impact:**
- CLI-based grouping silently fails for viz
- Only config-based grouping works
- Core feature is broken
- Users get wrong results without error

**Why this wasn't caught:**
- Integration tests only tested config-based grouping
- No tests for CLI flag override
- No tests for `run-all --group-by` with config that has no group_by

### Bug #3: NaN Group Values Silently Dropped (DATA LOSS BUG)
**File:** `src/sleap_roots_analyze/pipeline/utils.py:378`
**Severity:** MEDIUM - Silent data loss

`split_data_by_group` uses `df[df[col] == value]` to filter each group. This doesn't match NaN rows (NaN != NaN in pandas). Samples with missing group labels are silently excluded from all groups.

**Impact:**
- Data loss without warning
- No visibility into dropped samples
- Incorrect sample counts
- Silent failure mode

## What Changes

### Fix #1: Persist Group CSVs or Keep Original Path

**Option A (PREFERRED):** Write group CSV to the group's output directory instead of temp file:
```python
# Instead of temp file:
group_csv_path = group_output_dir / f"00_input_data_{group_label}.csv"
group_df.to_csv(group_csv_path, index=False)
```

**Benefits:**
- Saved config points to a real, persistent file
- Input data is preserved in output directory
- Reproducible: can re-run using saved config
- Traceable: can inspect exact input data used

**Option B:** Keep config pointing to original CSV, filter in-memory in pipeline
- More complex, requires pipeline changes
- Defer to Option A

### Fix #2: Use Effective Group-By Column for Viz Fan-Out Detection

Track the effective grouping column (CLI flag takes precedence over config) and use it for viz fan-out detection:

```python
def _run_qc_pipelines(self):
    for config_path in self.qc_configs:
        # Determine effective group_by (CLI > config)
        effective_group_by = self.group_by  # CLI flag
        if effective_group_by is None:
            # Fall back to config file value
            effective_group_by = self._get_qc_config_group_by(config_path)

        # Run pipeline (respects CLI flag via _run_pipeline_command)
        result = self._run_pipeline_command(...)

        # Detect grouped outputs using EFFECTIVE group_by
        if effective_group_by is not None:
            grouped_outputs = self._find_grouped_qc_outputs(
                base_dir, effective_group_by
            )
            if grouped_outputs:
                self.qc_grouped_outputs[config_rel] = grouped_outputs
```

### Fix #3: Handle NaN Group Values Explicitly

Use `groupby()` with `dropna=False` to handle NaN values explicitly:

```python
def split_data_by_group(
    df: pd.DataFrame,
    group_by_column: str,
    handle_na: str = "warn_and_drop"  # or "treat_as_group"
) -> dict[Any, pd.DataFrame]:
    """Split DataFrame by unique values in group_by_column.

    Args:
        handle_na: How to handle NaN group values
            - "warn_and_drop": Log warning and exclude NaN rows (default)
            - "treat_as_group": Treat NaN as its own group
    """
    if group_by_column not in df.columns:
        raise KeyError(f"Column '{group_by_column}' not found")

    # Check for NaN values
    na_mask = df[group_by_column].isna()
    n_na = na_mask.sum()

    if n_na > 0:
        if handle_na == "warn_and_drop":
            logger.warning(
                f"Dropping {n_na}/{len(df)} samples with missing '{group_by_column}' "
                f"values during grouping"
            )
            df = df[~na_mask].copy()
        elif handle_na == "treat_as_group":
            # pandas groupby with dropna=False will treat NaN as a group
            pass

    # Use groupby to split (handles NaN correctly)
    groups = {}
    for group_value, group_df in df.groupby(group_by_column, dropna=(handle_na == "warn_and_drop")):
        groups[group_value] = group_df.copy()

    return groups
```

### Bug #4: ANOVA Error Handling Crashes Pipeline (BLOCKING BUG)
**File:** `src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py:113`
**Severity:** HIGH - Crashes pipeline, blocks tests

When ANOVA calculation fails (e.g., insufficient data), `calculate_anova_by_genotype()` returns an error message string instead of a result dict. The code then calls `.get()` on this string, causing an `AttributeError`.

```python
# Line 113 in statistical_analysis.py
"f_statistic": result.get("f_statistic"),  # ❌ result is a string, not a dict
```

**Impact:**
- Pipeline crashes instead of handling error gracefully
- Blocks tests from running (including our new grouped pipeline tests)
- No visibility into which traits failed ANOVA
- Poor user experience

**Why this wasn't caught:**
- Tests use simple data that always succeeds ANOVA
- No tests for ANOVA failure cases (low sample count, zero variance, etc.)

**Fix:** Check if result is a string (error message) before calling `.get()`:

```python
if isinstance(result, str):
    # Error case: result is an error message
    anova_records.append({
        "trait": trait,
        "f_statistic": None,
        "p_value": None,
        "eta_squared": None,
        "significant": None,
        "n_groups": None,
        "total_n": None,
        "error": result,  # Store the error message
    })
else:
    # Success case: result is a dict
    anova_records.append({
        "trait": trait,
        "f_statistic": result.get("f_statistic"),
        "p_value": result.get("p_value"),
        "eta_squared": result.get("eta_squared"),
        "significant": result.get("significant"),
        "n_groups": result.get("n_groups"),
        "total_n": result.get("total_n"),
        "error": None,
    })
```

## Impact

**Affected specs:**
- None (bug fixes, no spec changes)

**Affected code:**
- `src/sleap_roots_analyze/pipeline/utils.py` - Fix bugs #1 and #3
- `src/sleap_roots_analyze/pipeline_runner.py` - Fix bug #2
- `src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py` - Fix bug #4

**Breaking changes:** None (purely fixes)

**Migration:** No user action required. Existing configs work correctly after fixes.

## Testing Strategy

### TDD Approach (Write Tests FIRST)

**Phase 1: Write Failing Tests**
- Bug #1: Test that saved config.yaml has csv_path pointing to existing file
- Bug #2: Test CLI --group-by triggers viz fan-out when config has no group_by
- Bug #3: Test NaN handling logs warning and drops/includes based on option
- Bug #4: Test ANOVA error handling (string result instead of dict)

**Phase 2: Implement Fixes**
- Fix each bug to make tests pass

**Phase 3: Integration Tests**
- End-to-end test: run-all --group-by with config that has no group_by
- Verify: grouped QC outputs, grouped viz outputs, saved configs are valid
- Manual testing: Run real pipeline, inspect outputs, metadata, configs

**Phase 4: Manual Verification**
- Run actual pipeline with real data
- Inspect: output directories, saved configs, metadata files
- Verify: configs are reproducible (can re-run using saved config.yaml)
- Check: no data loss, all groups processed correctly
