# Proposal: Address Copilot Review Comments for PR #63

## Why

GitHub Copilot review of PR #63 (grouped pipeline execution) identified 8 code quality and robustness issues:

### Code Quality Issues
1. **Unused variable** - `original_pipeline_name` in utils.py (never used)
2. **Import organization** - `tempfile` and `copy` imported inline instead of at module top
3. **Documentation accuracy** - CI workflow references wrong issue number (#70 instead of #69)

### Robustness Improvements
4. **Error handling** - CSV read/group split failures lack contextual error messages
5. **Empty groups warning** - No warning when all groups filtered out (likely config error)
6. **Mixed-type groups** - `sorted()` fails on mixed string/numeric group values
7. **Silent failures** - Group pipeline failures don't log before continuing
8. **Early validation** - `group_by` column existence not validated until runtime

All issues are LOW-MEDIUM priority but should be addressed before merge for code quality.

## What Changes

### 1. Code Cleanup (Issues 1-3)

**src/sleap_roots_analyze/pipeline/utils.py:**
- Remove unused `original_pipeline_name` variable assignment
- Move `import tempfile` and `import copy` to top of file with other stdlib imports

**.github/workflows/ci.yml:**
- Fix issue reference in comment: `#70` → `#69`

### 2. Error Handling Improvements (Issue 4)

**src/sleap_roots_analyze/pipeline/utils.py - `run_grouped_pipelines()`:**

Current (line ~478-482):
```python
# Load data
df = pd.read_csv(config.data.csv_path)
logger.info(f"Loaded {len(df)} samples from {config.data.csv_path}")

# Split by group column
groups = split_data_by_group(df, group_by_column=group_by_column)
```

Enhanced:
```python
try:
    # Load data
    df = pd.read_csv(config.data.csv_path)
    logger.info(f"Loaded {len(df)} samples from {config.data.csv_path}")

    # Split by group column
    groups = split_data_by_group(df, group_by_column=group_by_column)
    logger.info(f"Split data into {len(groups)} groups: {list(groups.keys())}")
except FileNotFoundError as exc:
    msg = (
        f"Failed to read data CSV at '{config.data.csv_path}' while preparing "
        f"grouped pipelines (group_by='{group_by_column}')."
    )
    logger.error(msg)
    raise FileNotFoundError(msg) from exc
except pd.errors.EmptyDataError as exc:
    msg = (
        f"Data CSV at '{config.data.csv_path}' is empty or invalid while preparing "
        f"grouped pipelines (group_by='{group_by_column}')."
    )
    logger.error(msg)
    raise pd.errors.EmptyDataError(msg) from exc
except KeyError:
    # Preserve the original helpful KeyError message from split_data_by_group
    raise
except Exception as exc:
    msg = (
        f"Failed to load or split data for grouped pipelines "
        f"(csv_path='{config.data.csv_path}', group_by='{group_by_column}'): {exc}"
    )
    logger.error(msg)
    raise
```

### 3. Empty Groups Warning (Issue 5)

**src/sleap_roots_analyze/pipeline/utils.py - `run_grouped_pipelines()`:**

After line ~492 (`valid_groups = filter_valid_groups(...)`), add:
```python
logger.info(f"Retained {len(valid_groups)} valid groups after filtering")
if not valid_groups:
    logger.warning(
        "No valid groups remain after filtering with min_samples_per_trait=%s. "
        "This likely indicates a configuration issue or incompatible data.",
        min_samples,
    )
    if skipped_groups:
        logger.info(f"Skipped {len(skipped_groups)} groups: {skipped_groups}")
    return {}
```

### 4. Mixed-Type Group Handling (Issue 6)

**src/sleap_roots_analyze/pipeline/utils.py - `run_grouped_pipelines()`:**

Around line ~520, replace:
```python
for group_value in sorted(valid_groups.keys()):
```

With:
```python
try:
    sorted_group_values = sorted(valid_groups.keys())
except TypeError:
    logger.warning(
        "Mixed-type group values detected for '%s'; sorting by string "
        "representation to ensure consistent processing order.",
        group_by_column,
    )
    sorted_group_values = sorted(valid_groups.keys(), key=lambda v: str(v))

for group_value in sorted_group_values:
```

### 5. Log Group Pipeline Failures (Issue 7)

**src/sleap_roots_analyze/pipeline/utils.py - `run_grouped_pipelines()`:**

Around line ~557, the except block should log before continuing:
```python
        logger.info(f"Group {group_by_column}={group_value} completed successfully")
    except Exception:
        logger.exception(
            f"Group {group_by_column}={group_value} failed during pipeline execution"
        )
        # Continue processing remaining groups
```

**Note**: Current code has `except Exception: pass` which silently continues. The fix adds logging.

### 6. Early group_by Validation (Issue 8)

**src/sleap_roots_analyze/pipeline/config/utils.py - `validate_qc_config()`:**

Add validation after existing checks:
```python
def validate_qc_config(config: QCConfig, check_files: bool = True) -> None:
    """Validate QC configuration.

    Args:
        config: QC configuration to validate
        check_files: If True, verify data files exist (default: True)

    Raises:
        ValueError: If configuration is invalid
    """
    # ... existing validation ...

    # Validate group_by column exists in data (if group_by specified and check_files enabled)
    if check_files and config.data.group_by is not None:
        if config.data.csv_path and Path(config.data.csv_path).exists():
            try:
                df = pd.read_csv(config.data.csv_path, nrows=0)  # Read only header
                if config.data.group_by not in df.columns:
                    available = list(df.columns)
                    raise ValueError(
                        f"group_by column '{config.data.group_by}' not found in CSV. "
                        f"Available columns: {available}"
                    )
            except pd.errors.EmptyDataError:
                # Empty CSV will fail at runtime anyway, skip validation
                pass
```

## Impact

**Affected specs:**
- None (internal code quality improvements)

**Affected code:**
- `src/sleap_roots_analyze/pipeline/utils.py` - error handling, logging, robustness
- `src/sleap_roots_analyze/pipeline/config/utils.py` - early validation
- `.github/workflows/ci.yml` - documentation fix

**Tests to add:**
- `tests/test_grouped_pipeline_execution.py` - error handling tests
- `tests/test_pipeline_config.py` - group_by validation tests

**Breaking changes:** None. All changes are internal improvements.

**Migration:** No user action required.

## Validation

### Tests Required

1. **Error handling tests** (Issue 4):
   - Test FileNotFoundError with helpful message
   - Test EmptyDataError with helpful message
   - Test KeyError from split_data_by_group preserved
   - Test generic Exception with contextual message

2. **Empty groups warning test** (Issue 5):
   - Test warning logged when all groups filtered out
   - Test early return when no valid groups

3. **Mixed-type group test** (Issue 6):
   - Test sorting with mixed string/int group values
   - Test fallback to string-based sorting

4. **Failure logging test** (Issue 7):
   - Test exception logged when group pipeline fails
   - Verify remaining groups still processed

5. **Early validation test** (Issue 8):
   - Test ValueError raised when group_by column missing
   - Test validation skipped when check_files=False
   - Test validation skipped when csv_path doesn't exist yet

### Manual Testing

Run existing integration tests to ensure no regressions:
```bash
uv run pytest tests/test_grouped_pipeline_integration.py -v
```
