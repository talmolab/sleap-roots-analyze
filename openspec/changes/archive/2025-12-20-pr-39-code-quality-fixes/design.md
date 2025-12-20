# Design: PR #39 Code Quality Fixes

## Overview

This document details the technical approach for fixing code quality issues identified in PR #39.

## Issue 1: Logger Handler Accumulation

### Problem

In `base_pipeline.py`, the `_setup_logger()` method adds handlers to the logger without checking if equivalent handlers already exist. In interactive sessions (notebooks, REPL), this causes duplicate log messages.

```python
# Current problematic code (lines 104-131)
logger = logging.getLogger(f"{self.pipeline_name}")
# ... adds StreamHandler and FileHandler without clearing existing
```

### Solution

Clear non-FileHandler handlers at the start, and only add StreamHandler if none exists:

```python
def _setup_logger(self) -> logging.Logger:
    logger = logging.getLogger(f"{self.pipeline_name}")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Check for existing StreamHandler (excluding FileHandlers)
    # Note: FileHandler inherits from StreamHandler, so we need explicit exclusion
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler is per-run, so always add a new one
    # (existing FileHandlers point to old run directories)
    log_file_path = self.run_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    self._file_handler = file_handler
    return logger
```

The current code already has this logic - verify it's working correctly and add a clarifying comment.

## Issue 2: Mutable Default Argument

### Problem

In `cross_experiment_analysis.py`, `calculate_genotype_statistics()` uses a mutable default:

```python
def calculate_genotype_statistics(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "genotype",
    statistics: List[str] = ["mean", "median", "min", "max", "std"],  # PROBLEM
) -> Dict[str, pd.DataFrame]:
```

If any caller mutates this list (e.g., `stats.append("count")`), future calls get the mutated default.

### Solution

Use `None` as default with runtime initialization:

```python
def calculate_genotype_statistics(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "genotype",
    statistics: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Calculate multiple statistics per genotype for all traits.

    Args:
        ...
        statistics: List of statistics to compute.
            Default: ["mean", "median", "min", "max", "std"]
    """
    if statistics is None:
        statistics = ["mean", "median", "min", "max", "std"]
    # ... rest of function
```

## Issue 3: Replicate Column Ambiguity

### Problem

In `cross_experiment_analysis.py`, `load_and_align_experiments()` searches for replicate columns:

```python
for possible_rep_col in [rep_col1, "Replicate", "replicate", "Rep", "rep"]:
    if possible_rep_col in exp1_df.columns and possible_rep_col != "replicate":
        rep_renames_exp1[possible_rep_col] = "replicate"
        break
```

If a dataset has both "Replicate" and "rep" columns, the first match wins silently.

### Solution

Add validation to detect multiple replicate column variants:

```python
def _find_replicate_column(df: pd.DataFrame, primary: str) -> str:
    """Find the replicate column, warning if multiple variants exist."""
    candidates = [primary, "Replicate", "replicate", "Rep", "rep"]
    found = [c for c in candidates if c in df.columns and c != "replicate"]

    if len(found) > 1:
        warnings.warn(
            f"Multiple replicate column variants found: {found}. Using '{found[0]}'.",
            UserWarning
        )

    return found[0] if found else None
```

## Issue 4: Log Path Error Handling

### Problem

In `cli.py`, log path creation can fail with OSError:

```python
log_path = output_dir / cfg.logging.log_file
log_path.parent.mkdir(parents=True, exist_ok=True)  # Can raise OSError
effective_log_file = str(log_path)
```

### Solution

Add try/except with user-friendly message:

```python
try:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    effective_log_file = str(log_path)
except OSError as e:
    console.print(f"[yellow]Warning: Could not create log directory: {e}[/yellow]")
    console.print("[yellow]Logging to console only.[/yellow]")
    effective_log_file = None
```

## Issue 5: Redundant Import

### Problem

```python
from scipy import stats
from scipy.stats import spearmanr  # Redundant - stats.spearmanr works
```

### Solution

Remove line 18: `from scipy.stats import spearmanr`

All usages already go through `stats.spearmanr()` or `spearmanr()` can be replaced with `stats.spearmanr()`.

## Testing Strategy

1. **Logger leak test**: Create multiple pipeline instances in same session, verify no duplicate messages
2. **Mutable default test**: Call `calculate_genotype_statistics` with default, verify isolation
3. **Replicate ambiguity test**: Create DataFrame with both "Replicate" and "rep", verify warning

## Files Modified

1. `src/sleap_roots_analyze/pipeline/pipelines/base_pipeline.py` - Add clarifying comment
2. `src/sleap_roots_analyze/cross_experiment_analysis.py` - Fix mutable default, add column validation
3. `src/sleap_roots_analyze/cli.py` - Add OSError handling
