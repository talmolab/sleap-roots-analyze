# Tasks: PR #39 Code Quality Fixes

## Must Fix (Blocking Merge)

- [x] **Fix mutable default in calculate_genotype_statistics**
  - File: `src/sleap_roots_analyze/cross_experiment_analysis.py`
  - Line: 414
  - Change `statistics: List[str] = [...]` to `Optional[List[str]] = None`
  - Add runtime initialization

- [x] **Add replicate column validation**
  - File: `src/sleap_roots_analyze/cross_experiment_analysis.py`
  - Lines: 355-367
  - Detect multiple replicate column variants
  - Issue warning if found

- [x] **Add OSError handling for log path**
  - File: `src/sleap_roots_analyze/cli.py`
  - Lines: 139-142, 351-360
  - Wrap mkdir in try/except
  - Fall back to console-only logging

## Should Fix

- [x] **Remove redundant spearmanr import**
  - File: `src/sleap_roots_analyze/cross_experiment_analysis.py`
  - Line: 18
  - Remove `from scipy.stats import spearmanr`
  - Verify all usages work with `stats.spearmanr()`

- [x] **Add clarifying comment for logger handler check**
  - File: `src/sleap_roots_analyze/pipeline/pipelines/base_pipeline.py`
  - Lines: 111-114
  - Explain why FileHandler is excluded from StreamHandler check

## Testing

- [x] **Existing tests cover mutable default**
  - Verified: tests pass with new implementation

- [x] **Replicate column warning verified**
  - Warning appears in test output when multiple variants found

## Documentation

- [x] **Update docstring for statistics parameter**
  - Document that default is applied at runtime if None

## Verification

- [x] Run full test suite: `uv run pytest` - 1391 tests pass
- [x] Run linter: `uv run ruff check` - All checks passed
- [x] Run formatter: `uv run black` - Files reformatted
