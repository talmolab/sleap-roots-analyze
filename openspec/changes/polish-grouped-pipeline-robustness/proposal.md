# Polish Grouped Pipeline Robustness

## Status
Proposed

## Summary
Address 7 code quality and robustness improvements identified in GitHub Copilot code review of the grouped pipeline implementation. These improvements enhance error handling, logging, edge case handling, and code cleanliness without changing core functionality.

## Why
**Context**: GitHub Copilot reviewed PR #63 (grouped pipeline implementation + 4 critical bug fixes) and identified 10 issues. Three were already fixed by our bug fix work (config persistence, CLI group-by detection, NaN handling). The remaining 7 issues range from critical error handling gaps to code quality improvements.

**Problem**:
1. **Silent Failures**: Pipeline execution failures during grouped runs are not logged, making debugging difficult
2. **Poor Edge Case Handling**: Empty groups after filtering and mixed-type group values cause silent failures or crashes
3. **Unclear Errors**: Generic errors when CSV loading or grouping fails don't provide actionable context
4. **Code Cleanliness**: Unused imports, variables, and incorrect issue references reduce code maintainability

**Impact**:
- Users experience silent failures without diagnostic information
- Edge cases (all groups filtered, mixed string/numeric IDs) cause cryptic errors
- Code reviewers and maintainers face unnecessary cognitive load from dead code

**User Experience Gap**: When a grouped pipeline fails, users get no feedback about which group failed or why, forcing manual log inspection.

## What
Implement 7 improvements to grouped pipeline robustness and code quality:

### HIGH Priority (Critical for Production)
1. **Pipeline Failure Logging** (`src/sleap_roots_analyze/pipeline/utils.py:671`)
   - Add exception logging with group context when pipeline.run() fails
   - Re-raise exception to fail fast (don't continue with partial results)

### MEDIUM Priority (Robustness)
2. **Empty Groups Warning** (`src/sleap_roots_analyze/pipeline/utils.py:607`)
   - Warn when all groups filtered out by min_samples_per_trait threshold
   - Log filtered group names for diagnostic context

3. **Mixed-Type Group Sorting** (`src/sleap_roots_analyze/pipeline/utils.py:609`)
   - Handle mixed string/numeric group values without TypeError
   - Fallback to string sorting with warning when types incompatible

4. **Contextualized Error Messages** (`src/sleap_roots_analyze/pipeline/utils.py:595`)
   - Wrap CSV loading and grouping in try-except blocks
   - Provide clear error messages mentioning group_by context

### LOW Priority (Code Quality)
5. **Remove Unused Imports** (`src/sleap_roots_analyze/pipeline/utils.py`)
   - Remove `tempfile` and `copy` imports (no longer needed after Bug #1 fix)

6. **Remove Unused Variable** (`src/sleap_roots_analyze/pipeline/utils.py`)
   - Remove `original_pipeline_name` variable (leftover from development)

7. **Fix Issue Reference** (`.github/workflows/ci.yml`)
   - Update comment from issue #70 to #69 (correct grouped pipeline issue)

## How

### Implementation Strategy: Test-Driven Development (TDD)

**Phase 1: Write Failing Tests (RED)**

Test file: `tests/test_grouped_pipeline_robustness.py`

**Test 1.1: Pipeline Failure Logging**
```python
def test_pipeline_failure_logs_group_context(self, caplog):
    """Pipeline failures should log the failing group for diagnostics."""
    # Mock pipeline that raises exception
    class FailingPipeline:
        def __init__(self, config, output_dir, **kwargs):
            self.config = config
            self.run_dir = output_dir
        def run(self):
            raise ValueError("Simulated pipeline failure")

    config = get_default_qc_config()
    config.data.group_by = "age"

    with pytest.raises(ValueError, match="Simulated pipeline failure"):
        with caplog.at_level(logging.ERROR):
            run_grouped_pipelines(
                config=config,
                output_dir=tmp_path / "output",
                pipeline_class=FailingPipeline,
                validate=False,
            )

    # Should have logged the failing group
    error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("age=" in r.message and "failed" in r.message for r in error_logs)
```

**Test 1.2: Empty Groups Warning**
```python
def test_empty_groups_after_filtering_warns(self, caplog):
    """When all groups filtered out, should log warning with context."""
    # Create data where all groups have < min_samples
    csv_path = tmp_path / "sparse_data.csv"
    rows = ["barcode,genotype,replicate,age,trait1"]
    for i in range(6):  # 3 groups with 2 samples each
        rows.append(f"p{i},A,1,{i % 3},1.0")
    csv_path.write_text("\n".join(rows))

    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    config.data.group_by = "age"
    config.cleanup.min_samples_per_trait = 5  # All groups have only 2 samples

    with caplog.at_level(logging.WARNING):
        result = run_grouped_pipelines(config, tmp_path / "output", MockPipeline, validate=False)

    # Should warn about no valid groups
    assert any("No valid groups remain" in r.message for r in caplog.records)
    assert result == {}  # Empty result
```

**Test 1.3: Mixed-Type Group Sorting**
```python
def test_mixed_type_group_values_sorted_gracefully(self):
    """Mixed string/numeric group values should not crash sorting."""
    csv_path = tmp_path / "mixed_types.csv"
    rows = [
        "barcode,genotype,replicate,experiment_id,trait1",
        "p1,A,1,exp1,1.0",  # String
        "p2,A,2,exp1,1.1",
        "p3,B,1,7,2.0",      # Numeric
        "p4,B,2,7,2.1",
        "p5,C,1,exp2,3.0",   # String
        "p6,C,2,exp2,3.1",
    ]
    csv_path.write_text("\n".join(rows))

    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    config.data.group_by = "experiment_id"
    config.cleanup.min_samples_per_trait = 2

    # Should not crash with TypeError
    result = run_grouped_pipelines(config, tmp_path / "output", MockPipeline, validate=False)

    # Should have all 3 groups
    assert len(result) == 3
    assert 7 in result or "7" in result
    assert "exp1" in result
    assert "exp2" in result
```

**Test 1.4: CSV Error Messages**
```python
def test_missing_csv_error_message_includes_context(self):
    """FileNotFoundError should mention group_by context."""
    config = get_default_qc_config()
    config.data.csv_path = "nonexistent.csv"
    config.data.group_by = "plant_age_days"

    with pytest.raises(FileNotFoundError, match="group_by='plant_age_days'"):
        run_grouped_pipelines(config, tmp_path / "output", MockPipeline, validate=False)

def test_missing_group_column_error_message_clear(self):
    """KeyError for missing column should be descriptive."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("barcode,trait1\np1,1.0")

    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    config.data.group_by = "nonexistent_column"

    with pytest.raises(KeyError, match="nonexistent_column"):
        run_grouped_pipelines(config, tmp_path / "output", MockPipeline, validate=False)
```

**Run tests to confirm RED**:
```bash
pytest tests/test_grouped_pipeline_robustness.py -v
```

Expected: All tests fail as functionality not yet implemented.

---

**Phase 2: Implement Fixes (GREEN)**

**Fix 1: Pipeline Failure Logging** (`src/sleap_roots_analyze/pipeline/utils.py:665-672`)
```python
try:
    result = pipeline.run()
    # ... existing success code ...
except Exception:
    logger.exception(
        f"Group {group_by_column}={group_value} failed during pipeline execution"
    )
    raise  # Re-raise to fail fast
```

**Fix 2: Empty Groups Warning** (`src/sleap_roots_analyze/pipeline/utils.py:606-610`)
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

**Fix 3: Mixed-Type Sorting** (`src/sleap_roots_analyze/pipeline/utils.py:609`)
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
```

**Fix 4: Contextualized Errors** (`src/sleap_roots_analyze/pipeline/utils.py:595`)
```python
try:
    df = pd.read_csv(config.data.csv_path)
    logger.info(f"Loaded {len(df)} samples from {config.data.csv_path}")

    groups = split_data_by_group(df, group_by_column=group_by_column)
    logger.info(f"Split data into {len(groups)} groups: {list(groups.keys())}")
except FileNotFoundError as exc:
    msg = (
        f"Failed to read data CSV at '{config.data.csv_path}' while preparing "
        f"grouped pipelines (group_by='{group_by_column}')."
    )
    logger.error(msg)
    raise FileNotFoundError(msg) from exc
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

**Fix 5-7: Code Cleanup**
- Remove `import tempfile` and `from copy import deepcopy` from line ~20
- Remove `original_pipeline_name = config.pipeline_name` from line ~630
- Update `.github/workflows/ci.yml` comment from issue #70 to #69

**Run tests to confirm GREEN**:
```bash
pytest tests/test_grouped_pipeline_robustness.py -v
```

Expected: All tests pass.

---

**Phase 3: Refactor**

No refactoring needed - fixes are minimal and localized.

---

**Phase 4: Integration Testing**

Run full grouped pipeline test suite to ensure no regressions:
```bash
pytest tests/test_grouped_pipeline*.py -v
```

Expected: All existing tests still pass + 7 new tests pass.

---

**Phase 5: Commit**

Single commit with all fixes:
```bash
git add src/sleap_roots_analyze/pipeline/utils.py .github/workflows/ci.yml tests/test_grouped_pipeline_robustness.py
git commit -m "polish: Improve grouped pipeline robustness per Copilot review

Address 7 code quality improvements from GitHub Copilot PR review:

HIGH Priority:
- Add pipeline failure logging with group context (fail fast)

MEDIUM Priority:
- Warn when all groups filtered out by min_samples threshold
- Handle mixed-type group values with graceful sorting fallback
- Provide contextualized error messages for CSV/grouping failures

LOW Priority:
- Remove unused imports (tempfile, copy) after Bug #1 fix
- Remove unused variable (original_pipeline_name)
- Fix issue reference (#70 -> #69 in CI comment)

All changes TDD-verified with 7 new tests.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

## Scope

### In Scope
- Error handling improvements (logging, warnings, exceptions)
- Edge case robustness (empty groups, mixed types)
- Code cleanup (dead code removal, comment corrections)
- Test coverage for all improvements

### Out of Scope
- Performance optimization
- New features or functionality changes
- Refactoring beyond dead code removal
- Changes to existing test behavior

## Dependencies

**Upstream**: PR #63 (grouped pipeline implementation + bug fixes) must be merged first

**Downstream**: None - these are polish improvements that don't affect API

## Testing Strategy

**TDD Approach**: Write failing tests first (Phase 1 RED), implement fixes (Phase 2 GREEN), verify no regressions (Phase 4).

**Test Coverage**:
- 7 new unit tests in `test_grouped_pipeline_robustness.py`
- All existing grouped pipeline tests must still pass
- Manual verification: Run grouped pipeline with edge cases (sparse data, mixed types)

**Acceptance Criteria**:
- All 7 new tests pass
- All existing tests pass (no regressions)
- Code coverage for utils.py remains >90%

## Risks

**Risk 1: Logging Overhead**
- Mitigation: Log only at WARNING/ERROR levels for exceptional cases

**Risk 2: TypeError in String Sorting Fallback**
- Mitigation: Use `str(v)` which works for all Python types

**Risk 3: Breaking Exception Handling**
- Mitigation: Re-raise original exceptions to preserve stack traces

## Alternatives Considered

**Alternative 1: Skip LOW priority fixes**
- Rejected: Code cleanup is trivial and improves maintainability

**Alternative 2: Defer MEDIUM priority to follow-up**
- Rejected: Edge cases should be fixed now while context is fresh

**Alternative 3: Don't re-raise exceptions after logging**
- Rejected: Silent partial failures are worse than explicit failures

## Success Metrics

- Zero grouped pipeline failures without diagnostic logging
- Zero edge case crashes (empty groups, mixed types)
- Zero dead code in pipeline/utils.py
- Maintainer feedback: "Code is cleaner and errors are actionable"

## Timeline

- **Phase 1 (TDD RED)**: 30 minutes (write 7 tests)
- **Phase 2 (TDD GREEN)**: 45 minutes (implement 7 fixes)
- **Phase 3 (Refactor)**: 0 minutes (not needed)
- **Phase 4 (Integration)**: 15 minutes (run full test suite)
- **Phase 5 (Commit)**: 10 minutes (single commit with tests + fixes)

**Total**: 1.5-2 hours

## Related

- **PR #63**: Grouped pipeline implementation (requires merge first)
- **Issue #69**: Viz fan-out bug (fixed by PR #63, referenced in CI comment)
- **OpenSpec Change**: `fix-grouped-pipeline-bugs` (archived 2026-02-25)
