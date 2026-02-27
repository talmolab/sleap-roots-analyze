# Design: Address Copilot Review Comments

## TDD Workflow

Following test-driven development:
1. Write failing tests for each issue
2. Run tests to confirm they fail
3. Implement fixes
4. Run tests to confirm they pass
5. Commit with descriptive message

## Implementation Order

### Phase 1: Simple Code Cleanup (No Tests Required)
Issues 1-3 are trivial fixes with no behavioral changes.

**1. Remove unused variable**
- File: `src/sleap_roots_analyze/pipeline/utils.py`
- Search for: `original_pipeline_name = `
- Action: Delete the line

**2. Move imports to top**
- File: `src/sleap_roots_analyze/pipeline/utils.py`
- Search for inline: `import tempfile`, `import copy`
- Action: Move to top-level imports section (after other stdlib imports)

**3. Fix issue reference**
- File: `.github/workflows/ci.yml`
- Line: ~87
- Change: `issue #70` → `issue #69`

### Phase 2: Error Handling (Issues 4-7) - TDD Required

**Test file:** `tests/test_grouped_pipeline_execution.py`

#### Test 2.1: Better error messages for data loading failures (Issue 4)

```python
class TestErrorHandling:
    """Tests for error handling in run_grouped_pipelines."""

    def test_file_not_found_error_includes_context(self, tmp_path):
        """FileNotFoundError includes group_by context in message."""
        config = create_mock_qc_config(csv_path="nonexistent.csv", group_by="age")

        with pytest.raises(FileNotFoundError) as exc_info:
            run_grouped_pipelines(
                config=config,
                pipeline_class=MockPipeline,
                group_by_column="age",
                base_run_dir=tmp_path,
            )

        # Error message should mention both the file path and group_by context
        assert "nonexistent.csv" in str(exc_info.value)
        assert "group_by='age'" in str(exc_info.value)

    def test_empty_data_error_includes_context(self, tmp_path):
        """EmptyDataError includes group_by context in message."""
        # Create empty CSV file
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        config = create_mock_qc_config(csv_path=str(empty_csv), group_by="age")

        with pytest.raises(pd.errors.EmptyDataError) as exc_info:
            run_grouped_pipelines(
                config=config,
                pipeline_class=MockPipeline,
                group_by_column="age",
                base_run_dir=tmp_path,
            )

        assert str(empty_csv) in str(exc_info.value)
        assert "group_by='age'" in str(exc_info.value)

    def test_key_error_from_split_preserved(self, tmp_path):
        """KeyError from split_data_by_group is preserved with original message."""
        # CSV without the group_by column
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("barcode,genotype,trait1\np1,A,1.0\n")

        config = create_mock_qc_config(csv_path=str(csv_path), group_by="nonexistent_col")

        with pytest.raises(KeyError) as exc_info:
            run_grouped_pipelines(
                config=config,
                pipeline_class=MockPipeline,
                group_by_column="nonexistent_col",
                base_run_dir=tmp_path,
            )

        # Should be the original KeyError from split_data_by_group
        assert "nonexistent_col" in str(exc_info.value)
```

#### Test 2.2: Empty groups warning (Issue 5)

```python
def test_empty_groups_warning_logged(tmp_path, caplog):
    """Warning logged when all groups filtered out."""
    # Create CSV with groups that will all be filtered
    csv_path = tmp_path / "small_groups.csv"
    csv_path.write_text(
        "barcode,genotype,replicate,age,trait1\n"
        "p1,A,1,0,1.0\n"  # age=0, n=1 (will be filtered with min_samples=10)
        "p2,B,1,1,2.0\n"  # age=1, n=1 (will be filtered)
    )

    config = create_mock_qc_config(
        csv_path=str(csv_path),
        group_by="age",
        min_samples_per_trait=10,
    )

    with caplog.at_level(logging.WARNING):
        result = run_grouped_pipelines(
            config=config,
            pipeline_class=MockPipeline,
            group_by_column="age",
            base_run_dir=tmp_path,
        )

    # Should return empty dict
    assert result == {}

    # Should log warning about no valid groups
    assert any(
        "No valid groups remain" in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )
    assert any(
        "min_samples_per_trait=10" in record.message
        for record in caplog.records
    )
```

#### Test 2.3: Mixed-type group handling (Issue 6)

```python
def test_mixed_type_groups_sorted_safely(tmp_path, caplog):
    """Mixed string/int group values handled gracefully."""
    # Create CSV with mixed-type group column
    csv_path = tmp_path / "mixed_groups.csv"
    csv_path.write_text(
        "barcode,genotype,replicate,site,trait1\n"
        + "\n".join([
            f"p{i},A,1,site{i % 3 if i % 2 == 0 else i},1.0"
            for i in range(15)
        ])
    )

    config = create_mock_qc_config(csv_path=str(csv_path), group_by="site")

    with caplog.at_level(logging.WARNING):
        result = run_grouped_pipelines(
            config=config,
            pipeline_class=MockPipeline,
            group_by_column="site",
            base_run_dir=tmp_path,
        )

    # Should complete successfully
    assert len(result) > 0

    # Should log warning about mixed types
    assert any(
        "Mixed-type group values" in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )
```

#### Test 2.4: Log group failures (Issue 7)

```python
def test_group_failure_logged_and_continues(tmp_path, caplog):
    """Pipeline failure for one group is logged, but other groups continue."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "barcode,genotype,replicate,age,trait1\n"
        + "\n".join([f"p{i},A,1,{i % 3},1.0" for i in range(30)])
    )

    config = create_mock_qc_config(csv_path=str(csv_path), group_by="age")

    # Mock pipeline that fails for age=1
    class FailingPipeline:
        def __init__(self, config, run_dir):
            self.config = config
            self.run_dir = run_dir

        def run(self):
            # Check if this is the age=1 group by looking at data
            df = pd.read_csv(self.config.data.csv_path)
            if df["age"].iloc[0] == 1:
                raise ValueError("Simulated pipeline failure for age=1")
            return {"status": "success"}

    with caplog.at_level(logging.ERROR):
        result = run_grouped_pipelines(
            config=config,
            pipeline_class=FailingPipeline,
            group_by_column="age",
            base_run_dir=tmp_path,
        )

    # Should have results for age=0 and age=2, but not age=1
    assert len(result) == 2
    assert 0 in result
    assert 2 in result
    assert 1 not in result

    # Should log exception for failed group
    assert any(
        "age=1" in record.message and "failed" in record.message
        for record in caplog.records
        if record.levelname == "ERROR"
    )
```

### Phase 3: Early Validation (Issue 8) - TDD Required

**Test file:** `tests/test_pipeline_config.py`

#### Test 3.1: Validate group_by column exists

```python
def test_validate_qc_config_group_by_column_missing(tmp_path):
    """Validation fails when group_by column not in CSV."""
    # Create CSV without 'age' column
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("barcode,genotype,replicate,trait1\np1,A,1,1.0\n")

    config_dict = {
        "pipeline_name": "test",
        "data": {
            "csv_path": str(csv_path),
            "group_by": "age",  # Column doesn't exist
        },
        # ... other required fields ...
    }

    config = QCConfig(**config_dict)

    with pytest.raises(ValueError) as exc_info:
        validate_qc_config(config, check_files=True)

    assert "group_by column 'age' not found" in str(exc_info.value)
    assert "Available columns:" in str(exc_info.value)

def test_validate_qc_config_group_by_validation_skipped_when_check_files_false(tmp_path):
    """Validation skipped when check_files=False."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("barcode,genotype,replicate,trait1\np1,A,1,1.0\n")

    config_dict = {
        "pipeline_name": "test",
        "data": {
            "csv_path": str(csv_path),
            "group_by": "nonexistent",  # Would fail if validated
        },
        # ... other required fields ...
    }

    config = QCConfig(**config_dict)

    # Should not raise when check_files=False
    validate_qc_config(config, check_files=False)

def test_validate_qc_config_group_by_validation_skipped_when_csv_missing(tmp_path):
    """Validation skipped when CSV file doesn't exist yet."""
    csv_path = tmp_path / "nonexistent.csv"  # Doesn't exist

    config_dict = {
        "pipeline_name": "test",
        "data": {
            "csv_path": str(csv_path),
            "group_by": "age",
        },
        # ... other required fields ...
    }

    config = QCConfig(**config_dict)

    # Should not raise even with check_files=True (CSV doesn't exist)
    # This allows creating configs before data is available
    validate_qc_config(config, check_files=True)

def test_validate_qc_config_group_by_null_skips_validation():
    """No validation when group_by is null."""
    config_dict = {
        "pipeline_name": "test",
        "data": {
            "csv_path": "any_path.csv",
            "group_by": None,  # No grouping
        },
        # ... other required fields ...
    }

    config = QCConfig(**config_dict)

    # Should not attempt validation when group_by is None
    validate_qc_config(config, check_files=True)
```

## Implementation Details

### Error Handling Patterns

**Preserve original errors:**
```python
except KeyError:
    # Re-raise original KeyError from split_data_by_group
    # It already has helpful message about missing column
    raise
```

**Enhance with context:**
```python
except FileNotFoundError as exc:
    msg = f"Failed to read data CSV at '{config.data.csv_path}' while preparing grouped pipelines (group_by='{group_by_column}')."
    logger.error(msg)
    raise FileNotFoundError(msg) from exc  # Preserve traceback
```

### Logging Best Practices

**Use appropriate levels:**
- `logger.error()` - Errors that prevent operation
- `logger.warning()` - Issues that don't prevent operation but indicate problems
- `logger.info()` - Normal operation information
- `logger.exception()` - Errors with full traceback

**Example:**
```python
except Exception:
    logger.exception(f"Group {group_by_column}={group_value} failed during pipeline execution")
    # Continue processing remaining groups
```

### Validation Strategy

**Early validation priorities:**
1. Validate column existence ONLY when CSV exists and check_files=True
2. Allow config creation before data exists (for template workflows)
3. Provide helpful error messages with available columns
4. Use `nrows=0` to read only header (fast validation)

## Files Modified

1. `src/sleap_roots_analyze/pipeline/utils.py` - Main changes
2. `src/sleap_roots_analyze/pipeline/config/utils.py` - Add validation
3. `.github/workflows/ci.yml` - Fix comment
4. `tests/test_grouped_pipeline_execution.py` - Add error handling tests
5. `tests/test_pipeline_config.py` - Add validation tests

## Commit Strategy

**Commit 1: Code cleanup**
```
chore: remove unused variable and fix imports in pipeline utils

- Remove unused original_pipeline_name variable
- Move tempfile and copy imports to top of file
- Fix issue reference in CI workflow (#70 -> #69)
```

**Commit 2: Error handling improvements (TDD)**
```
feat: improve error handling in grouped pipeline execution

- Add contextual error messages for CSV read failures
- Warn when all groups filtered out (likely config issue)
- Handle mixed-type group values gracefully
- Log failures during group pipeline execution

Tests:
- test_file_not_found_error_includes_context
- test_empty_data_error_includes_context
- test_key_error_from_split_preserved
- test_empty_groups_warning_logged
- test_mixed_type_groups_sorted_safely
- test_group_failure_logged_and_continues
```

**Commit 3: Early validation (TDD)**
```
feat: validate group_by column exists in config validation

- Add validation to check group_by column exists in CSV
- Skip validation when check_files=False or CSV missing
- Provide helpful error with available columns

Tests:
- test_validate_qc_config_group_by_column_missing
- test_validate_qc_config_group_by_validation_skipped_when_check_files_false
- test_validate_qc_config_group_by_validation_skipped_when_csv_missing
- test_validate_qc_config_group_by_null_skips_validation
```
