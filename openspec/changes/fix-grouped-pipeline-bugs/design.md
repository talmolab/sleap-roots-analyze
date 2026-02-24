# Design: Fix Critical Bugs in Grouped Pipeline Execution

## Implementation Strategy

Following strict TDD: Write all tests FIRST, watch them FAIL, then implement fixes to make them PASS.

---

## Bug #1: Saved Configs Reference Non-Existent CSV Files

### Test Specification (Write FIRST)

**File:** `tests/test_grouped_pipeline_config_persistence.py` (new file)

```python
class TestGroupedPipelineConfigPersistence:
    """Test that grouped pipelines save configs with valid, persistent CSV paths."""

    def test_saved_config_csv_path_exists(self, tmp_path):
        """Saved config.yaml must reference an existing CSV file."""
        # Create test data with multiple groups
        csv_path = tmp_path / "data.csv"
        # ... create CSV with group column

        config = get_default_qc_config()
        config.data.csv_path = str(csv_path)
        config.data.group_by = "age"

        # Run grouped pipelines
        result = run_grouped_pipelines(...)

        # For each group, verify saved config points to existing file
        for group_label, group_result in result.items():
            saved_config_path = group_result.run_dir / "config.yaml"
            assert saved_config_path.exists()

            # Load saved config
            saved_config = load_qc_config(saved_config_path)

            # CRITICAL: csv_path must exist
            csv_file = Path(saved_config.data.csv_path)
            assert csv_file.exists(), (
                f"Saved config references non-existent CSV: {csv_file}"
            )

            # CSV should be in the group's output directory (not /tmp)
            assert csv_file.parent == group_result.run_dir

    def test_saved_config_is_reproducible(self, tmp_path):
        """Saved config can be used to re-run the exact same analysis."""
        # Run pipeline once
        result1 = run_grouped_pipelines(...)
        group_result = list(result1.values())[0]

        # Load the saved config
        saved_config_path = group_result.run_dir / "config.yaml"
        saved_config = load_qc_config(saved_config_path)

        # Re-run using saved config
        rerun_output_dir = tmp_path / "rerun"
        pipeline = QCPipeline(config=saved_config, output_dir=rerun_output_dir)
        result2 = pipeline.run()

        # Should succeed (config is valid and data file exists)
        assert result2["status"] == "success"

    def test_input_csv_preserved_in_output_directory(self, tmp_path):
        """Each group's input CSV is saved in its output directory."""
        # Run grouped pipelines
        result = run_grouped_pipelines(...)

        for group_label, group_result in result.items():
            # Check for input CSV in output dir
            input_csv_pattern = group_result.run_dir / "00_input_data_*.csv"
            input_csvs = list(group_result.run_dir.glob("00_input_data_*.csv"))

            assert len(input_csvs) == 1, (
                f"Expected 1 input CSV in {group_result.run_dir}, found {len(input_csvs)}"
            )

            input_csv = input_csvs[0]
            # Verify it contains only the group's data
            df = pd.read_csv(input_csv)
            assert all(df["age"] == group_label.split("_")[-1])
```

### Implementation Plan

**Step 1:** Modify `run_grouped_pipelines()` to persist group CSVs

```python
# In run_grouped_pipelines(), line ~540

for group_value in sorted_group_values:
    group_df = valid_groups[group_value]
    group_label = f"{group_by_column}_{group_value}"

    # Create group-specific output directory
    group_output_dir = output_dir / f"{group_label}_{timestamp}"
    group_output_dir.mkdir(parents=True, exist_ok=True)

    # Write group CSV to OUTPUT DIRECTORY (not temp file)
    group_csv_filename = f"00_input_data_{group_label}.csv"
    group_csv_path = group_output_dir / group_csv_filename
    group_df.to_csv(group_csv_path, index=False)
    logger.info(f"Saved group data to {group_csv_path}")

    # Update config to point to persistent CSV
    group_config = copy.deepcopy(config)
    group_config.data.csv_path = str(group_csv_path)

    try:
        # Run pipeline
        pipeline = pipeline_class(
            config=group_config,
            output_dir=group_output_dir,
            validate=validate,
        )
        result = pipeline.run()
        grouped_results[group_label] = result
        logger.info(f"Group {group_by_column}={group_value} completed successfully")
    except Exception:
        logger.exception(f"Group {group_by_column}={group_value} failed")
    # NO finally block to delete CSV - it stays in output dir
```

---

## Bug #2: CLI --group-by Flag Doesn't Trigger Viz Fan-Out

### Test Specification (Write FIRST)

**File:** `tests/test_run_all_cli_group_by.py` (new file)

```python
class TestRunAllCLIGroupBy:
    """Test that CLI --group-by flag correctly triggers viz fan-out."""

    def test_cli_group_by_triggers_viz_fanout_when_config_has_no_group_by(self, tmp_path):
        """run-all --group-by should trigger viz fan-out even when config has no group_by."""
        # Create QC config WITHOUT group_by
        qc_config_path = tmp_path / "qc_config.yaml"
        qc_config = get_default_qc_config()
        qc_config.data.csv_path = str(tmp_path / "data.csv")
        qc_config.data.group_by = None  # NO group_by in config
        save_qc_config(qc_config, qc_config_path)

        # Create viz config
        viz_config_path = tmp_path / "viz_config.yaml"
        # ... create viz config

        # Create manifest
        manifest_path = tmp_path / "manifest.yaml"
        # ... create manifest referencing qc_config and viz_config

        # Create test data with groups
        create_test_csv_with_groups(tmp_path / "data.csv")

        # Run pipeline runner with CLI --group-by flag
        runner = PipelineRunner(
            manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            group_by="plant_age_days"  # CLI flag
        )
        runner.run_all_pipelines()

        # CRITICAL: Verify viz ran for EACH group (not just once)
        viz_results = runner.run_results["viz"]

        # Should have N viz results (one per group), not 1
        expected_groups = ["plant_age_days_0", "plant_age_days_3", "plant_age_days_5"]
        for group in expected_groups:
            result_key = f"viz_config.yaml:{group}"
            assert result_key in viz_results, (
                f"Viz fan-out failed for group {group}. "
                f"Found keys: {list(viz_results.keys())}"
            )

    def test_cli_group_by_overrides_config_group_by_for_viz_fanout(self, tmp_path):
        """CLI --group-by should override config group_by for viz fan-out detection."""
        # Create QC config with group_by="site"
        qc_config = get_default_qc_config()
        qc_config.data.group_by = "site"  # Config says "site"
        # ... save config

        # Run with CLI --group-by plant_age_days (overrides config)
        runner = PipelineRunner(
            manifest_path=manifest_path,
            group_by="plant_age_days"  # CLI overrides config
        )
        runner.run_all_pipelines()

        # Viz fan-out should detect "plant_age_days" groups (from CLI)
        # NOT "site" groups (from config)
        viz_results = runner.run_results["viz"]
        assert any("plant_age_days_" in key for key in viz_results.keys())
        assert not any("site_" in key for key in viz_results.keys())

    def test_effective_group_by_tracked_correctly(self, tmp_path):
        """Runner should track effective group_by (CLI > config)."""
        runner = PipelineRunner(
            manifest_path=manifest_path,
            group_by="age"  # CLI flag
        )

        # Before running QC
        qc_config_path = runner.qc_configs[0]
        config_group_by = runner._get_qc_config_group_by(qc_config_path)  # Returns "site"
        effective_group_by = runner.group_by or config_group_by  # Should be "age" (CLI)

        assert effective_group_by == "age", "CLI flag should take precedence"
```

### Implementation Plan

**Step 1:** Track effective group_by in `_run_qc_pipelines()`

```python
def _run_qc_pipelines(self):
    """Run all QC pipelines and detect grouped outputs."""
    for config_path in self.qc_configs:
        config_rel = self._relative_to_manifest_dir(config_path)

        # Determine EFFECTIVE group_by (CLI > config)
        config_group_by = self._get_qc_config_group_by(config_path)
        effective_group_by = self.group_by if self.group_by is not None else config_group_by

        logger.info(
            f"Running QC: {config_rel} "
            f"(group_by: CLI={self.group_by}, config={config_group_by}, effective={effective_group_by})"
        )

        # Run pipeline (CLI flag is passed via _run_pipeline_command)
        result = self._run_pipeline_command(
            command="qc",
            config_path=config_path,
            output_dir=self.run_dir / "qc"
        )
        self.run_results["qc"][config_rel] = result

        # Detect grouped outputs using EFFECTIVE group_by
        if effective_group_by is not None and result["exit_code"] == 0:
            base_dir = self.run_dir / "qc"
            grouped_outputs = self._find_grouped_qc_outputs(base_dir, effective_group_by)

            if grouped_outputs:
                logger.info(
                    f"Detected {len(grouped_outputs)} group outputs for {config_rel} "
                    f"(group_by='{effective_group_by}')"
                )
                self.qc_grouped_outputs[config_rel] = grouped_outputs
```

---

## Bug #3: NaN Group Values Silently Dropped

### Test Specification (Write FIRST)

**File:** `tests/test_grouped_pipeline_nan_handling.py` (new file)

```python
class TestGroupedPipelineNaNHandling:
    """Test that NaN group values are handled explicitly (not silently dropped)."""

    def test_nan_group_values_logged_and_dropped_by_default(self, tmp_path, caplog):
        """Samples with NaN group values should be logged and dropped (default behavior)."""
        # Create CSV with NaN group values
        csv_path = tmp_path / "data.csv"
        rows = [
            "barcode,genotype,replicate,age,trait1",
            "p1,A,1,7,1.0",
            "p2,A,2,7,1.1",
            "p3,A,3,7,1.2",
            "p4,B,1,14,2.0",
            "p5,B,2,,2.1",  # NaN age
            "p6,B,3,,2.2",  # NaN age
        ]
        csv_path.write_text("\n".join(rows))

        df = pd.read_csv(csv_path)
        assert df["age"].isna().sum() == 2, "Test setup: should have 2 NaN rows"

        # Split by age column
        with caplog.at_level(logging.WARNING):
            groups = split_data_by_group(df, group_by_column="age")

        # Should log warning about NaN values
        assert any(
            "Dropping 2/6 samples with missing 'age' values" in record.message
            for record in caplog.records
        )

        # NaN rows should NOT be in any group
        all_group_rows = sum(len(group_df) for group_df in groups.values())
        assert all_group_rows == 4, "Only 4 non-NaN rows should be in groups"

        # Verify specific groups
        assert len(groups) == 2  # 7.0 and 14.0
        assert 7.0 in groups or "7.0" in groups
        assert 14.0 in groups or "14.0" in groups

    def test_nan_group_values_can_be_treated_as_group(self, tmp_path):
        """Optional: NaN values can be treated as their own group."""
        csv_path = tmp_path / "data.csv"
        # ... same CSV with NaN values

        df = pd.read_csv(csv_path)

        # Split with handle_na="treat_as_group"
        groups = split_data_by_group(
            df,
            group_by_column="age",
            handle_na="treat_as_group"
        )

        # Should have 3 groups: 7.0, 14.0, and NaN
        assert len(groups) == 3

        # Check for NaN group (pandas represents NaN key in groupby)
        has_nan_group = any(pd.isna(k) for k in groups.keys())
        assert has_nan_group, "Should have a group for NaN values"

        # All 6 rows should be in groups
        all_group_rows = sum(len(group_df) for group_df in groups.values())
        assert all_group_rows == 6

    def test_grouped_pipeline_with_nan_values(self, tmp_path, caplog):
        """Integration: run_grouped_pipelines handles NaN values correctly."""
        # Create test CSV with NaN in group column
        # ... create data

        config = get_default_qc_config()
        config.data.group_by = "age"

        # Run grouped pipelines
        with caplog.at_level(logging.WARNING):
            result = run_grouped_pipelines(...)

        # Should log warning
        assert any("missing 'age' values" in record.message for record in caplog.records)

        # Should only process non-NaN groups
        assert len(result) == 2  # Only groups 7 and 14 (not NaN)
```

### Implementation Plan

**Step 1:** Modify `split_data_by_group()` to handle NaN explicitly

```python
def split_data_by_group(
    df: pd.DataFrame,
    group_by_column: str,
    handle_na: str = "warn_and_drop",
) -> dict[Any, pd.DataFrame]:
    """Split DataFrame by unique values in a column.

    Args:
        df: Input DataFrame
        group_by_column: Column name to group by
        handle_na: How to handle NaN values:
            - "warn_and_drop": Log warning and exclude NaN rows (default)
            - "treat_as_group": Include NaN as its own group

    Returns:
        Dictionary mapping group values to DataFrames

    Raises:
        KeyError: If group_by_column not in DataFrame
    """
    if group_by_column not in df.columns:
        raise KeyError(
            f"Column '{group_by_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    # Check for NaN values
    na_mask = df[group_by_column].isna()
    n_na = na_mask.sum()

    if n_na > 0 and handle_na == "warn_and_drop":
        logger.warning(
            f"Dropping {n_na}/{len(df)} samples with missing '{group_by_column}' "
            f"values during grouping"
        )

    # Use pandas groupby (handles NaN correctly with dropna parameter)
    groups = {}
    dropna = (handle_na == "warn_and_drop")

    for group_value, group_df in df.groupby(group_by_column, dropna=dropna):
        groups[group_value] = group_df.copy()

    return groups
```

**Step 2:** Save dropped NaN samples for traceability

In `run_grouped_pipelines()`, save dropped NaN samples to a separate CSV file:

```python
def run_grouped_pipelines(...):
    # Load data
    df = pd.read_csv(config.data.csv_path)
    logger.info(f"Loaded {len(df)} samples from {config.data.csv_path}")

    # Check for NaN values in group column BEFORE splitting
    na_mask = df[group_by_column].isna()
    n_na = na_mask.sum()

    if n_na > 0:
        # Save dropped samples to output directory for traceability
        dropped_csv_path = output_dir / f"00_dropped_samples_missing_{group_by_column}.csv"
        dropped_df = df[na_mask].copy()
        dropped_df.to_csv(dropped_csv_path, index=False)

        logger.warning(
            f"Dropped {n_na}/{len(df)} samples with missing '{group_by_column}' values. "
            f"Saved to: {dropped_csv_path}"
        )

        # Also create a metadata file explaining why they were dropped
        metadata_path = output_dir / f"00_dropped_samples_missing_{group_by_column}.txt"
        metadata_path.write_text(
            f"Dropped Samples Metadata\n"
            f"========================\n\n"
            f"Reason: Missing values in group_by column '{group_by_column}'\n"
            f"Count: {n_na}/{len(df)} samples\n"
            f"Dropped CSV: {dropped_csv_path.name}\n\n"
            f"Barcodes of dropped samples:\n"
            + "\n".join(f"  - {bc}" for bc in dropped_df[config.columns.barcode].tolist())
        )

    # Continue with grouping (NaN rows are already saved)
    groups = split_data_by_group(df, group_by_column=group_by_column)
    # ... rest of function
```

**Test for traceability:**

```python
def test_dropped_nan_samples_saved_for_traceability(self, tmp_path):
    """Dropped NaN samples must be saved to a CSV file with metadata."""
    # Create CSV with NaN values
    csv_path = tmp_path / "data.csv"
    # ... create data with 2 NaN rows

    config = get_default_qc_config()
    config.data.group_by = "age"

    result = run_grouped_pipelines(
        config=config,
        output_dir=tmp_path / "output",
        pipeline_class=MockPipeline,
    )

    output_dir = tmp_path / "output"

    # Check for dropped samples CSV
    dropped_csv = output_dir / "00_dropped_samples_missing_age.csv"
    assert dropped_csv.exists(), "Dropped samples CSV must be saved"

    # Verify dropped CSV contains exactly the NaN rows
    dropped_df = pd.read_csv(dropped_csv)
    assert len(dropped_df) == 2
    assert set(dropped_df["barcode"]) == {"p5", "p6"}

    # Check for metadata file
    metadata_file = output_dir / "00_dropped_samples_missing_age.txt"
    assert metadata_file.exists(), "Metadata file must be saved"

    metadata_text = metadata_file.read_text()
    assert "2/6 samples" in metadata_text
    assert "p5" in metadata_text
    assert "p6" in metadata_text
```

**Step 3:** Update pipeline summary to track dropped samples

The pipeline summary JSON should include metadata about dropped samples:

```python
# In run_grouped_pipelines(), return additional metadata
dropped_samples_info = {
    "column": group_by_column,
    "count": n_na,
    "total": len(df),
    "fraction": n_na / len(df) if len(df) > 0 else 0,
    "csv_path": str(dropped_csv_path) if n_na > 0 else None,
    "metadata_path": str(metadata_path) if n_na > 0 else None,
}

# Include in each group's result
for group_label, group_result in grouped_results.items():
    group_result["dropped_samples"] = dropped_samples_info
```

**Test for summary tracking:**

```python
def test_dropped_samples_tracked_in_summary(self, tmp_path):
    """Pipeline summary must include dropped sample metadata."""
    # Run grouped pipelines with NaN values
    result = run_grouped_pipelines(...)

    # Each group result should have dropped_samples metadata
    for group_label, group_result in result.items():
        assert "dropped_samples" in group_result

        dropped_info = group_result["dropped_samples"]
        assert dropped_info["column"] == "age"
        assert dropped_info["count"] == 2
        assert dropped_info["total"] == 32  # 30 valid + 2 NaN
        assert dropped_info["csv_path"] is not None
        assert Path(dropped_info["csv_path"]).exists()
```

---

## Bug #4: ANOVA Error Handling Crashes Pipeline

### Test Specification (Write FIRST)

**File:** `tests/test_statistical_analysis_error_handling.py` (new file)

```python
class TestStatisticalAnalysisErrorHandling:
    """Test that ANOVA errors are handled gracefully (not crash)."""

    def test_anova_string_error_handled_gracefully(self, tmp_path):
        """When ANOVA returns error string, should store it (not crash)."""
        from sleap_roots_analyze.pipeline.steps.statistical_analysis import (
            StatisticalAnalysisStep,
        )
        from sleap_roots_analyze.pipeline.config.utils import get_default_qc_config

        # Create minimal data that will fail ANOVA (insufficient samples)
        csv_path = tmp_path / "minimal_data.csv"
        rows = [
            "barcode,genotype,replicate,trait1,trait2",
            "p1,A,1,1.0,2.0",
            "p2,B,1,1.1,2.1",
            # Only 2 genotypes with 1 sample each - insufficient for ANOVA
        ]
        csv_path.write_text("\n".join(rows))

        df = pd.read_csv(csv_path)

        config = get_default_qc_config()
        config.columns.barcode = "Barcode"
        config.columns.genotype = "Genotype"
        config.columns.replicate = "Replicate"
        config.statistics.calculate_anova = True

        step = StatisticalAnalysisStep()

        # CRITICAL: This should NOT crash
        result = step.execute(
            config=config,
            run_dir=tmp_path,
            logger=logging.getLogger("test"),
            df=df,
            trait_cols=["trait1", "trait2"],
        )

        # Should return result (not crash)
        assert result is not None
        assert "anova_results" in result

        # Check that ANOVA results handle errors
        anova_df = pd.read_csv(result["files"]["anova_results"])

        # Should have error messages for traits that failed
        error_rows = anova_df[anova_df["error"].notna()]
        assert len(error_rows) > 0, (
            "Should have error entries for failed ANOVA calculations"
        )

        # Error rows should have None/NaN for statistical values
        for idx, row in error_rows.iterrows():
            assert pd.isna(row["f_statistic"])
            assert pd.isna(row["p_value"])
            assert row["error"] is not None

    def test_anova_mixed_success_and_failure(self, tmp_path):
        """Some traits succeed ANOVA, others fail - should handle both."""
        # Create data where some traits have enough variation, others don't
        csv_path = tmp_path / "mixed_data.csv"
        rows = ["barcode,genotype,replicate,good_trait,bad_trait"]

        # good_trait: sufficient data for ANOVA
        for i in range(30):
            genotype = ["A", "B", "C"][i % 3]
            rows.append(f"p{i},{genotype},{i % 3},{1.0 + i * 0.1},{1.0}")  # bad_trait has zero variance

        csv_path.write_text("\n".join(rows))

        df = pd.read_csv(csv_path)

        config = get_default_qc_config()
        config.columns.barcode = "Barcode"
        config.columns.genotype = "Genotype"
        config.columns.replicate = "Replicate"
        config.statistics.calculate_anova = True

        step = StatisticalAnalysisStep()

        result = step.execute(
            config=config,
            run_dir=tmp_path,
            logger=logging.getLogger("test"),
            df=df,
            trait_cols=["good_trait", "bad_trait"],
        )

        anova_df = pd.read_csv(result["files"]["anova_results"])

        # good_trait should have valid results
        good_row = anova_df[anova_df["trait"] == "good_trait"].iloc[0]
        assert not pd.isna(good_row["f_statistic"])
        assert pd.isna(good_row["error"]) or good_row["error"] is None

        # bad_trait might have error (zero variance)
        # Just verify it didn't crash - either succeeds or fails gracefully
```

### Implementation Plan

**Step 1:** Add type check before calling `.get()` on ANOVA result

```python
# In src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py, line ~110

for trait in trait_cols:
    result = calculate_anova_by_genotype(
        df, trait_col=trait, genotype_col=genotype_col
    )

    # CRITICAL: Check if result is an error message string
    if isinstance(result, str):
        # Error case: ANOVA failed, result is error message
        anova_records.append(
            {
                "trait": trait,
                "f_statistic": None,
                "p_value": None,
                "eta_squared": None,
                "significant": None,
                "n_groups": None,
                "total_n": None,
                "error": result,
            }
        )
    else:
        # Success case: result is a dict with statistics
        anova_records.append(
            {
                "trait": trait,
                "f_statistic": result.get("f_statistic"),
                "p_value": result.get("p_value"),
                "eta_squared": result.get("eta_squared"),
                "significant": result.get("significant"),
                "n_groups": result.get("n_groups"),
                "total_n": result.get("total_n"),
                "error": None,
            }
        )
```

---

## TDD Workflow

### Phase 1: Write All Tests (RED)
1. Create `tests/test_grouped_pipeline_config_persistence.py` with 4 tests
2. Create `tests/test_run_all_cli_group_by.py` with 4 tests
3. Create `tests/test_grouped_pipeline_nan_handling.py` with 7 tests
4. Create `tests/test_statistical_analysis_error_handling.py` with 2 tests
5. Run tests - **confirm they ALL FAIL**

### Phase 2: Implement Fixes (GREEN)
1. Fix Bug #4: Add type check for ANOVA result (PRIORITY - unblocks other tests)
2. Fix Bug #1: Modify `run_grouped_pipelines()` to persist CSVs
3. Fix Bug #2: Track effective group_by in `_run_qc_pipelines()`
4. Fix Bug #3: Add NaN handling to `split_data_by_group()`
5. Run tests - **confirm they ALL PASS**

### Phase 3: Integration Testing
1. Run existing integration tests - ensure no regressions
2. Run new integration tests
3. Full test suite: `uv run pytest tests/ -v`

### Phase 4: Manual Verification
1. Run real pipeline with `--group-by` flag
2. Inspect output directories
3. Verify saved configs are valid and point to existing files
4. Attempt to re-run using saved config.yaml
5. Check metadata and provenance
