"""Tests for pipeline core abstractions (BaseStep, StepResult)."""

from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import BaseStep, StepResult


def test_step_result_creation():
    """Test creating a StepResult."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = StepResult(
        data=df, metadata={"samples": 3}, files_generated=["output.csv"]
    )

    assert isinstance(result.data, pd.DataFrame)
    assert result.metadata["samples"] == 3
    assert len(result.files_generated) == 1


def test_step_result_defaults():
    """Test StepResult with default values."""
    result = StepResult(data=None)

    assert result.data is None
    assert result.metadata == {}
    assert result.files_generated == []


class ConcreteStep(BaseStep):
    """Concrete implementation of BaseStep for testing."""

    def execute(self, data, config, run_dir, prev_result=None):
        """Execute the step."""
        # Simple operation: add a column
        df = data.copy()
        df["processed"] = True

        # Save output
        output_file = self.save_dataframe(df, "processed.csv", run_dir)

        return StepResult(
            data=df,
            metadata={"rows": len(df)},
            files_generated=[output_file],
        )


def test_base_step_initialization():
    """Test BaseStep initialization."""
    step = ConcreteStep()
    assert step.step_name == "ConcreteStep"
    assert step.description == ""

    step = ConcreteStep(step_name="custom", description="Custom step")
    assert step.step_name == "custom"
    assert step.description == "Custom step"


def test_base_step_execute(tmp_path):
    """Test BaseStep execution."""
    step = ConcreteStep()
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = step.execute(df, None, tmp_path, None)

    assert "processed" in result.data.columns
    assert result.metadata["rows"] == 3
    assert len(result.files_generated) == 1
    assert (tmp_path / "processed.csv").exists()


def test_base_step_save_dataframe(tmp_path):
    """Test saving a DataFrame."""
    step = ConcreteStep()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    output_path = step.save_dataframe(df, "test.csv", tmp_path)

    assert output_path.exists()
    assert output_path.name == "test.csv"

    # Load and verify
    loaded = pd.read_csv(output_path)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["a", "b"]


def test_base_step_save_json(tmp_path):
    """Test saving a JSON file."""
    step = ConcreteStep()
    data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

    output_path = step.save_json(data, "test.json", tmp_path)

    assert output_path.exists()
    assert output_path.name == "test.json"

    # Load and verify
    import json

    with open(output_path) as f:
        loaded = json.load(f)

    assert loaded["key1"] == "value1"
    assert loaded["key2"] == 42
    assert loaded["key3"] == [1, 2, 3]


def test_base_step_must_implement_execute():
    """Test that BaseStep requires execute implementation."""

    class IncompleteStep(BaseStep):
        pass

    with pytest.raises(TypeError):
        IncompleteStep()


def test_step_chaining(tmp_path):
    """Test chaining multiple steps together."""

    class Step1(BaseStep):
        def execute(self, data, config, run_dir, prev_result=None):
            df = data.copy()
            df["step1"] = True
            return StepResult(data=df, metadata={"step": 1})

    class Step2(BaseStep):
        def execute(self, data, config, run_dir, prev_result=None):
            df = data.copy()
            df["step2"] = True
            # Access previous step metadata
            prev_step = prev_result.metadata.get("step", 0) if prev_result else 0
            return StepResult(data=df, metadata={"step": 2, "prev_step": prev_step})

    df = pd.DataFrame({"a": [1, 2, 3]})

    # Execute step 1
    step1 = Step1()
    result1 = step1.execute(df, None, tmp_path, None)
    assert "step1" in result1.data.columns
    assert result1.metadata["step"] == 1

    # Execute step 2 with step1's result
    step2 = Step2()
    result2 = step2.execute(result1.data, None, tmp_path, result1)
    assert "step1" in result2.data.columns
    assert "step2" in result2.data.columns
    assert result2.metadata["prev_step"] == 1
