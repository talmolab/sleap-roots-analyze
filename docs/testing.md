# Testing Guide

This document describes the testing infrastructure for `sleap-roots-analyze`.

## Test Structure

Tests are organized in the `tests/` directory:
- `tests/fixtures.py` - Centralized pytest fixtures for all tests
- `tests/conftest.py` - Pytest configuration that loads fixtures
- `tests/test_*.py` - Test modules for each source module

## Running Tests

### Basic Test Execution
```bash
# Run all tests
uv run pytest

# Run specific test module
uv run pytest tests/test_data_cleanup.py

# Run with verbose output
uv run pytest -v
```

### Coverage Analysis
```bash
# Run with coverage report
uv run pytest --cov --cov-branch

# Coverage for specific module
uv run pytest tests/test_data_cleanup.py --cov=src/sleap_roots_analyze/data_cleanup --cov-branch

# Show missing lines
uv run pytest --cov --cov-branch --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov --cov-branch --cov-report=html
```

### Code Formatting
```bash
# Format code with black
uv run black src/sleap_roots_analyze tests

# Check formatting without changes
uv run black --check src/sleap_roots_analyze tests

# Format with ruff
uv run ruff format src/sleap_roots_analyze tests
uv run ruff check --fix src/sleap_roots_analyze tests
```

## Test Data Fixtures

The `tests/fixtures.py` file provides centralized fixtures for test data:

### CSV Data Fixtures
- `features_df` - Features data from `features.csv`
- `traits_11dag_df` - 11 DAG traits data  
- `traits_summary_df` - Summarized trait data
- `traits_summary_lateral_df` - Lateral root summary data
- `turface_traits_df` - Turface experiment data
- `wheat_edpie_excel_df` - Wheat EDPIE Excel data

### Sample Data Fixtures
- `*_sample` fixtures provide first 10 rows for quick testing
- Mock data fixtures for unit testing without file I/O

### Utility Fixtures
- `test_data_dir` - Path to test data directory
- `*_csv_path` - Paths to CSV files
- Column name fixtures for validation
- Random number generator fixtures

## Writing Tests

### Example Test Structure
```python
import pytest
import pandas as pd
from src.sleap_roots_analyze.data_cleanup import load_trait_data

class TestDataCleanup:
    def test_load_valid_csv(self, tmp_path):
        """Test loading a valid CSV file."""
        # Create test data
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "Barcode": ["BC001"],
            "geno": ["G1"],
            "trait1": [1.0]
        })
        df.to_csv(csv_path, index=False)
        
        # Test function
        result = load_trait_data(csv_path)
        
        # Assertions
        assert len(result) == 1
        assert "Barcode" in result.columns
```

### Using Fixtures
```python
def test_with_features_data(features_df):
    """Test using features.csv fixture."""
    assert not features_df.empty
    assert "Total.Root.Length.mm" in features_df.columns
```

## Coverage Goals

- Target: 100% coverage for critical modules
- Current coverage:
  - `data_cleanup.py`: 99%
  - Additional modules: In progress

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Release builds

See `.github/workflows/ci.yml` for CI configuration.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `uv sync` has been run
2. **Missing test data**: Check that CSV files exist in `tests/data/`
3. **Coverage not working**: Use `--cov` flag with full module paths

### Performance Warnings

If you see warnings about DataFrame fragmentation:
```
PerformanceWarning: DataFrame is highly fragmented
```
This is expected in some test scenarios and can be ignored.

## Best Practices

1. **Organize tests by module**: Mirror source structure in tests
2. **Use fixtures**: Centralize test data in `fixtures.py`
3. **Test edge cases**: Include tests for error conditions
4. **Mock external dependencies**: Use unittest.mock for file I/O in unit tests
5. **Keep tests fast**: Use sample data fixtures for quick tests
6. **Document tests**: Use clear test names and docstrings