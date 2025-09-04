# Testing Guide

Comprehensive guide for running and writing tests for `sleap-roots-analyze`.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_statistics.py

# Run specific test class
uv run pytest tests/test_statistics.py::TestHeritabilityNumericalAccuracy

# Run specific test method
uv run pytest tests/test_statistics.py::TestCalculateTraitStatistics::test_basic_statistics
```

### Coverage Reports

```bash
# Run with coverage
uv run pytest --cov --cov-branch

# Generate detailed coverage report
uv run pytest --cov=src/sleap_roots_analyze --cov-branch --cov-report=html

# View coverage in terminal with missing lines
uv run pytest --cov=src/sleap_roots_analyze --cov-report=term-missing
```

### Test Selection

```bash
# Run only tests matching pattern
uv run pytest -k "heritability"

# Run tests marked with specific marker
uv run pytest -m "slow"  # If markers are defined

# Exclude specific tests
uv run pytest -k "not test_extreme_values"
```

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
├── fixtures.py              # Centralized test fixtures
├── test_data_cleanup.py    # Tests for data_cleanup module
├── test_statistics.py      # Tests for statistics module  
└── data/                    # Test data files
    ├── features.csv
    ├── traits_summary.csv
    ├── traits_summary_lateral.csv
    ├── traits_11DAG_cleaned_qc_scanner_independent.csv
    ├── Turface_all_traits_2024.csv
    └── Wheat_EDPIE_cylinder_master_data.xlsx
```

### Test Categories

#### 1. Unit Tests
Test individual functions in isolation:

```python
def test_basic_statistics(self):
    """Test calculation of basic statistics for traits."""
    df = pd.DataFrame({
        "trait1": [1, 2, 3, 4, 5],
        "trait2": [10, 20, 30, 40, 50],
    })
    trait_cols = ["trait1", "trait2"]
    
    stats = calculate_trait_statistics(df, trait_cols)
    
    assert stats["trait1"]["mean"] == 3.0
    assert stats["trait2"]["mean"] == 30.0
```

#### 2. Numerical Accuracy Tests
Verify calculations match expected mathematical results:

```python
def test_heritability_known_values(self, heritability_data_known_h2):
    """Test heritability calculation with known variance components."""
    df, expected_h2 = heritability_data_known_h2
    
    results = calculate_heritability_estimates(df, trait_cols)
    
    # Test relative ordering rather than exact values
    assert h2_high > h2_moderate > h2_low
```

#### 3. Edge Case Tests
Handle boundary conditions and special cases:

```python
def test_extreme_values(self, edge_case_extreme_values):
    """Test handling of infinity and extreme values."""
    df = edge_case_extreme_values  # Contains inf, -inf, tiny values
    
    stats = calculate_trait_statistics(df, ["trait_inf"])
    
    # Verify proper handling of infinity
    if np.isnan(stats["trait_inf"]["mean"]):
        assert True  # Expected behavior with inf values
```

#### 4. Integration Tests
Test complete workflows:

```python
def test_heritability_with_filtering(self):
    """Test integrated heritability calculation and filtering."""
    results = calculate_heritability_estimates(
        df, trait_cols,
        remove_low_h2=True,
        h2_threshold=0.3
    )
    
    h2_results, df_filtered, removed, details = results
    assert len(removed) > 0  # Some traits removed
```

## Writing Tests

### Test Fixtures

Fixtures provide reusable test data. Define in `fixtures.py`:

```python
@pytest.fixture
def heritability_data_known_h2():
    """Generate data with known heritability values.
    
    Returns:
        tuple: (DataFrame, dict of expected h2 values)
    """
    np.random.seed(42)  # Reproducible randomness
    
    # Generate data with known variance components
    # σ²_G = 4.0, σ²_E = 1.0 → H² = 0.8
    
    return df, expected_h2
```

### Best Practices

#### 1. Use Descriptive Names

```python
# Good
def test_remove_nan_samples_with_high_threshold():
    """Test NaN removal with 50% threshold."""
    
# Bad
def test_nan():
    """Test NaN."""
```

#### 2. Test One Thing

```python
# Good - tests one specific behavior
def test_zero_heritability(self):
    """Test that pure environmental variation gives H² ≈ 0."""
    
# Bad - tests multiple unrelated things
def test_statistics():
    """Test all statistics."""
```

#### 3. Use Fixtures for Complex Data

```python
def test_anova_known_effects(self, anova_data_known_effects):
    """Test ANOVA with fixture providing known group differences."""
    df, expected = anova_data_known_effects
    
    results = perform_anova_by_genotype(df, ["trait_anova"])
    
    assert results["trait_anova"]["p_value"] < 0.001
```

#### 4. Document Expected Failures

```python
def test_missing_required_columns(self):
    """Test that function fails appropriately with missing columns."""
    df = pd.DataFrame({"trait1": [1, 2, 3]})
    
    results = calculate_heritability_estimates(df, ["trait1"])
    
    assert "error" in results
    assert "Missing required columns" in results["error"]
```

#### 5. Handle Random Variation

```python
def test_with_random_data(self):
    """Test with random data using ranges rather than exact values."""
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Test ranges rather than exact values
    assert 0.7 < heritability < 0.9  # Not exact
```

## Current Test Coverage

### Module Coverage

| Module | Coverage | Tests | Notes |
|--------|----------|-------|-------|
| `data_cleanup.py` | 99% | 15 | Missing edge case for Excel loading |
| `statistics.py` | 95%+ | 45 | Full coverage with numerical tests |
| `data_utils.py` | 100% | 5 | Complete coverage |
| `outlier_detection.py` | N/A | 0 | Module in development |

### Test Statistics

- **Total Tests**: 65+
- **Pass Rate**: 100% (all tests passing)
- **Execution Time**: ~1.5 seconds
- **Fixtures**: 30+ reusable fixtures

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Solution: Ensure dependencies installed
uv sync --group dev
```

#### 2. Test Discovery Issues

```bash
# Ensure __init__.py exists in tests/
touch tests/__init__.py
```

#### 3. Fixture Not Found

```python
# Import fixtures in conftest.py
from tests.fixtures import *  # noqa
```

#### 4. Numerical Precision Warnings

These warnings are expected for edge cases:
- `RuntimeWarning: invalid value encountered` - Expected with infinity
- `ConvergenceWarning` - Expected with insufficient data
- `Precision loss` - Expected with constant values

### Debugging Tests

```bash
# Run with debugging output
uv run pytest -vvs

# Stop on first failure
uv run pytest -x

# Enter debugger on failure
uv run pytest --pdb

# Show local variables on failure
uv run pytest -l
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --group dev
      - run: uv run pytest --cov --cov-branch
```

## Performance Testing

### Benchmarking

```python
import pytest
import time

def test_heritability_performance(benchmark):
    """Benchmark heritability calculation."""
    df = create_large_dataset(1000, 50)  # 1000 samples, 50 traits
    
    result = benchmark(calculate_heritability_estimates, df, trait_cols)
    
    assert result is not None
```

### Memory Testing

```python
import tracemalloc

def test_memory_usage():
    """Test memory consumption stays reasonable."""
    tracemalloc.start()
    
    # Run memory-intensive operation
    process_large_dataset()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 1_000_000_000  # Less than 1GB
```

## Contributing Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Add fixtures** for complex test data
3. **Test edge cases** and error conditions
4. **Verify numerical accuracy** for calculations
5. **Update this documentation** with new patterns

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://realpython.com/pytest-python-testing/)
- [Numerical Testing Guide](https://numpy.org/doc/stable/reference/testing.html)