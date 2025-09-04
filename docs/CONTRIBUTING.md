# Contributing to SLEAP Roots Analyze

Thank you for your interest in contributing to `sleap-roots-analyze`! We welcome contributions from the community.

## Getting Started

### Prerequisites

- Python 3.9+
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/))
- Git

### Development Setup

1. **Fork the repository**
   - Click the "Fork" button on GitHub
   - Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sleap-roots-analyze.git
   cd sleap-roots-analyze
   ```

2. **Set up development environment**
   ```bash
   # Install with development dependencies
   uv sync --group dev
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Write Tests First (TDD)

Before implementing a feature, write tests:

```python
# tests/test_your_feature.py
def test_new_functionality():
    """Test the new feature."""
    result = new_function(input_data)
    assert result == expected_output
```

### 2. Implement the Feature

Write clean, documented code:

```python
def new_function(data: pd.DataFrame) -> pd.DataFrame:
    """Brief description of function.
    
    Args:
        data: Input DataFrame with trait data
        
    Returns:
        Processed DataFrame
        
    Raises:
        ValueError: If data is invalid
    """
    # Implementation
    return processed_data
```

### 3. Run Tests

Ensure all tests pass:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov --cov-branch

# Run specific tests
uv run pytest tests/test_your_feature.py -v
```

### 4. Format and Lint

Format your code:

```bash
# Auto-format with black
uv run black src/sleap_roots_analyze tests

# Check linting
uv run ruff check src/sleap_roots_analyze tests

# Fix linting issues
uv run ruff check --fix src/sleap_roots_analyze tests
```

### 5. Update Documentation

- Add docstrings to all functions
- Update relevant `.md` files
- Add examples if introducing new features

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) with these specifications:

- **Line length**: 88 characters (black default)
- **Docstrings**: Google style
- **Type hints**: Use when beneficial for clarity

### Docstring Example

```python
def calculate_metric(
    data: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate metrics for specified columns.
    
    Args:
        data: Input DataFrame containing trait data
        columns: List of column names to process
        threshold: Cutoff threshold for filtering (default: 0.5)
        
    Returns:
        Dictionary mapping column names to calculated metrics
        
    Raises:
        ValueError: If columns are not found in DataFrame
        TypeError: If data is not a DataFrame
        
    Example:
        >>> df = pd.DataFrame({'trait1': [1, 2, 3]})
        >>> result = calculate_metric(df, ['trait1'])
        >>> print(result['trait1'])
        2.0
    """
    # Implementation
```

### Import Organization

```python
# Standard library imports
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

# Local imports
from sleap_roots_analyze.data_utils import helper_function
```

## Testing Guidelines

### Test Organization

```
tests/
├── fixtures.py          # Shared test fixtures
├── test_feature.py      # Tests for specific feature
└── data/               # Test data files
```

### Writing Good Tests

1. **Test one thing**: Each test should verify a single behavior
2. **Use descriptive names**: `test_remove_nan_with_high_threshold` not `test_nan`
3. **Test edge cases**: Empty data, single values, extreme values
4. **Use fixtures**: Share complex test data via fixtures
5. **Document intent**: Add docstrings explaining what's being tested

### Test Example

```python
def test_heritability_with_perfect_genetic_determination(heritability_perfect_data):
    """Test that H² = 1.0 when all variation is genetic."""
    df = heritability_perfect_data
    
    results = calculate_heritability_estimates(df, ["trait"])
    
    assert abs(results["trait"]["heritability"] - 1.0) < 0.001
    assert results["trait"]["var_residual"] < 0.001
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: `uv run pytest`
2. **Check coverage**: `uv run pytest --cov --cov-branch`
3. **Format code**: `uv run black src tests`
4. **Lint code**: `uv run ruff check src tests`
5. **Update documentation**: Especially if adding new features
6. **Update CHANGELOG.md**: Add entry under "Unreleased"

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Coverage maintained or improved

## Checklist
- [ ] Code formatted with black
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Submit PR against `main` branch
2. Ensure CI checks pass
3. Respond to reviewer feedback
4. Once approved, PR will be merged

## Reporting Issues

### Bug Reports

Include:
- Python version
- Package version
- Minimal reproducible example
- Error messages
- Expected vs actual behavior

### Feature Requests

Include:
- Use case description
- Proposed API
- Example code
- Alternative solutions considered

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information
- Other unprofessional conduct

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in relevant documentation

## Questions?

- Open a [GitHub Discussion](https://github.com/talmolab/sleap-roots-analyze/discussions)
- Check existing [Issues](https://github.com/talmolab/sleap-roots-analyze/issues)
- Review [Documentation](docs/)

Thank you for contributing to make `sleap-roots-analyze` better!