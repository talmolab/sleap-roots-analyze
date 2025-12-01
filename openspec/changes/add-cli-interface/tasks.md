# Implementation Tasks

## 1. Add Dependencies

**File**: `pyproject.toml`

**Changes**:
```toml
[project.dependencies]
# Add after existing dependencies
click = ">=8.0.0"
rich = ">=13.0.0"
```

**Estimate**: 0.5 hours (including testing that all deps resolve)

## 2. Create CLI Module

**File**: `src/sleap_roots_analyze/cli.py` (NEW)

**Changes**:
- Create main Click group
- Implement `qc` command with all options using `@click.command()`
- Implement `viz` command with all options
- Implement `config` command group using `@click.group()`:
  - `validate` - Validate a config file
  - `show` - Display resolved config
  - `list` - List example configs
- Implement `version` command
- Helper functions:
  - `setup_logging()` - Configure logging based on flags
  - `display_config_summary()` - Pretty print config summary
  - `main()` - Entry point function that calls `cli()`

**Estimate**: 5 hours

## 3. Create __main__ Module

**File**: `src/sleap_roots_analyze/__main__.py` (NEW)

**Changes**:
```python
"""Allow running: python -m sleap_roots_analyze"""
from sleap_roots_analyze.cli import main

if __name__ == "__main__":
    main()
```

**Estimate**: 0.25 hours

## 4. Update Package __init__

**File**: `src/sleap_roots_analyze/__init__.py`

**Changes**:
- Import `main` from cli module
- Add `main` to `__all__` exports

```python
from sleap_roots_analyze.cli import main

__all__ = [
    "main",  # Add this
    # ... existing exports
]
```

**Estimate**: 0.25 hours

## 5. Create CLI Tests

**File**: `tests/test_cli.py` (NEW)

**Changes**:
- Fixtures:
  - `cli_runner` - Typer CliRunner
  - `sample_qc_config` - Minimal valid QC config
  - `sample_viz_config` - Minimal valid Viz config
- Test classes:
  - `TestQCCommand` - All qc command tests
  - `TestVizCommand` - All viz command tests  
  - `TestConfigCommands` - Config subcommand tests
  - `TestVersionCommand` - Version command test
  - `TestGlobalOptions` - Help, no command, etc.
  - `TestIntegration` - Full pipeline execution tests
- Test coverage:
  - Missing config file
  - Invalid config
  - Dry run mode
  - Verbose/quiet flags
  - Custom output directory
  - Log file option
  - Help text
  - All subcommands

**Estimate**: 4 hours

## 6. Deprecate run_turface_qc.py

**File**: `run_turface_qc.py`

**Changes**:
- Add deprecation warning at top of file
- Keep functionality working
- Update docstring with migration instructions

```python
#!/usr/bin/env python
"""DEPRECATED: Use 'sleap-roots-analyze qc' instead.

This script is deprecated and will be removed in v0.1.0.
Please use the new CLI:

    sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml -o ./qc_runs

For more info: sleap-roots-analyze qc --help
"""
import warnings

warnings.warn(
    "run_turface_qc.py is deprecated. Use 'sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml'",
    DeprecationWarning,
    stacklevel=2,
)

# Existing code continues to work...
```

**Estimate**: 0.5 hours

## 7. Update Documentation

### 7.1 Update README.md

**File**: `README.md`

**Changes**:
- Add "Command-Line Interface" section
- Document all commands with examples
- Add quick start section
- Document migration from run_turface_qc.py

**Estimate**: 1.5 hours

### 7.2 Update CLAUDE.md

**File**: `CLAUDE.md`

**Changes**:
- Add CLI development guidelines
- Document testing approach for CLI
- Add examples of CLI usage

**Estimate**: 0.5 hours

### 7.3 Update TURFACE_QC_README.md

**File**: `TURFACE_QC_README.md`

**Changes**:
- Update "Quick Start" section to use new CLI
- Add deprecation notice for run_turface_qc.py
- Update all command examples

**Estimate**: 0.5 hours

## 8. Manual Testing

**Tasks**:
- Install package in editable mode
- Test `sleap-roots-analyze --help`
- Test `sleap-roots-analyze qc` with Turface config
- Test `sleap-roots-analyze config validate`
- Test error cases (missing files, invalid configs)
- Test on Windows and Unix-like systems
- Test `python -m sleap_roots_analyze`

**Estimate**: 1 hour

## 9. Update pyproject.toml

**File**: `pyproject.toml`

**Changes**:
- Verify entry point is correct:
```toml
[project.scripts]
sleap-roots-analyze = "sleap_roots_analyze.cli:main"
```
- Add test markers for CLI tests:
```toml
[tool.pytest.ini_options]
markers = [
    "cli: marks tests as CLI tests",
]
```

**Estimate**: 0.25 hours

## 10. Run Full Test Suite

**Tasks**:
- Run all tests: `uv run pytest tests/`
- Verify 95%+ coverage on cli.py: `uv run pytest --cov=sleap_roots_analyze.cli tests/test_cli.py`
- Run CLI-specific tests: `uv run pytest -m cli`
- Fix any failures

**Estimate**: 1 hour

## Total Effort Estimate

| Task | Hours |
|------|-------|
| 1. Add dependencies | 0.5 |
| 2. Create CLI module | 5.0 |
| 3. Create __main__ | 0.25 |
| 4. Update __init__ | 0.25 |
| 5. Create CLI tests | 4.0 |
| 6. Deprecate script | 0.5 |
| 7. Documentation | 2.5 |
| 8. Manual testing | 1.0 |
| 9. Update pyproject.toml | 0.25 |
| 10. Run test suite | 1.0 |
| **Total** | **15.25** |

## Implementation Order

1. **Dependencies** (task 1) - Required for all other work
2. **CLI core** (tasks 2, 3, 4) - Core functionality
3. **Testing** (task 5) - Verify everything works
4. **Deprecation** (task 6) - Handle migration
5. **Documentation** (task 7) - User-facing docs
6. **Validation** (tasks 8, 9, 10) - Final verification

## Validation Checklist

- [x] `sleap-roots-analyze qc <config>` works
- [x] `sleap-roots-analyze viz <config>` works
- [x] `sleap-roots-analyze config validate <config>` works
- [x] `sleap-roots-analyze --help` shows useful information
- [x] `python -m sleap_roots_analyze` works
- [x] Turface QC config runs via CLI
- [x] Custom output directory works
- [x] Verbose/quiet flags work
- [x] Dry run mode works
- [x] Invalid config shows clear error
- [x] Missing file shows clear error
- [ ] All tests pass (1109+ tests) - Skipped comprehensive CLI tests for initial implementation
- [ ] CLI tests achieve 95%+ coverage - Skipped for initial implementation
- [x] Documentation is complete
- [x] run_turface_qc.py shows deprecation warning
- [x] `openspec validate` passes
