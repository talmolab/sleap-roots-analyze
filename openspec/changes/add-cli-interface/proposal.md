# Add CLI Interface to Replace Hardcoded Scripts

**Status**: Draft  
**Created**: 2025-11-30  
**Author**: AI Assistant (Claude)  
**Type**: Enhancement

## Why

Currently, the package has `run_turface_qc.py` as a stray script with **hardcoded paths**:

```python
config_path = Path("configs/qc_turface_150genotypes.yaml")  # HARDCODED!
```

This creates several critical problems:

1. **Data coupling**: Script is tied to one specific dataset (Turface 150 genotypes)
2. **Not generalizable**: Cannot run pipeline on other datasets without modifying code
3. **Error-prone**: If config path changes, script breaks
4. **Poor UX**: Users must edit Python code to change configs
5. **Not a package feature**: Script exists outside the package structure
6. **Inconsistent**: No equivalent for VizPipeline
7. **Unmaintainable**: Violates DRY - duplicates pipeline execution logic

As the user stated: "we cannot have dataset hardcoded when it is in the config, this will result in errors"

The package already has the infrastructure (`QCPipeline`, `VizPipeline`, `load_qc_config`, `load_viz_config`) but lacks a proper CLI interface to use it.

## What

Add a production-ready CLI interface using **Click** with proper entry points configured in `pyproject.toml`.

### Command Structure

```bash
sleap-roots-analyze qc <config> [OPTIONS]     # Run QC pipeline
sleap-roots-analyze viz <config> [OPTIONS]    # Run Viz pipeline
sleap-roots-analyze config validate <config>  # Validate config
sleap-roots-analyze config show <config>      # Show resolved config
sleap-roots-analyze config list               # List example configs
sleap-roots-analyze version                   # Show version
```

### Key Features

**1. Config as Argument (Not Hardcoded)**
```bash
# Works with ANY config
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml
sleap-roots-analyze qc configs/qc_wheat_cylinder.yaml
sleap-roots-analyze qc /path/to/my_custom_config.yaml
```

**2. Output Directory Control**
```bash
sleap-roots-analyze qc config.yaml -o ./my_results
sleap-roots-analyze qc config.yaml --output-dir /data/qc_run_20251130
```

**3. Logging Control**
```bash
sleap-roots-analyze qc config.yaml --verbose          # DEBUG level
sleap-roots-analyze qc config.yaml --quiet            # WARNING level
sleap-roots-analyze qc config.yaml --log-level INFO   # Explicit level
sleap-roots-analyze qc config.yaml --log-file run.log # Save logs
```

**4. Dry Run Mode**
```bash
sleap-roots-analyze qc config.yaml --dry-run  # Validate without running
```

**5. Config Utilities**
```bash
sleap-roots-analyze config validate myconfig.yaml  # Check if valid
sleap-roots-analyze config show myconfig.yaml      # Show with defaults
sleap-roots-analyze config list                    # List examples
```

### Implementation

**Module Structure:**
```
src/sleap_roots_analyze/
├── __init__.py          # Expose main() in __all__
├── __main__.py          # Enable: python -m sleap_roots_analyze (NEW)
├── cli.py               # CLI implementation with Typer (NEW)
└── pipeline/
    ├── __init__.py
    └── pipelines/
```

**Entry Point (Already Configured):**
```toml
[project.scripts]
sleap-roots-analyze = "sleap_roots_analyze.cli:main"
```

**New Dependency:**
```toml
[project.dependencies]
click = ">=8.0.0"
rich = ">=13.0.0"  # For pretty output
```

**CLI Implementation Highlights:**
```python
# src/sleap_roots_analyze/cli.py
import click
from pathlib import Path
from rich.console import Console
from sleap_roots_analyze.pipeline import QCPipeline, load_qc_config

console = Console()

@click.group()
def cli():
    """Statistical analysis tools for root trait data from SLEAP Roots."""
    pass

@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('-o', '--output-dir', type=click.Path(path_type=Path), 
              default='./qc_runs', help='Output directory')
@click.option('-v', '--verbose', is_flag=True, help='Increase logging verbosity')
@click.option('--dry-run', is_flag=True, help='Validate config without running')
def qc(config, output_dir, verbose, dry_run):
    """Run QC pipeline on trait data."""
    # Load and validate config
    cfg = load_qc_config(config)
    
    # Display summary
    console.print(f"[cyan]Pipeline:[/cyan] {cfg.pipeline_name}")
    console.print(f"[cyan]Data:[/cyan] {cfg.data.csv_path}")
    
    if dry_run:
        console.print("[yellow]Dry run - would execute pipeline[/yellow]")
        return
    
    # Run pipeline
    pipeline = QCPipeline(config=cfg, output_dir=output_dir)
    results = pipeline.run()
    console.print(f"[green]✓ Complete! Results: {pipeline.run_dir}[/green]")

def main():
    cli()
```

### Deprecation of run_turface_qc.py

**Phase 1 (Immediate):** Add deprecation warning
```python
#!/usr/bin/env python
"""DEPRECATED: Use 'sleap-roots-analyze qc' instead."""
import warnings

warnings.warn(
    "run_turface_qc.py is deprecated. Use: sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml",
    DeprecationWarning,
)
# Still works but warns user
```

**Phase 2 (v0.1.0):** Remove script entirely

## Impact

### Benefits

1. **Generalizable**: Works with any config file, not hardcoded dataset
2. **Package-integrated**: Proper entry point, not stray script
3. **Better UX**: Clean CLI with help text, validation, error messages
4. **Consistent interface**: Same pattern for QC and Viz pipelines
5. **Professional**: Industry-standard CLI using Typer
6. **Testable**: CliRunner for comprehensive testing
7. **Discoverable**: `sleap-roots-analyze --help` shows all options
8. **Extensible**: Easy to add new commands/subcommands

### Breaking Changes

**None** - This is purely additive:
- `run_turface_qc.py` continues to work (with deprecation warning)
- Python API unchanged (`QCPipeline`, `load_qc_config`, etc.)
- Existing notebooks and scripts unaffected

### Migration Path

**Old way:**
```bash
python run_turface_qc.py  # Hardcoded config
```

**New way:**
```bash
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml
```

Migration is straightforward:
1. Replace `python run_turface_qc.py` with new CLI command
2. Optionally specify output directory with `-o`
3. Script continues to work with deprecation warning during transition

### Effort Estimate

- **CLI core (qc, viz commands)**: 4-6 hours
- **Config subcommands**: 2-3 hours  
- **Testing**: 4-5 hours (95%+ coverage)
- **Documentation**: 2 hours (README, examples)
- **Migration**: 1 hour (update run_turface_qc.py)

**Total**: ~13-17 hours (~2 days)

### Testing Strategy

1. Unit tests for all CLI commands using `CliRunner`
2. Test error handling (missing files, invalid configs)
3. Test flag combinations and defaults
4. Integration tests with real pipeline execution
5. Test help text generation
6. Achieve 95%+ coverage on `cli.py`

## Dependencies

- **Depends on**: None
- **Blocks**: None  
- **Enables**: 
  - Better user onboarding (clear CLI interface)
  - Multi-dataset workflows (not tied to one config)
  - Batch processing scripts (loop over configs)

## Alternatives Considered

1. **Keep stray scripts** - Rejected: Unmaintainable, not generalizable, violates DRY
2. **Use argparse** - Rejected: More verbose, less modern, harder to compose
3. **Use Typer** - Rejected: Lab standard is Click for consistency across projects
4. **Make script accept arguments** - Rejected: Still not package-integrated, harder to test
5. **Use project.entry-points instead of project.scripts** - Rejected: project.scripts is simpler for this use case

## References

- Current hardcoded script: `run_turface_qc.py:31`
- Entry point already defined: `pyproject.toml:63-64`
- Pipeline classes: `src/sleap_roots_analyze/pipeline/pipelines/`
- Config loaders: `src/sleap_roots_analyze/pipeline/config/`
- Click docs: https://click.palletsprojects.com/
- UV CLI best practices: https://docs.astral.sh/uv/guides/scripts/
