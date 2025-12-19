## Context

Users need to run multiple pipelines (QC, Viz, Cross-Platform) together and generate publication-ready summaries. Currently this requires:
1. Manually running each pipeline
2. Updating paths between dependent pipelines
3. Manually collecting results into a summary document

The existing `configs/` directory contains both example configs (for documentation/templates) and active project configs (for actual analysis runs).

## Goals / **Non-Goals:**
- Modifying the underlying QC/Viz/Cross-Platform pipeline code
- Automatic config validation (existing `--dry-run` serves this)

## Decisions

### Decision 1: Config Directory Structure

Organize configs into subfolders:

```
configs/
├── active/                    # Configs for actual project runs
│   ├── qc/                   # QC pipeline configs
│   │   ├── qc_turface_150genotypes.yaml
│   │   ├── qc_turface_19genotypes.yaml
│   │   └── ...
│   ├── viz/                  # Visualization configs
│   │   ├── viz_turface_150genotypes.yaml
│   │   └── ...
│   └── cross_platform/       # Cross-platform analysis configs
│       ├── cross_platform_turface_150vs19.yaml
│       └── ...
├── templates/                 # Example/template configs (already exists)
│   ├── qc_cleanup_only_template.yaml
│   └── qc_full_pipeline_template.yaml
└── examples/                  # Example configs for documentation
    └── ...
```

**Rationale:**
- Clear separation of active project configs from examples
- Easy to glob all configs to run: `configs/active/**/*.yaml`
- Templates preserved separately for user reference
- Prevents accidentally running example configs

**Alternative considered:** Using naming conventions (e.g., `_active` suffix)
- Rejected: Harder to manage, clutters filenames

### Decision 2: Run Output Organization

Create timestamped run folders:

```
pipeline_runs/                 # All pipeline outputs (gitignored)
├── 2025-12-15_130000/        # Timestamped run folder
│   ├── qc/                   # QC outputs
│   │   ├── turface_150genotypes_qc/
│   │   └── ...
│   ├── viz/                  # Viz outputs
│   │   ├── viz_turface_150genotypes/
│   │   └── ...
│   ├── cross_platform/       # Cross-platform outputs
│   │   └── ...
│   └── SUMMARY.md            # Run summary
└── latest -> 2025-12-15_130000/  # Symlink to latest run
```

**Rationale:**
- All outputs from a single run grouped together
- Easy to find latest run via symlink
- Timestamps enable comparing historical runs
- Single gitignore entry for `pipeline_runs/`

**Alternative considered:** Keep current `qc_runs/`, `viz_runs/`, `cross_platform_runs/`
- Rejected: Harder to correlate outputs from same run session

### Decision 3: Config Grouping via Manifest File

Create a manifest file listing configs to run:

```yaml
# configs/active/run_manifest.yaml
run_name: "EDPIE Full Analysis"
description: "Complete QC, Viz, and Cross-Platform analysis for EDPIE paper"

# Configs are run in order: qc first, then viz/cross_platform
qc_configs:
  - qc/qc_turface_150genotypes.yaml
  - qc/qc_turface_19genotypes.yaml
  - qc/qc_cylinder_edpie.yaml
  - qc/qc_root_core_edpie.yaml
  - qc/qc_field_2024_clean.yaml

viz_configs:
  - viz/viz_turface_150genotypes.yaml
  - viz/viz_turface_19genotypes.yaml
  - viz/viz_cylinder_edpie.yaml
  - viz/viz_root_coring.yaml
  - viz/viz_field_2024_clean.yaml

cross_platform_configs:
  - cross_platform/cross_platform_turface_150vs19_genotypes.yaml
  - cross_platform/cross_platform_turface19_vs_cylinder.yaml
  - cross_platform/cross_platform_turface19_vs_field.yaml
  - cross_platform/cross_platform_field_vs_cylinder.yaml
```

**Rationale:**
- Explicit control over which configs to run
- Can have multiple manifests for different analysis sets
- Documents the intended grouping for reproducibility
- Paths relative to `configs/active/` directory

**Alternative considered:** Auto-discover all configs in `configs/active/`
- Rejected: Less control, may run unwanted configs

### Decision 4: Summary Document with Timestamp

Include generation metadata in summary:

```markdown
# EDPIE Phenotyping Pipeline Runs Summary

**Generated:** 2025-12-15 13:00:00
**Run Directory:** pipeline_runs/2025-12-15_130000
**Git Commit:** 0fb6f35...
**Manifest:** configs/active/run_manifest.yaml

---
[Rest of summary content]
```

**Rationale:**
- Clear provenance for reproducibility
- Easy to match summary to run outputs
- Git commit ties results to code version

### Decision 5: Gitignore Pattern

Add to `.gitignore`:

```gitignore
# Pipeline run outputs (timestamped)
pipeline_runs/

# Keep existing patterns for backward compatibility
qc_runs/
viz_runs/
cross_platform_runs/
```

**Rationale:**
- New structure gitignored by default
- Backward compatibility with existing runs
- Users can selectively add specific runs to git if needed

### Decision 6: Dual Interface (CLI + Slash Command)

Provide both a CLI command and a Claude Code slash command:

```bash
# CLI command (for scripts, CI/CD, terminal users)
sleap-roots-analyze run-all --manifest configs/active/run_manifest.yaml

# Slash command (for Claude Code interactive sessions)
/run-pipelines --manifest custom_manifest.yaml
```

**CLI Command Structure:**
```
sleap-roots-analyze run-all [OPTIONS]

Options:
  --manifest PATH    Path to run manifest file [default: configs/active/run_manifest.yaml]
  --output PATH      Output directory [default: pipeline_runs/]
  --dry-run          Validate and show execution plan without running
  --qc-only          Run only QC pipelines
  --viz-only         Run only Viz pipelines (requires existing QC outputs)
  --cross-only       Run only Cross-Platform pipelines (requires existing QC outputs)
  --no-summary       Skip summary generation
  -v, --verbose      Increase output verbosity
```

**Rationale:**
- CLI command enables scripting, CI/CD integration, and terminal workflows
- Slash command provides interactive progress tracking via TodoWrite
- Same underlying logic (pipeline_runner module) for both interfaces
- CLI is the "source of truth" implementation; slash command wraps it

**Alternative considered:** Slash command only
- Rejected: Limits automation, CI/CD, and non-Claude Code usage

### Decision 7: Pipeline Runner Module Architecture

Create a new Python module for the core logic:

```python
# src/sleap_roots_analyze/pipeline_runner.py

class PipelineRunner:
    """Orchestrates multi-pipeline execution with dependency management."""
    
    def __init__(self, manifest_path: Path, output_dir: Path):
        self.manifest = self._load_manifest(manifest_path)
        self.output_dir = output_dir
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    def run_all(self, qc_only=False, viz_only=False, cross_only=False):
        """Execute pipelines in dependency order."""
        ...
    
    def _run_qc_pipelines(self) -> Dict[str, Path]:
        """Run all QC pipelines, return mapping of config -> output path."""
        ...
    
    def _update_dependent_configs(self, qc_outputs: Dict[str, Path]):
        """Update Viz/Cross-Platform configs with new QC output paths."""
        ...
    
    def generate_summary(self) -> Path:
        """Generate comprehensive markdown summary."""
        ...
```

**Rationale:**
- Clean separation of concerns
- Testable module independent of CLI
- Reusable by both CLI and slash command
- Follows existing codebase patterns

## Risks / Trade-offs

1. **Migration effort**: Existing configs need to be moved to new structure
   - Mitigation: Provide migration script, keep old locations working temporarily

2. **Breaking existing workflows**: Users may have scripts pointing to old paths
   - Mitigation: Document changes, keep old `qc_runs/` etc. working

3. **Complexity**: More directory structure to understand
   - Mitigation: Clear documentation, sensible defaults

## Migration Plan

1. Create new directory structure (`configs/active/`, `pipeline_runs/`)
2. Move existing active configs to `configs/active/`
3. Create `run_manifest.yaml` listing current active configs
4. Update `.gitignore`
5. Document new workflow in README/CLAUDE.md
6. Keep old paths working for backward compatibility

## Open Questions

1. Should we support multiple manifest files for different analysis subsets?
   - Recommendation: Yes, allow specifying manifest via command argument

2. Should viz configs auto-update to point to latest QC run, or require explicit mapping?
   - Recommendation: Auto-update during run, with option to specify explicit mapping in manifest

3. Should we generate HTML summary in addition to markdown?
   - Recommendation: Start with markdown only, add HTML later if needed
