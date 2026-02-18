# Run All Pipelines

Execute all QC, Viz, and Cross-Platform pipelines defined in the run manifest.

## Usage

This command runs all pipelines in the correct order with automatic path updates between dependent pipelines.

## Arguments

$ARGUMENTS

If no arguments provided, uses default manifest at `configs/active/run_manifest.yaml`.

Supported arguments:
- `--manifest <path>` - Custom manifest file
- `--dry-run` - Validate without running
- `--qc-only` - Run only QC pipelines
- `--viz-only` - Run only Viz pipelines
- `--cross-only` - Run only Cross-Platform pipelines
- `--no-summary` - Skip summary generation

## Workflow

1. Read the run manifest file (default: `configs/active/run_manifest.yaml`)
2. Create timestamped output directory in `pipeline_runs/`
3. Run all QC pipelines first
4. Update Viz and Cross-Platform config paths to point to new QC outputs
5. Run all Viz pipelines
   - If a QC config uses `group_by`, viz fans out automatically — one run per group,
     each written to `run_dir/viz/{group_label}/`. No manual workaround needed.
6. Run all Cross-Platform pipelines
7. Generate comprehensive summary document
8. Create `latest` symlink to most recent run

## Instructions

Run the following command based on the arguments provided:

```bash
sleap-roots-analyze run-all $ARGUMENTS
```

Track progress using TodoWrite for each pipeline being executed.

After completion, report:
- Number of successful/failed pipelines per type (QC, Viz, Cross-Platform)
- Output directory path
- Summary file location
- Any errors encountered

## Examples

```bash
# Run all pipelines with default manifest
sleap-roots-analyze run-all

# Dry run to validate manifest
sleap-roots-analyze run-all --dry-run

# Run only QC pipelines
sleap-roots-analyze run-all --qc-only

# Use custom manifest
sleap-roots-analyze run-all --manifest configs/active/custom_manifest.yaml
```