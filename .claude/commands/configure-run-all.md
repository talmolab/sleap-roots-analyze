# Configure Run-All

Interactively guide the user through creating a complete, scientifically sound set of pipeline
configuration files (QC config, Viz config, run manifest) for a new analysis.

This command uses **golden templates** from `configs/templates/` to ensure all required schema
fields are present. You customize only the fields that need to change for the user's dataset.

This is the companion to `/run-pipelines` — **configure first, then run.**

## Arguments

$ARGUMENTS

If a CSV path is provided as an argument, skip directly to the dataset inspection step.

---

## Workflow

Work through the following steps **one at a time**, waiting for user input at each stage.

---

### Step 1: Dataset Inspection

Ask the user for the path to their trait CSV file if not already provided.

Once you have the path, call the inspection utility:

```python
from sleap_roots_analyze.config_authoring import inspect_dataset
result = inspect_dataset("<csv_path>")
```

Report to the user:
- Total sample count (`result["n_samples"]`)
- All column names (`result["columns"]`)
- Number of numeric trait columns (`result["n_numeric_cols"]`)
- Candidate group_by columns — columns with ≤20 unique values (`result["group_by_candidates"]`)

**Statistical guardrails at inspection time:**

1. For each group_by candidate, show the group sizes. Flag any group with n < 30 with:
   ```python
   from sleap_roots_analyze.config_authoring import warn_mahalanobis_small_n
   # for each group: warn_mahalanobis_small_n(group_size)
   ```

2. Check minimum replicates per genotype (if genotype column is identifiable):
   ```python
   from sleap_roots_analyze.config_authoring import warn_heritability_low_replicates
   # min_reps = df.groupby(genotype_col)[replicate_col].nunique().min()
   # warn_heritability_low_replicates(min_reps)
   ```

---

### Step 2: Choose Template Variant

Ask the user two questions to determine which golden templates to use:

**2.1 — Grouped or ungrouped analysis?**
- Ask: "Do you want to analyze subgroups separately (e.g., different timepoints, sites)?"
- If YES: will use `qc_template_grouped.yaml`
- If NO: will use `qc_template_ungrouped.yaml`

**2.2 — Are images available?**
- Ask: "Are plant images available for visualization?"
- If YES: will use `viz_template_with_images.yaml`, ask for `image_dir` path
- If NO: will use `viz_template_no_images.yaml`

Tell the user which templates will be used:
```
Using templates:
  - QC: configs/templates/<qc_template_grouped or qc_template_ungrouped>.yaml
  - Viz: configs/templates/<viz_template_with_images or viz_template_no_images>.yaml
  - Manifest: configs/templates/run_manifest_template.yaml
```

---

### Step 3: Collect Required Customizations

Collect ONLY the fields that must be customized in the templates. Show the recommended default
and brief rationale for each question. Do NOT present all questions at once.

**3.1 — Output directory**
- Ask: "Where should pipeline outputs be written?" (suggest `./pipeline_runs/`)

**3.2 — Analysis name**
- Ask: "What is a short name for this analysis?" (used as `pipeline_name` and `run_name`)
- This will be used in config filenames: `qc/<name>.yaml`, `viz/<name>.yaml`, `run_manifest_<name>.yaml`

**3.3 — Column assignments** (for UNGROUPED template only)
- The ungrouped template has placeholders for column names
- Ask which column is the **barcode** (sample ID / plant QR code)
- Ask which column is the **genotype** (accession, variety, line)
- Ask which column is the **replicate** (plant ID, rep, block)
- Show detected candidates from the CSV to help the user choose

**3.4 — Group-by column** (for GROUPED template only)
- Ask which column to group by (show `group_by_candidates` from Step 1)
- Show group sizes; surface any statistical guardrail warnings (n < 30)

**3.5 — Image directory** (if images available)
- Ask for the path to the image directory
- Explain: "This directory should contain plant images referenced in the CSV"

**3.6 — Optional: Configurable parameters**
Ask if the user wants to customize these (default: use template values):

- `heritability.threshold` (template default: 0.30 for grouped, 0.40 for ungrouped)
  - 0.30: permissive (exploratory)
  - 0.40: moderate
  - 0.50–0.60: strict (only strongly heritable traits)
  - Surface heritability warnings from Step 1 if applicable

- `cleanup.min_samples_per_trait` (template default: 10)
  - Recommend: `max(10, n_samples_in_smallest_group // 4)`

- `pca.n_top_features` (template default: 5)
  - Controls how many traits are selected for UMAP coloring and metadata storage

- `umap.n_neighbors` (template default: 10 for grouped, 10 for ungrouped)
  - Recommend:
    ```python
    from sleap_roots_analyze.config_authoring import recommend_umap_n_neighbors
    n, warning = recommend_umap_n_neighbors(n_samples)
    ```

- `static_viz.pca_biplot_top_features` (template default: 1)
  - Controls biplot arrow count (independent of `n_top_features`)
  - For "extreme" strategy: 1 → 2 arrows/PC, 2 → 4 arrows/PC
  - Keep small (1–5) for high-dimensional datasets

---

### Step 4: Critical Parameter Review

Before writing any files, present a review table:

```
CRITICAL PARAMETER REVIEW
═══════════════════════════════════════════════════════════════
Parameter                  Value      Status
───────────────────────────────────────────────────────────────
csv_path                   <path>     OK
group_by                   <col>      OK (n=25, 30, 35)
columns.barcode            <col>      OK
heritability.threshold     0.30       OK (permissive)
min_samples_per_trait      10         OK
umap.n_neighbors           10         OK for n=90
image_dir                  <path>     OK
═══════════════════════════════════════════════════════════════
```

Flag (⚠) any parameter that deviates from recommended values based on the dataset.
Ask the user to confirm or modify each flagged parameter before proceeding.

---

### Step 5: Backup Check

For each config file that will be written, check if it already exists in `configs/active/`.

If a file exists:
1. Inform the user: "A config file already exists at `configs/active/<path>`."
2. Ask: "Would you like to save a backup before overwriting?"
3. If yes, create the backup:
   ```python
   from sleap_roots_analyze.config_authoring import make_backup_path
   from datetime import datetime
   from pathlib import Path
   import shutil

   archive_dir = Path("configs/archive")
   archive_dir.mkdir(parents=True, exist_ok=True)
   backup_path = make_backup_path(source, archive_dir)
   shutil.copy2(source, backup_path)
   # Report: f"Backed up to {backup_path}"
   ```
4. Do NOT overwrite without explicit user confirmation.

---

### Step 6: Copy and Customize Templates

**CRITICAL**: Use Read → Edit → Write workflow to preserve all template fields.
**NEVER** write configs from scratch — always start from a golden template.

**6.1 — QC Config** → `configs/active/qc/<analysis_name>.yaml`

1. **Read** the appropriate QC template:
   ```python
   # For grouped:
   template_path = "configs/templates/qc_template_grouped.yaml"
   # For ungrouped:
   template_path = "configs/templates/qc_template_ungrouped.yaml"
   ```

2. **Edit** ONLY the placeholders:
   - Replace `FILL_IN_PIPELINE_NAME` with the analysis name
   - Replace `FILL_IN_CSV_PATH` with the dataset CSV path
   - For UNGROUPED: Replace `FILL_IN_BARCODE_COLUMN`, `FILL_IN_GENOTYPE_COLUMN`, `FILL_IN_REPLICATE_COLUMN`
   - For GROUPED: The `group_by` column is already set in the template, but you may need to update it
   - If user customized optional parameters (heritability threshold, min_samples, etc.), update those values

3. **Add** a self-documenting header (replace the template comment header):
   ```yaml
   # QC Pipeline Configuration: <Analysis Name>
   #
   # Dataset: <csv_path>
   # Analysis date: <YYYY-MM-DD>
   # Generated from: <qc_template_grouped.yaml or qc_template_ungrouped.yaml>
   #
   # Key choices:
   #   heritability.threshold: <value> — <brief rationale>
   #   group_by: <column or null> — <brief rationale if grouped>
   ```

4. **Write** the modified config to `configs/active/qc/<analysis_name>.yaml`

**6.2 — Viz Config** → `configs/active/viz/<analysis_name>.yaml`

1. **Read** the appropriate Viz template:
   ```python
   # For with images:
   template_path = "configs/templates/viz_template_with_images.yaml"
   # For no images:
   template_path = "configs/templates/viz_template_no_images.yaml"
   ```

2. **Edit** ONLY the placeholders:
   - Replace `FILL_IN_PIPELINE_NAME` with the analysis name
   - Replace `FILL_IN_CSV_PATH` with a placeholder (explain it will be auto-updated by run-all)
   - For WITH IMAGES: Replace `FILL_IN_IMAGE_DIR` with the image directory path
   - If user customized optional parameters (n_top_features, n_neighbors, pca_biplot_top_features), update those values

3. **Use sanitized column names** in the Viz config:
   - `columns.barcode: "Barcode"`
   - `columns.genotype: "Genotype"`
   - `columns.replicate: "Replicate"`
   - `columns.image_path: "Image_Path"` (if images available, else `null`)
   - Explain: "These are the standardized names that the QC pipeline outputs after column renaming."

4. **Add** a self-documenting header (replace the template comment header):
   ```yaml
   # Visualization Pipeline Configuration: <Analysis Name>
   #
   # Dataset: QC output from qc/<analysis_name>.yaml
   # Analysis date: <YYYY-MM-DD>
   # Generated from: <viz_template_with_images.yaml or viz_template_no_images.yaml>
   #
   # Note: csv_path will be auto-updated to QC output when run via run-all.
   ```

5. **Write** the modified config to `configs/active/viz/<analysis_name>.yaml`

**6.3 — Run Manifest** → `configs/active/run_manifest_<analysis_name>.yaml`

1. **Read** the manifest template:
   ```python
   template_path = "configs/templates/run_manifest_template.yaml"
   ```

2. **Edit** ONLY the placeholders:
   - Replace `FILL_IN_RUN_NAME` with the analysis name
   - Replace `FILL_IN_DESCRIPTION` with: "QC and Viz for <dataset description>. [Groups: <group info if applicable>]"
   - Replace `FILL_IN_QC_CONFIG_PATH` with: `qc/<analysis_name>.yaml`
   - Replace `FILL_IN_VIZ_CONFIG_PATH` with: `viz/<analysis_name>.yaml`
   - Update the `qc_mapping` dictionary: `{"viz/<analysis_name>.yaml": "qc/<analysis_name>.yaml"}`

3. **Add** a self-documenting header (replace the template comment header):
   ```yaml
   # Run Manifest: <Analysis Name>
   #
   # Dataset: <csv_path>
   # Analysis date: <YYYY-MM-DD>
   # Generated from: run_manifest_template.yaml
   #
   # Reproduce this run:
   #   sleap-roots-analyze run-all configs/active/run_manifest_<analysis_name>.yaml
   #
   # All paths are relative to configs/active/
   ```

4. **Write** the modified manifest to `configs/active/run_manifest_<analysis_name>.yaml`

---

### Step 7: Validate Configs

**CRITICAL**: Validate the QC and Viz configs BEFORE showing them to the user.

```python
from sleap_roots_analyze.pipeline.config.utils import load_qc_config, load_viz_config, validate_qc_config, validate_viz_config

# Validate QC config
qc_config = load_qc_config("configs/active/qc/<analysis_name>.yaml")
validate_qc_config(qc_config)  # Will raise ValueError if invalid

# Validate Viz config
viz_config = load_viz_config("configs/active/viz/<analysis_name>.yaml")
validate_viz_config(viz_config)  # Will raise ValueError if invalid
```

If validation fails:
- Show the error message to the user
- Explain which field(s) are invalid
- Offer to fix and re-validate
- DO NOT proceed to user validation gate until configs pass validation

If validation succeeds, proceed to Step 8.

---

### Step 8: User Validation Gate

After validation passes, display a summary of each config file (not the full content — just key parameters).

Highlight (in bold text) the most consequential parameters:
- **QC config**:
  - `data.csv_path`
  - `data.group_by` (if grouped)
  - `columns.barcode`, `columns.genotype`, `columns.replicate` (if ungrouped)
  - `heritability.threshold`
  - `cleanup.min_samples_per_trait`

- **Viz config**:
  - `data.image_dir` (if images available)
  - `pca.n_top_features`
  - `umap.n_neighbors`
  - `static_viz.pca_biplot_top_features`

- **Run manifest**:
  - `run_name`
  - `qc_configs`
  - `viz_configs`

Ask the user: "Do these configs look correct? Confirm with 'yes' / 'looks good' to commit and proceed."

Wait for explicit confirmation before continuing.

---

### Step 9: Git Commit

After user approval, commit the config files:

```bash
git add configs/active/
git commit -m "chore: configure analysis \"<run_name>\" (<YYYY-MM-DD>)

Dataset: <csv_path>
Config files:
  - configs/active/qc/<analysis_name>.yaml
  - configs/active/viz/<analysis_name>.yaml
  - configs/active/run_manifest_<analysis_name>.yaml

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

Report the resulting SHA to the user:
```bash
git rev-parse HEAD
```

Tell the user: "**Reproducibility anchor**: commit SHA `<sha>`. This SHA permanently
links these configs to their exact codebase state."

If the git commit fails (e.g., no changes, detached HEAD):
- Warn clearly: "Git commit failed. Configs are written but NOT yet anchored to git."
- Instruct: "Run manually: `git add configs/active/ && git commit -m 'chore: configure ...'`"
- Do NOT crash or refuse to continue.

---

### Step 10: Handoff

Tell the user:
1. The configs are written and committed.
2. To run the analysis: `/run-pipelines --manifest configs/active/run_manifest_<analysis_name>.yaml`
3. The reproducibility SHA for their records.

Do NOT invoke `/run-pipelines` automatically.

---

## Examples

```bash
# Start fresh (will ask for CSV path)
/configure-run-all

# Provide CSV path upfront
/configure-run-all /path/to/traits.csv
```

## Reference: Statistical Guardrail Thresholds

| Check | Threshold | Function |
|---|---|---|
| Mahalanobis n | n ≥ 30 | `warn_mahalanobis_small_n(n)` |
| Heritability replicates | ≥ 3 reps/genotype | `warn_heritability_low_replicates(n)` |
| UMAP n_neighbors | min(15, n//4) | `recommend_umap_n_neighbors(n)` |

## Golden Template Reference

| Template | Use When | Placeholders |
|---|---|---|
| `qc_template_grouped.yaml` | Multi-group analysis (timepoints, sites, batches) | `FILL_IN_PIPELINE_NAME`, `FILL_IN_CSV_PATH` |
| `qc_template_ungrouped.yaml` | Single-group analysis | `FILL_IN_PIPELINE_NAME`, `FILL_IN_CSV_PATH`, `FILL_IN_*_COLUMN` |
| `viz_template_with_images.yaml` | Images available | `FILL_IN_PIPELINE_NAME`, `FILL_IN_CSV_PATH`, `FILL_IN_IMAGE_DIR` |
| `viz_template_no_images.yaml` | No images | `FILL_IN_PIPELINE_NAME`, `FILL_IN_CSV_PATH` |
| `run_manifest_template.yaml` | Always | `FILL_IN_RUN_NAME`, `FILL_IN_*_CONFIG_PATH` |
