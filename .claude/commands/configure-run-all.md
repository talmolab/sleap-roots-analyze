# Configure Run-All

Interactively guide the user through creating a complete, scientifically sound set of pipeline
configuration files (QC config, Viz config, run manifest) for a new analysis.

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

### Step 2: Interactive Q&A Sequence

Collect answers **one topic at a time**. Show the recommended default and a brief rationale
for each question. Do NOT present all questions at once.

**2.1 — Output directory**
- Ask: "Where should pipeline outputs be written?" (suggest `./pipeline_runs/`)

**2.2 — Analysis name**
- Ask: "What is a short name for this analysis?" (used as `pipeline_name` and `run_name`)

**2.3 — Column assignments**
- Ask which column is the **barcode** (sample ID / plant QR code)
- Ask which column is the **genotype** (accession, variety, line)
- Ask which column is the **replicate** (plant ID, rep, block)
- Show detected candidates from the CSV to help the user choose

**2.4 — Grouped analysis**
- Ask: "Do you want to analyze subgroups separately?" (e.g., different timepoints)
- If yes: ask which column to group by (show `group_by_candidates`)
- Show group sizes; surface any statistical guardrail warnings

**2.5 — Cleanup thresholds**
- `max_nan_fraction` (default: 0.0 = drop any sample with a NaN; 0.25 = up to 25% NaN allowed)
- `max_zeros_per_trait` (default: 0.5 = drop traits with >50% zeros)
- `max_nans_per_trait` (default: 0.2 = drop traits with >20% NaN)
- `min_samples_per_trait` (default: max(10, n_samples_in_smallest_group // 4))
  - Explain: also controls minimum group size for grouping

**2.6 — Outlier detection**
- Ask: "Enable outlier detection?" (recommended: yes for n ≥ 30)
- If yes: recommend Mahalanobis (explain chi-squared approximation requires n ≥ 30)
- Surface any Mahalanobis small-n warnings from Step 1
- Show default: `chi2_percentile: 99.0`, suggest 95.0 for small groups

**2.7 — Heritability filtering**
- Ask: "Enable heritability filtering?" (recommended: yes when ≥ 3 replicates per genotype)
- Surface any heritability warnings from Step 1
- If enabled: ask threshold (default: 0.30 — permissive, show range 0.30–0.60)
  - 0.30: retains most heritable traits (exploratory)
  - 0.40: moderate
  - 0.50–0.60: only strongly heritable traits

**2.8 — PCA settings**
- `n_components` (default: 0.95 = keep PCs explaining 95% of variance)
- `feature_selection_strategy`:
  - `"top_variance"` — traits with highest total variance contribution (good for general exploration)
  - `"extreme"` — traits with most extreme positive AND negative PC loadings (better for mechanistic interpretation)
  - **Clarify**: this controls which traits are selected for UMAP coloring and metadata storage.
    It does NOT affect the feature contribution bar chart, which always ranks all traits by variance.
- `n_top_features` (default: 5) — how many traits the strategy selects for UMAP coloring
- `pca_biplot_top_features` (default: 1–2 for >100 traits, up to 5 for smaller datasets)
  - **Clarify**: this is INDEPENDENT of `n_top_features`. It controls biplot arrow count only.
  - For `"extreme"` strategy: a value of 1 gives 2 arrows per PC (one positive, one negative)
  - Keep small (1–5) for high-dimensional datasets to avoid arrow crowding

**2.9 — UMAP**
- Ask: "Enable UMAP?" (recommended: yes when n ≥ 15)
- If yes: recommend n_neighbors:
  ```python
  from sleap_roots_analyze.config_authoring import recommend_umap_n_neighbors
  n, warning = recommend_umap_n_neighbors(n_samples)
  ```
  Show the recommendation and any warning.
- `min_dist` (default: 0.1), `random_state` (default: 42)

**2.10 — Images**
- Ask: "Are plant images available for visualization?"
- If yes: ask for `image_dir` path
- If no: set `image_dir: null`

---

### Step 3: Critical Parameter Review

Before writing any files, present a review table:

```
CRITICAL PARAMETER REVIEW
═══════════════════════════════════════════════════════════════
Parameter                  Value      Status
───────────────────────────────────────────────────────────────
heritability.threshold     0.30       OK (permissive)
outlier chi2_percentile    99.0       ⚠ CHECK: n=25 < 30
min_samples_per_trait      10         OK
umap.n_neighbors           5          OK for n=25
group sizes                25, 30     ⚠ CHECK: day 0 has n=25
═══════════════════════════════════════════════════════════════
```

Flag (⚠) any parameter that deviates from recommended values based on the dataset.
Ask the user to confirm or modify each flagged parameter before proceeding.

---

### Step 4: Backup Check

For each config file that will be written, check if it already exists in `configs/active/`.

If a file exists:
1. Inform the user: "A config file already exists at `configs/active/<name>.yaml`."
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

### Step 5: Write Config Files

Write three files using the Write tool:

**5.1 — QC Config** → `configs/active/qc/<analysis_name>.yaml`

Include a self-documenting header:
```yaml
# QC Pipeline Configuration: <Analysis Name>
#
# Dataset: <csv_path>
# Analysis date: <YYYY-MM-DD>
# Generated by: /configure-run-all
#
# Key choices:
#   heritability.threshold: <value> — <brief rationale>
#   group_by: <column or "not used"> — <brief rationale>
#   chi2_percentile: <value> — <brief rationale>
```

**5.2 — Viz Config** → `configs/active/viz/<analysis_name>.yaml`

Include a self-documenting header:
```yaml
# Visualization Pipeline Configuration: <Analysis Name>
#
# Dataset: QC output from qc/<analysis_name>.yaml
# Analysis date: <YYYY-MM-DD>
# Generated by: /configure-run-all
#
# Note: When run via run-all, csv_path is auto-updated to the QC output.
#
# Key choices:
#   feature_selection_strategy: <value> — <brief rationale>
#   pca_biplot_top_features: <value> — <brief rationale>
```

Use sanitized column names in the Viz config (Barcode, Genotype, Replicate — these are
what the QC pipeline outputs after column renaming).

**5.3 — Run Manifest** → `configs/active/run_manifest_<analysis_name>.yaml`

Include a self-documenting header:
```yaml
# Run Manifest: <Analysis Name>
#
# Dataset: <csv_path>
# Analysis date: <YYYY-MM-DD>
# Generated by: /configure-run-all
#
# Reproduce this run:
#   sleap-roots-analyze run-all configs/active/run_manifest_<analysis_name>.yaml
#
# All paths are relative to configs/active/
```

---

### Step 6: User Validation Gate

After writing the files, display the full content of each config file.

Highlight (in bold text) the most consequential parameters:
- `heritability.threshold`
- `outlier_detection.traditional_methods` and `chi2_percentile`
- `data.group_by`
- `cleanup.min_samples_per_trait`
- `pca.feature_selection_strategy` and `pca_biplot_top_features`

Ask the user: "Do these configs look correct? Confirm with 'yes' / 'looks good' to commit and proceed."

Wait for explicit confirmation before continuing.

---

### Step 7: Git Commit

After user approval, commit the config files:

```bash
git add configs/active/
git commit -m "chore: configure analysis \"<run_name>\" (<YYYY-MM-DD>)

Dataset: <csv_path>
Config files:
  - configs/active/qc/<analysis_name>.yaml
  - configs/active/viz/<analysis_name>.yaml
  - configs/active/run_manifest_<analysis_name>.yaml"
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

### Step 8: Handoff

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
