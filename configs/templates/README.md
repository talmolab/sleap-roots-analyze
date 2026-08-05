# Pipeline Configuration Templates

Golden configuration templates for the QC and Viz pipelines.

These templates are **complete and schema-validated** — they include all required fields with clear placeholders for customization. Use them as starting points to ensure your configs are structurally correct.

## Recommended Starting Point: `/configure-run-all`

For new analyses, use the interactive slash command instead of editing templates manually:

```
/configure-run-all
```

This command:
- Inspects your CSV and reports sample counts, column names, and group sizes
- Guides you through parameter choices with statistical rationale and recommended defaults
- Warns when your sample size is too small for Mahalanobis chi-squared or heritability
- **Copies golden templates** and customizes only the fields that need to change
- **Validates configs** before writing them (catches schema errors early)
- Creates QC config, Viz config, and run manifest in one session
- Backs up any existing configs before overwriting
- Commits the final configs to git for reproducibility

After configuring, run with:
```
/run-pipelines --manifest configs/active/run_manifest_<your_analysis>.yaml
```

---

## Golden Templates

These templates are the source of truth for schema completeness. They are derived from known-working configs in `configs/active/` and validated against the config schema before each release.

### QC Templates

**`qc_template_grouped.yaml`** — For multi-group analyses (e.g., multiple timepoints, sites, batches)
- Uses `data.group_by` to split data by a metadata column
- Each group is analyzed independently (separate statistics, PCA, outlier detection)
- Heritability threshold default: 0.30 (permissive, suitable for exploratory grouped analyses)
- **Use when**: You have multiple timepoints, sites, or conditions and want to prevent confounding

**`qc_template_ungrouped.yaml`** — For single-group analyses
- All samples analyzed together as one dataset
- Heritability threshold default: 0.40 (moderate)
- **Use when**: All samples are from the same condition/timepoint

### Viz Templates

**`viz_template_with_images.yaml`** — When plant images are available
- `data.image_dir` enabled
- `interactive_viz.show_images_on_hover: true` (hover over plot points to see images)
- `interesting_genotypes.generate_image_grids: true` (creates image grids for extreme genotypes)
- **Use when**: You have a directory of plant images referenced in your CSV

**`viz_template_no_images.yaml`** — When images are NOT available
- `data.image_dir: null`
- Image-related features disabled
- All other viz features still available (PCA, UMAP, clustering, statistics)
- **Use when**: You only have trait data, no images

### Run Manifest Template

**`run_manifest_template.yaml`** — Orchestrates QC → Viz pipeline execution
- Lists QC configs to run first
- Lists Viz configs to run after (with csv_path auto-updated to QC output)
- Maps viz configs to their corresponding QC configs for path auto-update

---

## Manual Quick Start

If you prefer to create configs manually instead of using `/configure-run-all`:

### Step 1: Choose Templates

Based on your analysis needs, choose:

| Your Analysis | QC Template | Viz Template |
|---|---|---|
| Multi-timepoint, with images | `qc_template_grouped.yaml` | `viz_template_with_images.yaml` |
| Multi-timepoint, no images | `qc_template_grouped.yaml` | `viz_template_no_images.yaml` |
| Single timepoint, with images | `qc_template_ungrouped.yaml` | `viz_template_with_images.yaml` |
| Single timepoint, no images | `qc_template_ungrouped.yaml` | `viz_template_no_images.yaml` |

### Step 2: Copy and Customize

```bash
# Example: grouped analysis with images
cp configs/templates/qc_template_grouped.yaml configs/active/qc/my_analysis.yaml
cp configs/templates/viz_template_with_images.yaml configs/active/viz/my_analysis.yaml
cp configs/templates/run_manifest_template.yaml configs/active/run_manifest_my_analysis.yaml
```

### Step 3: Replace Placeholders

Edit each file and replace the `FILL_IN_*` placeholders:

**In QC config:**
- `FILL_IN_PIPELINE_NAME` → e.g., `"my_analysis_qc"`
- `FILL_IN_CSV_PATH` → path to your trait CSV
- For **ungrouped** template only:
  - `FILL_IN_BARCODE_COLUMN` → your sample ID column (e.g., `"barcode"`)
  - `FILL_IN_GENOTYPE_COLUMN` → your genotype column (e.g., `"geno"`)
  - `FILL_IN_REPLICATE_COLUMN` → your replicate column (e.g., `"rep"`)

**In Viz config:**
- `FILL_IN_PIPELINE_NAME` → e.g., `"my_analysis_viz"`
- `FILL_IN_CSV_PATH` → placeholder (will be auto-updated by run-all)
- For **with_images** template only:
  - `FILL_IN_IMAGE_DIR` → path to your image directory

**In Run Manifest:**
- `FILL_IN_RUN_NAME` → e.g., `"My Analysis Run"`
- `FILL_IN_DESCRIPTION` → brief description of your analysis
- `FILL_IN_QC_CONFIG_PATH` → e.g., `"qc/my_analysis.yaml"`
- `FILL_IN_VIZ_CONFIG_PATH` → e.g., `"viz/my_analysis.yaml"`
- Update `qc_mapping` dictionary to map viz → qc

### Step 4: Validate

Before running, validate your configs:

```bash
/validate-config configs/active/qc/my_analysis.yaml
```

This checks:
- All required fields are present
- Field types are correct
- Outlier methods are valid
- Data files exist (if paths specified)

### Step 5: Run

```bash
sleap-roots-analyze run-all configs/active/run_manifest_my_analysis.yaml
```

---

## Required Customizations

These fields **must** be replaced in the templates (marked with `FILL_IN_*` placeholders):

### QC Config (All Templates)
- `pipeline_name` - Unique name for this analysis
- `data.csv_path` - Path to your trait CSV file

### QC Config (Ungrouped Only)
- `columns.barcode` - Your sample ID column name
- `columns.genotype` - Your genotype column name
- `columns.replicate` - Your replicate column name

### Viz Config (All Templates)
- `pipeline_name` - Unique name for this analysis
- `data.csv_path` - Path to QC output (use placeholder for run-all)

### Viz Config (With Images Only)
- `data.image_dir` - Path to image directory

### Run Manifest
- `run_name` - Display name for this run
- `description` - Brief description
- `qc_configs` - List of QC config paths (relative to configs/active/)
- `viz_configs` - List of Viz config paths
- `qc_mapping` - Maps each viz config to its corresponding QC config

---

## Optional Customizations

These parameters have sensible defaults in the templates, but you may want to customize them:

### Heritability Threshold

- **0.30** (grouped template default) - Permissive (retains more traits, good for exploration)
- **0.40** (ungrouped template default) - Moderate (balanced)
- **0.50-0.60** - Stringent (only highly heritable traits)

Choose based on your downstream analysis requirements. Higher thresholds give you fewer but more reliable traits.

### Cleanup Thresholds

- **`max_nan_fraction`** (template default: 0.0 for ungrouped, varies for grouped)
  - 0.0 = drop any sample with missing data (strict)
  - 0.25 = allow up to 25% missing data per sample (permissive)

- **`max_zeros_per_trait`** (template default: 0.5)
  - 0.5 = drop traits with >50% zeros
  - Lower (0.3-0.4) = stricter, higher (0.6-0.7) = more permissive

- **`min_samples_per_trait`** (template default: 10)
  - Recommend: `max(10, n_samples_in_smallest_group // 4)`
  - Also controls minimum group size when using `data.group_by`

### PCA Settings (viz templates only — QC templates have no PCA settings)

- **`n_components`** (viz template default: 0.95)
  - 0.95 = retain PCs explaining 95% of variance
  - Higher (0.99) = more components retained, lower (0.90) = fewer components

- **`feature_selection_strategy`** (viz template default: varies)
  - `"top_variance"` - Traits with highest total variance contribution
  - `"extreme"` - Traits with most extreme PC loadings (both positive and negative)

- **`n_top_features`** (viz template default: 5)
  - Controls how many traits are selected for UMAP coloring and metadata storage
  - Does NOT affect the feature contribution bar chart (which always shows all traits)

- **`pca_biplot_top_features`** (viz template default: 1)
  - Controls biplot arrow count (INDEPENDENT of `n_top_features`)
  - For "extreme" strategy: 1 → 2 arrows/PC, 2 → 4 arrows/PC
  - Keep small (1–5) for high-dimensional datasets to avoid crowding

### UMAP Settings

- **`n_neighbors`** (template default: 10)
  - Recommend: `min(15, max(2, n_samples // 4))`
  - Smaller n → more local structure, larger n → more global structure

- **`min_dist`** (template default: 0.1)
  - Controls how tightly UMAP packs points together
  - 0.0 = very tight, 1.0 = very spread out

---

## Grouped Analysis by Timepoint

For multi-timepoint experiments (e.g., plants measured at 7, 14, 21 days), use `data.group_by` to analyze each timepoint independently.

### Why Use Grouped Analysis?

Combining data across timepoints can **confound temporal and genetic effects**, making heritability estimates invalid. Grouping ensures:
- Independent statistics per timepoint (ANOVA, heritability)
- Separate PCA analyses (PC loadings differ by developmental stage)
- Clean comparison of genetic effects within homogeneous groups

### Configuration

Use `qc_template_grouped.yaml` and set:

```yaml
data:
  csv_path: "multi_timepoint_data.csv"
  group_by: "plant_age_days"  # Column containing timepoint values
```

### CLI Usage

```bash
# Group by config value
sleap-roots-analyze qc my_config.yaml

# Override with CLI flag
sleap-roots-analyze qc my_config.yaml --group-by plant_age_days

# Run-all with grouping (applies to all pipelines in manifest)
sleap-roots-analyze run-all manifest.yaml --group-by plant_age_days
```

### run-all with group_by: Automatic Viz Fan-Out

When a QC config uses `group_by`, `run-all` automatically fans out the downstream viz pipeline
to run once per QC group output. Each group gets its own viz subdirectory and updated config:

```
run_dir/
├── qc/
│   ├── plant_age_days_7_20260217_143052/    # QC output for day 7
│   ├── plant_age_days_14_20260217_143108/   # QC output for day 14
│   └── plant_age_days_21_20260217_143124/   # QC output for day 21
└── viz/
    ├── plant_age_days_7/
    │   ├── _updated_my_viz_config.yaml      # csv_path → day 7 10_final_data.csv
    │   └── viz_output_20260217_144000/
    ├── plant_age_days_14/
    │   └── ...
    └── plant_age_days_21/
        └── ...
```

No manual workaround is needed. `run-all` handles the fan-out natively.

### Group Validation

Groups with fewer than `cleanup.min_samples_per_trait` samples are automatically skipped with a warning:

```
WARNING: Skipping group plant_age_days=28 (3 samples < 10 minimum)
```

### When to Use Grouping

✅ **Use grouping when:**
- Data has multiple timepoints/developmental stages
- Samples were collected at different sites/batches
- You need per-group heritability estimates

❌ **Don't use grouping when:**
- All samples are from the same timepoint/condition
- You intentionally want to analyze temporal trends
- Groups would have insufficient samples

---

## Real-World Configs

The parent `configs/active/` directory contains several real-world configurations you can learn from:

```bash
# Browse available configs
ls configs/active/qc/
ls configs/active/viz/

# View a specific config
cat configs/active/qc/qc_turface_150genotypes.yaml
```

These configs show real parameter choices from actual analyses. Feel free to use them as reference or starting points for your own experiments.

---

## Validation

When you run the pipeline, configs are validated automatically. Common issues:

### Error: Missing FILL_IN_ Placeholders

```
ValueError: data.csv_path is required
```

**Fix**: Replace all `FILL_IN_*` placeholders with actual values before running.

### Error: Invalid Outlier Method

```
ValueError: outlier_detection.traditional_methods contains invalid method 'mahalanobis_pca'
```

**Fix**: Check the template for valid method names. Use `"mahalanobis"`, not `"mahalanobis_pca"`.

### Warning: No Outlier Detection

```
UserWarning: No outlier detection methods configured...
```

**This is normal** if `traditional_methods` and `clustering_methods` are both empty. The warning ensures you're making a conscious choice. You can still run the pipeline with cleanup only (no outlier detection).

---

## Getting Help

If you encounter issues:

1. Use `/configure-run-all` instead of editing templates manually (recommended)
2. Use `/validate-config` to check your config before running
3. Check that all `FILL_IN_*` placeholders are replaced
4. Review the examples in `configs/active/` directory
5. Read the error messages carefully - they include recommended fixes

---

## Tips

1. **Start with `/configure-run-all`** - It's faster and catches errors earlier than manual editing
2. **Use golden templates for manual configs** - They're complete and schema-validated
3. **Validate before running** - Use `/validate-config` to catch errors early
4. **Document your choices** - Add comments explaining why you chose specific values
5. **Test with small datasets** - Validate your config before running on full data
6. **Review outputs** - Check the generated plots and summaries to ensure QC worked as expected
7. **Learn from examples** - The configs in `configs/active/` show real-world usage patterns
8. **Use grouping for multi-timepoint data** - Prevents confounding temporal and genetic effects
