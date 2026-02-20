# Design: Restore Interactive Parameter Q&A

## Problem Analysis

The current `/configure-run-all` command (after golden templates implementation) fails to walk users through critical parameters interactively. This creates a poor user experience where users must:
1. Know which parameters exist
2. Know what values are appropriate for their dataset
3. Manually edit configs or re-run the command

The original implementation (commit `b5e7166^`) had Steps 2.5-2.9 that comprehensively covered all parameters. This was lost in commit `2f46e5b`.

## Solution Design

### Restore Original Interactive Workflow

Copy the Q&A structure from the original implementation (pre-golden-templates) and integrate it with the current golden template workflow.

**Integration point**: After Step 3.4 (Group-by column / Column assignments), before Step 4 (Critical Parameter Review).

**New steps**:
- **Step 3.5**: Cleanup Thresholds (4 parameters)
- **Step 3.6**: Outlier Detection (enable?, method, chi2_percentile)
- **Step 3.7**: Heritability (enable?, threshold with null option)
- **Step 3.8**: PCA Settings (n_components, strategy, n_top_features, biplot_top_features)
- **Step 3.9**: UMAP (enable?, n_neighbors, min_dist, random_state)

### Parameter Organization

Group parameters by pipeline stage:

**QC Parameters** (affect data filtering):
- Cleanup thresholds
- Outlier detection
- Heritability filtering

**Viz Parameters** (affect visualization):
- PCA settings (both QC and Viz use PCA, but Viz has additional biplot settings)
- UMAP settings

### Interactive Prompting Pattern

For each parameter:
```
1. Present the question with context
2. Show recommended value based on dataset
3. Explain what different values mean
4. Surface any warnings (e.g., n < 30 for Mahalanobis)
5. Wait for user input
6. Validate input
7. Proceed to next parameter
```

### Template Customization Workflow

When user provides values that differ from template defaults:
1. Remember the customized values
2. When copying template (Step 6), Edit tool replaces BOTH placeholders AND customized parameters
3. Preserve all other template fields unchanged

Example:
```python
# User wants: max_nan_fraction=0.0, pca.n_components=0.75, pca.feature_selection_strategy="extreme"

# When editing QC template:
Edit(
    file_path="configs/active/qc/alfalfa.yaml",
    old_string="max_nan_fraction: 0.0",  # Template already has 0.0 for ungrouped
    new_string="max_nan_fraction: 0.0",  # No change needed
)

Edit(
    file_path="configs/active/qc/alfalfa.yaml",
    old_string="n_components: 0.95",  # Template default
    new_string="n_components: 0.75",  # User customization
)

Edit(
    file_path="configs/active/qc/alfalfa.yaml",
    old_string='feature_selection_strategy: "top_variance"',  # Template default
    new_string='feature_selection_strategy: "extreme"',  # User customization
)
```

### Context-Aware Recommendations

Use results from Step 1 (Dataset Inspection) to provide smart defaults:

**For cleanup.min_samples_per_trait**:
```python
if group_by_column:
    smallest_group = min(group_sizes.values())
    recommended = max(10, smallest_group // 4)
else:
    recommended = max(10, n_samples // 4)
```

**For umap.n_neighbors**:
```python
from sleap_roots_analyze.config_authoring import recommend_umap_n_neighbors
if group_by_column:
    smallest_group = min(group_sizes.values())
    n, warning = recommend_umap_n_neighbors(smallest_group)
else:
    n, warning = recommend_umap_n_neighbors(n_samples)
```

**For outlier_detection.mahalanobis.chi2_percentile**:
```python
if any(group_size < 30 for group_size in group_sizes.values()):
    recommended = 95.0  # More permissive for small groups
    warning = "Some groups have n<30; chi-squared approximation less reliable"
else:
    recommended = 99.0  # Strict for large groups
    warning = None
```

### Heritability Special Case

User may want heritability **calculation and visualization** but **not filtering**. Support this with:
```
Step 3.7.2: If enabled, filter by heritability threshold?
  Options:
    - null (calculate and visualize H² but don't drop low-heritability traits)
    - 0.30 (permissive filtering)
    - 0.40 (moderate filtering)
    - 0.50-0.60 (strict filtering)
```

When user selects `null`:
```python
# In QC config
heritability:
  enabled: true
  threshold: 0.0  # Set to 0.0 instead of null (schema may not allow null)
  generate_diagnostics: true

# OR: Disable filtering entirely but keep calculation
heritability:
  enabled: true
  threshold: null  # If schema allows null
  generate_diagnostics: true
```

Check QC config schema to determine if `threshold: null` is allowed or if we need a separate `filter_enabled: false` flag.

## Implementation Plan

### Phase 1: Restore Q&A Steps

1. Copy Steps 2.5-2.9 from original implementation (`b5e7166^:.claude/commands/configure-run-all.md`)
2. Renumber them to 3.5-3.9 (to fit after current Step 3.4)
3. Integrate with golden template workflow (references to templates, not Write-from-scratch)

### Phase 2: Update Template Customization Logic

Modify Step 6 (Copy and Customize Templates) to handle all collected customizations:
- QC customizations: cleanup thresholds, outlier settings, heritability threshold, PCA (n_components, strategy, n_top_features)
- Viz customizations: PCA (biplot_top_features), UMAP (n_neighbors, min_dist, random_state)

### Phase 3: Update Critical Parameter Review

Step 4 (Critical Parameter Review) should include ALL collected parameters, not just a subset:
```
CRITICAL PARAMETER REVIEW
═══════════════════════════════════════════════════════════════
Parameter                      Value        Status
───────────────────────────────────────────────────────────────
csv_path                       <path>       OK
group_by                       plant_age_days OK (n=42-72)
columns.barcode                plant_qr_code OK
columns.genotype               accession_id  OK
columns.replicate              plant_id      OK

[Cleanup Thresholds]
max_nan_fraction               0.0          OK (strict)
max_zeros_per_trait            0.5          OK (default)
max_nans_per_trait             0.2          OK (default)
min_samples_per_trait          10           OK

[Outlier Detection]
method                         mahalanobis  OK
chi2_percentile                99.0         OK (n≥42)

[Heritability]
enabled                        true         OK
threshold                      null         OK (viz-only)

[PCA]
n_components                   0.75         ⚠ CUSTOM (default: 0.95)
feature_selection_strategy     extreme      ⚠ CUSTOM (default: top_variance)
n_top_features                 5            OK
pca_biplot_top_features        1            OK

[UMAP]
n_neighbors                    10           OK (recommended: 10 for n=42)
min_dist                       0.1          OK
═══════════════════════════════════════════════════════════════
```

Flag parameters that deviate from template defaults so user is aware of customizations.

## Testing Strategy

### Manual Testing

Run `/configure-run-all` with test dataset and verify:
1. All parameters from Steps 3.5-3.9 are asked
2. Prompts appear one-at-a-time (not batched)
3. Recommendations are dataset-aware
4. Warnings surface correctly (n<30, low replicates, etc.)
5. Final configs contain all customized values

### Regression Testing

Ensure golden template workflow still works:
1. Templates are read (not written from scratch)
2. All schema fields are preserved
3. Validation passes before user confirmation
4. Backups are created if overwriting existing configs

## Rollout Plan

1. Implement changes to `.claude/commands/configure-run-all.md`
2. Test with real dataset (user's Alfalfa GWAS data)
3. Verify all parameters collected correctly
4. Verify generated configs are valid and complete
5. Commit changes
6. Document improvement in changelog

## Open Questions

1. **Heritability threshold = null**: Does the QC config schema allow `threshold: null`, or do we need `threshold: 0.0` to disable filtering?
   - **Resolution**: Check `src/sleap_roots_analyze/pipeline/config/qc_config.py` for schema definition

2. **Parameter order**: Should PCA settings be split between QC (n_components, strategy, n_top_features) and Viz (biplot_top_features)?
   - **Resolution**: Yes, ask about QC PCA settings in Step 3.8, then ask about Viz-specific biplot setting later or combine them

3. **Batch vs one-at-a-time**: Should sub-parameters (e.g., all 4 cleanup thresholds) be asked in one batch or truly one-at-a-time?
   - **Resolution**: Compromise: group related parameters in one prompt (e.g., all cleanup thresholds together) but ask each group separately
