# Proposal: Restore Interactive Parameter Q&A to `/configure-run-all`

## Why

The `/configure-run-all` command had a comprehensive interactive Q&A workflow (Steps 2.5-2.9) that walked users through ALL scientifically critical parameters one-by-one. This was removed in commit `2f46e5b` when implementing golden templates, causing a **major regression**:

### What Was Lost

**Removed from original implementation:**
- **Step 2.5 — Cleanup thresholds**: `max_nan_fraction`, `max_zeros_per_trait`, `max_nans_per_trait`, `min_samples_per_trait`
- **Step 2.6 — Outlier detection**: Enable?, method choice (mahalanobis/pca/isolation_forest), `chi2_percentile`
- **Step 2.7 — Heritability filtering**: Enable?, threshold with guidance (0.30/0.40/0.50)
- **Step 2.8 — PCA settings**: `n_components`, `feature_selection_strategy`, `n_top_features`, `pca_biplot_top_features` with detailed explanations
- **Step 2.9 — UMAP**: Enable?, `n_neighbors` with recommendation function, `min_dist`, `random_state`

**Current broken state (Step 3.6):**
- Collapsed into "Ask if the user wants to customize these (default: use template values)"
- Lists only 5 parameters as bullet points
- Does NOT actually walk through each parameter interactively
- Missing critical cleanup thresholds entirely
- No guidance on when to use each value

### Real-World Impact

User attempting to configure Alfalfa GWAS analysis discovered:
1. Command didn't ask about `max_nan_fraction` (user wants 0.0 to drop all NaN samples)
2. Command didn't ask about `max_zeros_per_trait` or `max_nans_per_trait`
3. Command didn't ask about outlier detection method choice
4. Command didn't ask about PCA variance threshold (user wants 0.75, not default 0.95)
5. Command didn't ask about PCA feature selection strategy (user wants "extreme", not "top_variance")
6. Command didn't ask about heritability enable vs. threshold (user wants visualization without filtering)

**Result**: User had to manually discover what parameters exist and what they should be set to, defeating the purpose of the interactive command.

## What Changes

### 1. Restore Full Interactive Q&A Workflow

Replace current Step 3.6 ("Optional: Configurable parameters") with the original comprehensive workflow:

**Step 3.5 — Cleanup Thresholds** (ask one-by-one):
```
1. max_nan_fraction (per sample)
   - Explain: "Drop samples with more than X% missing trait values"
   - Show current dataset NaN fraction distribution
   - Recommended: 0.0 (strict), 0.25 (permissive)
   - User specifies or accepts default

2. max_zeros_per_trait (per trait column)
   - Explain: "Drop trait columns with more than X% zero values"
   - Recommended: 0.5 (drop traits with >50% zeros)
   - User specifies or accepts default

3. max_nans_per_trait (per trait column)
   - Explain: "Drop trait columns with more than X% NaN values"
   - Recommended: 0.2 (drop traits with >20% NaNs)
   - User specifies or accepts default

4. min_samples_per_trait
   - Explain: "Minimum non-NaN samples required per trait (also controls minimum group size)"
   - Recommended: max(10, n_samples_in_smallest_group // 4)
   - User specifies or accepts default
```

**Step 3.6 — Outlier Detection** (ask one-by-one):
```
1. Enable outlier detection?
   - Explain: "Detect and optionally remove statistical outliers"
   - Recommended: yes when n ≥ 30 per group
   - Surface Mahalanobis warnings from Step 1 if any group has n < 30

2. If enabled, which method?
   - "mahalanobis" (recommended for n ≥ 30, uses chi-squared approximation)
   - "pca" (PCA-based outlier detection)
   - "isolation_forest" (good for small datasets)
   - Can select multiple

3. If mahalanobis selected, chi2_percentile?
   - Explain: "Outlier threshold: higher = stricter (fewer outliers flagged)"
   - 99.0 = strict (top 1% flagged)
   - 95.0 = permissive (top 5% flagged)
   - User specifies or accepts default (99.0)
```

**Step 3.7 — Heritability** (ask one-by-one):
```
1. Enable heritability calculation?
   - Explain: "Calculate broad-sense heritability (H²) per trait"
   - Recommended: yes when ≥ 3 replicates per genotype
   - Surface heritability warnings from Step 1 if insufficient replicates

2. If enabled, filter by heritability threshold?
   - Explain: "Drop traits with H² below threshold (set to null for visualization-only)"
   - Options:
     - null (calculate and visualize but don't filter)
     - 0.30 (permissive, exploratory)
     - 0.40 (moderate)
     - 0.50-0.60 (strict, only highly heritable traits)
   - User specifies or accepts default (0.30 for grouped, 0.40 for ungrouped)
```

**Step 3.8 — PCA Settings** (ask one-by-one):
```
1. n_components (variance threshold or fixed count)
   - Explain: "How many principal components to retain"
   - Float (0.0-1.0): retain PCs explaining X% of variance
   - Int: retain exactly X components
   - Recommended: 0.95 (retain PCs explaining 95% variance)
   - User specifies or accepts default

2. feature_selection_strategy
   - Explain: "How to select top traits for UMAP coloring and metadata storage"
   - Options:
     - "top_variance": traits with highest total variance contribution (general exploration)
     - "extreme": traits with most extreme positive AND negative PC loadings (mechanistic interpretation)
   - Clarify: Does NOT affect feature contribution bar chart (always shows all traits)
   - User specifies or accepts default ("top_variance")

3. n_top_features
   - Explain: "How many traits to select for UMAP coloring and metadata storage"
   - Recommended: 5
   - User specifies or accepts default

4. pca_biplot_top_features (for viz config only)
   - Explain: "How many trait arrows to show per PC in biplots (INDEPENDENT of n_top_features)"
   - For "extreme" strategy: 1 = 2 arrows/PC (one positive, one negative)
   - Recommended: 1 for high-dimensional datasets (>100 traits) to avoid crowding
   - User specifies or accepts default (1)
```

**Step 3.9 — UMAP** (ask one-by-one):
```
1. Enable UMAP?
   - Explain: "Create UMAP dimensionality reduction plots"
   - Recommended: yes when n ≥ 15
   - User confirms or declines

2. If enabled, n_neighbors
   - Explain: "UMAP neighborhood size (larger = more global structure)"
   - Use recommend_umap_n_neighbors() to suggest value
   - Surface warning if n_samples too small
   - User specifies or accepts recommendation

3. min_dist
   - Explain: "How tightly UMAP packs points (0.0 = tight, 1.0 = spread out)"
   - Recommended: 0.1
   - User specifies or accepts default

4. random_state
   - Explain: "Random seed for reproducibility"
   - Recommended: 42
   - User specifies or accepts default
```

### 2. Update Step Numbers

Current steps after the broken "3.6" should be renumbered:
- Current Step 4 (Critical Parameter Review) → Step 4 (unchanged, now includes all params)
- Current Step 5 (Backup Check) → Step 5 (unchanged)
- Current Step 6 (Copy and Customize Templates) → Step 6 (unchanged)
- Current Step 7 (Validate Configs) → Step 7 (unchanged)
- Current Step 8 (User Validation Gate) → Step 8 (unchanged)
- Current Step 9 (Git Commit) → Step 9 (unchanged)
- Current Step 10 (Handoff) → Step 10 (unchanged)

### 3. Workflow Requirements

**Interactive one-at-a-time prompting**:
- Present ONE question at a time
- Show recommended value and brief rationale
- Wait for user input before proceeding to next parameter
- Do NOT batch questions or skip any parameter

**Context-aware recommendations**:
- Use inspection results from Step 1 to inform recommendations
- Surface statistical guardrail warnings at relevant points
- Adjust recommendations based on dataset size and structure

**Clear explanations**:
- Explain what each parameter controls
- Show the effect of different values
- Clarify when parameters are independent (e.g., pca_biplot_top_features vs n_top_features)

## Impact

**Affected specs:**
- `developer-tooling` — MODIFIED: "Interactive Analysis Configuration Command" (restore comprehensive Q&A)

**Affected code:**
- `.claude/commands/configure-run-all.md` — restore Steps 2.5-2.9 with full interactive workflow

**Breaking changes:** None. This restores previously removed functionality.

**Migration:** No user action required. Users will get a better interactive experience.

## Validation

Test that the restored workflow:
1. Asks about ALL parameters that were in the original implementation
2. Prompts one-at-a-time (not batched)
3. Provides clear guidance and recommendations
4. Surfaces statistical warnings at appropriate points
5. Doesn't skip any scientifically critical parameters
