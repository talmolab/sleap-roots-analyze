# Proposal: Fix Numeric Genotype IDs Rendering as Continuous Colormap

## Why

PCA biplots (and other scatter functions that accept `color_by`) branch on
`dtype == "object"` to decide categorical vs continuous coloring. When the genotype column
contains numeric IDs (e.g., int64 USDA accession numbers like `12305183`), points are
rendered with the viridis continuous colormap and a numeric colorbar, making genotypes
visually indistinguishable from one another.

`color_by` is exclusively a grouping/label parameter — every call site passes `"Genotype"`
or an equivalent categorical column. Continuous trait coloring (e.g., UMAP trait overlays)
uses separate code paths (`c=trait_values`) that are unrelated to `color_by`. The cast to
string is therefore unconditionally safe.

## What Changes

- In `create_pca_biplot` (visualization.py ~line 2073): cast `df_pca[color_by]` to string
  before the dtype check so all `color_by` columns route through the categorical (tab10)
  branch regardless of original dtype.
- The string cast preserves display (`12305183` → `"12305183"`) and produces a discrete
  legend instead of a continuous colorbar.

**Scope limitations (defer to future):**
- No `color_mode: "continuous" | "discrete"` config override — always discrete for
  `color_by` is correct given current call sites.

## Impact

**Affected specs:**
- `visualization-pipeline` - MODIFIED: Genotype Highlighting Configuration

**Affected code:**
- `src/sleap_roots_analyze/visualization.py` — `create_pca_biplot` (~line 2073)

**Breaking changes:** None — string-typed genotype columns already work correctly; numeric
columns currently produce a broken continuous plot. UMAP and other continuous coloring
paths are unaffected (they do not use `color_by`).

**Migration:** No config changes needed.
