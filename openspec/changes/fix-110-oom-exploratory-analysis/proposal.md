## Why

`ExploratoryAnalysisStep.execute()` (`src/sleap_roots_analyze/pipeline/steps/exploratory_analysis.py`)
generates all step-4 figures — summary plots, EDA plots, the full correlation heatmap, and every
batched histogram/boxplot — into a single `all_figures` dict, and only saves+closes them in one
final loop at the end. Peak memory is the **sum** of every figure held at once, not the size of the
single largest one.

GitHub Issue #110's own repro (478 samples × 904 traits × 94 genotypes, TTC-SALK soybean) hit
`bad allocation` after ~455s, with an estimated ~6.5 GB peak dominated by 57 batched boxplot figures
(94 genotypes × 0.3"/genotype × 4 rows = 112.8" tall each, ~72 MB apiece).

This is not hypothetical: `configs/active/qc/mo_soybean_2021_grouped.yaml` (MO Soybean 2021
Diversity Screen) hit the identical `"not enough free memory for image buffer"` failure at step
`04_exploratory_analysis` after ~1858s, with the system down to ~210 MB free RAM out of 32 GB. That
dataset — 2,675 samples × up to ~1,061 columns × **489 unique genotypes** — is over 5× the
genotype count of #110's own repro case, confirmed by direct inspection of the run's
`pipeline_summary.json` (`status: "failed"`) and the underlying data. It is a real, currently-blocked
scientist-facing experiment, not a synthetic stress test.

A second, related defect compounds this: the horizontal-orientation boxplot sizing (used in both
`create_trait_boxplots_by_genotype()` and its batched wrapper
`create_trait_boxplots_by_genotype_batched()` in `src/sleap_roots_analyze/visualization.py`) has no
cap on subplot height, unlike the vertical-orientation branch, which already caps adaptive subplot
width at 20 inches (`min(20.0, max(subplot_size[0], n_genotypes * 0.5))`). The horizontal branch —
which is what actually runs once `n_genotypes > horizontal_threshold` (default 8, so true for
essentially any real multi-genotype screen) — computes
`subplot_height = max(subplot_size[1], n_genotypes * height_per_genotype)` with no upper bound. At
489 genotypes × 0.3"/genotype, that is ~147" per subplot, large enough to fail to allocate **on its
own**, independent of the accumulation bug above. Issue #110 itself flags this under its P1
suggestions ("cap subplot height ... when genotypes > 50"). The cap must be applied inside
`create_trait_boxplots_by_genotype()` itself (not only in the batched wrapper's local size
calculation) — see `design.md` Decision 2 for why a cap applied only in the batched wrapper would be
silently discarded before the figure is rendered.

A third site with the identical accumulate-then-close pattern as the P0 defect was found during
review: `GenerateStaticFiguresStep` (`src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py`)
calls the same batched creator functions and holds the full batch list in memory before closing
anything, one batch at a time, in a loop. Per explicit scoping decision, this is fixed in the same
change rather than filed separately, since the fix reuses the exact generator functions introduced
for `ExploratoryAnalysisStep` — see `design.md` Decision 1.

GitHub Issue: #110

## What Changes

- **Incremental save+close (P0, from #110 directly)**: both `ExploratoryAnalysisStep.execute()` and
  `GenerateStaticFiguresStep` save and close each figure immediately after it is produced, instead
  of accumulating every figure (either explicitly, via `all_figures`, or implicitly, via a fully-
  materialized batch list) before saving any of them. Peak concurrently-open figures of a given kind
  drops from "every figure/batch generated" to a small constant. Both steps consume new private
  generator functions (`_generate_trait_histogram_batches`, `_generate_trait_boxplot_batches`) that
  yield one batch figure at a time; the existing public list-returning functions become thin
  wrappers over these generators, so external callers of the public API are unaffected.
- **Cap horizontal boxplot subplot height (P1, from real-world reproduction)**: add a
  `max_subplot_height` cap (default 20.0", mirroring the vertical branch's existing width cap) to
  `create_trait_boxplots_by_genotype()`'s horizontal-orientation branch — the actual point where
  figsize is finalized before rendering — and thread the same parameter through
  `create_trait_boxplots_by_genotype_batched()` so both entry points are protected consistently. See
  `design.md` Decision 2 for why the cap must live there rather than only in the batched wrapper.
- **Genotype pagination for readability (added per explicit user decision)**: the height cap alone
  keeps high-genotype-count figures memory-safe but not readable — 489 genotype labels squeezed
  into a fixed 20" axis is ~0.041"/genotype, illegible regardless of cap value. Per explicit user
  decision, `create_trait_boxplots_by_genotype_batched()` (and its generator) gain a
  `max_genotypes_per_page` parameter (auto-derived from the cap when not set: ~66 genotypes/page
  horizontal, ~40 vertical). When genotype count exceeds that page capacity, genotypes are split
  into consecutive alphabetically-sorted pages and rendered as separate figures per (trait batch,
  genotype page) — every genotype still appears, at the same readable per-genotype spacing used
  today below the cap, just across more output files. See `design.md` Decision 3.
- **Regression test fixture**: a synthetic DataFrame with a pinned shape (480 genotypes × 300 trait
  columns) is added to assert (a) the rendered boxplot figure's subplot height stays bounded at high
  genotype counts, for both the direct and batched entry points, (b) the cap boundary and a custom
  override are respected, (c) the new generators are lazy and produce output equivalent to the
  current list-returning functions, (d) the peak number of concurrently-open matplotlib figures
  during `execute()` and `GenerateStaticFiguresStep`'s figure generation never scales with the total
  number of figures generated, and (e) genotype pagination covers every genotype exactly once across
  pages and preserves the pre-cap readable spacing within each page.

## Out of Scope (raise separately if warranted)

- `create_correlation_heatmap`'s existing 60" figsize cap and #110's P2 suggestion (300→150 DPI for
  step-4-only diagnostic plots) are not touched by this change. They are independent memory
  reductions that can be evaluated on their own merits in a follow-up change; bundling them here
  would widen this change's blast radius beyond the two root causes above.
- Issue #202 (`feature_selection_strategy` silently ignored in two visualization functions) is a
  different bug in the same file and is explicitly not part of this change.
- `create_exploratory_summary_plots()`'s single combined-genotype boxplot (step 2 of
  `ExploratoryAnalysisStep.execute()`, `"trait_ranges_by_genotype"`) is bounded by a separate,
  pre-existing, already-safe code path (`AdaptiveSizingConfig.max_height`, default 16") when
  adaptive sizing is enabled, and is not paginated by this change — see `design.md` Decision 3's
  final note.
- Smarter-than-alphabetical genotype pagination (e.g. clustering similar genotypes onto the same
  page, or a sampling strategy that shows a representative subset instead of every genotype) is
  out of scope — this change does the simplest thing that preserves completeness (every genotype
  appears somewhere) and readability (fixed per-genotype spacing per page); a smarter strategy can
  be a follow-up if the basic pagination proves insufficient in practice.

## Impact

- Affected specs: `visualization-pipeline` (figure generation / memory-safety requirements,
  boxplot genotype label readability)
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/exploratory_analysis.py` (incremental save+close)
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py` (incremental save+close,
    bundled per explicit scoping decision)
  - `src/sleap_roots_analyze/visualization.py` (boxplot batch generation, height cap, new private
    generator functions)
  - `tests/test_step_exploratory_analysis.py`, `tests/test_step_generate_static_figures.py`,
    `tests/test_visualization.py` (new regression tests)
  - `docs/CHANGELOG.md` (new `[Unreleased]` entry under `### Fixed`)
- No breaking changes to the public API: `create_trait_histograms_batched()` and
  `create_trait_boxplots_by_genotype_batched()` keep their existing signatures and list-returning
  behavior for any external callers; `max_subplot_height` and `max_genotypes_per_page` are new
  optional parameters (the former also added to `create_trait_boxplots_by_genotype()`) with
  defaults that only change behavior for `n_genotypes` high enough to have hit the uncapped case
  (i.e., callers who were already at risk of the OOM this fixes).
- Visual output changes for high genotype counts: previously-uncapped horizontal subplot height is
  now capped, and datasets above the per-page genotype capacity (~66 horizontal / ~40 vertical by
  default) now produce more boxplot figures than before (one per genotype page, not one total) —
  both are intentional, documented changes, not regressions. `create_trait_boxplots_by_genotype_batched()`'s
  return value (a flat `List[Figure]`) can therefore be longer than before for high-genotype-count
  inputs; this is a behavior change worth calling out to any external caller, though not a signature
  break.
