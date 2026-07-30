## Context

Two independent memory defects compound in `ExploratoryAnalysisStep.execute()`:

1. Every figure the step generates — summary plots, EDA plots, the full correlation heatmap, and
   every batched histogram/boxplot — is accumulated into one `all_figures` dict before any of them
   is saved or closed. Peak memory is the sum of every figure held at once.
2. `create_trait_boxplots_by_genotype_batched()`'s horizontal-orientation branch has no cap on
   subplot height, so a single figure can itself be large enough to fail to allocate, independent
   of (1).

Review of this proposal surfaced a third, related site: `GenerateStaticFiguresStep`
(`pipeline/steps/generate_static_figures.py:607-664`) calls the same
`create_trait_histograms_batched()` / `create_trait_boxplots_by_genotype_batched()` and has the
identical accumulate-then-close pattern (the full batch list is built and held before the first
`plt.close()` runs). Per explicit user decision, this is bundled into the same change rather than
filed as a separate follow-up, since it is fixed by consuming the same new generator functions
introduced for (1) — see Decision 1.

A prior, non-TDD session in this same working tree drafted a fix for (1) and (2) (now stashed, not
applied, on this branch — `git stash show -p` on `stash@{0}` if it needs to be inspected). Its
shape is used as a starting point for the design questions below, but every claim here is
independently re-derived, and the implementation itself will be written test-first via `/tdd`, not
copied from the stash. In particular, the stash's approach to (2) — adding a cap parameter only to
`create_trait_boxplots_by_genotype_batched()` — is not sufficient; see Decision 2's correctness
finding below.

## Decision 1: How to make the batched-figure loops save+close incrementally

**Problem**: `create_trait_histograms_batched()` and `create_trait_boxplots_by_genotype_batched()`
build and return a full `List[Figure]` internally — by the time a caller gets the list back, all
batches for that call already coexist in memory. Simply moving the caller's save+close loop
earlier (right after the `create_*_batched()` call, instead of at the very end) does **not** fix
this — the expensive accumulation already happened inside the creator function before it returned.
This is true for both current callers: `ExploratoryAnalysisStep.execute()` (accumulates into
`all_figures` before any save) and `GenerateStaticFiguresStep` (saves+closes one at a time in a
loop, but only after the full list already exists in memory).

**Decision**: Split each batched creator into:
- A private generator (`_generate_trait_histogram_batches`, `_generate_trait_boxplot_batches`) that
  `yield`s one figure at a time as it's built.
- The existing public function becomes a thin `list(...)` wrapper over the generator, so any
  external caller relying on the current list-returning signature is unaffected.

Both `ExploratoryAnalysisStep.execute()` and `GenerateStaticFiguresStep` are updated to iterate the
generators directly and save+close each figure as it's yielded, so at most one batch figure of a
given kind is ever open at once (down from up to 57 in the #110 repro). `GenerateStaticFiguresStep`
keeps its existing periodic `gc.collect()` calls (harmless, now redundant safety margin rather than
the only defense).

**Alternative considered**: keep the list-returning functions and just have each caller close
figures "as soon as possible" after the call returns. Rejected — by construction, the whole list
already exists in memory by the time the function returns, so this doesn't change peak memory at
all for the batched-figure case, which is the dominant contributor in #110's own memory table (57
boxplot batches ≈ 4.1 GB of the ~6.5 GB total).

For the smaller, non-batched figures in `ExploratoryAnalysisStep` (summary plots, EDA plots,
correlation heatmap — a handful of figures, not dozens), `execute()` saves and closes each one
immediately after creation rather than merging into `all_figures` first.

## Decision 2: Where the horizontal boxplot subplot height cap must actually live

**Problem**: horizontal orientation's `subplot_height = max(subplot_size[1], n_genotypes *
height_per_genotype)` has no upper bound in either place it's computed. The vertical branch already
caps its analogous growing dimension (subplot width) at 20 inches.

**Correctness finding (verified directly against the code, not assumed)**: it is not enough to cap
`batch_figsize` inside `create_trait_boxplots_by_genotype_batched()` (`visualization.py:398-412`)
alone. That function passes `figsize=batch_figsize` into `create_trait_boxplots_by_genotype()`
(`visualization.py:416-424`) — but that inner function's own horizontal-orientation branch
(`visualization.py:188-194`, reached because the batched wrapper never passes `adaptive_config`)
**unconditionally recomputes and overwrites** `figsize` with the same uncapped formula:

```python
elif actual_orientation == "horizontal":
    n_rows = (n_traits + n_cols - 1) // n_cols
    height_per_genotype = 0.3
    min_subplot_height = max(4, n_genotypes * height_per_genotype)
    figsize = (figsize[0], min_subplot_height * n_rows)   # discards the caller's figsize height
```

Contrast with the vertical branch (`visualization.py:195-203`), which only *conditionally* widens
(`if adaptive_subplot_width > current_subplot_width`) — a no-op once the caller already applied the
same cap, which is why the vertical cap actually works end-to-end today. A cap added only to the
batched wrapper's local variable would be silently discarded before the figure is actually rendered
— the fix would look correct in isolation (the batched wrapper's own `batch_figsize` variable is
capped) but have zero effect on the real output.

This inner function is also called directly by `create_exploratory_summary_plots()`
(`visualization.py:593-597`, used in `ExploratoryAnalysisStep.execute()`'s step 2, unconditionally
on every run) — but that call site passes `adaptive_config`, so when adaptive sizing is enabled
(true for the real MO Soybean config that hit this failure) it takes the `adaptive_config is not
None` branch instead, which is already bounded by `adaptive_config.max_height`. It is only unbounded
when adaptive sizing is disabled. The batched-boxplot path is unbounded unconditionally, since
`create_trait_boxplots_by_genotype_batched()` has no `adaptive_config` parameter at all and never
passes one through.

**Decision**: add `max_subplot_height: float = 20.0` to `create_trait_boxplots_by_genotype()`
itself, and apply it in that function's horizontal-orientation branch:

```python
elif actual_orientation == "horizontal":
    n_rows = (n_traits + n_cols - 1) // n_cols
    height_per_genotype = 0.3
    min_subplot_height = min(max_subplot_height, max(4, n_genotypes * height_per_genotype))
    figsize = (figsize[0], min_subplot_height * n_rows)
```

`create_trait_boxplots_by_genotype_batched()` (and its new generator counterpart) also gains the
same `max_subplot_height: float = 20.0` parameter, uses it consistently in its own `batch_figsize`
precomputation for the `figsize is not None` scaling path, and passes it through to the inner
`create_trait_boxplots_by_genotype()` call — so the cap holds regardless of which code path a
caller exercises, and a direct caller of `create_trait_boxplots_by_genotype()` (not just the
batched wrapper) is protected too.

**Cap value: 20.0 inches**, matching the vertical branch's existing width cap, not a different
value:
- It is the direct mirror of the existing, already-reviewed vertical cap (same role: bound the
  dimension that scales with genotype count), keeping the two branches consistent instead of
  introducing a second unexplained magic number.
- At `n_rows` (typically ≤ 4 for a 16-trait batch in a 4-column grid) subplots stacked vertically,
  a 20" per-subplot cap bounds total figure height at ≤ 80", matching the vertical branch's ≤ 80"
  total-width bound under the same batch geometry. `GenerateStaticFiguresStep`'s adaptive batch
  sizing can push `n_rows` up to 9 (batch_size up to 36), giving ≤ 180" — at 300 DPI that is
  54,000 px, still comfortably under the Agg backend's ~65,536 px per-dimension ceiling, though it
  uses most of that margin; this is a pre-existing batch-size choice, not something this change
  alters.
- The discarded draft used `max_subplot_height=40.0` (2× the vertical cap) without a stated
  rationale. Doubling the cap does not meaningfully improve readability at the genotype counts
  where the cap actually binds (at 489 genotypes, 20" ÷ 489 ≈ 0.041"/genotype vs. 40" ÷ 489 ≈
  0.082"/genotype — both are far below any legible tick-label spacing), so it only buys back
  memory risk with no real readability benefit. 20.0 is used instead.

**Readability at extreme genotype counts is not solved by the cap alone**: at 489 genotypes
squeezed into a fixed 20" axis, that's ~0.041"/genotype — far below any legible tick-label spacing,
regardless of cap value. Per explicit user decision during proposal review, this change adds a
basic pagination safeguard (Decision 3) rather than deferring readability entirely to a follow-up
issue.

## Decision 3: Genotype pagination for readability at high genotype counts

**Problem**: the height cap (Decision 2) stops the crash but, by itself, produces an illegible
chart once genotype count is high enough that the per-genotype spacing that already looks fine
below the cap (`0.3"/genotype` horizontal, `0.5"/genotype` vertical) no longer fits within
`max_subplot_height`. That crossover point is itself computable: `max_subplot_height /
per_genotype_size` — 20 / 0.3 ≈ 66 genotypes (horizontal), 20 / 0.5 = 40 genotypes (vertical).

**Decision**: when `n_genotypes` exceeds that per-page capacity, split genotypes into consecutive
pages of at most `max_genotypes_per_page` genotypes each (sorted the same way the horizontal
renderer already sorts them: `sorted(df[genotype_col].unique())`), and render one figure per
(trait batch, genotype page) combination instead of one figure per trait batch. Every genotype
still appears, in some output figure, at the same readable per-genotype spacing used today below
the cap — completeness is preserved (no genotype is silently dropped/sampled out), just spread
across more files. This is intentionally the simple option: alphabetical paging, not smart
sampling/clustering — "basic safeguard," not a new visualization feature.

- New parameter `max_genotypes_per_page: Optional[int] = None` on
  `create_trait_boxplots_by_genotype_batched()` / `_generate_trait_boxplot_batches()`. When `None`,
  auto-derived from the cap and the orientation actually in effect:
  `max(1, int(max_subplot_height // per_genotype_size))` where `per_genotype_size` is 0.3
  (horizontal) or 0.5 (vertical) — so the default is self-consistent with Decision 2's cap rather
  than a second independent magic number.
- Because each page's genotype count is bounded by construction (`≤ max_genotypes_per_page`), the
  per-figure height computed for that page is always `≤ max_subplot_height` — the Decision 2 cap
  never actually needs to engage in the paginated path. The cap remains a defense-in-depth backstop
  (protects a direct call to `create_trait_boxplots_by_genotype()` that bypasses pagination, or a
  bug in the pagination math), while pagination is the mechanism that actually keeps charts
  readable.
- Where it lives: `create_trait_boxplots_by_genotype()` itself stays simple — it always renders
  "every genotype present in the DataFrame it's given." Pagination is the batched wrapper's
  responsibility: it computes genotype pages, filters the DataFrame to each page's genotype values,
  and calls the unchanged single-figure renderer once per page.
- Naming/discoverability: figures stay flatly, sequentially numbered
  (`04_trait_boxplots_batch_{i+1}`), matching the existing convention where descriptive detail
  (trait range) lives in the figure's `suptitle`, not the filename. The `suptitle` gains a
  `" | Genotypes {start}-{end} of {n_genotypes}"` segment whenever more than one genotype page
  exists, so an opened figure is self-describing.
- Orientation coupling: since `orientation="auto"` already switches to horizontal above
  `horizontal_threshold` (default 8), and pagination only meaningfully engages above ~40-66
  genotypes, the common case (default auto orientation, high genotype count) always pages using the
  horizontal per-genotype size. An explicit `orientation="vertical"` override with very high genotype
  counts is handled by the same logic using the vertical per-genotype size (0.5) — a rarer path,
  still covered, not left as a gap.
- **Orientation consistency across pages**: `actual_orientation` (resolved once from the *full*
  dataset's genotype count, before pagination) must be passed explicitly to each per-page call into
  `create_trait_boxplots_by_genotype()` — not the original possibly-`"auto"` `orientation` argument.
  Otherwise a small last page (e.g. 3 leftover genotypes after several full pages of 66) could
  independently re-resolve `"auto"` to vertical for that one page based on its own small count,
  producing an orientation that's inconsistent with the rest of the same trait batch's pages. This
  is a genuine gap in the *current* code today (harmless today only because, pre-pagination, every
  batch already always sees the full dataset's genotype count, so the inner re-resolution always
  agrees with the outer one) that pagination would otherwise reintroduce; fixed by always passing
  the pre-resolved `actual_orientation` through, for every page.
- **Missing-column and NaN safety (round-2 review finding, verified against the code)**: the
  existing per-genotype rendering path only ever sorts a `.dropna()`'d subset
  (`visualization.py:217,224`), and only guards `genotype_col in df.columns` in one place
  (`visualization.py:370`, the `n_genotypes` computation) — there is no existing precedent for an
  unconditional, unguarded `df[genotype_col].unique()` call. Pagination's genotype-list
  construction must not introduce one: it SHALL (a) guard `genotype_col in df.columns` before
  accessing it at all — if absent, pagination is a no-op (single page, matching today's "0
  genotypes" behavior) rather than raising `KeyError`; (b) drop NaN genotype values before sorting
  (`df[genotype_col].dropna().unique()`) — sorting a mixed NaN/string array raises `TypeError` in
  Python, and a NaN "page" would be meaningless anyway.

**Explicitly still out of scope**: `create_exploratory_summary_plots()`'s call into
`create_trait_boxplots_by_genotype()` (used in `ExploratoryAnalysisStep.execute()`'s step 2,
`"trait_ranges_by_genotype"`) passes `adaptive_config` when adaptive sizing is enabled, which takes
a separate code path (`calculate_barplot_size` + `AdaptiveSizingConfig.max_height`, default 16.0") —
already bounded, unrelated to the horizontal-branch cap/pagination this change adds, and not
paginated. At very high genotype counts that single combined-genotype figure is still memory-safe
(bounded at 16") but not addressed for readability by this change; left as-is since it's a
pre-existing, independently-configured, already-safe code path, not part of the crash this change
fixes.

## Decision 4: Regression test shape

A synthetic fixture (no proprietary data needed) with a pinned size of **480 genotypes × 300 trait
columns** (chosen to be in the "hundreds of genotypes" real-failure range while keeping test
runtime and memory bounded, using the existing low-DPI test convention of `dpi=100`) reproduces the
shape of the real failure. Properties asserted:

1. **Bounded figure size**: for `n_genotypes` in the hundreds, the horizontal-orientation figure's
   subplot height stays at or under the cap (20") regardless of genotype count — asserted through
   the actual rendered `fig.get_size_inches()` of the figure produced by
   `create_trait_boxplots_by_genotype()` and by `create_trait_boxplots_by_genotype_batched()`, not
   just an internal local variable, since Decision 2's finding shows those can disagree.
2. **Cap boundary and override behavior**: exact-equality at the cap boundary
   (`n_genotypes * 0.3 == max_subplot_height`), and a custom `max_subplot_height` value is honored
   (not just the default).
3. **Generator/list parity**: `list(_generate_trait_boxplot_batches(...))` and
   `create_trait_boxplots_by_genotype_batched(...)` (and the histogram equivalents) produce the
   same number of figures with the same sizes, and the generator is confirmed to yield lazily (one
   figure existing at a time), not just be generator syntax around eager work.
4. **Bounded concurrency**: during `ExploratoryAnalysisStep.execute()` and
   `GenerateStaticFiguresStep`, the peak number of simultaneously-open matplotlib figures never
   scales with the total number of figures generated. Measured by tracking `plt.get_fignums()` at
   **both** figure-creation time and close time (not close time alone — sampling only at close
   would miss a figure that's created but never explicitly closed, silently undercounting a real
   leak). The exact small constant this pins to (expected to be a single-digit number, not on the
   order of dozens) is determined and locked in during the `/tdd` red/green cycle rather than
   guessed here.
5. **Empty-input safety**: zero genotypes and zero traits are handled without error for both the
   capped function and its batched/generator wrappers.
6. **Pagination correctness**: at 480 genotypes (default `max_genotypes_per_page=66` for
   horizontal), the batched boxplot output contains multiple genotype pages per trait batch; every
   one of the 480 genotypes appears in exactly one page's rendered figure (no genotype dropped or
   duplicated across pages); each individual page's figure height stays at the pre-cap "readable"
   size (`page_genotype_count * 0.3`, not the 20" cap, since pages are sized to never need it); the
   `suptitle` of a multi-page batch includes the genotype range; and a genotype count at or below
   the auto-derived page capacity produces exactly one page (no behavior change from before this
   decision). Pagination tests use a small, fixed trait count (not the full 300-trait fixture) —
   pagination is a property of the genotype axis, not the trait axis, and keeping the trait count
   small keeps these tests fast and keeps a single trait batch unambiguous (needed for tests that
   check "every genotype appears across pages of the same batch").
7. **CI runtime budget for the end-to-end test (Task 5)**: with pagination active, the 480x300
   fixture produces roughly `n_trait_batches × n_genotype_pages` figures (≈8 genotype pages ×
   ~19 trait batches at `batch_size=16` ≈ 150 small figures). Each is cheap once capped/paginated,
   but Task 5 must measure actual wall-clock time when first implemented and reduce the trait count
   if it meaningfully slows the non-integration CI suite — do not assume the estimate above without
   measuring.

   **Update after measuring in CI, not just locally**: the 480x300 fixture (and the analogous
   480x30 fixture used for `GenerateStaticFiguresStep`'s equivalent test) ran fine standalone
   locally (~170-180s each), which read as an acceptable, budgeted cost. It was not acceptable in
   practice: combined with the rest of the suite, it pushed CI's 30-minute `tests` job timeout on
   Ubuntu and Windows (confirmed via an actual CI run on the opened PR — both failed at the
   identical 30m16s mark, mid-suite, not from a real test failure). Reduced both fixtures to
   100 genotypes x 40 traits (`ExploratoryAnalysisStep`) and 100 genotypes x 12 traits
   (`GenerateStaticFiguresStep`) — still large enough to trigger genotype pagination (2 pages) and
   multiple trait batches, still large enough to clearly distinguish a bounded (~4 peak) from an
   unbounded (near-total-figure-count peak) implementation, at roughly 1/12th the runtime (~15s
   each). Lesson: for a regression test meant to run in CI's shared job budget, a local-only
   wall-clock measurement is not sufficient evidence of CI feasibility — the fixture size needs to
   be validated against an actual CI run, since CI's aggregate job budget (the whole test suite
   sharing one 30-minute window) is a stricter constraint than "is this one test fast enough on its
   own."
