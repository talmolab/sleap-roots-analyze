# Proposal: Public Outlier-Plotting Entry Point — `plot_outlier_analysis`

## Why

`remove_outlier_samples` (#165, landed on `main` via #172) gives downstream consumers a tested,
IO-free way to **trim** outlier samples — but it **explicitly deferred plots** ("outlier
visualization dashboards … unless cheap to include" was an out-of-scope item). It is now cheap:
the figures already exist and are already public. What is missing is the **orchestration** — the
mapping from a detection `method` to *which* figures to draw, fed the right detector inputs.

The nine `create_*_outlier` figure functions are public in `__all__`
(`src/sleap_roots_analyze/__init__.py:305-313`), but the composition that decides *which* figures
belong to *which* method lives **only inside `VisualizeOutliersStep`**
(`src/sleap_roots_analyze/pipeline/steps/visualize_outliers.py`), coupled to a pipeline `run_dir`
+ `savefig` and to upstream pipeline metadata (`prev_result.metadata["outlier_results"]`,
`config.visualization.*`). `create_comprehensive_outlier_comparison` is just one cross-method
comparison figure, not an umbrella that draws a method's full figure set.

So an external consumer that wants "give me the figures for this method" today must either fake a
`StepResult` + pipeline `config` + `run_dir` and then scrape PNGs off disk by filename, or
re-stitch the per-method `create_*` calls by hand — **untested, duplicated** outside `analyze`,
the same re-stitching trap #164/#165 removed for cleanup and trimming.

This adds **one public, tested, IO-free entry point** that *imports and composes those
already-public figure functions* — introducing **no new plotting** — mirroring the #164/#165
pattern. It is the plotting sibling of `remove_outlier_samples`, and the immediate consumer is the
bloom-mcp `remove_outliers` tool's optional `include_plots`.

Tracked by [talmolab/sleap-roots-analyze#173](https://github.com/talmolab/sleap-roots-analyze/issues/173)
(follow-up to #165).

> **Two decisions here reinterpret the original #173 and were RATIFIED by Elizabeth in the updated
> #173 (2026-07-01):** (1) **B1** — `create_pca_outlier_plot` and the pca/kmeans/gmm/hierarchical
> plots stay pipeline-only; the public entry point covers only `mahalanobis` + `isolation_forest`
> (the original mapping was not buildable against the real code — see design.md "Decision B1");
> (2) **D3** — the pipeline step delegates to a *shared selection helper* (`select_outlier_figures`)
> with its own pre-computed results rather than calling `plot_outlier_analysis` (see design.md
> "Decision D3"). The updated #173 states both explicitly.

## What Changes

### Add the entry point that composes the existing figure functions

Add public **`plot_outlier_analysis`** in `src/sleap_roots_analyze/outlier_visualization.py`
(alongside the `create_*` functions it composes) that:

1. **Re-detects deterministically, under the same preconditions as `remove_outlier_samples`.**
   Takes the clean (NaN-free) trait frame and the same `method` / `random_state` / per-method
   `**detect_kwargs`, and re-runs the **same** detector (`detect_outliers_mahalanobis` /
   `detect_outliers_isolation_forest`). To make "the figures match the trimmed table" a sound
   guarantee — and to keep the per-sample index alignment the figures rely on — it enforces the
   **same two preconditions `remove_outlier_samples` enforces**: NaN-free trait columns (via
   `validate_clean_traits`, error pointing to `clean_traits_for_analysis`) and a **unique index**.
   Without these, the detectors' PCA silently `dropna()`s and reports indices against the
   post-`dropna` frame, and `create_outliers_per_genotype_plot`'s `df.loc[idx, genotype_col]`
   mis-indexes on a duplicate label. The signature is plain-data-in — no `run_dir`, `config`, or
   pipeline context.
2. **Selects the method-appropriate figure set** (small — not all nine `create_*`):
   - `method="mahalanobis"` → `create_mahalanobis_outlier_plots` (its
     `mahalanobis_outlier_detection` / `mahalanobis_pc_analysis` / `mahalanobis_threshold_analysis`
     figures), plus — when a genotype column is present — `create_outliers_per_genotype_plot`.
   - `method="isolation_forest"` → `create_isolation_forest_plots` (`isolation_forest_analysis`),
     plus — when a genotype column is present — `create_outliers_per_genotype_plot`.

   **It does NOT draw `create_pca_outlier_plot`.** #173's suggested mapping listed it under
   `mahalanobis`, but that function consumes a `detect_outliers_pca` **reconstruction** result
   (keys `reconstruction_errors`, `cumulative_variance`, …) that a Mahalanobis result does not
   contain — feeding it a Mahalanobis result raises `KeyError` / yields a blank figure, and
   `remove_outlier_samples` never runs `detect_outliers_pca`, so no matching PCA outlier set exists
   under the determinism contract. The Mahalanobis PCA **projection** view is already drawn by
   `create_mahalanobis_outlier_plots` (`mahalanobis_pc_analysis`). *(B1 — ratified in updated #173.)*
3. **Filters by `which`.** An optional `which` selector (a single figure-key string **or** a list
   of keys; `None` = the method's full available set) narrows the returned figures, mapping 1:1 to
   the bloom-mcp tool's `plots` parameter. An unrecognized key — or a key guarded out for this
   frame (e.g. `outliers_per_genotype` with no genotype column) — is rejected with an actionable
   `ValueError` naming the keys **actually available** for this method and frame (fail-fast,
   mirroring `remove_outlier_samples`'s unknown-`method` / unknown-`detect_kwargs` guards).
4. **Returns a `{stable_name: plt.Figure}` dict** and performs **no IO** — it never writes files,
   picks a format/DPI, or closes the figures. The **caller** does all IO: the pipeline step
   `savefig`s with its `config`; bloom-mcp persists via its `ResultStore`. The dict keys are stable
   identifiers suitable as artifact names / filename stems (the pipeline maps them onto its
   existing *prefixed* filenames — `outliers_mahal_*`, `outliers_if_*` — so they are stable but not
   byte-identical to the pipeline stems).
5. **Surfaces a detector failure up front.** It raises a `ValueError` on a detector `error` key /
   missing `outlier_indices` **before delegating**, because the composed `create_*` figure
   functions silently return `{}` on such input — so the raise must happen on the detector result,
   not be inferred from an empty figure dict.
6. **Rejects malformed input** (empty frame, duplicate column names, explicit `trait_cols` missing
   or non-numeric) with actionable `ValueError`s up front — mirroring `remove_outlier_samples`'s
   "Input Misuse Diagnostics".

### Refactor `VisualizeOutliersStep` to the single source of truth (B3 → `visualization-pipeline`)

Extract the per-method **figure-selection** into a shared **public** helper (PR #175 review)
`select_outlier_figures(df, results, method, which=None, genotype_col=None)` (in `__all__`) that
dispatches the pre-computed `results[method]` to that method's `create_*` figure function(s) and,
when `genotype_col` is present in `df`, adds the per-genotype figure over the full `results` dict.
Public so a consumer holding a detector result can select figures without a redundant re-detection.
Both the new entry point and the pipeline step call it:

- `plot_outlier_analysis` re-detects, then calls the helper with `results={method: <result>}` and
  `genotype_col="geno"` (so the per-genotype figure, when drawn, is single-method).
- `VisualizeOutliersStep`'s `mahalanobis` / `isolation_forest` blocks call the helper with their
  **already-computed** `outlier_results` and `genotype_col=None` (no re-detection, no per-genotype
  from the helper — the step keeps its own cross-method per-genotype block), `savefig`ing under the
  existing filenames.

The step **retains unchanged**: its `pca` block, its `kmeans`/`gmm`/`hierarchical` blocks, its
`len(methods_run) > 1` comparison figures, and its **cross-method** per-genotype block (which passes
the full multi-method `outlier_results`, one bar-group per method — structurally different from the
entry point's single-method per-genotype figure, so that figure is **not** part of the shared
helper). Its rendered output — figures, filenames, count — is **byte-for-byte unchanged**. Because
the requirement constrains the existing `visualization-pipeline` capability's step (co-located with
its `Outlier Method Comparison Summary` requirement), the step-refactor requirement lives in a
`visualization-pipeline` spec delta, not in the new `outlier-visualization` capability.

Note: `plot_outlier_analysis` is therefore exercised only by its own unit tests, not by the
pipeline path (the step shares only the helper). This is the accepted cost of preserving the
pipeline's configured detector params. *(D3 — ratified in updated #173.)*

### PR #175 review follow-ups (single-detection reuse, correctness, lifecycle)

Landed in response to Elizabeth's PR #175 review, before the a4 public API locks (see design.md D6):

- **Public `select_outlier_figures`** (in `__all__`) + **`remove_outlier_samples(return_detector_result=True)`**
  (additive; default keeps the compact 2-tuple) so the #378 `remove_outliers` consumer detects **once**
  and feeds the raw result to both the trim and the plots — no redundant second detection.
- **Metadata-column params** on `plot_outlier_analysis` (`barcode_col`/`genotype_col`/`replicate_col`)
  matching `remove_outlier_samples`, so the plotted trait/outlier set cannot silently diverge from the
  trimmed one (a correctness fix; also removes the hardcoded `"geno"`).
- **Figure lifecycle**: a `which`-narrowed call closes the figures it excluded (and skips the
  per-genotype render when not requested) — no figure leak in a long-running server.
  `random_state` is `Optional[int]` (matches docs/tests).

### Public API + docs

7. **Export `plot_outlier_analysis`** from `__init__.py` / `__all__`, with a Google-style docstring
   (Args/Returns/Raises) and `typing.get_type_hints()`-resolvable hints (the #116 acceptance bar
   enforced by the `test_public_api_docs` audit). If `**detect_kwargs` is annotated (`: Any`),
   `Any` MUST be imported, since `get_type_hints()` evaluates stringized annotations under
   `from __future__ import annotations` (the #117 `NameError: 'Any'` class of failure). The nine
   composed `create_*` functions are already in `__all__`, so no new figure function is exported.
8. **Backfill `docs/API.md`.** `docs/API.md` currently has **no `outlier_visualization` section**
   — none of the composed `create_*` figure functions are documented. Mirroring #165's backfill of
   its cross-referenced primitives, add `plot_outlier_analysis` **and** the figure functions it
   composes (`create_mahalanobis_outlier_plots`, `create_isolation_forest_plots`,
   `create_outliers_per_genotype_plot`) so the new entry's references resolve. Add a
   `docs/CHANGELOG.md` `[Unreleased] → Added` entry with the `(#173)` suffix. Update the stale
   `__all__` count in `docs/public_api_audit_2026.md` (currently "112", now 114 with #165 + this).
9. **Register for the reproducibility gate.** Because `plot_outlier_analysis` exposes
   `random_state`, the package-wide sweep (`tests/test_reproducibility.py`) auto-discovers it and
   the coverage guard goes **red the moment the function lands** — so registration MUST land in the
   **same commit** as the implementation. Register it in `tests/reproducibility_cases.py` either as
   a `CASES` entry whose `run` returns the **re-detected `outlier_indices`** (the `Case` comparator
   cannot compare `Figure`s) and whose `compare` is exact on those indices, **or** as an `EXCLUDED`
   entry with the documented reason that determinism is fully delegated to the already-swept
   `detect_outliers_*` (the function adds no new stochastic step). Update the pinned
   `EXPECTED_QUALNAMES` / case-count anchors in `tests/test_reproducibility.py` in lockstep.

### What "single source of truth" means here

The entry point and the pipeline step call the **same per-method figure-selection helper** for the
two detect-methods, so the "which `create_*` figures for this method" mapping cannot drift. It does
**not** promise byte-identical rendered pixels, nor does it share the cross-method comparison /
per-genotype figures (those stay in the step). The shared truth is the per-method **selection**.

### Composition

`clean_traits_for_analysis` → (optional) `remove_outlier_samples` (report) → (optional)
**`plot_outlier_analysis`** (figures) → caller persists. The bloom-mcp `remove_outliers` tool's
optional `include_plots` delegates to this function so removal and plots come from **one pinned
`analyze` version**.

### Out of scope (explicitly)

- **Plotting logic inside `remove_outlier_samples`** — it stays plot-free (the only change is the
  additive, opt-in `return_detector_result` flag; the default return is unchanged).
- **New plot types / restyling** the existing `create_*` figures. This composes them as-is.
- **`create_pca_outlier_plot`** (a `detect_outliers_pca`-fed figure) — excluded per B1.
- **File / format / saving** — returns `Figure` objects; the caller writes files.
- **Clustering-method figures** (`kmeans`/`gmm`/`hierarchical`) and **multi-method comparison**
  figures — the step keeps drawing these.
- **Interactive / HTML dashboards.**
- **Provenance payload** (returning the re-detected `outlier_indices` / effective params alongside
  the figures for auditability) — a reasonable follow-up (would also make the reproducibility case
  genuine), but out of scope here to keep the surface minimal.
- The downstream **bloom-mcp** `remove_outliers` tool — consumes this function; not built here.

## Impact

- Affected specs:
  - **outlier-visualization** — new capability (the `plot_outlier_analysis` entry point).
  - **visualization-pipeline** — ADDED requirement for the shared per-method selection helper the
    step delegates to (behavior unchanged).
  - **reproducibility-gates / stochastic-determinism** — existing gates; the new `random_state`
    function must be registered (no new requirement, satisfied by registration).
  - **public-api-introspection** — existing gate; new `__all__` entry + docstring audit.
- Affected code:
  - `src/sleap_roots_analyze/outlier_visualization.py` — new public `plot_outlier_analysis` + public
    `select_outlier_figures` helper (both fully type-annotated).
  - `src/sleap_roots_analyze/outlier_removal.py` — additive `return_detector_result` flag on
    `remove_outlier_samples` (default 2-tuple unchanged; PR #175 review).
  - `src/sleap_roots_analyze/pipeline/steps/visualize_outliers.py` — the two shared methods' blocks
    delegate to the helper (behavior unchanged; pca/clustering/comparison/cross-method-genotype
    retained).
  - `src/sleap_roots_analyze/__init__.py` — import + `__all__` entries for `plot_outlier_analysis`
    and `select_outlier_figures`.
  - `docs/API.md` — `plot_outlier_analysis`, `select_outlier_figures`, `return_detector_result`, +
    backfill of the composed `create_*` figure functions; `docs/CHANGELOG.md` — `[Unreleased]` entry.
  - `tests/test_plot_outlier_analysis.py` (new) — TDD coverage (incl. figure-leak + metadata-col +
    single-detection-reuse follow-ups).
  - `tests/test_step_visualize_outliers.py` — pinned filename/count baseline for the two methods.
  - `tests/test_remove_outlier_samples.py` — `return_detector_result` 3-tuple coverage.
  - `tests/reproducibility_cases.py` — register the new `random_state` function (`EXCLUDED`).
- **No behavior change** to `remove_outlier_samples`'s default return, the detectors, or
  `VisualizeOutliersStep`'s output. The step refactor is transparent; the rest is additive.
- **Depends on #165** (`remove_outlier_samples`, the detectors, the `create_*` exports) — on
  `main` via #172, **not yet on PyPI**.
- **Release coupling:** rides the same next `analyze` pre-release as #165 (`0.1.0a4`); `[Unreleased]`
  until then.
