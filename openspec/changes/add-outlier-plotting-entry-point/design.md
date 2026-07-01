## Context

`remove_outlier_samples` (#165/#172) is a pure, IO-free composition returning `(trimmed_df,
outlier_report)`; it discards the detector's per-sample arrays and draws no figures. Issue #173
asks for the plotting sibling: a public, IO-free function returning the method-appropriate outlier
figures so the bloom-mcp `remove_outliers` tool can offer optional plots without re-stitching
pipeline internals.

Two facts constrain the design (both verified against the real code during review):

1. The nine `create_*_outlier` figure functions exist and are public. Their return shapes differ:
   `create_mahalanobis_outlier_plots` and `create_isolation_forest_plots` return
   `Dict[str, plt.Figure]`; `create_pca_outlier_plot` and `create_outliers_per_genotype_plot`
   return a **bare `plt.Figure`**. Each reads specific detector-result keys.
2. `VisualizeOutliersStep` already encodes "which figures per method," but coupled to `run_dir`,
   `savefig`, `config`, and pre-computed `prev_result.metadata["outlier_results"]`, spanning six
   methods plus cross-method comparison and per-genotype figures.

## Goals / Non-Goals

- **Goals:** one public IO-free `plot_outlier_analysis` returning `{name: Figure}`; deterministic
  figures matching what `remove_outlier_samples` trims (under the same preconditions); a single
  source of truth for the per-method figure selection shared with the pipeline step; zero behavior
  change to `remove_outlier_samples` and to the step's rendered output.
- **Non-Goals:** new/restyled plots; file IO; clustering-method or multi-method-comparison figures
  in the entry point; changing the detectors or the report schema.

## Decisions

### D1 — Re-detect inside the entry point, under the sibling's preconditions

`plot_outlier_analysis(clean_df, trait_cols=None, *, method="mahalanobis", random_state=42,
which=None, **detect_kwargs)` (no `genotype_col` param — it uses `"geno"` internally, matching the
#173 signature) re-runs the selected detector internally, and
**enforces the same NaN-free + unique-index preconditions as `remove_outlier_samples`**
(`validate_clean_traits` + `index.is_unique`), plus the same input-misuse guards (empty frame,
duplicate columns, non-numeric explicit `trait_cols`).

- **Why re-detect:** #173 mandates a plain-data-in signature so an external consumer needs only the
  frame it has; deterministic detection reproduces the exact set `remove_outlier_samples` removed.
- **Why the preconditions are load-bearing (not defense):** the detectors run PCA that silently
  `dropna()`s and report `outlier_indices` against the post-`dropna` frame; removal and
  `create_outliers_per_genotype_plot`'s `df.loc[idx, genotype_col]` are label-based. On a
  NaN-carrying or duplicate-indexed frame the re-detected set can diverge from removal (which would
  have *rejected* that frame) and the per-genotype figure mis-indexes. So the "figures match the
  trimmed table" guarantee holds precisely on the inputs both functions share.
- **Alternatives rejected:** adding raw detector arrays to `remove_outlier_samples`'s report
  (touches the #165 public API + its compact-report invariant); accepting a pre-computed
  detector-result dict as the public input (pushes the re-stitching #173 removes back on the
  consumer — it survives only as the *internal* helper input, see D3).

### D2 — Return `Figure` objects; the caller owns all IO

Returns `Dict[str, plt.Figure]`; never writes files, never calls `plt.close`. The step `savefig`s +
closes with its `config`; bloom-mcp persists via `ResultStore`. Stable dict keys are the contract.

### D3 — Single source of truth = a shared per-method **selection** helper (RATIFIED in #173)

#173 says "refactor `VisualizeOutliersStep` to call **this function** … behavior unchanged … keeps
its configured [detector] params." Those two clauses are **mutually incompatible**: if the step
called `plot_outlier_analysis` it would **re-detect** with the entry point's params, discarding the
pipeline's already-computed `outlier_results[method]` (built from `config.outlier_detection.*`) —
risking drift and a redundant in-pipeline detector run. `CleanupTraitsStep → clean_traits_for_analysis`
is not a perfect mirror because cleanup has no upstream pre-computed detector state to preserve.

**Decision:** extract `_select_outlier_figures(df, results, method, which=None, genotype_col=None)
-> dict[str, plt.Figure]` (private) as the one place mapping a method + its pre-computed
detector-result dict to the method's `create_*` figures (and, when `genotype_col` is present in
`df`, the per-genotype figure over the full `results` dict):
- `plot_outlier_analysis` = validate + preconditions → re-detect → `_select_outlier_figures(clean_df,
  {method: result}, method, which=which, genotype_col="geno")` (per-genotype single-method).
- `VisualizeOutliersStep` calls `_select_outlier_figures(df, outlier_results, method)` (default
  `genotype_col=None` → core figures only) with its **pre-computed** results for `mahalanobis` /
  `isolation_forest`, then `savefig`s under its existing filenames. It does **not** call
  `plot_outlier_analysis`, does **not** re-detect, and keeps its own cross-method per-genotype block.

Consequence accepted: `plot_outlier_analysis` is covered only by its own unit tests, not the
pipeline path — the accepted cost of preserving configured detector params. The updated #173 states
this design (shared helper on the step's own pre-computed results).

### D4 — Method → figure-key map (corrected; B1 RATIFIED in #173)

| `method` | figure keys returned | source (return shape) |
|---|---|---|
| `mahalanobis` | `mahalanobis_outlier_detection`, `mahalanobis_pc_analysis`, `mahalanobis_threshold_analysis` | `create_mahalanobis_outlier_plots` (**dict** — its own keys) |
| `mahalanobis` (+ genotype) | `outliers_per_genotype` | `create_outliers_per_genotype_plot` (**bare Figure** — key assigned by the composer) |
| `isolation_forest` | `isolation_forest_analysis` | `create_isolation_forest_plots` (**dict** — its own key) |
| `isolation_forest` (+ genotype) | `outliers_per_genotype` | `create_outliers_per_genotype_plot` (**bare Figure** — composer-assigned key) |

The dict-returning functions contribute their own keys unprefixed; the bare-`Figure` per-genotype
function gets a composer-assigned key (`outliers_per_genotype`). `_select_outlier_figures` adds the
per-genotype figure only when `genotype_col` is present in `df` — the entry point passes
`genotype_col="geno"`, the step passes `genotype_col=None`. So for the step the helper is a pure
extraction of its per-method `create_*` calls (no per-genotype) → byte-identical step output, and
the step keeps drawing its own cross-method per-genotype figure separately.

**Decision B1 — drop `create_pca_outlier_plot` from the Mahalanobis set.** #173 listed it, but
`create_pca_outlier_plot` reads `reconstruction_errors` / `cumulative_variance` /
`explained_variance_threshold` — keys produced by `detect_outliers_pca`, **not** by
`detect_outliers_mahalanobis` (which has `mahalanobis_distances`, `cumulative_variance_explained`,
`feature_fraction_explained`, `degrees_of_freedom`). Feeding a Mahalanobis result raises `KeyError`
/ yields a blank figure, the pipeline never makes this pairing (it draws `create_pca_outlier_plot`
only under a separate `method=="pca"` branch), and `remove_outlier_samples` never runs
`detect_outliers_pca` — so there is no PCA outlier set to match the trimmed table. The Mahalanobis
PCA **projection** view is already in `create_mahalanobis_outlier_plots` (`mahalanobis_pc_analysis`).
Options were: (a) drop it [chosen]; (b) run a second `detect_outliers_pca` under the hood — rejected
(new detection scope; its outlier set ≠ the trimmed Mahalanobis set, breaking "figures match");
(c) key-adapt the Mahalanobis dict — rejected (it lacks the reconstruction data the plot needs, not
just renamed keys). The updated #173 adopts (a): those plots stay pipeline-only.

### D5 — Reproducibility-gate registration returns indices, not Figures

The sweep auto-discovers any public `random_state`-bearing function, so the coverage guard breaks
the moment `plot_outlier_analysis` lands — registration must ship in the **same commit**. The `Case`
comparator cannot compare `Figure`s. Two acceptable forms:
- **CASES with an indices adapter:** `run` re-detects and returns the `outlier_indices`; `compare`
  is exact on those (mirrors `_compare_remove_outlier_samples`). The `run` must avoid leaking
  figures (return before rendering).
- **EXCLUDED with reason:** "determinism is fully delegated to the already-swept `detect_outliers_*`;
  `plot_outlier_analysis` adds no new stochastic step (Figure composition is deterministic given the
  detected set)." Honest, since the registered CASES form would only re-exercise the detector.

**Implemented as EXCLUDED-with-reason** (truthful; the function adds no new randomness and returns
Figures the sweep can't compare). Because the entry point is `EXCLUDED` (not added to `CASES`), the
`EXPECTED_QUALNAMES` / `len(CASES)` anchors are unchanged; `test_excluded_set_is_consistent` verifies
the excluded name is a discoverable `random_state` function disjoint from `CASES`.

## Risks / Trade-offs

- **B1 overrules the original #173 figure list** → mitigated by the `mahalanobis_pc_analysis` figure
  already covering the PCA projection; ratified in the updated #173.
- **D3 leaves `plot_outlier_analysis` outside the pipeline path** → its determinism/validation/`which`
  logic has only unit-test coverage; accepted to preserve configured detector params.
- **Per-genotype figure differs step vs entry point** (multi-method grouping in the step;
  single-method in the entry point) → it is deliberately **not** shared; the step's cross-method
  block is untouched, and the entry point composes its own single-method version. Documented so MCP
  consumers don't expect the multi-method view.
- **`genotype_col` handling:** the entry point uses `"geno"` internally (matching
  `remove_outlier_samples`) and passes it to the helper; the step passes `genotype_col=None` so the
  helper draws no per-genotype figure, and the step keeps its own per-genotype block using its
  `column_mapping.get("genotype", "Genotype")` column. So the helper's per-genotype path is
  entry-point-only and the step's genotype column name is unchanged — no divergence in behavior.
- **Re-detection cost:** one extra detector run when a consumer calls both functions — cheap and
  deterministic.
- **`which` against a guarded-out key** → actionable `ValueError` naming the keys available for that
  frame+method (not a silent empty dict).

## Migration Plan

Additive: new public function + one `__all__` entry + a transparent step refactor. No consumer
migration. Ships in `0.1.0a4` alongside #165; `[Unreleased]` until then. Rollback = revert the
commit (the step's pre-refactor per-method blocks restored; no data/format change).

## Resolved (updated #173, 2026-07-01)

- **Q1 — B1 ratified:** `create_pca_outlier_plot` (and pca/kmeans/gmm/hierarchical plots) stay
  pipeline-only; the entry point covers `mahalanobis` + `isolation_forest`.
- **Q2 — D3 ratified:** the step delegates to `_select_outlier_figures` with its own pre-computed
  results; it does not call `plot_outlier_analysis`.
- **Q3 — reproducibility registry:** implemented as `EXCLUDED`-with-reason (the function returns
  Figures and adds no new stochastic step over the already-swept detectors).
