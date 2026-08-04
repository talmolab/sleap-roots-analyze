## Why

Fixes #206 and resolves the open "decide" checkbox in #203.

`pca.n_top_features` (`components.py:422`, default `10`) is a plain integer
count. It has no visible effect on any plot a user actually looks at — its
only real consumer is `PCAAnalysisStep` (`pca_analysis.py:109-115`), which
writes `top_features.csv` and a metadata list consumed only by a summary
count. Worse, for `feature_selection_strategy: "extreme"` (used by most
active viz configs), the count is fundamentally arbitrary — every config
hand-picks a different value (`1` in some, `5` in others; see
`configs/active/viz/*.yaml`).

Separately (#203), `PCAAnalysisStep` never passes `pc_indices` to
`select_top_features_from_pca()`, so it silently defaults to `[0, 1]`
(`pca.py:64-65`) regardless of how many PCs `n_components` actually
retained. For `"extreme"`, `"top_absolute"`, and `"top_contribution"` — all
of which respect `pc_indices` — a run with `n_components: 0.75` (5+ PCs)
still only considers PC1/PC2, silently discarding every other retained
component's contribution.

## What Changes

- **Fix #203**: `PCAAnalysisStep` always passes
  `pc_indices=list(range(n_components))` explicitly to
  `select_top_features_from_pca()` — never relies on the `[0, 1]` default.
- **`extreme` gets no count parameter** (#206 Part 1): when
  `feature_selection_strategy == "extreme"`, `PCAAnalysisStep` calls
  `select_top_features_from_pca(n_features_to_select=1, ...)` unconditionally
  — `config.pca.n_top_features` is not read for this method. Combined with
  the `pc_indices` fix above, this always selects exactly the single most-
  positive and single most-negative loading trait per retained PC.
  **Design decision** (see `design.md` for the full justification): the
  now-inert `n_top_features` value under `"extreme"` is handled by
  documentation, not a validation error — `PCAConfig.n_top_features`'s
  docstring states explicitly that it is ignored for `"extreme"`, mirroring
  the existing precedent in the same file
  (`select_top_features_from_pca`'s own docstring already documents that
  `pc_indices` is "ignored entirely by `top_variance`" with no runtime
  warning). The 28 active configs that currently pair
  `feature_selection_strategy: "extreme"` with an explicit `n_top_features`
  (27 files under `configs/active/viz/`, plus the flat pre-reorg duplicate
  `configs/active/viz_turface_150genotypes.yaml`; **zero** under
  `configs/active/qc/` — none of those pair the two fields) have that
  now-meaningless line removed as part of this change. As a cheap
  additional safeguard (added after adversarial review — see `design.md`
  Decision 1), `PCAAnalysisStep` also emits a `logger.info()` whenever
  `feature_selection_strategy == "extreme"`, stating that `n_top_features`
  is not read for this method — not a validation error, but not silent
  either, so a config from outside this repo (a fork, a collaborator's
  local copy) that still sets the now-inert field gets a visible runtime
  signal, not just a docstring.
- **`top_variance` gets a variance-threshold option** (#206 Part 2):
  `PCAConfig.n_top_features` changes type from `int` to `float`, reusing the
  exact overload convention already used by `PCAConfig.n_components`
  ("Number of components (or variance ratio if < 1)"): a value `< 1` means
  "stop once the selected features' cumulative `fractional_contribution`
  reaches this fraction"; a value `>= 1` preserves today's fixed-count
  behavior. A new helper, `select_n_features_by_variance()` in `pca.py`,
  resolves a `< 1` threshold to a concrete feature count, structurally
  mirroring the existing `select_n_components()` pattern
  (`np.argmax(cumulative >= threshold) + 1`) but walking
  `feature_contributions["fractional_contribution"]` instead of
  `explained_variance_ratio_`. `PCAAnalysisStep` calls this helper only when
  `feature_selection_strategy == "top_variance"` and `n_top_features < 1`,
  then calls `select_top_features_from_pca()` exactly as before with the
  resolved integer.
  - **Scope**: only `"extreme"` and `"top_variance"` appear in any active
    config today. The threshold option is implemented for `"top_variance"`
    only; `"top_absolute"` and `"top_contribution"` keep today's
    count-only behavior as an explicit, documented follow-up (not silently
    unsupported — see the new config validation below). `"vector_length"`
    is **not** part of this scope note: `pca.feature_selection_strategy`'s
    existing validation enum (`validate_qc_config()` / `validate_viz_config()`,
    `pipeline/config/utils.py`) has never accepted `"vector_length"` as a
    value for *this* field — that string is only a valid value for the
    separate `create_pca_biplot(feature_selection=...)` parameter. Widening
    the `pca.feature_selection_strategy` enum is unrelated scope creep and
    is not done here.
- **Config validation** (`validate_qc_config()` / `validate_viz_config()`):
  add two new checks alongside the existing PCA validation block:
  1. reject `n_top_features < 1` when `feature_selection_strategy` is
     `"top_absolute"` or `"top_contribution"` (methods that still require a
     plain count and would silently misbehave — e.g. `int(0.5) == 0`
     selected features — if given a fractional value). No restriction is
     added for `"extreme"` (the field is ignored regardless of value) or
     `"top_variance"` (this is exactly the new supported case).
  2. reject a non-whole-number `n_top_features >= 1` (checked with a small
     floating-point tolerance, not exact equality) for every strategy
     except `"extreme"` (i.e. for `"top_variance"`'s count branch,
     `"top_absolute"`, and `"top_contribution"`) — e.g. `n_top_features: 5.7`
     would otherwise silently truncate to `int(5.7) == 5` with no warning,
     the same class of silent-no-op this change is otherwise trying to
     eliminate.
- **Docs**: rewrite `PCAConfig.n_top_features`'s docstring, and update
  `configs/active/viz/*.yaml`/`configs/examples/viz_*.yaml` comments to
  match the new semantics — this supersedes patching the individual wrong
  "UMAP coloring" / "interesting genotypes" claims found during the original
  investigation (they are rewritten here rather than fixed in place).
- **Close #203** with a comment cross-linking to this PR once merged (same
  pattern as #64/#68).

## Impact

- Affected specs: `visualization-pipeline` (a new, standalone requirement
  for `PCAAnalysisStep`'s feature-selection resolution — see `design.md`
  Decision 4 for why this is `ADDED` rather than folded into the existing,
  already-large "Pipeline Step Parameter Passing" requirement),
  `config-management` (new cross-field validation for `n_top_features` vs.
  `feature_selection_strategy`).
- Affected code:
  - `src/sleap_roots_analyze/pca.py` (`select_n_features_by_variance`, new)
  - `src/sleap_roots_analyze/pipeline/config/components.py` (`PCAConfig`)
  - `src/sleap_roots_analyze/pipeline/config/utils.py`
    (`validate_qc_config`, `validate_viz_config`)
  - `src/sleap_roots_analyze/pipeline/steps/pca_analysis.py`
  - `configs/active/viz/*.yaml` (27 files) and
    `configs/active/viz_turface_150genotypes.yaml` (1 flat pre-reorg
    duplicate) — **not** `configs/active/qc/*.yaml`, which has no file
    pairing `"extreme"` with an explicit `n_top_features`
  - `configs/examples/viz_*.yaml`
  - `tests/test_pca.py`, `tests/test_step_pca_analysis.py`,
    `tests/test_pipeline_config.py`, `tests/test_viz_pipeline_config.py`
    (the last two are where `validate_qc_config`/`validate_viz_config`'s
    existing PCA-validation tests actually live today — there is no
    separate `test_pca_validation.py`)
  - `docs/CHANGELOG.md` `[Unreleased]` — new `### Changed`/`### Fixed`
    entries
- **Behavior change (not purely a bug fix)**: `top_features.csv` output
  changes for every active config using `"extreme"` (now scoped to all
  retained PCs, not just PC1/PC2, and always exactly 1 per direction per PC
  regardless of any prior `n_top_features` value) and for any config that
  adopts the new `< 1` threshold under `"top_variance"`. No output change
  for configs using `"top_variance"` with the existing `>= 1` count
  behavior, or for `"top_absolute"`/`"top_contribution"` configs (none
  exist today). This is also why the change is named `fix-` rather than
  `refactor-` — it changes observable output, not just internal structure.
- **bloommcp compatibility**: already verified safe (see #206's comment
  thread). `bloommcp` calls `perform_pca_analysis`/`create_pca_biplot`/
  `create_feature_contribution_plot`/`create_umap_colored_by_top_traits`
  directly, bypassing the QC/Viz YAML config entirely — it has no
  `n_top_features`/`feature_selection_strategy` fields, and this change
  does not touch `select_top_features_from_pca()`'s public signature.
- Explicitly out of scope (tracked separately, not touched here):
  - `create_pca_biplot`'s `top_n_features` / `static_viz.pca_biplot_top_features`
    — a separate config field controlling biplot display, unrelated to
    `pca.n_top_features`.
  - #207's fix in `create_umap_colored_by_top_traits` (already merged).
  - Threshold support for `"top_absolute"`/`"top_contribution"` — left
    count-based, flagged as a follow-up.
  - Widening `pca.feature_selection_strategy`'s validation enum to accept
    `"vector_length"` — that string is currently only reachable through
    `create_pca_biplot`'s separate `feature_selection` parameter, not
    through this config field; not touched here.
  - A step-level (`PCAAnalysisStep`) defense-in-depth guard for the new
    validation, analogous to `FilterHeritabilityStep`'s guard for bypassed
    config validation — see `design.md` Decision 3 for why this is
    explicitly not added.
