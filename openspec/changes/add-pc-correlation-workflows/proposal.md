# Add PC-Correlation and Trait-Enrichment Workflows (issue #119)

## Why

The wheat EDPIE paper's central analysis is a **cross-platform PC-correlation
workflow** plus a downstream **trait-level enrichment test**. Today both live as
paper-specific scripts (`run_analysis.py`, `run_trait_enrichment.py`) outside
`sleap-roots-analyze`. Reproducing them means composing 6+ lower-level calls in
the right order and hand-wiring file IO — fragile, and not something the
`bloom-mcp` reproduction tool (Phase 2 of the Metcalf 2026 project) can wrap
cleanly.

Issue #119 asks for a single public entry point so the reproduction becomes a
one-call test and the MCP gets a clean tool. The paper code is already factored
into DAG-style nodes (`aggregate` → `correlate` → `visualize`; `enrichment`),
so the work is to port those nodes into the public API, deduplicate against
existing helpers, and expose two orchestrator functions.

## What Changes

- **New subpackage** `src/sleap_roots_analyze/pc_correlations/` holding the
  DAG-node modules ported from the paper: `aggregate.py`, `correlate.py`,
  `enrichment.py`, `visualize.py`, and a `workflow.py` with the two
  orchestrators.
- **New public capability — PC-level cross-platform correlations.** Public
  function `cross_platform_pc_correlations(pipeline_run, platform_config,
  output_dir, ...)` runs the full DAG: load per-platform sample PC scores + QC
  genotype labels → aggregate sample PC scores to genotype means → align on
  common genotypes → correlate every PC × every PC per platform pair → FDR
  correction at **both** `combined` and `per_pair` scopes → export tidy CSVs,
  figures, and `metadata.json`. Returns a structured `dict` for inspection.
- **New public capability — trait-level correlation enrichment.** Public
  function `trait_correlation_enrichment(correlation_files, output_dir, ...)`
  runs an **independent** DAG over existing trait-level
  `cross_platform_correlations.csv` files: load → count nominally significant
  (p < α) → binomial enrichment/depletion test (per-pair + combined) → export
  `enrichment_results.csv`, a summary figure, and `metadata.json`.
- **Reuse, not duplicate.** `correlate.py` imports the existing
  `calculate_correlation_ci`, `achieved_power`, and
  `minimum_detectable_correlation` from `cross_experiment_analysis.py` instead
  of re-implementing them.
- **Public exports.** `cross_platform_pc_correlations`,
  `trait_correlation_enrichment`, and `EnrichmentResult` added to the package
  `__all__` with full type hints + Google docstrings.
- **CLI/script wrappers** (paper reproduction entry points) call the new public
  functions rather than duplicating logic; no hard-coded Windows paths, no
  paper-specific values baked into public functions (`WHEAT_EDPIE_PLATFORMS`
  ships as an example default config only).
- **Tests** with small synthetic fixtures: a synthetic PCA-run directory for
  workflow 1 and synthetic `cross_platform_correlations.csv` files for
  workflow 2, asserting the DAG through outputs and confirming the two
  workflows are independent. A skip-guarded golden regression reproduces the
  paper headline numbers (47 PC tests, 19 genotypes, 0 FDR-significant under
  combined `fdr_by`).

## Impact

- Affected specs: **new** `pc-correlation-workflow`, **new**
  `trait-correlation-enrichment`. The existing `cross-platform-analysis`
  (trait-level pipeline) is unchanged — workflow 2 consumes its output but does
  not modify it.
- Affected code: new `src/sleap_roots_analyze/pc_correlations/` subpackage;
  `__init__.py` exports; new reproduction scripts under `scripts/`; new tests
  and synthetic fixtures.
- No breaking changes; purely additive public API.
