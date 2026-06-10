## Why

The wheat EDPIE paper's central analysis is a cross-platform PC-correlation workflow, but today it
lives only in paper-specific scripts (`wheat-edpie-pc-correlations/scripts/run_analysis.py`).
Reproducing it means composing 6+ lower-level functions (`calculate_genotype_means`,
`perform_pca_analysis`, `calculate_cross_experiment_correlations`,
`identify_significant_correlations`, CI/power helpers) in exactly the right order — fragile, and a
poor surface for the bloom-mcp wheat EDPIE reproduction tool (Phase 2 of the Metcalf 2026 project).
Exposing a single public function turns the reproduction into a one-call test and a clean MCP tool.

Tracks `talmolab/sleap-roots-analyze#119`.

## What Changes

- Add a public workflow function `cross_platform_pc_correlations(...)` (new module
  `src/sleap_roots_analyze/cross_platform_pc.py`) that runs the full per-platform PCA →
  cross-platform PC-correlation workflow and returns a typed `CrossPlatformPCResult`.
- **Correctness-critical ordering** (per the maintainer's reference note on #119): *sample-level
  PCA → genotype-mean PC scores → correlate*, **not** average-then-PCA. For each platform the
  function fits PCA on sample-level traits, then aggregates the resulting sample PC scores to
  genotype means; correlations are computed on those genotype-mean PC scores.
- Add `CrossPlatformPCResult` (frozen dataclass) bundling: per-platform PCA results, genotype-aligned
  PC-score matrices, per-platform-pair correlation tables, pooled FDR-corrected p-values, Fisher-z
  confidence intervals, power statistics, and the significant-correlations list.
- Pool **all** cross-platform PC tests into a single FDR family (Turface×Cylinder + Turface×Field +
  Cylinder×Field = 12 + 15 + 20 = 47 tests for the paper's 3/4/5-PC configuration) and apply one
  multiple-testing correction across the family.
- Export `cross_platform_pc_correlations` and `CrossPlatformPCResult` from the package root
  (`__init__.py` + `__all__`), with full type hints and Google docstrings (introspection-ready).
- Add a synthetic 3-platform unit test. Scaffold a wheat-EDPIE regression test that validates the
  golden 47-test result (19 genotypes, 0 FDR-significant); it is **skipped when the post-QC fixture
  is absent** (the fixture ships via the related issue #120 / Box reference data).

## Impact

- Affected specs: `cross-platform-analysis` (ADDED requirements).
- Affected code:
  - `src/sleap_roots_analyze/cross_platform_pc.py` (new) — `cross_platform_pc_correlations`,
    `CrossPlatformPCResult`.
  - `src/sleap_roots_analyze/__init__.py` — import + `__all__` entries.
  - `tests/test_cross_platform_pc.py` (new) — synthetic unit test + (skip-guarded) golden regression.
- Composes existing functions unchanged: `calculate_genotype_means`, `perform_pca_analysis`,
  `calculate_correlation_ci`, `achieved_power`, `minimum_detectable_correlation`, and
  `statsmodels.stats.multitest.multipletests`.
- No pipeline/config changes; this is a pure, importable workflow function. The config-driven
  `CrossPlatformPipeline` is untouched.

## Open question for review

- The wheat-EDPIE **regression** test (acceptance items "reproduces paper numbers" + "regression
  test using the fixture") depends on post-QC input data not in this repo (Box / issue #120). This
  change ships the function + synthetic test now and a skip-guarded regression test; the exact-number
  validation lands when the fixture is committed. Acceptable, or should the fixture be vendored here
  first?
