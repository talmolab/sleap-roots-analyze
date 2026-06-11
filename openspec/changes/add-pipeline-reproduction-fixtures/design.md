## Context

The full golden pipeline run (4 platforms × QC+viz+cross-platform) is ~109 MB on Box,
dominated by non-assertable artifacts (`pipeline_summary.json` up to 52 MB,
`code_snapshot.tar.gz`, `pipeline.log`). Committing it verbatim would bloat git history.
This repo owns the **full real reproduction data** (original column names
`Genotype`/`Barcode`/`Replicate`). The canonical synthetic analysis-input examples are
owned by `sleap-roots-contracts` (contracts#3) and are consumed only in the follow-up
contract-conformance change, not here.

## Goals / Non-Goals

- Goals: all four EDPIE platforms (`turface_19`, `turface_150`, `cylinder`,
  `root_core`) with QC + viz golden, post-QC + raw inputs, and the four cross-platform
  pairings; per-stage tests parametrized over platforms; curated commit size; a single
  shared cross-platform source with #119.
- Non-Goals: full `run-all` re-execution in CI; Git LFS; figures; contract conformance.

## Decisions

- **Curate, don't mirror.** Commit only artifacts a test asserts against. Exclude
  `code_snapshot.tar.gz`, `pipeline.log`/`viz_pipeline.log`, oversized per-row
  intermediates (`cross_platform_exp*_loaded.csv`, per-step `*_data_*.csv`), and the
  oversized per-stage `pipeline_summary.json` (up to **52 MB** for cylinder viz / 13 MB
  for turface_150). Extract the assertable viz metrics into a compact
  `viz_pca_metadata.json` (trait roster, PCA explained variance, component count, top
  features) + `viz_umap_embedding.csv` (where the run produced one). Committed real
  golden for all four platforms ≈ 6 MB, well under 10 MB.
  - Alternatives: commit-all (rejected: ~109 MB history); Git LFS (rejected: CI/setup
    cost not justified for a curated ~6 MB set).
- **PCA reproduction via the deterministic spectrum.** The viz pipeline's
  component-selection rule is platform-specific (turface_19 reaches the 0.95 threshold
  at 3 components; others report fewer). Rather than re-derive it, the test re-runs
  `perform_pca_analysis` and sums the first `n_pca_components` (golden) explained-variance
  ratios — the eigenvalue spectrum is deterministic, so this matches the golden
  `pca_explained_variance` to ~1e-16 within an environment.
- **Per-stage assertions over full reproduction in CI.** Feed each stage its input
  fixture and assert its committed golden. Isolates failures and runs fast; the
  `harness/` configs keep a full reproduction runnable but out of CI.
- **Split on the contracts dependency.** The reproduction harness imports no
  `sleap-roots-contracts` and merges now. Analysis-input contract conformance (synthetic
  examples from the package accessor + canonicalize-then-validate the post-QC fixture,
  asserting the returned `ValidationResult`) moves to a follow-up change opened once
  `sleap-roots-contracts[pandas]>=0.1.0a1` is released — it adds a dev dependency and one
  test file, reusing the post-QC fixture committed here (canonicalizing a *copy*, never
  the frame that feeds the golden harness).
- **Source = the lab Box bundle.** The full real golden set (324 files, ~113 MB) is the
  `wheat-edpie-pc-correlations` Box bundle; curate from it into the fixture tree rather
  than committing it verbatim.

## Risks / Trade-offs

- Golden values are environment-sensitive (BLAS/lib versions) → compare with
  `numpy.allclose` at the tolerance from `docs/reproducibility.md`, not exact equality.
- Curation could omit an artifact a future test needs → README documents what was
  excluded and how to regenerate from the harness.

## Migration Plan

Additive (new `tests/fixtures/**` + one test module + loaders). No `src/` changes, no
rollback concerns. All four platforms ship in this change; the only follow-up is the
contracts-gated conformance tests.

## Open Questions

- None blocking; remaining-platform follow-ups tracked against #120.
