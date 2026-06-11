## Context

The full golden pipeline run (4 platforms × QC+viz+cross-platform) is ~109 MB on Box,
dominated by non-assertable artifacts (`pipeline_summary.json` up to 52 MB,
`code_snapshot.tar.gz`, `pipeline.log`). Committing it verbatim would bloat git history.
The canonical synthetic analysis-input examples are owned by `sleap-roots-contracts`
(contracts#3); this repo owns the **full real reproduction data** (original column names
`Genotype`/`Barcode`/`Replicate`).

## Goals / Non-Goals

- Goals: a reviewable vertical slice (`turface_19`) proving the layout + loaders +
  per-stage tests + README policy on real data; curated commit size; a single shared
  cross-platform source with #119.
- Non-Goals: other platforms; full `run-all` re-execution in CI; Git LFS; figures.

## Decisions

- **Curate, don't mirror.** Commit only artifacts a test asserts against. Exclude
  `code_snapshot.tar.gz`, `pipeline.log`/`viz_pipeline.log`, and oversized per-row
  intermediates (`cross_platform_exp*_loaded.csv`, multi-MB per-platform summaries that
  embed raw rows). Keep `turface_19`'s small `pipeline_summary.json` (347 KB QC /
  1.6 MB viz) — it is the carrier of the assertable PCA/UMAP/ANOVA/heritability metrics.
  Committed slice ≈ 3 MB, well under 10 MB.
  - Alternatives: commit-all (rejected: 109 MB history); Git LFS (rejected: CI/setup
    cost not justified for a curated <10 MB slice).
- **Per-stage assertions over full reproduction in CI.** Feed each stage its input
  fixture and assert its committed golden. Isolates failures and runs fast; the
  `harness/` configs keep a full reproduction runnable but out of CI.
- **Synthetic derives from contracts.** Reference/derive from the contracts canonical
  examples rather than maintaining a second copy, to avoid divergence.
- **Contract validation is soft-optional.** `validate_analysis_input()` runs when
  `sleap-roots-contracts` is importable; otherwise the test skips, so the analyze suite
  does not hard-depend on the contracts package.
- **Source = local contracts repo.** The full set is already at
  `../sleap-roots-contracts/tests/fixtures/real/`; copy from there rather than
  re-downloading from Box.

## Risks / Trade-offs

- Golden values are environment-sensitive (BLAS/lib versions) → compare with
  `numpy.allclose` at the tolerance from `docs/reproducibility.md`, not exact equality.
- Curation could omit an artifact a future test needs → README documents what was
  excluded and how to regenerate from the harness.

## Migration Plan

Additive (new `tests/fixtures/**` + one test module + loaders). No `src/` changes, no
rollback concerns. Follow-up changes add remaining platforms under the same tree.

## Open Questions

- None blocking; remaining-platform follow-ups tracked against #120.
