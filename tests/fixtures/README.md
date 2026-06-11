# Pipeline reproduction fixtures

Version-controlled golden fixtures that back the full `sleap-roots-analyze` pipeline
(`run-all` = **QC → viz → cross-platform**) for the real wheat-EDPIE experiment. They
are the single shared source for reproductions, downstream tool tests, and generated
tests, so the published numbers do not silently drift.

Backs issue [#120](https://github.com/talmolab/sleap-roots-analyze/issues/120). The
cross-platform slice is the single source shared with
[#119](https://github.com/talmolab/sleap-roots-analyze/issues/119).

> **Scope of the current set.** This is the **infrastructure + `turface_19` vertical
> slice** (one of the four EDPIE platforms). The layout, loaders, tolerance/regenerate
> policy, harness, and per-stage tests are proven end-to-end on `turface_19`. Follow-up
> changes add `turface_150`, `cylinder`, and `root_core` against the same scaffold.

## Layout

```
tests/fixtures/
├── README.md                 ← this file
├── harness/                  ← the runnable EDPIE recipe (all 4 platforms)
│   ├── run_manifest.yaml      run-all manifest
│   ├── qc/                    per-platform QC configs
│   ├── viz/                   per-platform viz configs
│   └── cross_platform/        per-pairing cross-platform configs
└── real/wheat_edpie/
    ├── inputs/
    │   ├── post_qc/           boundary-A analysis inputs (10_final_data per platform)
    │   └── raw/               raw pre-QC inputs (turface single CSV here)
    └── expected/              curated golden outputs, per stage, per platform
        ├── qc/turface_19/         QC per-step outputs (final data, removed-detail, heritability filter, summaries)
        ├── viz/turface_19/        viz outputs (summary, heritability, figure manifests, full step summary)
        └── cross_platform/        per-pairing correlations + alignment (turface_19 pairings)
```

> Analysis-input contract conformance — synthetic examples and validating the post-QC
> table against `sleap_roots_contracts.validate_analysis_input()` — is **not** part of
> this fixture set. It is a follow-up change gated on the `sleap-roots-contracts`
> release; this tree adds no dependency on that package.

## What is committed (curation policy)

The full golden pipeline run is ~109 MB, dominated by **non-assertable** artifacts. We
commit **only what a test asserts against** (the `turface_19` slice is ~3 MB). The
following are intentionally **excluded** and never committed:

- `pipeline.log` / `viz_pipeline.log` — run logs, no assertion value.
- `code_snapshot.tar.gz` — per-run source tarball.
- `cross_platform_exp{1,2}_loaded.csv` — large per-sample intermediates (the
  `cross_platform_correlations.csv` is the golden).
- The multi-MB per-platform `pipeline_summary.json` for platforms whose summaries embed
  raw per-row data. (`turface_19`'s summaries are small enough — QC 347 KB, viz 1.6 MB —
  and carry the assertable PCA/UMAP/ANOVA/heritability metrics, so they are kept.)

The committed run records (`config.yaml`, `pipeline_summary.json`) are **verbatim** from
the original run and contain absolute Windows paths and git metadata from that machine;
they are historical records, not runnable configs. The runnable recipe lives in
`harness/`.

## Provenance

- **Real wheat EDPIE** golden was produced by the EDPIE paper run (Phase 1, Metcalf
  2026) and staged on Box; copied here from the lab fixture bundle. This repo owns the
  **full real reproduction data** (original column names `Barcode`/`Genotype`/`Replicate`).
  The post-QC `inputs/post_qc/turface_19_final_data.csv` is reused by the follow-up
  contract-conformance change (canonicalized to the contract's role names there).

## Tolerance policy

Numeric golden comparisons follow [`docs/reproducibility.md`](../../docs/reproducibility.md)
(#118):

- **Integer counts, labels, indices, and trait-name rosters:** compare for **exact**
  equality.
- **Floating-point values** (PCA explained variance / loadings / scores, heritability
  H², ANOVA statistics, correlations): compare with **`numpy.allclose(rtol=1e-6,
  atol=1e-9)`**.
- **UMAP embeddings** are the most environment-sensitive output (numba/BLAS dependent
  across OSes); assert their **shape and finiteness**, not exact coordinates, in
  cross-platform CI.

Reproductions that re-run a stage (e.g. PCA on the post-QC `10_final_data.csv`) match the
committed golden to ~`1e-16` within a single environment; `rtol=1e-6` absorbs
cross-OS / BLAS reordering. See the BLAS caveat in `docs/reproducibility.md`.

## Regenerate policy

**Golden values do not change as a quiet bugfix drift.** To regenerate any golden
artifact:

1. Re-run the relevant `harness/` config(s) (the `run_manifest.yaml` recipe).
2. If the only differences are within the tolerance above, no update is needed.
3. If a **method change** moved the numbers, the regeneration requires **reviewer
   approval** and a corresponding **paper-supplementary update** — it must not be folded
   silently into an unrelated change. Record the reason (method change + reviewer) in the
   PR that updates the golden.

## Tests

`tests/test_pipeline_reproduction.py` loads these fixtures via `scope="session"` pytest
loaders (in `tests/fixtures.py`) and, for `turface_19`, asserts each stage against its
golden: QC (final-data shape/roster, removed-detail counts, heritability filter), viz
(PCA explained-variance **re-run** vs golden, heritability/ANOVA summary), and
cross-platform (correlations structure + alignment). It also checks the harness configs
validate. The module imports no `sleap-roots-contracts` and needs nothing beyond this
repo's own dependencies.

Analysis-input contract conformance is a **follow-up change** (it depends on the
unreleased `sleap-roots-contracts`): it will canonicalize a *copy* of the post-QC
fixture committed here — native roles renamed to `genotype`/`sample_id`/`replicate`,
cast to string, metadata dropped via `get_trait_columns` (the analyze#144 boundary) —
and assert `validate_analysis_input(...).raise_for_status()`, plus validate the
package's canonical examples loaded from `sleap_roots_contracts.examples`. That change
adds `sleap-roots-contracts[pandas]>=0.1.0a1` to the dev group. The golden reproduction
tests here (native names, `rtol=1e-6`) are the proof the pipeline is unchanged, so they
must stay green — contract checks always run on a copy, never the frame that feeds the
pipeline.
