# Pipeline reproduction fixtures

Version-controlled golden fixtures that back the full `sleap-roots-analyze` pipeline
(`run-all` = **QC → viz → cross-platform**) for the real wheat-EDPIE experiment. They
are the single shared source for reproductions, downstream tool tests, and generated
tests, so the published numbers do not silently drift.

Backs issue [#120](https://github.com/talmolab/sleap-roots-analyze/issues/120). The
cross-platform slice is the single source shared with
[#119](https://github.com/talmolab/sleap-roots-analyze/issues/119).

**Platforms covered (all four EDPIE):** `turface_19`, `turface_150`, `cylinder`, and
`root_core` (the field root-core platform) — each with QC + viz golden, post-QC + raw
inputs. The four `run_manifest` cross-platform pairings ship too.

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
    │   ├── post_qc/           boundary-A analysis inputs (<platform>_final_data.csv ×4)
    │   └── raw/<platform>/     raw pre-QC inputs (turface/turface_150/cylinder single CSV;
    │                           root_core = 3-file ingest)
    └── expected/              curated golden outputs, per stage, per platform
        ├── qc/<platform>/         final data, removed trait/sample/outlier details,
        │                          heritability filter + diagnostics, trait_statistics, config
        ├── viz/<platform>/        summary.json, heritability filter, config, and the compact
        │                          viz_pca_metadata.json (+ viz_umap_embedding.csv where UMAP ran)
        └── cross_platform/        per-pairing correlations + alignment (4 pairings)
```

`<platform>` ∈ {`turface_19`, `turface_150`, `cylinder`, `root_core`}. `root_core` has
no UMAP embedding (its viz run disabled UMAP), so it has no `viz_umap_embedding.csv`.

> Analysis-input contract conformance — synthetic examples and validating the post-QC
> table against `sleap_roots_contracts.validate_analysis_input()` — is **not** part of
> this fixture set. It is a follow-up change gated on the `sleap-roots-contracts`
> release; this tree adds no dependency on that package.

## What is committed (curation policy)

The full golden pipeline run is ~109 MB, dominated by **non-assertable** artifacts. We
commit **only what a test asserts against** (all four platforms together are ~6 MB). The
following are intentionally **excluded** and never committed:

- `pipeline.log` / `viz_pipeline.log` — run logs, no assertion value.
- `code_snapshot.tar.gz` — per-run source tarball.
- `cross_platform_exp{1,2}_loaded.csv` — large per-sample intermediates (the
  `cross_platform_correlations.csv` is the golden).
- The per-stage `pipeline_summary.json` and the per-step data CSVs (`00_data_loaded.csv`,
  `01_data_traits_cleaned.csv`, …) — these embed raw per-row data and reach **52 MB**
  (cylinder viz) / 13 MB (turface_150). Instead, the assertable viz metrics are extracted
  into a **compact `viz_pca_metadata.json`** (trait roster, PCA explained variance,
  component count, top features) and **`viz_umap_embedding.csv`** (the Nx2 embedding).

The committed `config.yaml` run records are **verbatim** from the original run and
contain absolute Windows paths and git metadata from that machine; they are historical
records, not runnable configs. The runnable recipe lives in `harness/`. The compact
`viz_pca_metadata.json` / `viz_umap_embedding.csv` are **derived** from the original
`pipeline_summary.json` (faithful subset, no transform) — regenerate them by re-running
the relevant `harness/` viz config and re-extracting.

## Provenance

- **Real wheat EDPIE** golden was produced by the EDPIE paper run (Phase 1, Metcalf
  2026) and staged on Box; copied here from the lab fixture bundle. This repo owns the
  **full real reproduction data** (original column names `Barcode`/`Genotype`/`Replicate`).
  The post-QC `inputs/post_qc/<platform>_final_data.csv` tables are reused by the
  follow-up contract-conformance change (canonicalized to the contract's role names there).

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
cross-OS / BLAS reordering. See the BLAS caveat in `docs/reproducibility.md`. The PCA
reproduction re-runs `perform_pca_analysis` and sums the first `n_pca_components`
explained-variance ratios from the **deterministic eigenvalue spectrum** to match the
golden `pca_explained_variance` (the pipeline's own component-selection rule is not
re-derived; the golden component count indexes the reproduced spectrum).

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
loaders (in `tests/fixtures.py`) and asserts each stage against its golden,
**parametrized over all four platforms**: QC (final-data sample count/roles, removed
outlier/trait/sample counts, heritability filter), viz (PCA explained-variance **re-run**
vs golden, heritability/ANOVA summary with cross-stage H² consistency, UMAP shape where
present), and cross-platform (correlations structure + alignment over the 4 pairings). It
also checks every harness config validates. The module imports no `sleap-roots-contracts`
and needs nothing beyond this repo's own dependencies.

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
