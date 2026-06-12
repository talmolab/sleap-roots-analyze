## 1. Fixture tree + harness

- [x] 1.1 Create `tests/fixtures/{harness,real/wheat_edpie/{inputs,expected}}/`
- [x] 1.2 Copy EDPIE `run_manifest.yaml` + `harness/{qc,viz,cross_platform}/` configs from the contracts repo
- [x] 1.3 Verify committed `harness/qc` + `harness/viz` configs pass `validate_qc_config()` / `validate_viz_config()`

## 2. Curated real golden (all 4 platforms)

- [x] 2.1 Copy post-QC `inputs/post_qc/<platform>_final_data.csv` + raw inputs for turface_19, turface_150, cylinder, root_core (3-file ingest)
- [x] 2.2 Copy curated QC golden per platform (final_data, removed-trait/sample/outlier details, heritability filter + diagnostics, trait_statistics, config) — exclude `code_snapshot.tar.gz`, logs, per-step `*_data_*.csv`, and oversized `pipeline_summary.json`
- [x] 2.3 Copy curated viz golden per platform (summary, heritability filter, config) + extract compact `viz_pca_metadata.json` (+ `viz_umap_embedding.csv` where UMAP ran; root_core has none) from the oversized `pipeline_summary.json`
- [x] 2.4 Copy the four cross-platform pairings incl. `root_core_vs_cylinder` (shared single source with #119)
- [x] 2.5 Confirm committed real fixtures are well under the curation budget (~6 MB, no excluded large blobs)

## 3. Loaders + tests (TDD)

- [x] 3.1 Add `scope="session"` loaders in `tests/fixtures.py` keyed by platform (final_data, heritability, removed counts, viz summary, PCA metadata, UMAP embedding, cross-platform dir)
- [x] 3.2 Per-stage reproduction tests parametrized over all 4 platforms (QC / viz / cross-platform), `allclose` within documented tolerance; PCA re-run via deterministic eigenvalue spectrum
- [x] 3.3 Harness-config validity test parametrized over all 4 platforms (`validate_qc_config()` / `validate_viz_config()`)

> Analysis-input contract conformance (synthetic examples + canonicalize-then-validate
> the post-QC fixture) is deferred to a follow-up change gated on the
> `sleap-roots-contracts` release — this change imports no contracts package.

## 5. Docs + policy

- [x] 5.1 Write `tests/fixtures/README.md`: layout, tolerance (link `docs/reproducibility.md`), regenerate policy (reviewer approval + supplementary update)

## 6. Verify

- [x] 6.1 `uv run pytest tests/test_pipeline_reproduction.py` green
- [x] 6.2 `/pre-merge-check` (black + ruff + full pytest + coverage + OpenSpec validate)
- [x] 6.3 `openspec validate add-pipeline-reproduction-fixtures --strict`
