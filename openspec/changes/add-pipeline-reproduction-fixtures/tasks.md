## 1. Fixture tree + harness

- [x] 1.1 Create `tests/fixtures/{harness,real/wheat_edpie/{inputs,expected}}/`
- [x] 1.2 Copy EDPIE `run_manifest.yaml` + `harness/{qc,viz,cross_platform}/` configs from the contracts repo
- [x] 1.3 Verify committed `harness/qc` + `harness/viz` configs pass `validate_qc_config()` / `validate_viz_config()`

## 2. Curated real golden (turface_19)

- [x] 2.1 Copy `turface_19` post-QC `inputs/post_qc/turface_19_final_data.csv`
- [x] 2.2 Copy `turface_19` curated QC golden (final_data, removed-trait/sample/outlier details, heritability filter, summaries) — exclude `code_snapshot.tar.gz`, `pipeline.log`
- [x] 2.3 Copy `turface_19` curated viz golden (PCA explained variance, UMAP, heritability H², ANOVA, trait statistics via small `pipeline_summary.json`) — same exclusions
- [x] 2.4 Copy the cross-platform correlations slice involving `turface_19` (shared single source with #119)
- [x] 2.5 Confirm committed real fixtures are well under the curation budget (no excluded large blobs)

## 3. Loaders + tests (TDD)

- [x] 3.1 Add `scope="session"` loaders in `tests/fixtures.py` for the turface_19 fixture tables + harness config paths
- [x] 3.2 Per-stage reproduction tests for turface_19 (QC / viz / cross-platform), `allclose` within documented tolerance
- [x] 3.3 Harness-config validity test (`validate_qc_config()` / `validate_viz_config()`)

> Analysis-input contract conformance (synthetic examples + canonicalize-then-validate
> the post-QC fixture) is deferred to a follow-up change gated on the
> `sleap-roots-contracts` release — this change imports no contracts package.

## 5. Docs + policy

- [x] 5.1 Write `tests/fixtures/README.md`: layout, tolerance (link `docs/reproducibility.md`), regenerate policy (reviewer approval + supplementary update)

## 6. Verify

- [x] 6.1 `uv run pytest tests/test_pipeline_reproduction.py` green
- [x] 6.2 `/pre-merge-check` (black + ruff + full pytest + coverage + OpenSpec validate)
- [x] 6.3 `openspec validate add-pipeline-reproduction-fixtures --strict`
