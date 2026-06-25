## ADDED Requirements

### Requirement: Reproduction Fixture Layout

The repository SHALL provide a version-controlled fixture tree under `tests/fixtures/`
that backs the full pipeline (QC → viz → cross-platform) and is shared across
reproductions, downstream tool tests, and generated tests.

The tree SHALL contain: a top-level `README.md`; a `harness/` directory holding the
EDPIE `run_manifest.yaml` plus `qc/`, `viz/`, and `cross_platform/` configs; and a
`real/wheat_edpie/` directory with `inputs/` and `expected/`. Golden artifacts SHALL be
present for all four EDPIE platforms — `turface_19`, `turface_150`, `cylinder`, and
`root_core` — under `expected/qc/<platform>/` and `expected/viz/<platform>/`. (Synthetic
analysis-input coverage and the contract-conformance tests are deferred to a follow-up
change that depends on the unreleased `sleap-roots-contracts`; this change adds no
dependency on it.)

#### Scenario: Fixture tree is present and structured

- **WHEN** the repository is checked out
- **THEN** `tests/fixtures/README.md`, `tests/fixtures/harness/run_manifest.yaml`,
  `tests/fixtures/harness/qc/`, `tests/fixtures/harness/viz/`,
  `tests/fixtures/harness/cross_platform/`, `tests/fixtures/real/wheat_edpie/inputs/`,
  and `tests/fixtures/real/wheat_edpie/expected/` all exist

#### Scenario: Every platform has golden directories

- **WHEN** the `expected/` tree is inspected
- **THEN** `expected/qc/<platform>/` and `expected/viz/<platform>/` exist for each of
  `turface_19`, `turface_150`, `cylinder`, and `root_core`
- **AND** `inputs/post_qc/<platform>_final_data.csv` and `inputs/raw/<platform>/` exist
  for each platform

#### Scenario: Harness configs are valid

- **WHEN** each committed `harness/qc/` and `harness/viz/` config is validated
- **THEN** it passes `validate_qc_config()` / `validate_viz_config()` respectively

### Requirement: Curated Real Golden Artifacts

The fixture set SHALL commit only the **assertable** real wheat-EDPIE golden artifacts —
per-stage CSV/JSON outputs that tests compare against — and SHALL exclude non-assertable
artifacts: per-run logs (`pipeline.log`, `viz_pipeline.log`), source tarballs
(`code_snapshot.tar.gz`), oversized per-row intermediates (`cross_platform_exp{1,2}_loaded.csv`,
the per-step `*_data_*.csv` tables), and the oversized per-stage `pipeline_summary.json`
(up to 52 MB for cylinder viz / 13 MB for turface_150). The assertable viz metrics SHALL
instead be committed as a **compact `viz_pca_metadata.json`** (trait roster, PCA explained
variance, component count, top features) and, where the viz run produced one, a
**`viz_umap_embedding.csv`** — both faithful subsets extracted from the original summary.

Committed artifacts SHALL include, per platform: the post-QC `10_final_data.csv`; QC
removed-trait/sample/outlier details; the heritability-filter result + diagnostics; viz
`summary.json`, `viz_pca_metadata.json`, and `viz_umap_embedding.csv` (where present);
and the cross-platform correlations + alignment for the four manifest pairings (the
single source shared with #119).

#### Scenario: Only assertable artifacts are committed

- **WHEN** the `real/wheat_edpie/expected/` tree is inspected
- **THEN** the per-stage assertable CSV/JSON outputs are present for every platform
- **AND** no `pipeline.log`, `viz_pipeline.log`, or `code_snapshot.tar.gz` files are
  committed
- **AND** no `pipeline_summary.json` is committed under `expected/qc/` or `expected/viz/`
- **AND** the committed real golden (all four platforms) stays well under the curation
  budget (~6 MB)

### Requirement: Per-Stage Reproduction Tests

The test suite SHALL load fixture tables via `scope="session"` pytest fixtures and
assert each pipeline stage's committed output against its golden artifact, **parametrized
over all four platforms**, comparing numeric values with `numpy.allclose` within the
tolerance documented in `tests/fixtures/README.md`. The viz PCA assertion SHALL re-run
`perform_pca_analysis` and match the golden `pca_explained_variance` by summing the first
`n_pca_components` explained-variance ratios of the deterministic eigenvalue spectrum.
UMAP assertions SHALL be skipped for platforms whose viz run produced no embedding
(`root_core`).

#### Scenario: Per-stage assertions pass within tolerance for every platform

- **WHEN** the per-stage reproduction tests run
- **THEN** for each of `turface_19`, `turface_150`, `cylinder`, and `root_core`, each
  stage's numeric outputs match the committed golden within the documented tolerance

#### Scenario: Fixture tables loaded once per session

- **WHEN** multiple per-stage tests read the same platform table
- **THEN** the table is loaded once by a `scope="session"` fixture (keyed by platform)
  and reused

### Requirement: Tolerance and Regenerate Policy

`tests/fixtures/README.md` SHALL document the numerical tolerance used for golden
comparisons (reusing `docs/reproducibility.md`, #118) and the regenerate policy: a
method change that alters golden values requires reviewer approval and a
paper-supplementary update, and SHALL NOT be applied as a quiet bugfix drift.

#### Scenario: Policy is documented

- **WHEN** `tests/fixtures/README.md` is read
- **THEN** it states the comparison tolerance and the reviewer-approval + supplementary
  update requirement for regenerating golden values
