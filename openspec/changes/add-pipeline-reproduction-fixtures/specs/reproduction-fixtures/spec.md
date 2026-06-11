## ADDED Requirements

### Requirement: Reproduction Fixture Layout

The repository SHALL provide a version-controlled fixture tree under `tests/fixtures/`
that backs the full pipeline (QC → viz → cross-platform) and is shared across
reproductions, downstream tool tests, and generated tests.

The tree SHALL contain: a top-level `README.md`; a `harness/` directory holding the
EDPIE `run_manifest.yaml` plus `qc/`, `viz/`, and `cross_platform/` configs; a
`real/wheat_edpie/` directory with `inputs/` and `expected/`; and a `synthetic/`
directory.

#### Scenario: Fixture tree is present and structured

- **WHEN** the repository is checked out
- **THEN** `tests/fixtures/README.md`, `tests/fixtures/harness/run_manifest.yaml`,
  `tests/fixtures/harness/qc/`, `tests/fixtures/harness/viz/`,
  `tests/fixtures/harness/cross_platform/`, `tests/fixtures/real/wheat_edpie/inputs/`,
  `tests/fixtures/real/wheat_edpie/expected/`, and `tests/fixtures/synthetic/` all exist

#### Scenario: Harness configs are valid

- **WHEN** the committed `harness/qc/` and `harness/viz/` configs are validated
- **THEN** they pass `validate_qc_config()` / `validate_viz_config()` respectively

### Requirement: Curated Real Golden Artifacts

The fixture set SHALL commit only the **assertable** real wheat-EDPIE golden artifacts
for the `turface_19` platform — per-stage CSV/JSON outputs that tests compare against —
and SHALL exclude non-assertable artifacts: per-run logs (`pipeline.log`,
`viz_pipeline.log`), source tarballs (`code_snapshot.tar.gz`), and oversized per-row
intermediates (e.g. `cross_platform_exp{1,2}_loaded.csv`). The per-platform
`pipeline_summary.json` is committed **only when** it is small enough to be the
practical carrier of assertable metrics (as for `turface_19`); the multi-MB summaries
that embed raw per-row data for other platforms are excluded.

Committed artifacts SHALL include, for `turface_19`: the post-QC `10_final_data.csv`;
QC removed-trait, removed-sample, and outlier-removal details/counts; the
heritability-filter result; viz PCA explained variance, UMAP, heritability H², ANOVA,
and trait statistics; and the cross-platform correlations slice shared with #119.

#### Scenario: Only assertable artifacts are committed

- **WHEN** the `real/wheat_edpie/expected/` tree is inspected for `turface_19`
- **THEN** the per-stage assertable CSV/JSON outputs are present
- **AND** no `pipeline.log`, `viz_pipeline.log`, or `code_snapshot.tar.gz` files are
  committed
- **AND** the committed `turface_19` slice stays well under the curation budget (~3 MB)

### Requirement: Synthetic Per-Step Fixtures

The fixture set SHALL provide synthetic per-step fixtures covering both a
replicate-present and a replicate-absent analysis-input shape, derived from the
canonical examples owned by `sleap-roots-contracts` (contracts#3) rather than a second
hand-maintained copy.

#### Scenario: Replicate-present and replicate-absent shapes exist

- **WHEN** the `synthetic/` directory is inspected
- **THEN** a replicate-present fixture and a replicate-absent fixture are both present

### Requirement: Analysis-Input Contract Validation

The post-QC `10_final_data.csv` and the synthetic analysis-input fixtures SHALL pass
`sleap_roots_contracts.validate_analysis_input()` when the `sleap-roots-contracts`
package is installed. When the package is not installed, the validation test SHALL be
skipped cleanly rather than fail.

#### Scenario: Inputs validate when contracts is installed

- **WHEN** `sleap-roots-contracts` is installed and the validation test runs
- **THEN** the post-QC `10_final_data.csv` and each synthetic analysis-input fixture
  pass `validate_analysis_input()`

#### Scenario: Validation test skips without contracts

- **WHEN** `sleap-roots-contracts` is not installed
- **THEN** the contract-validation test is skipped, not failed

### Requirement: Per-Stage Reproduction Tests

The test suite SHALL load fixture tables via `scope="session"` pytest fixtures and
assert each pipeline stage's committed output against its golden artifact for the
`turface_19` platform, comparing numeric values with `numpy.allclose` within the
tolerance documented in `tests/fixtures/README.md`.

#### Scenario: Per-stage assertions pass within tolerance

- **WHEN** the per-stage reproduction tests run for `turface_19`
- **THEN** each stage's numeric outputs match the committed golden within the documented
  tolerance

#### Scenario: Fixture tables loaded once per session

- **WHEN** multiple per-stage tests read the same platform table
- **THEN** the table is loaded once by a `scope="session"` fixture and reused

### Requirement: Tolerance and Regenerate Policy

`tests/fixtures/README.md` SHALL document the numerical tolerance used for golden
comparisons (reusing `docs/reproducibility.md`, #118) and the regenerate policy: a
method change that alters golden values requires reviewer approval and a
paper-supplementary update, and SHALL NOT be applied as a quiet bugfix drift.

#### Scenario: Policy is documented

- **WHEN** `tests/fixtures/README.md` is read
- **THEN** it states the comparison tolerance and the reviewer-approval + supplementary
  update requirement for regenerating golden values
