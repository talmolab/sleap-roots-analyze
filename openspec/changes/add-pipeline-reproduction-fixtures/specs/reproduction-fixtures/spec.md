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

### Requirement: Synthetic Analysis-Input Coverage From Contracts

The synthetic analysis-input coverage SHALL be sourced from the canonical examples
owned by `sleap-roots-contracts` (contracts#3) via its
`examples.load_analysis_input_example()` accessor (the single source of truth),
covering both a replicate-present and a replicate-absent shape. The repository SHALL NOT
maintain a divergent committed copy of those tables; the `synthetic/` directory
documents the accessor instead.

#### Scenario: Replicate-present and replicate-absent shapes are covered via the accessor

- **WHEN** the synthetic contract test runs with `sleap-roots-contracts` installed
- **THEN** it loads a replicate-present example (`turface`) and a replicate-absent
  example (`cylinder_no_replicate`, #142) from `examples.load_analysis_input_example()`
  and each validates

#### Scenario: No divergent synthetic copy is committed

- **WHEN** the `synthetic/` directory is inspected
- **THEN** it contains documentation pointing at the contracts accessor
- **AND** it does not commit a second copy of the canonical example CSVs

### Requirement: Analysis-Input Contract Validation

The contract tests SHALL validate the post-QC `10_final_data.csv` (after
canonicalization) and the canonical synthetic examples with
`sleap_roots_contracts.validate_analysis_input()`, and SHALL **assert the returned
`ValidationResult`** (via `raise_for_status()` / `ok`) — the validator returns a result
rather than raising, so a bare call would pass vacuously. The post-QC table SHALL be canonicalized first — native role columns renamed
to `genotype`/`sample_id`/`replicate` and cast to string, with non-trait metadata
dropped via `get_trait_columns` (the analyze#144 boundary) — because the contract takes
fixed canonical role names and no column-mapping parameter. When
`sleap-roots-contracts` (or `validate_analysis_input`) is unavailable, the test SHALL be
skipped cleanly rather than fail.

#### Scenario: Canonicalized inputs validate when contracts is installed

- **WHEN** `sleap-roots-contracts` is installed and the validation tests run
- **THEN** the canonicalized post-QC `10_final_data.csv` and each canonical example
  produce a `ValidationResult` with `ok` true (asserted via `raise_for_status()`)

#### Scenario: A non-conforming table fails the assertion

- **WHEN** an analysis-input table is missing a required canonical role (e.g. the raw,
  un-canonicalized post-QC table with native names)
- **THEN** `validate_analysis_input(...).raise_for_status()` raises and the test fails

#### Scenario: Validation test skips without contracts

- **WHEN** `sleap-roots-contracts` or `validate_analysis_input` is not importable
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
