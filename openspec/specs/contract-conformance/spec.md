# contract-conformance Specification

## Purpose
TBD - created by archiving change add-contract-conformance-tests. Update Purpose after archive.
## Requirements
### Requirement: Analysis-Input Contract Dev Dependency

The repository SHALL declare `sleap-roots-contracts[pandas]>=0.1.0a1` in the `dev`
dependency group, with `uv.lock` resolving the pre-release, so the analysis-input contract
conformance tests run in CI without being skipped.

#### Scenario: Contract package available under dev install

- **WHEN** the project is installed with its dev dependencies
- **THEN** `sleap_roots_contracts` (with `validate_analysis_input`,
  `canonicalize_role_dtypes`, and `examples.load_analysis_input_example`) is importable
- **AND** the contract conformance tests execute rather than `importorskip`-skipping.

### Requirement: Post-QC Fixture Contract Conformance

The conformance tests SHALL prove that every committed EDPIE post-QC fixture
(`turface_19`, `turface_150`, `cylinder`, `root_core`) conforms to the analysis-input
contract, validating only a transformed **copy** so the frame that feeds
QC/viz/cross-platform is never mutated.

#### Scenario: Post-QC fixture validates after canonicalization

- **WHEN** a post-QC fixture is loaded and a separate frame is built by renaming native
  role columns (`Genotype`→`genotype`, `Barcode`→`sample_id`, `Replicate`→`replicate`),
  dropping non-trait metadata via `get_trait_columns` called with the renamed role kwargs
  (`barcode_col="sample_id"`, `genotype_col="genotype"`, `replicate_col="replicate"`), and
  applying `canonicalize_role_dtypes`
- **THEN** `validate_analysis_input(copy).raise_for_status()` succeeds for every platform.

#### Scenario: Rename and trait selection are non-vacuous

- **WHEN** the canonicalized copy is built
- **THEN** `genotype` and `sample_id` are present and the native names are gone
- **AND** the selected trait columns exclude the `replicate` and `genotype` role columns,
  so a no-op rename or default `get_trait_columns` kwargs cannot make the check pass
  vacuously.

#### Scenario: Original fixture frame is not mutated

- **WHEN** the conformance test has built and validated the canonicalized copy
- **THEN** the originally loaded fixture frame is unchanged under
  `pd.testing.assert_frame_equal` against a deep pre-copy.

### Requirement: Canonical Example Contract Conformance

The conformance tests SHALL validate every canonical example exposed by the contract
package registry (`analysis_input_example_names()`), loaded from
`sleap_roots_contracts.examples`, so the repository tracks the contract's single source of
truth and auto-covers future examples.

#### Scenario: Packaged canonical examples validate as-is

- **WHEN** each name from `analysis_input_example_names()` is loaded via
  `load_analysis_input_example`
- **THEN** `validate_analysis_input(example).raise_for_status()` succeeds with no
  modification to the example.

### Requirement: Validation Is Asserted and Non-Vacuous

The conformance tests SHALL assert the `ValidationResult` returned by
`validate_analysis_input` (via `raise_for_status()` or `.ok`) and SHALL NOT discard it, and
SHALL include a negative control proving the validator can fail.

#### Scenario: Result is checked, not discarded

- **WHEN** a conformance test calls `validate_analysis_input(...)`
- **THEN** the returned `ValidationResult` is asserted via `raise_for_status()` or `.ok`.

#### Scenario: Negative control proves the validator can fail

- **WHEN** a deliberately malformed frame (e.g. missing the `genotype` role) is validated
- **THEN** `validate_analysis_input(bad).raise_for_status()` raises `ValueError` and
  `bad`'s result `.ok is False`.

### Requirement: No Stale Validation Artifacts

The repository SHALL NOT retain `*_validation.json` expected files under `tests/fixtures/`,
whose `summary` shape does not match `ValidationResult`; the contract is asserted live, not
against stored JSON.

#### Scenario: No validation JSON remains in the fixture tree

- **WHEN** the fixture tree is inspected (`git ls-files 'tests/**_validation.json'`)
- **THEN** no such file exists.

