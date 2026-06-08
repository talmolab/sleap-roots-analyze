# statistics-api Specification

## ADDED Requirements

### Requirement: Public Statistics API Surface

The package SHALL export all heritability, ANOVA, and trait-variance functions
defined in `statistics.py` from the top-level `sleap_roots_analyze` namespace so
downstream code can import them without reaching into internal modules.

#### Scenario: All eight statistics functions are importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import (calculate_trait_statistics, perform_anova_by_genotype, calculate_heritability_estimates, identify_high_heritability_traits, analyze_heritability_thresholds, analyze_trait_variance, diagnose_heritability_issues, compare_trait_heritabilities)`
- **THEN** the import SHALL succeed
- **AND** each imported object SHALL be identity-equal (`is`) to the function
  defined in `sleap_roots_analyze.statistics`

#### Scenario: Each function is listed in `__all__` and bound by star import

- **WHEN** `sleap_roots_analyze.__all__` is inspected
- **THEN** it SHALL contain the name of each of the eight statistics functions
- **AND** SHALL contain no duplicate entries
- **AND** `from sleap_roots_analyze import *` SHALL bind each of those names

### Requirement: Resolvable Statistics Type Hints

Each public statistics function SHALL have type hints that resolve at runtime, so
downstream tool-schema generators that call `typing.get_type_hints()` do not fail.

#### Scenario: get_type_hints succeeds on every public function

- **WHEN** `typing.get_type_hints(fn)` is called on each of the eight functions
- **THEN** it SHALL return without raising `NameError`
- **AND** every parameter and the return value SHALL carry a type annotation

### Requirement: Documented Statistics Functions

Each public statistics function SHALL have a complete Google-style docstring, and
the module SHALL describe its scope, so consumers and tooling can understand the
contract.

#### Scenario: Every public function has Args and Returns sections

- **WHEN** any of the eight public statistics functions is inspected
- **THEN** its docstring SHALL include populated Args and Returns sections in
  Google style (and a Raises section where the function raises)
- **AND** functions returning structured dictionaries SHALL enumerate the returned
  keys in the Returns section

#### Scenario: Module docstring distinguishes statistics from cross-experiment analysis

- **WHEN** the `statistics.py` module docstring is read
- **THEN** it SHALL describe the module's scope (single-experiment heritability,
  ANOVA, and trait-variance analysis)
- **AND** it SHALL name `cross_experiment_analysis` to clarify how that scope differs

### Requirement: Statistics API Reference Documentation

The project documentation SHALL keep its hand-maintained API reference and
changelog in sync with the newly-public statistics functions.

#### Scenario: API reference lists all eight statistics functions

- **WHEN** `docs/API.md` is viewed
- **THEN** the `## statistics Module` section SHALL include a reference entry for
  each of the eight statistics functions
- **AND** each entry's documented signature and defaults SHALL match the code

#### Scenario: Changelog records the newly-public API

- **WHEN** `docs/CHANGELOG.md` `[Unreleased]` section is viewed
- **THEN** it SHALL include an `### Added` entry noting the eight statistics
  functions are now importable from `sleap_roots_analyze`
