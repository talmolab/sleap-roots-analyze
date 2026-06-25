# public-api-introspection Specification

## Purpose
TBD - created by archiving change audit-public-api-introspection. Update Purpose after archive.
## Requirements
### Requirement: Introspectable Public Function Signatures

Every callable function listed in `sleap_roots_analyze.__all__` SHALL carry
complete type information so downstream tooling can derive a parameter schema by
introspection.

#### Scenario: Every parameter and the return value is annotated

- **WHEN** `inspect.signature()` is taken for each public function in `__all__`
- **THEN** every parameter (excluding `*args`/`**kwargs`) SHALL have a non-empty
  annotation
- **AND** the function SHALL have a non-empty return annotation

#### Scenario: Type hints resolve at runtime

- **WHEN** `typing.get_type_hints()` is called on each public function
- **THEN** it SHALL return without raising (no `NameError` from unimported names
  such as `Any` under `from __future__ import annotations`)

### Requirement: Parsable Public Function Docstrings

Every public function in `__all__` SHALL have a Google-style docstring that names
each parameter, so tooling can derive parameter descriptions.

#### Scenario: Docstring has Args and Returns sections

- **WHEN** the docstring of each public function is inspected
- **THEN** it SHALL be non-empty
- **AND** it SHALL contain a `Returns:` section
- **AND** it SHALL contain an `Args:` section when the function takes at least one
  parameter

#### Scenario: Every parameter is documented

- **WHEN** a public function declares parameters (excluding `self`/`cls` and
  `*args`/`**kwargs`)
- **THEN** each parameter name SHALL appear in the docstring body

#### Scenario: Raising functions document their exceptions

- **WHEN** a public function contains a `raise` of a non-trivial exception in its
  own body
- **THEN** its docstring SHALL contain a `Raises:` section

### Requirement: Introspectable Public Classes

Every class listed in `sleap_roots_analyze.__all__` SHALL be documented so tooling
can describe it and its constructor.

#### Scenario: Class has a docstring and documented constructor parameters

- **WHEN** a public class in `__all__` is inspected
- **THEN** the class SHALL have a non-empty docstring
- **AND** where its `__init__` declares parameters beyond `self`, those parameters
  SHALL be annotated and named in the class or `__init__` docstring

### Requirement: Enforced Introspection Contract

The project SHALL provide an automated check that fails when any `__all__` entry
violates the introspection contract, so the contract holds as the API evolves.

#### Scenario: Audit script reports and gates violations

- **WHEN** `scripts/check_public_api_docs.py` is run
- **THEN** it SHALL iterate every name in `sleap_roots_analyze.__all__`
- **AND** it SHALL print a per-symbol pass/fail report
- **AND** it SHALL exit with a non-zero status if any symbol violates the contract,
  and zero when all entries pass

#### Scenario: Contract is enforced in the test suite

- **WHEN** the pytest suite runs
- **THEN** a test SHALL execute the audit and fail if any `__all__` entry violates
  the contract

### Requirement: Public API Audit Report

The project SHALL document the audit outcome in a committed report.

#### Scenario: Audit report records findings and result

- **WHEN** `docs/public_api_audit_2026.md` is viewed
- **THEN** it SHALL describe the audit methodology and criteria
- **AND** it SHALL list the symbols that failed before the change and what was
  changed to fix them
- **AND** it SHALL state the post-change result that all `__all__` entries pass

