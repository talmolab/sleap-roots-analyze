# Spec Delta: reproducibility-gates

## ADDED Requirements

### Requirement: Whole-package determinism coverage guard

The determinism coverage guard SHALL ensure every stochastic function in the package is
covered by the determinism sweep, enforced automatically rather than maintained by hand.
The guard SHALL walk all modules under `sleap_roots_analyze` and collect every
module-level function defined in the package whose signature accepts `random_state`, and
SHALL fail unless each such function is present in the sweep's case registry or in an
explicit, documented exclusion set. The guard SHALL NOT rely on top-level re-exports
alone, so stochastic functions defined in submodules are covered.

#### Scenario: All stochastic functions in the package are covered

- **GIVEN** the package defines functions accepting `random_state` across its modules
- **WHEN** the coverage guard runs
- **THEN** it passes only if every such function is in the case registry or the documented exclusion set

#### Scenario: A stochastic submodule helper is added without a case

- **GIVEN** a new function accepting `random_state` is added in a submodule and not re-exported at top level
- **WHEN** the coverage guard runs in CI
- **THEN** the guard fails, blocking the change until a determinism case or a documented exclusion is added

#### Scenario: The guard can report a missing function (negative test)

- **GIVEN** a function-set that contains a name absent from the registry and the exclusion set
- **WHEN** the guard's coverage comparison is evaluated
- **THEN** it reports that name as uncovered, proving the guard can fail rather than shipping born-green

#### Scenario: The exclusion set cannot silently rot

- **GIVEN** the documented exclusion set of intentionally-uncovered functions
- **WHEN** the guard's consistency check runs
- **THEN** it fails if any excluded name no longer exists or also appears in the registry

### Requirement: Same-seed determinism

Each function in the determinism case registry SHALL produce identical output across two
runs with the same `random_state`. Integer labels and indices SHALL be exactly equal;
floating-point arrays SHALL match within `rtol=1e-6`. Each seeded function SHALL also
accept `random_state=None` without raising.

#### Scenario: Same seed reproduces identical output

- **GIVEN** a registered stochastic function and a fixed dataset
- **WHEN** it is called twice with the same `random_state`
- **THEN** integer labels/indices are exactly equal and float arrays match within `rtol=1e-6`

#### Scenario: Function accepts an unset seed

- **GIVEN** a registered seeded function
- **WHEN** it is called with `random_state=None`
- **THEN** it returns a result without raising

### Requirement: Result-object round-trip gate

The result round-trip gate SHALL assert that every analytical function in its case list,
when it returns a serializable dataclass result object, serializes to JSON and round-trips
without loss. Its case list SHALL span the analytical surface — stochastic functions and
the statistics/heritability functions — so result types from issues #127, #128, and #129
are covered when they land. The gate SHALL be opt-in by construction: functions returning
plain dictionaries SHALL be skipped, so the gate never blocks functions that have not
adopted a result object, and SHALL begin asserting automatically when a listed function
returns a dataclass, without test edits. The gate SHALL guard at least one shipped
serializable dataclass so it is not vacuous before new result types land.

#### Scenario: Function returns a dataclass result object

- **GIVEN** a listed analytical function whose return value is a dataclass result object
- **WHEN** the round-trip gate serializes its JSON-native projection and parses it back
- **THEN** the parsed projection equals the original projection, and `from_dict` (when defined) reconstructs an equal object

#### Scenario: Function has not adopted a result object yet

- **GIVEN** a listed analytical function that still returns a plain dictionary
- **WHEN** the round-trip gate runs
- **THEN** that function is skipped (recorded as skipped) and CI stays green

#### Scenario: Result object contains numpy types and NaN

- **GIVEN** a result object holding numpy scalars, ndarrays, nested dicts, and NaN
- **WHEN** the gate projects it with `convert_to_json_serializable` and JSON round-trips it
- **THEN** numpy types become JSON-native values and the projection compares equal NaN-aware

#### Scenario: A shipped serializable dataclass round-trips today

- **GIVEN** a directly-constructed `PipelineSummary` containing a `Path` and a numpy value
- **WHEN** the gate round-trips its serialized form
- **THEN** the projection survives JSON encode/decode unchanged

#### Scenario: Lossy stringification fails the gate

- **GIVEN** a dataclass field holding an object the serializer can only stringify to a `"<TypeName>"` placeholder
- **WHEN** the gate inspects the JSON-native projection
- **THEN** the gate fails rather than reporting a vacuous round-trip

### Requirement: Gates enforced as a required pull-request check

The determinism sweep and the result round-trip gate SHALL run in pull-request CI as a
dedicated job that can be configured as a required status check, not as a step inside the
matrix test job and not behind the `integration` marker, so any non-determinism or
serialization regression fails the pull request.

#### Scenario: Determinism regression on a pull request

- **GIVEN** a change makes a stochastic function non-deterministic under a fixed seed
- **WHEN** pull-request CI runs
- **THEN** the dedicated reproducibility-gates job fails

#### Scenario: Serialization regression on a pull request

- **GIVEN** a change breaks JSON round-tripping of a result object
- **WHEN** pull-request CI runs
- **THEN** the dedicated reproducibility-gates job fails

#### Scenario: The gate is an independently requireable check

- **GIVEN** the reproducibility-gates job is defined as its own CI job
- **WHEN** branch protection is configured
- **THEN** the job can be selected as a required status check independent of the matrix tests
