## ADDED Requirements

### Requirement: Slow Test CI Partitioning

Tests whose individual runtime meaningfully erodes the `tests` CI job's `timeout-minutes` margin SHALL be marked with a registered `slow` pytest marker (in `pyproject.toml`, alongside the existing `integration` marker) and SHALL run in a dedicated `slow-tests` CI job on the same three-OS matrix as `tests`, rather than inside the main `tests` job. The main `tests` job SHALL exclude them via `-m "not integration and not slow"`. No test coverage SHALL be dropped by this partitioning — every `slow`-marked test SHALL still run on every pull request.

#### Scenario: Slow test runs in its own job, not the main tests job

- **GIVEN** a test is decorated with `@pytest.mark.slow` (or carries `pytestmark =
  pytest.mark.slow` at class level)
- **WHEN** CI runs on a pull request
- **THEN** the `tests` job's pytest invocation (`-m "not integration and not slow"`) SHALL NOT
  execute that test
- **AND** the `slow-tests` job's pytest invocation (`-m "slow"`) SHALL execute it on all three
  matrix OSes (ubuntu, windows, mac)

#### Scenario: Main tests job recovers timeout margin

- **GIVEN** the `slow`-marked tests are excluded from the `tests` job
- **WHEN** the `tests` job runs on Windows (the tightest-margin OS, ~28.5-minute pre-partition
  baseline against a 30-minute `timeout-minutes` budget)
- **THEN** its wall-clock duration SHALL be at least 5 minutes below that baseline, restoring at
  least a 6-minute margin under the 30-minute budget

#### Scenario: A new large-dataset regression test is added later

- **GIVEN** a contributor adds a new test whose runtime is large enough to be a `slow` candidate
  (e.g. a future OOM/large-dataset regression test, following the precedent of PR #210)
- **WHEN** they want it included in the partitioned slow suite instead of the main `tests` job
- **THEN** marking it `@pytest.mark.slow` is sufficient — no CI workflow changes are needed, since
  the `slow-tests` job already selects on the marker
