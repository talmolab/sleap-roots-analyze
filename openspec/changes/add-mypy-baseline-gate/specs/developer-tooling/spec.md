## ADDED Requirements

### Requirement: Type-Check Gate

CI SHALL run `mypy` against `src/sleap_roots_analyze` on every pull request and filter its
output through a committed baseline file so that pre-existing type errors do not block PRs
while newly introduced type errors do. The gate SHALL run as a dedicated CI job (modeled on
the existing reproducibility/serialization gates) so it can serve as a required status check.
The gate SHALL use the standard `mypy-baseline` tool (no bespoke type-checking scripts). The
mypy configuration SHALL start lenient — targeting the package only, tolerating untyped
third-party imports, with `disallow_untyped_defs` as the single initial ratchet knob.

#### Scenario: Pre-existing type debt does not block a PR

- **GIVEN** the committed `.mypy-baseline.txt` records the current set of mypy errors
- **WHEN** a PR is opened that does not introduce new type errors
- **THEN** the `type-check` CI job runs `mypy src/sleap_roots_analyze` piped through
  `mypy-baseline filter`
- **AND** the job passes because every reported error matches the baseline

#### Scenario: A new untyped def fails the gate

- **GIVEN** a PR adds a function (public or private) without type annotations in `src/sleap_roots_analyze`
- **WHEN** the `type-check` CI job runs
- **THEN** mypy reports a `disallow_untyped_defs` error that is absent from the baseline
- **AND** `mypy-baseline filter` exits non-zero, failing CI
- **AND** adding the missing annotations makes the job pass without editing the baseline

#### Scenario: Baseline regenerates when existing debt is paid down

- **GIVEN** a contributor fixes a pre-existing type error that is recorded in the baseline
- **WHEN** they run `mypy src/sleap_roots_analyze | mypy-baseline sync` and commit the updated
  `.mypy-baseline.txt`
- **THEN** the baseline shrinks to reflect the resolved error
- **AND** the gate continues to pass, having ratcheted tighter

### Requirement: Type-Check Ratchet Documentation

`docs/CONTRIBUTING.md` SHALL document the mypy ratchet in at least one paragraph, covering: the
command to run mypy locally, what the frozen baseline means, the expectation that new
definitions (public or private) are typed, and how to regenerate the baseline when existing debt
is resolved.

#### Scenario: Contributor learns the ratchet from CONTRIBUTING

- **WHEN** a contributor reads `docs/CONTRIBUTING.md`
- **THEN** they find the local mypy command and an explanation that pre-existing errors are
  frozen in `.mypy-baseline.txt`
- **AND** they learn that new defs (public or private) must be typed or CI fails
- **AND** they learn to regenerate the baseline with `mypy-baseline sync` when they fix
  existing debt
