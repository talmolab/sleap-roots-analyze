## MODIFIED Requirements

### Requirement: Type-Check Gate

CI SHALL run `mypy` against `src/sleap_roots_analyze` on every pull request and filter its
output through a committed baseline file so that pre-existing type errors do not block PRs
while newly introduced type errors do. The gate SHALL run as a dedicated CI job (modeled on
the existing reproducibility/serialization gates) so it can serve as a required status check.
The gate SHALL use the standard `mypy-baseline` tool (no bespoke type-checking scripts). The
mypy configuration SHALL start lenient — targeting the package only, tolerating untyped
third-party imports, with `disallow_untyped_defs` as the single initial ratchet knob.

The gate SHALL also inspect mypy's own process exit code (independent of `mypy-baseline
filter`'s exit code) and fail CI immediately when mypy exits with any code other than `0`
(clean) or `1` (errors reported, its normal exit whenever type errors exist) — for example a
fatal/usage/internal-error exit (`2`). This guard SHALL apply regardless of what
`mypy-baseline filter` reports, so that a crashed mypy invocation cannot produce a false-green
result once the baseline reaches zero recorded errors.

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

#### Scenario: A fatal mypy exit fails CI even when the baseline is empty

- **GIVEN** `.mypy-baseline.txt` currently records zero errors (the ratchet's end state)
- **WHEN** mypy itself crashes or exits fatally (exit code other than `0` or `1`) and emits no
  parseable error output
- **THEN** the `type-check` CI job fails on mypy's own exit code
- **AND** it does so even though `mypy-baseline filter` would otherwise report `new: 0` on the
  empty output and exit `0`

#### Scenario: Normal error-bearing runs are unaffected by the guard

- **GIVEN** mypy exits `0` (no errors) or `1` (errors reported, matching mypy's documented
  convention)
- **WHEN** the `type-check` CI job runs
- **THEN** the guard takes no action beyond confirming the exit code is expected
- **AND** the job's pass/fail outcome is determined by `mypy-baseline filter`'s exit code exactly
  as before this change
