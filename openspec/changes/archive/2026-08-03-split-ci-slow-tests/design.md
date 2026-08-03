## Context

The `tests` CI job runs the full non-integration suite (~2900 tests) on three OSes under a single
shared `timeout-minutes: 30`. A small number of large-dataset regression tests added by PR #210
(fix #110, OOM in exploratory analysis) account for a disproportionate share of wall-clock time —
roughly 14 of the ~28.5 minutes on Windows come from ~20 tests (`--durations=20` from the PR #215
CI run). This proposal partitions those tests out of the main job.

## Goals / Non-Goals

**Goals:**
- Restore a comfortable timeout margin on the main `tests` job.
- Keep 100% of existing test coverage running on every PR/push — nothing gets skipped.
- Avoid a single slow test class masking or gating the rest of the suite's pass/fail signal.

**Non-Goals:**
- Reducing the actual runtime of any individual slow test (separate future work if needed).
- Introducing test parallelization (`pytest-xdist`) — a larger, riskier change (shared fixtures,
  `tmp_path` isolation assumptions, flaky-test surface) out of scope for this fix.
- Changing the `integration` marker's semantics or CI treatment.

## Decisions

### Decision 1: New `slow` marker + separate CI job, not a higher timeout or xdist

**Alternatives considered:**
- **Just raise `timeout-minutes`** — simplest, but only delays the next collision as the suite
  keeps growing; doesn't address that one test (198s) dominates the tail, and a bigger timeout
  means a genuinely-hung job burns more CI minutes before anyone notices.
- **`pytest-xdist` parallelization** — would cut wall-clock time broadly, but is a bigger change
  with its own risk surface (shared `tmp_path`/fixture state across workers, nondeterministic test
  ordering interacting with existing reproducibility/determinism gates) and doesn't specifically
  target the actual cause (a handful of oversized tests).
- **Delete/shrink the slow tests** — would touch tests written intentionally to catch OOM
  regressions; shrinking their data size risks weakening the exact regression coverage PR #210
  added them for. Left as an explicitly out-of-scope future option in `proposal.md`.

**Chosen: partition by marker into a separate job.** Directly targets the tests actually causing
the problem, preserves their current assertions/data sizes untouched, and requires no changes to
fixtures or test infra — only a marker and a workflow job, mirroring the existing `integration`
marker precedent already in `pyproject.toml`.

### Decision 2: `slow-tests` job runs the full three-OS matrix, not Windows-only

The Windows job was the one that got cancelled, but the CI log from the same run shows Ubuntu
finished at 28m34s — equally close to the 30-minute ceiling. Scoping the new job to Windows-only
would leave Ubuntu one slow addition away from repeating this exact failure. Running `slow-tests`
on all three OSes costs more total CI minutes but keeps the safety margin symmetric across the
matrix, consistent with how the existing `tests` job already runs on all three.

**Implementation note**: the `tests` job's mac leg is pinned to `runs-on: macos-14`, not
`macos-latest` (that's used by the unrelated `numerical-stability` job for a documented arm64/
golden-generation reason). The new job's `include:` block must be copied verbatim — including
`macos-14` — or the two "mirrored" jobs would silently run on different macOS versions.

**`--cov` decision**: the existing `tests` job's pytest invocation includes
`--cov=src/sleap_roots_analyze --cov-report=xml`, but the new `slow-tests` job deliberately
omits `--cov` — coverage upload is currently commented out in `tests` (a dead TODO stub), so
no CI gate depends on it today, and the slow tests' coverage contribution is not currently
counted anywhere. This is a conscious scope decision, not an oversight: if/when coverage upload
is ever re-enabled, whoever does that should also decide then whether `slow-tests` needs `--cov`
so the reported number isn't silently missing ~20 tests' worth of coverage.

### Decision 3: Selection is by measured duration, not by which PR/file introduced them

Rather than marking every test in the specific files touched by PR #210, the concrete list (in
`tasks.md`) is exactly the tests observed above a threshold in the actual `--durations=20` output
from a real CI run. This is deliberately data-driven — the goal is timeout-margin recovery, not
attribution to that PR (`test_qc_pipeline.py`'s slow integration-style tests, for instance,
predate #210 and were already contributing to the margin problem).

### Decision 4: Class-level marker only where every test in the class is slow

`TestRunAllCLIGroupBy` (`tests/test_run_all_cli_group_by.py`) has exactly 4 tests and all 4 are in
the slow set, so it gets a class-level `pytestmark = pytest.mark.slow`. Every other affected file
has a mix of fast and slow tests in the same class (e.g. `TestBatchedFigureGenerators` has 3 tests,
only 1 of which is slow) — those get per-test `@pytest.mark.slow` decorators so fast tests in the
same class stay in the main job.

## Risks / Trade-offs

- The `slow-tests` job adds a new required-status-check surface; if it's flaky or slow to start,
  it could itself become a merge bottleneck. Mitigated by keeping its own `timeout-minutes: 30` (a
  full budget, since it now only carries ~15 min of tests) and `fail-fast: false`, matching the
  existing `tests` job's pattern.
- Total CI minutes consumed per PR increases slightly (a new job across 3 OSes), trading CI cost
  for a lower risk of a misleading spurious-timeout "failure" gating unrelated PRs.
- If the test suite keeps growing, the `slow` set will need periodic re-review — this is an
  accepted ongoing maintenance cost, not a one-time fix; not automated by this change.
- **Class-level marker forward-drift**: `TestRunAllCLIGroupBy` and
  `TestGroupedPipelineConfigPersistence` get `pytestmark = pytest.mark.slow` at the class level
  because every current test in each qualifies. A future *fast* test added to either class would
  silently inherit `slow` too — pytest has no per-test override for a class-level `pytestmark`.
  This wouldn't drop coverage (the test would still run, just in `slow-tests` instead of `tests`),
  but it would quietly slow down `slow-tests` for no reason. Mitigation is process, not code: a
  comment next to each `pytestmark` line flags that a new fast test belongs in a separate class/
  module instead.
- **Marker double-tagging**: no test today is marked both `slow` and `integration`, and CI
  currently runs no job selecting `-m "integration"` at all (a separate, tracked gap — issue #69).
  If a test is ever marked both, it would newly start executing in CI (inside `slow-tests`, since
  `-m "slow"` catches it) — surprising if that test was excluded from `integration` specifically
  because it needs resources unavailable in CI runners. Task 4's overlap check
  (`-m "slow and integration"` expecting 0 results) catches this at review time for this PR and
  should be re-run if a contributor ever considers dual-marking a test.

## Migration Plan

Additive only: register the marker, tag the identified tests, add the new job, narrow the
existing job's `-m` filter. No test removal, no behavior change, no config schema change. Safe to
land in a single PR.

## Open Questions

None — this is a self-contained CI/test-infra change with no external dependencies.
