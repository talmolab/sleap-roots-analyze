## Why

The Windows `Tests (windows, Python 3.11)` CI job (`.github/workflows/ci.yml`, `timeout-minutes:
30`) has been running increasingly close to its budget over the past few days and was cancelled
outright on PR #215 — even though `pytest` itself had already completed successfully (2873
passed, 37 skipped) in 28m30s before the job hit the timeout during cleanup (issue #217).

Job duration jumped from a comfortable ~13–22 min (mid-July) to ~27–30 min starting 2026-07-29,
traced to commit `c379c1f` (`fix(#110): OOM in exploratory analysis for large genotype/trait
counts`, PR #210), which added large-dataset regression tests to guard against the OOM bug it
fixed. These tests are appropriately thorough for their purpose but disproportionately expensive:
one test alone (`TestBatchedFigureGenerators::test_boxplot_generator_matches_list_wrapper_output`)
takes ~198s on Windows — the single slowest test in the suite by a wide margin — and a handful of
others in `tests/test_run_all_cli_group_by.py` (~70-77s each) and `tests/test_qc_pipeline.py`
(~51-55s each) add another ~5 minutes combined. The Ubuntu `tests` job is equally close to the
timeout (28m34s in the same run) — this is not Windows-specific, just Windows-visible because it
tipped over first.

Any PR can now receive a misleading "failed" CI signal purely from job-timeout margin
exhaustion, unrelated to its own changes, as happened to PR #215 (a type-annotation-only fix).

## What Changes

- Add a `slow` pytest marker (alongside the existing `integration` marker in
  `pyproject.toml`) for tests whose individual runtime is large enough to meaningfully erode the
  `tests` job's timeout margin (using the ~20 tests identified from the PR #215 CI run's
  `--durations=20` output as the initial concrete set — see `tasks.md` for the exact list).
- Mark those tests `@pytest.mark.slow` (or `pytestmark = pytest.mark.slow` at class level where
  every test in the class qualifies, e.g. `TestRunAllCLIGroupBy`).
- Change the main `tests` job's pytest invocation from `-m "not integration"` to
  `-m "not integration and not slow"`.
- Add a new `slow-tests` CI job that runs `-m "slow"` on the same three-OS matrix
  (ubuntu/windows/mac) as the `tests` job, so slow tests keep running on every PR — nothing loses
  coverage — just isolated into a job whose own budget isn't shared with the fast suite.
- No change to test behavior, fixtures, or assertions — this is a CI-partitioning change only.

## Impact

- Affected specs: `developer-tooling` (ADDED: `Slow Test CI Partitioning` requirement).
- Affected code:
  - `pyproject.toml` (register the `slow` marker)
  - `.github/workflows/ci.yml` (new `slow-tests` job; `tests` job's pytest invocation)
  - `tests/test_visualization.py`, `tests/test_run_all_cli_group_by.py`,
    `tests/test_qc_pipeline.py`, `tests/test_grouped_pipeline_config_persistence.py`,
    `tests/test_step_exploratory_analysis.py`, `tests/test_step_generate_static_figures.py`,
    `tests/test_step_visualize_prediction.py`, `tests/test_viz_pipeline_zero_variance.py`
    (add `@pytest.mark.slow` to the identified slow tests/classes only)
- Affected docs:
  - `docs/CHANGELOG.md` (`[Unreleased]` `### Fixed` entry, see `tasks.md` Task 5.3)
  - `docs/testing.md` (two lines this change makes newly stale: a `uv run pytest -m
    "slow"  # If markers are defined` hedge that becomes false once the marker is
    registered, and a documented coverage command whose `-m "not integration"` filter
    must become `-m "not integration and not slow"` to keep matching what CI actually
    runs — see `tasks.md` Task 5.5)
- Not in scope: reducing the actual runtime of `test_boxplot_generator_matches_list_wrapper_output`
  or any other individual test (a valid separate follow-up if the isolated `slow-tests` job is
  itself later found to be too slow, but isolation alone solves the immediate CI-signal problem);
  changing the `integration` marker's existing scope or CI treatment.
