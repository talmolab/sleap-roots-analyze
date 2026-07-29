# Tasks: fix-post-pca-trait-names-propagation

**Suggested commit grouping**: Tasks 1-3 → one `test(#80):` commit (red);
Task 4 → one `fix(#80):` commit (green); Task 5 → folded into whichever
commit is last, plus a final `docs(#80):` commit for the checklist. (Note:
the `fix-clustering-feature-names-mismatch` precedent this pattern was
originally modeled on actually landed as 6 interleaved commits and was
squash-merged into `main` as a single commit — the fine-grained sequence
matters less than it might seem, since squash-merge collapses it anyway.
What matters is: never leave a red/broken state as the branch tip while the
PR is open for review, and land Tasks 4.1+4.2 together, never split across
commits — see the note in Task 4.) Recent merged PRs in this repo (#185,
#195, #199, #201) all target `main` directly, including release-cut PRs —
this PR should too.

## Task 1: Fixtures
- [x] 1.1 For the `test_step_pca_analysis.py` unit test in Task 2, reuse an
      existing interleaved-trait fixture from `tests/fixtures.py` instead of
      hand-rolling new data — `pca_zero_std_features` (`fixtures.py:1608`,
      alternating normal/zero-std columns) or `pca_constant_feature_data`
      (`fixtures.py:1269`, constant/constant/variable/variable/constant) both
      already interleave constant and variable columns, which is exactly
      what's needed to distinguish "filtered by value" from "filtered by
      trailing-slice coincidence". Used `pca_constant_feature_data`.
- [x] 1.2 For the `test_viz_pipeline_zero_variance.py` pipeline-level CSV
      fixture (Task 3), add a new variant of `csv_with_zero_variance_traits`
      with constant traits interleaved among the variable ones (no existing
      fixture carries the Barcode/Genotype/Replicate columns a full
      `VizPipeline` run needs) — use `viz_constant_trait_data`
      (`fixtures.py:1880`) as precedent for column shape/order, not
      `csv_with_zero_variance_traits`'s current trailing-only order. Added
      `csv_with_interleaved_zero_variance_traits`.

## Task 2: Write failing tests — PCA step metadata contract (TDD Red Phase)
- [x] 2.1 In `test_step_pca_analysis.py`, add a test asserting
      `metadata["trait_names"] == metadata["valid_trait_names"] == pca_results["feature_names"]`
      after running `PCAAnalysisStep` on the interleaved fixture (Task 1.1)
- [x] 2.2 Assert `metadata["original_trait_names"] == trait_cols` (the
      pre-filter list, same order as input)
- [x] 2.3 Add a test for the no-exclusions case (spec scenario "trait_names
      unchanged when nothing is excluded"): using the existing non-zero-variance
      `sample_data` fixture, assert `trait_names == valid_trait_names ==
      original_trait_names == trait_cols`
- [x] 2.4 Run tests, confirm 2.1/2.2 FAIL on current code (`trait_names`
      still equals the unfiltered `trait_cols`, and `original_trait_names`
      doesn't exist yet); confirm 2.3 already PASSES on current code (it's a
      regression guard, not a red test — nothing changes in the
      no-exclusions path). Confirmed exactly as predicted.

## Task 3: Write failing tests — downstream propagation (TDD Red Phase)
- [x] 3.1 In `test_step_umap_analysis.py`, add a test that **actually
      executes** `PCAAnalysisStep` on the interleaved fixture (Task 1.1) and
      feeds its real resulting `StepResult` into `UMAPAnalysisStep` — do NOT
      reuse the file's existing hand-mocked `prev_result` fixture (it
      hard-codes `trait_names` to the full list and would pass both before
      and after the fix, never exercising the bug). Assert the UMAP
      params/logged `n_traits` reflect the filtered trait count, not the
      original.
- [x] 3.2 In `test_viz_pipeline_zero_variance.py`, add a test using the
      interleaved fixture (Task 1.2) with `config.umap.enabled = True`
      asserting `umap_parameters.json`'s `n_traits` equals 4 (filtered), not
      8 (original)
- [x] 3.3 In `test_viz_pipeline_zero_variance.py`, add a test asserting
      `results["09_generate_static_figures"].data.metadata["trait_names"]`
      equals the PCA-filtered set — confirm it currently equals the
      pre-PCA set relayed from `08_genotype_aggregation` instead
- [x] 3.4 There is no `config.pca.enabled` flag in this codebase (unlike
      `umap`/`clustering`/`heritability`) — PCA always executes when
      scheduled, so "PCA disabled" isn't reachable through config. Instead,
      add a focused unit test (in `test_viz_pipeline_zero_variance.py` or a
      new `test_viz_pipeline_orchestrator.py`) that calls
      `VizPipeline._run_generate_static_figures(config, run_dir, logger, **kwargs)`
      directly with `"08_genotype_aggregation"` present but `"03_pca_analysis"`
      omitted from `kwargs` (simulating the DAG executor never producing a
      PCA task result, e.g. after an upstream failure) and
      `config.static_viz.enabled = False` (so the step returns
      `prev_result.metadata` verbatim with no figure generation needed).
      Assert the returned metadata's `trait_names` equals the
      `08_genotype_aggregation` branch's own value, unmodified — this should
      already PASS on current code (guards against the Task 4.2 merge
      accidentally overwriting with a missing value when the PCA task result
      isn't in kwargs). Added `TestGenerateStaticFiguresMetadataMergeGuard`
      in `test_viz_pipeline_zero_variance.py`; confirmed it passes both
      before and after Task 4.
- [x] 3.5 Do not add `@pytest.mark.integration` to any of the new
      `test_viz_pipeline_zero_variance.py` tests — the existing tests in
      that file already run a full 12-step pipeline without that marker and
      execute in CI's default `-m "not integration"` job on all 3 OSes;
      adding the marker would silently skip these new regression tests in CI
- [x] 3.6 Run tests, confirm 3.1-3.3 FAIL on current code and 3.4 PASSES.
      Confirmed exactly as predicted (3.1: n_traits=6 got vs 4 expected;
      3.2: n_traits=8 vs 4; 3.3: full pre-PCA 8-trait list vs the 4-trait
      filtered list; 3.4 passed pre-fix as a true no-op regression guard).

## Task 4: Implement the fix (TDD Green Phase)
- [x] 4.1 `pca_analysis.py` (~L166-174): set `metadata["trait_names"]` and
      `metadata["valid_trait_names"]` to `feature_names`; add
      `metadata["original_trait_names"] = list(trait_cols)`
- [x] 4.2 `viz_pipeline.py` (`_run_generate_static_figures`, ~L427-442):
      extend the PCA-branch cherry-pick block to also copy `trait_names` and
      `original_trait_names` from `pca_step_result.metadata` into
      `combined_metadata`, inside the existing `if pca_task_result:` guard
      (so Task 3.4's PCA-disabled regression stays green)
- [x] 4.3 Commit 4.1 and 4.2 together, never split across commits — Task
      3.3's test stays red with only 4.1 applied (see `design.md`)
- [x] 4.4 Run all Task 2-3 tests, confirm PASS (green). All 8 new tests
      pass; the 2 sanity/regression-guard tests continued passing throughout.
- [x] 4.5 Run the full existing suite for the three affected test files plus
      `test_step_generate_static_figures.py`, confirm no regressions (in
      particular, `test_viz_pipeline_zero_variance.py`'s existing
      trailing-only-fixture tests must still pass unchanged). 115/115 passed.

## Task 5: Verify no regressions
- [x] 5.1 Full test suite: **could not be completed locally.** Four separate
      attempts to run the full ~1939-test suite (and a ~46-file
      `test_step_*.py` subset) in this environment — via background
      execution, foreground execution with an explicit 590-600s timeout, and
      with output redirected to a file — all failed to return usable output
      (empty output, non-pytest exit codes), while the same invocation
      pattern reliably succeeds for smaller/faster runs (confirmed for the
      115-test directly-affected-files run, which completed in ~123s with
      full output). This looks like an environment/tooling limitation on
      long-running commands in this session, not a test failure — CI runs
      this exact suite (`pytest -m "not integration" tests/` per
      `.github/workflows/ci.yml`) across Ubuntu/Windows/macOS with a 30
      minute budget per OS, and is the authoritative full-suite gate for
      this PR. Local verification is scoped to: the three directly-modified
      test files, `test_step_generate_static_figures.py` (115 tests, all
      pass, Task 4.5), ruff, black, and mypy (below) — the actual code
      change is two small, targeted edits (`pca_analysis.py` metadata dict,
      `viz_pipeline.py` cherry-pick block) with no plausible mechanism for
      breaking unrelated modules.
- [x] 5.2 Linting, formatting, and the frozen mypy baseline pass
      (`uv run ruff check`, `uv run black --check`,
      `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt`).
      `ruff check src/sleap_roots_analyze` (CI's actual scope): all checks
      passed. `black --check src/sleap_roots_analyze tests`: clean (after
      running `black` once to reformat the two new/modified test files).
      `ruff check` on the modified test files (not CI-gated, since CI only
      lints `src/`, but fixed anyway to match repo docstring conventions):
      clean. mypy vs. baseline: 0 new errors (375 pre-existing/unresolved,
      unchanged; 0 fixed, 0 new).
- [x] 5.3 Update `docs/CHANGELOG.md` `[Unreleased]` with a `### Fixed` entry.
      No existing #76 entry was found in `docs/CHANGELOG.md` to distinguish
      from (unlike the `fix-clustering-feature-names-mismatch` precedent,
      #74/#76 were never given their own changelog entry), so the new entry
      stands alone; it explains both the `trait_names`/`original_trait_names`
      metadata correction and the `create_pca_biplot` mislabeling
      consequence.
