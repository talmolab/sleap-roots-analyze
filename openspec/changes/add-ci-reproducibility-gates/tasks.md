# Tasks: CI Reproducibility Gates

> Implement in an isolated `git worktree` to avoid the concurrent-session working-tree
> contention noted in design.md. Ship as a single PR; commit groups in order (group N
> depends on N-1). **Task 3.x must not land before group 2** — the CI job references
> `tests/test_result_serialization.py` and `pytest` exits non-zero on a missing path.

## 1. Shared registry + whole-package determinism coverage guard

- [ ] 1.1 Extract the determinism case registry from `tests/test_reproducibility.py`
  into `tests/reproducibility_cases.py`, generalized to `(label, call, compare)` cases so
  it can hold both DataFrame-in/dict-out functions and the array-in/tuple-or-scalar-out
  helpers (`pca.fit_pca`, `pca.select_n_components`,
  `pca.perform_pca_with_variance_threshold`, `clustering.calculate_optimal_k_kmeans`).
  Re-import it in `tests/test_reproducibility.py`; existing determinism behavior unchanged.
- [ ] 1.2 Add determinism cases for the four (and any other) module-level stochastic
  helpers the guard discovers, with same-seed reproducibility and `random_state=None`
  smoke coverage.
- [ ] 1.3 Pin coverage against silent drops: assert the registry's label set equals an
  explicit expected set, and that `pytest tests/test_reproducibility.py --collect-only -q`
  still collects the expected number of tests after the extraction + additions.
- [ ] 1.4 Write `test_sweep_covers_all_stochastic_functions`: walk all modules under
  `sleap_roots_analyze` (`pkgutil.walk_packages`), collect module-level functions whose
  `__module__` is in-package and whose signature accepts `random_state`, and assert each
  is in the registry or in a documented `EXCLUDED` set (default empty). Add a consistency
  check that `EXCLUDED` names still exist and don't overlap the registry.
- [ ] 1.5 Write the negative test `test_coverage_guard_detects_missing_function`:
  evaluate the guard's comparison against a synthetic `found` set containing a name absent
  from the registry/exclusion set and assert it is reported uncovered (proves red).
- [ ] 1.6 Confirm `uv run pytest tests/test_reproducibility.py` is green.

## 2. Result-object round-trip gate

- [ ] 2.1 Add `tests/test_result_serialization.py` with a `_json_equal(a, b)` helper
  (NaN-aware, recurses dict/list) and a synthetic dataclass case: fields covering numpy
  scalars, `np.ndarray`, nested dict, and `NaN`, plus a `from_dict` classmethod. Assert
  `projection = convert_to_json_serializable(asdict(obj))` survives `json.dumps`/`loads`
  unchanged and that `from_dict(loaded)` rebuilds an equal object. (Characterization test
  of the existing helper + new harness — no production code changes.)
- [ ] 2.2 Add the lossy-stringification guard: a dataclass field holding a
  non-serializable object projects to a `"<TypeName>"` placeholder, and the gate asserts
  the projection contains no such placeholder → fails loudly instead of passing vacuously.
- [ ] 2.3 Add the real-object case: directly construct a `PipelineSummary` (with a `Path`
  in `files_generated` and a numpy value in metadata) and assert
  `json.loads(summary.to_json())` round-trips its projection unchanged.
- [ ] 2.4 Add the analytical round-trip case list (decoupled from the determinism
  registry): list the stochastic functions *and* the public statistics/heritability
  functions with the fixtures each needs; for each, call once and assert the round-trip
  iff the return `is_dataclass`, else `pytest.skip`. Confirm all currently skip (dict
  returns) so the gate is non-vacuous only via 2.1–2.3 today.
- [ ] 2.5 Confirm the gate is green and that skips are visible under `pytest -rs`.

## 3. Enforce gates as a dedicated PR-CI job

- [ ] 3.1 Add a `reproducibility-gates` **job** to `.github/workflows/ci.yml`
  (single OS, `ubuntu-latest`, own `timeout-minutes`) running
  `uv run pytest tests/test_reproducibility.py tests/test_result_serialization.py`. Do
  **not** add an in-matrix step. Leave the matrix `-m "not integration"` step untouched.
- [ ] 3.2 Note in the PR description that `reproducibility-gates` should be added to
  branch-protection required checks (repo-admin action, not code).

> Integration-on-schedule/label (issue #133's optional item, #69) is **deferred to a
> follow-up change** — not in scope here.

## 4. Documentation & validation

- [ ] 4.1 Update `docs/reproducibility.md` (edit, not just append): reword the existing
  "Determinism guarantee" text to say coverage is now enforced by the whole-package guard;
  make `tests/reproducibility_cases.py` the single source of truth for the function
  inventory (do not add a fourth hand-maintained copy of the list); add a short "CI
  enforcement" note naming the `reproducibility-gates` job.
- [ ] 4.2 Add the **result-object serialization contract** to `docs/reproducibility.md`,
  written for #127–129 authors: return a `@dataclass`; optionally define `from_dict`; the
  gate covers it automatically via `convert_to_json_serializable`; add the function to the
  round-trip case list; plain-dict returns are skipped.
- [ ] 4.3 Update `docs/CONTRIBUTING.md` "Before Submitting" with the local gate command
  and a one-line note that new stochastic functions / result dataclasses are gated.
- [ ] 4.4 Add a `docs/CHANGELOG.md` `[Unreleased] → Added` entry (required by the repo's
  PR process).
- [ ] 4.5 `uv run black --check` + `uv run ruff check` clean on changed files.
- [ ] 4.6 `uv run pytest -m "not integration"` passes (full suite).
- [ ] 4.7 `openspec validate add-ci-reproducibility-gates --strict` passes.
