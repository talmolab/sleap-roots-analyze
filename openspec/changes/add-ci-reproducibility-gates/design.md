# Design: CI Reproducibility Gates

> Revised after the `review-openspec` subagent panel. Key changes from the first draft:
> the coverage guard now walks the **whole package** (not `dir(sra)`); the round-trip
> gate's discovery set is **decoupled** from the stochastic registry so it covers
> heritability/statistics result types; pipeline summaries are now **in** scope as the
> real-object teeth; CI enforcement is a **separate job**, not an in-matrix step.

## Decision 1 — A shared determinism registry, generalized to per-case callables

`tests/test_reproducibility.py` holds a `_CASES` literal: `(label, callable,
kwargs_builder, [(key, mode), ...])` for eight functions whose first arg is a DataFrame
and whose return is a dict. Extract it into a plain module `tests/reproducibility_cases.py`.

It must **generalize** to the stochastic helpers the whole-package guard now requires:
`fit_pca(X: ndarray, n_components, random_state) -> (PCA, ndarray)`,
`select_n_components(X: ndarray, ...)`, `perform_pca_with_variance_threshold(X: ndarray, ...)`,
`calculate_optimal_k_kmeans(data, ...)`. These take arrays (not DataFrames) and return
tuples/ints (not dicts). So a case becomes `(label, call, compare)` where `call(ctx)`
produces the result from shared fixtures and `compare(a, b)` asserts run-to-run equality
(dict-key comparison for dict returns; positional comparison for tuples; exact for ints).
This keeps one registry without forcing every function into a dict-shaped contract.

Why a shared registry: the coverage guard polices a single authoritative list, and the
determinism test parametrizes off it. (The round-trip gate uses a *separate* list — see
Decision 3 — because its membership question is different.)

## Decision 2 — Determinism coverage is enforced by whole-package introspection

Add `test_sweep_covers_all_stochastic_functions`. **Walk every module** under
`sleap_roots_analyze` (via `pkgutil.walk_packages`), collect module-level functions whose
signature contains `random_state` and whose `__module__` starts with the package name,
and assert each is covered by the registry — *or* is in an explicit, documented
`EXCLUDED` set. The `dir(sra)` approach from the first draft is rejected: it only sees
top-level re-exports and silently misses submodule helpers (verified: four such helpers
exist today), defeating the "self-enforcing" guarantee.

```python
found = {
    f"{fn.__module__}.{fn.__name__}"
    for _, mod, _ in pkgutil.walk_packages(sra.__path__, sra.__name__ + ".")
    for _, fn in inspect.getmembers(import_module(mod), inspect.isfunction)
    if fn.__module__.startswith("sra") and "random_state" in inspect.signature(fn).parameters
}
uncovered = found - covered_labels - EXCLUDED
assert not uncovered, f"stochastic functions missing a determinism case: {uncovered}"
```

`EXCLUDED` (if any) must be a named constant with a comment per entry, and a test asserts
`EXCLUDED` and the registry are disjoint and that every `EXCLUDED` name still exists
(so the exclusion can't rot into a typo that hides a real gap). Per the approved scope
decision, the default is **cover everything** — `EXCLUDED` starts empty; helpers get real
determinism cases.

### Negative test (the guard must be able to go red)

Add `test_coverage_guard_detects_missing_function`: build the guard's comparison against a
`found` set containing a synthetic `"sra.fake.fake_fn"` and assert the guard reports it as
uncovered. Without this the guard ships born-green and never proves it can fail — the only
behavior that matters.

## Decision 3 — Round-trip discovery is decoupled from the stochastic registry

The first draft drove the round-trip gate off the stochastic registry. Rejected: that set
excludes heritability/statistics functions, so #128 `HeritabilityResult` — named as a
target — would never be tested while the spec promised it was. Instead the round-trip gate
iterates its **own** case list spanning the result-bearing analytical surface (stochastic
functions *and* the public statistics/heritability functions), each with the fixtures it
needs. For each case it calls the function once and branches on the **actual return**:

- `dataclasses.is_dataclass(result)` → assert the round-trip (see Decision 4 for the exact
  equality definition), and if the class defines `from_dict`, that `from_dict(loaded)`
  rebuilds an equal object.
- otherwise (plain dict, the status quo) → `pytest.skip(...)`, recorded as skipped so the
  no-op state is visible under `-rs`.

This is opt-in by *return type* (no production base class / decorator needed — that's #130's
to design) and auto-extends: the instant a listed function returns a dataclass, its
assertion activates with no test edit. Adding a *new* analytical function to the list is
the one hand-maintained step; a companion note in the docs (the serialization contract)
tells #127–129 authors to do so.

## Decision 4 — "Lossless" is defined on the JSON-native projection (and must fail on lossy stringification)

`convert_to_json_serializable` (in `data_utils.py`) is deliberately **asymmetric**:
`ndarray`→`list`, `np.floating`→`float`, and **unknown objects → `"<TypeName>"` string**.
So "round-trips losslessly" cannot mean type-identical reconstruction. Define it as:

```python
projected = convert_to_json_serializable(asdict(result))
assert _json_equal(json.loads(json.dumps(projected)), projected)   # NaN-aware
```

i.e. the projection survives a JSON encode/decode unchanged. Two correctness guards on top:

- **NaN:** `json.dumps(float('nan'))` emits the non-standard token `NaN` and `json.loads`
  reads it back, but `nan == nan` is `False` — so `_json_equal` must compare floats with
  `math.isnan`/`np.isnan`, not `==`. (A stricter consumer rejecting non-standard JSON is a
  documented interoperability caveat, not something this gate forces.)
- **Lossy stringification:** if the projection contains a `"<...>"` placeholder string for
  a field that was supposed to be data, that is a serialization *failure*, not a pass. The
  gate asserts no field projects to a `"<TypeName>"` placeholder, so a result holding e.g.
  a raw sklearn estimator fails loudly instead of round-tripping vacuously.

## Decision 5 — Pipeline summaries are the real-object teeth (in scope)

Reversed from the first draft. `PipelineSummary` (`pipeline/summary.py`) is a shipped
serializable dataclass with `to_json()`/`to_dict()`. It can be constructed **directly** —
a `PipelineSummary` with one or two hand-built `StepSummary` entries (including a `Path` in
`files_generated` and a numpy value in metadata); no pipeline run is needed. Round-tripping
it gives the gate teeth on **real, shipped code today**, not only a synthetic stub. The
synthetic dataclass (Decision 6) remains, additionally, to exercise the `from_dict` branch.

## Decision 6 — A synthetic dataclass exercises the full contract, including `from_dict`

No result type defines `from_dict` yet, so that asserted branch would ship dead. Give the
synthetic dataclass a `from_dict` classmethod and fields covering numpy scalars, `ndarray`,
nested dict, and `NaN`, so the gate executes the `from_dict` round-trip and the NaN /
projection logic at least once. Note: this is a **characterization test** of the existing
`convert_to_json_serializable` plus the new gate harness — not red-green TDD of new
production code (there is none); the tasks label it as such.

## Decision 7 — CI enforcement is a separate job, not an in-matrix step

Reversed from the first draft. A *step* inside the matrix `tests` job **cannot be a
required status check** in branch protection (GitHub requires *jobs*), so an in-matrix
"Reproducibility gates" step could never be the "enforcing, clearly named check" the Why
asks for. It also wouldn't fail fast: placed after `uv sync`, it saves no install cost and
merely re-runs the two files a redundant time on every OS. Instead add a standalone job:

```yaml
  reproducibility-gates:
    name: Reproducibility gates
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with: { enable-cache: false }
      - run: uv python install 3.11
      - run: uv sync --group dev
      - run: uv run pytest tests/test_reproducibility.py tests/test_result_serialization.py
```

Single OS suffices (determinism is same-machine by Decision 1; serialization is
OS-independent). It runs in parallel with the matrix, is independently requireable, and the
matrix `-m "not integration"` step still covers these files for coverage — we do **not**
also add an in-matrix step (no quadruple execution).

## Decision 8 — Integration job deferred to a follow-up

Issue #133's third acceptance item (run the skipped integration suite on
`schedule`/`workflow_dispatch`/`run-integration`-label) is explicitly optional and
**deferred to a separate change**. Intended shape: a standalone `integration.yml` running
`uv run pytest -m integration`, kept out of the matrix so PR-CI latency is unchanged. The
matrix `-m "not integration"` filter is left untouched by this change.

## Operational note — working-tree contention

A concurrent agent session has been committing other branches in this same checkout and
once stashed this branch's files mid-work. Implement this change in an isolated
`git worktree` (`git worktree add ../sra-ci-repro add-ci-reproducibility-gates`) so the two
sessions cannot clobber each other's working tree. Commit at every pause; never leave
untracked files floating (the other session's `stash --include-untracked` will sweep them).

## Risks

- **Cross-platform float drift.** The determinism gate compares two runs on the *same*
  machine (bit-identical), so the single-OS job and the matrix both pose no risk here;
  cross-platform tolerance (`rtol=1e-6`) is the golden-fixture concern, already documented.
- **`random_state=None` smoke path** stays in the determinism test; the coverage guard
  does not touch it. Helpers added to the registry need a `random_state=None` smoke case
  too (or a documented reason they are excluded from that check).
- **Registry generalization** (Decision 1) touches the existing, passing determinism test.
  A case-count pin and a collected-test-count check (tasks) guard against silently dropping
  coverage during the refactor.
