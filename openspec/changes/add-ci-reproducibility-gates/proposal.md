# Proposal: CI Reproducibility Gates

## Why

Tracked by issue #133. The wheat EDPIE golden tests and bloom-mcp delegation depend
on two FAIR guarantees that nothing currently *enforces* as an auto-extending gate:

1. **Determinism** — every stochastic function returns identical output for a fixed
   `random_state`.
2. **Interoperability** — every analytical result object serializes cleanly and
   round-trips through JSON.

Where things stand after auditing the repo:

- The determinism *test* already exists (`tests/test_reproducibility.py`, merged in
  #141 / #118) and already runs in PR CI because it carries no `integration` marker.
  But its case list is a hand-maintained literal covering only the eight *top-level*
  functions, and it omits stochastic module-level helpers that also take a
  `random_state` (e.g. `pca.fit_pca`, `pca.select_n_components`,
  `pca.perform_pca_with_variance_threshold`, `clustering.calculate_optimal_k_kmeans`).
  A new stochastic function — top-level or helper — can be added with no test and CI
  stays green. The guarantee is partial and not *self-enforcing*.
- There is **no result-object round-trip gate**. The serializable dataclass result
  types it is meant to cover (#127 `PCAResult`, #128 `HeritabilityResult`,
  #129 `ClusterResult`, epic #130) have **not landed yet** — the analytical functions
  still return plain dicts. So the gate must be *opt-in by construction*: a no-op for
  functions that haven't adopted a result object, automatically asserting on each one
  the moment it returns a dataclass, with no test edits.
- Integration tests are **100% skipped in PR CI** (`pytest -m "not integration"`,
  issue #69), so the slowest reproducibility checks are silently untested.

This change closes the gaps: it makes the determinism sweep self-enforcing across the
*whole package*, adds the round-trip gate machinery (with teeth on a real object
today), and wires both as enforcing PR-CI checks.

## What Changes

1. **Self-enforcing determinism sweep (whole-package).** Extract the stochastic-function
   case registry into a shared test module, and add a **coverage guard** that walks every
   module in `sleap_roots_analyze` for module-level functions accepting `random_state`
   and fails if any is absent from the registry (with an explicit, asserted exclusion set
   for any function intentionally covered only transitively). This catches submodule
   helpers, not just top-level exports. The registry generalizes to per-case callables so
   helpers with array inputs / tuple or scalar returns (e.g. `fit_pca`) are covered too.
   A **negative test** proves the guard goes red when a function is missing.

2. **Result-object round-trip gate (`tests/test_result_serialization.py`).** Discovery is
   **decoupled from the determinism registry** so it is not limited to stochastic
   functions: it iterates a result-serialization case list spanning the analytical
   surface — the stochastic functions *and* the statistics/heritability functions — so
   #127/#128/#129 result types are all covered when they land. For each case it calls the
   function once and, *only if* the return is a dataclass, asserts the JSON round-trip;
   plain-dict returns are skipped (opt-in, recorded as skips). "Round-trip" is defined on
   the **JSON-native projection** (`convert_to_json_serializable(asdict(result))` →
   `json.dumps` → `json.loads` compares equal, NaN-aware), not type-identical
   reconstruction; a `from_dict` classmethod, when defined, must rebuild an equal object.
   The gate has teeth **today** via two real cases: a hand-built `PipelineSummary`
   (a shipped serializable dataclass) and a synthetic dataclass exercising numpy scalars,
   `ndarray`, nested dict, NaN, and a `from_dict` round-trip.

3. **Enforce both gates as a required PR-CI check.** Add a dedicated single-OS
   `reproducibility-gates` **job** (not a step inside the matrix) to `ci.yml` running the
   two gate files. A job can be made a required status check in branch protection and
   runs in parallel with the matrix; the determinism comparison is same-machine, so one
   OS suffices. The existing matrix `-m "not integration"` step is left untouched.

4. **Document** the gates: extend `docs/reproducibility.md` (coverage-guard model, CI
   enforcement, single-source-of-truth for the function inventory), add the
   **result-object serialization contract for #127–129 authors**, add a local-gate note
   to `docs/CONTRIBUTING.md`, and add a `docs/CHANGELOG.md` `[Unreleased]` entry.

The optional integration-on-schedule/label job (issue #133's third, explicitly optional
acceptance item, tied to #69) is **deferred to a follow-up change** to keep this PR
focused on the two core gates.

## Impact

- **Affected specs:** `reproducibility-gates` (new capability).
- **Affected code & docs:**
  - `tests/reproducibility_cases.py` (new — shared determinism case registry)
  - `tests/test_reproducibility.py` (import shared registry; whole-package coverage
    guard + negative guard test; pin case count)
  - `tests/test_result_serialization.py` (new — round-trip gate, real + synthetic cases)
  - `.github/workflows/ci.yml` (new `reproducibility-gates` job)
  - `docs/reproducibility.md`, `docs/CONTRIBUTING.md`, `docs/CHANGELOG.md` (update)
- **No production code change** unless the whole-package guard surfaces a stochastic
  function not yet in the registry — if so, the minimal fix is adding its determinism
  case (no behavior change).
- **No behavior change** to any analysis function.

## Notes / out of scope

- This change does **not** define the result dataclasses themselves — that is the
  #127/#128/#129/#130 work the round-trip gate is built to receive. It does verify
  serialization on the one shipped dataclass that already serializes (`PipelineSummary`).
- "Lossless round-trip" is defined on the JSON-native projection: `convert_to_json_serializable`
  is deliberately asymmetric (`ndarray`→list, unknown objects→`"<Type>"` string), so the
  gate compares the projected form and **fails on lossy stringification** of an
  unexpected object rather than passing vacuously. NaN is compared NaN-aware.
- The **integration-on-schedule/label job is deferred** to a follow-up change (issue
  #133's optional item, tied to #69).
- The merged `audit-stochastic-determinism` change is complete but **not yet archived**
  into `openspec/specs/`. Pre-existing housekeeping, tracked separately; this proposal
  introduces a distinct `reproducibility-gates` capability and does not depend on it.
