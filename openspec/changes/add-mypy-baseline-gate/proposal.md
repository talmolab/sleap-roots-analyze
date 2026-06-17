## Why

There is no static type checking in CI. Full `mypy --strict` on a numpy/pandas/sklearn
codebase would red-wall every PR on day one, but type quality directly enables the
introspection-readiness work (#117) and the result-object typing (#130). The middle path:
**freeze existing type debt in a committed baseline, fail CI only on new type errors, and
ratchet strictness later.** This teaches "improve-on-touch" hygiene without drowning a new
contributor in pre-existing errors. (Issue #132.)

## What Changes

- Add `mypy` and `mypy-baseline` to the `dev` dependency group.
- Add a `[tool.mypy]` configuration block to `pyproject.toml` that targets
  `src/sleap_roots_analyze`, tolerates untyped third-party libraries (numpy/pandas/sklearn/
  etc.), and turns on a single ratchet knob to start: `disallow_untyped_defs = true`.
- Commit a baseline file (`.mypy-baseline.txt`) that records every current error so existing
  debt does not block PRs.
- Add a dedicated `type-check` CI job to `.github/workflows/ci.yml` that runs mypy, filters
  its output through the committed baseline, and fails only on **new** errors (or on a stale
  baseline that no longer matches).
- Document the ratchet in `docs/CONTRIBUTING.md`: how to run mypy locally, what "frozen
  baseline" means, how to regenerate it when debt is paid down, and that new public defs must
  be typed.

No bespoke type-checking scripts: `mypy-baseline` is a standard, maintained PyPI tool, so the
gate survives maintainer handoff.

## Impact

- Affected specs: `developer-tooling` (ADDED: Type-Check Gate, Type-Check Ratchet
  Documentation)
- Affected code:
  - `pyproject.toml` — `dev` group + `[tool.mypy]` config
  - `.github/workflows/ci.yml` — new `type-check` job
  - `.mypy-baseline.txt` — new committed baseline (frozen existing debt)
  - `docs/CONTRIBUTING.md` — ratchet note
- Non-breaking: existing errors are frozen; only newly introduced untyped public code fails.
