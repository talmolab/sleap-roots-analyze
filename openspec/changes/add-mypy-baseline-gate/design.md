## Context

CI runs Black, Ruff, a determinism sweep, a serialization round-trip, and the test matrix
(`.github/workflows/ci.yml`). No static type checking exists. The codebase leans on
numpy/pandas/sklearn/statsmodels, which ship incomplete or absent stubs — a from-scratch
`mypy --strict` run would produce hundreds of errors and block every PR. Issue #132 asks for
the pragmatic middle: freeze the debt, block only new untyped public code, ratchet later, and
use **standard, documented** tooling (no bespoke scripts) so it survives maintainer handoff.

## Goals / Non-Goals

- Goals
  - Run mypy in CI against a committed baseline so existing errors do not block PRs.
  - Fail CI when changed/new code introduces type errors not in the baseline.
  - Start lenient with one ratchet knob (`disallow_untyped_defs`) on the package only.
  - Document the ratchet so a contributor knows how to react when CI flags them.
- Non-Goals
  - `mypy --strict` across the whole tree (future ratchet follow-ups).
  - Typing the test suite (CI checks `src/sleap_roots_analyze` only).
  - Writing custom baseline/diff scripts.

## Decisions

- **Decision: Use `mypy-baseline` to freeze existing debt.**
  mypy has no native baseline. `mypy-baseline` (PyPI, actively maintained) reads mypy output,
  stores known errors in `.mypy-baseline.txt`, and in CI filters them out so only new errors
  surface. Workflow:
  - Generate/refresh: `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline sync --baseline-path .mypy-baseline.txt`
  - CI gate: `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt`
    (`filter` exits non-zero on any unmatched/new error; `--baseline-path` targets the
    dot-prefixed file rather than mypy-baseline's no-dot default).
  - Alternatives considered:
    - *Per-module `[[tool.mypy.overrides]]` ignores* — pure-mypy, but freezing is coarse
      (whole modules go silent, so new errors inside an ignored module are missed; defeats the
      improve-on-touch goal).
    - *Hand-rolled diff script* — explicitly ruled out by the issue ("no bespoke tooling").
    - *Blanket `# type: ignore` sweep* — pollutes source and hides new errors at touched
      lines.
  `mypy-baseline` is the standard tool for exactly this pattern and keeps source clean.

- **Decision: Dedicated `type-check` CI job.**
  Mirrors the existing `reproducibility-gates` / `serialization-gate` pattern (own job, single
  OS, can be made a required status check in branch protection). Type checking is
  OS-independent, so `ubuntu-latest` only.

- **Decision: Lenient config, one ratchet knob.**
  `[tool.mypy]` targets the package, sets `python_version = "3.11"`,
  `ignore_missing_imports = true` (third-party libs lack stubs),
  `warn_unused_ignores = false` initially, and `disallow_untyped_defs = true` as the first
  ratchet. Existing untyped defs land in the baseline; *new* untyped public defs produce
  errors absent from the baseline and fail CI. Future follow-ups tighten incrementally
  (`disallow_incomplete_defs`, `check_untyped_defs`, dropping `ignore_missing_imports` per
  library) — each its own small PR with a baseline refresh.

- **Decision: Baseline regeneration is a documented, intentional act.**
  A drifting baseline (debt paid down but file not refreshed) makes `filter` flag resolved
  errors. CONTRIBUTING tells contributors to run `mypy-baseline sync` and commit the smaller
  baseline when they fix existing debt — the ratchet only tightens.

## Risks / Trade-offs

- **Baseline staleness/merge conflicts** → `.mypy-baseline.txt` is line-oriented and may
  conflict on busy branches. Mitigation: regenerate with `sync` rather than hand-merging;
  document it.
- **`ignore_missing_imports` hides real third-party-shaped bugs** → accepted for launch; a
  follow-up can enable stubs (`pandas-stubs`, `types-*`) per library and ratchet this off.
- **Contributor confusion when CI flags a baseline mismatch** → mitigated by the CONTRIBUTING
  note and the exact local command to reproduce.
- **Crash-safety becomes a hole at zero debt** → today a mypy crash that emits nothing makes the
  329/447 baselined errors vanish, which `filter` counts as new/unresolved and fails CI (verified).
  That safety relies on the baseline being non-empty; once it is paid down to zero, a silent crash
  yields `new: 0` → exit 0 → false green. **Tracked follow-up** (not this change): capture mypy's
  own exit code and fail on a fatal/internal-error (2) before piping to `filter`.

## Migration Plan

1. Add deps + `[tool.mypy]` config.
2. Run mypy once, `sync` the baseline, commit `.mypy-baseline.txt`.
3. Add the `type-check` CI job.
4. Document in CONTRIBUTING.
5. (Maintainer, out of scope) optionally add `type-check` to required status checks in branch
   protection.

Rollback: delete the job, the baseline file, and the config block; remove the deps.

## Open Questions

- ~~Should the gate also be wired into the `/run-ci-locally` and `/lint` slash commands now?~~
  Resolved: **follow-up** — this change stays scoped to CI + baseline + docs as the issue
  specifies.
