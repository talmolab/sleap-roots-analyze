## Why

The mypy frozen-baseline gate (#158, closes #132) runs `mypy | mypy-baseline filter` with
`set +o pipefail`, so the job's pass/fail comes entirely from `filter`'s exit code and mypy's own
exit code is discarded. That's safe today only because the baseline is non-empty: if mypy crashes
(bad config, a broken import, an internal error — exit 2, little or no stdout), the baselined
errors vanish from its output, `filter` sees them as "resolved", and CI correctly fails. But once
the baseline is paid down to zero — the explicit end-goal of the ratchet — a crashed mypy that
emits nothing also yields zero errors, `filter` sees `new: 0`, and CI passes: a false green with
type checking silently disabled. This was called out as a known, unticketed hole in #158's
`design.md` ("Crash-safety becomes a hole at zero debt"); #160 is that follow-up.

## What Changes

- In the `type-check` CI job (`.github/workflows/ci.yml`), capture mypy's own exit code via
  bash's `PIPESTATUS` (rather than only `mypy-baseline filter`'s exit code) and fail the job
  immediately — regardless of what `filter` reports — when mypy exits with anything other than
  `0` (clean) or `1` (errors found, mypy's normal exit when errors exist). Any other code (e.g.
  `2`, mypy's fatal/usage/internal-error exit) is treated as a crash, logged clearly, and fails
  CI.
- No change to behavior while the baseline is non-empty: a normal error-bearing run still exits
  `0`/`1` from mypy and defers to `filter`'s baseline diff exactly as before.
- Document the guard in `docs/CONTRIBUTING.md`'s existing mypy-ratchet section so a contributor
  seeing an unfamiliar CI failure mode ("mypy crash-guard tripped") understands it's distinct
  from a normal baseline diff failure.

No bespoke exit-code-parsing script beyond the guard itself: this stays a few lines of bash in
the existing job step, using bash's built-in `PIPESTATUS`, not a new tool or dependency.

## Impact

- Affected specs: `developer-tooling` (MODIFIED: Type-Check Gate)
- Affected code:
  - `.github/workflows/ci.yml` — extend the `type-check` job's mypy step with the exit-code guard
  - `docs/CONTRIBUTING.md` — note the crash-guard behavior in the mypy ratchet section
- Non-breaking: behavior is unchanged for every case CI exercises today (clean run, baseline-covered
  errors, new untyped def). The only newly-caught case is a fatal/internal mypy exit, which
  currently doesn't reliably fail CI once the baseline is empty.
