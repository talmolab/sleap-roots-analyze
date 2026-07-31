## Context

`.github/workflows/ci.yml`'s `type-check` job runs:

```bash
set +o pipefail
uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt
```

`pipefail` is disabled so the step's exit code is `filter`'s, not mypy's — mypy always exits
non-zero while the baseline holds errors, so without this the job would never pass. The trade-off,
flagged as a tracked-but-unticketed risk in #158's design.md, is that mypy's own exit code is now
invisible to the job: whatever mypy does, `filter` only ever sees stdout. A crash that prints
nothing is indistinguishable, to `filter`, from a clean run — and once the baseline is empty
(`new: 0` either way), the job can't tell them apart.

## Goals / Non-Goals

- Goals
  - Fail CI on a fatal/internal mypy exit (`2`), independent of the baseline's current size.
  - Leave every currently-passing/failing case (clean run, baseline-covered errors, new untyped
    def) exactly as it behaves today.
  - Keep the fix inside the existing job step; no new dependency, script, or job.
- Non-Goals
  - Auditing `mypy-baseline` itself for correctness — it's a maintained third-party tool.
  - Distinguishing *which* fatal error occurred (config vs. crash vs. OOM) beyond surfacing mypy's
    raw stderr/exit code in the log.

## Decisions

- **Decision: read mypy's exit code from bash's `PIPESTATUS` array, not a separate invocation.**
  The step already runs `shell: bash` and already disables `pipefail`, so `${PIPESTATUS[0]}`
  (mypy) and `${PIPESTATUS[1]}` (`mypy-baseline filter`) are available immediately after the
  pipeline runs, with no extra process. Fail with a clear `::error::` log line when
  `PIPESTATUS[0]` is not `0` or `1`; otherwise propagate `PIPESTATUS[1]` as the step's exit code,
  preserving today's behavior exactly.
  - Alternative considered: redirect mypy to a temp file, check `$?`, then `cat` the file into
    `filter`. Equivalent result, but adds a throwaway file and an extra step for no behavioral
    gain over `PIPESTATUS`, which the job's shell already supports.
  - Alternative considered: re-run `mypy` a second time without the pipe just to get its exit
    code. Doubles mypy's runtime for no benefit — `PIPESTATUS` gets the same information from the
    single invocation already happening.
- **Decision: treat exactly `{0, 1}` as "mypy ran"; anything else fails immediately.**
  mypy's documented convention is `0` = no errors, `1` = errors reported, and non-`0`/`1` (in
  practice `2`) for fatal/usage errors and unexpected crashes. Matching on the two expected codes
  (rather than denylisting `2` specifically) also catches other unexpected exits (e.g. a signal
  kill) without needing to enumerate them.

## Risks / Trade-offs

- **A legitimate future mypy exit code outside `{0, 1}` could false-positive.** Unlikely — this is
  mypy's own documented contract — but if it ever happens, the fix is a one-line change to the
  allowed set, not a redesign.
- **`PIPESTATUS` is bash-specific**, not POSIX `sh`. The step already declares `shell: bash`
  explicitly (see the existing step), so this is already assumed, not newly introduced.

## Migration Plan

1. Add the `PIPESTATUS` check to the `type-check` job's mypy step.
2. Confirm the three existing cases (clean tree, baseline-covered errors, new untyped def) still
   pass/fail exactly as before.
3. Confirm a simulated fatal exit (e.g. temporarily point mypy at a bogus `--config-file`) fails
   the job even when piped through a `filter` that would otherwise report `new: 0`.
4. Document the guard in `docs/CONTRIBUTING.md`.

Rollback: revert the step to the current two-line `set +o pipefail` form.
