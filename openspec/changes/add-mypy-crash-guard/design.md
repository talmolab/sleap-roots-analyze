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

- **Decision: read mypy's exit code from bash's `PIPESTATUS` array, captured atomically in a
  single statement, not a separate invocation.**
  The step already runs `shell: bash` and already disables `pipefail`. `PIPESTATUS` reflects only
  the *most recently executed* pipeline, so it must be copied into a plain array **in the very
  next statement** after the pipe — any intervening command (even a bare assignment referencing
  only one element, or an `if [[ ... ]]` test) silently resets it first. Verified empirically:
  reading `${PIPESTATUS[0]}` and `${PIPESTATUS[1]}` as two separate assignment lines loses the
  second value. The safe form:
  ```bash
  set +o pipefail
  uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt
  status=("${PIPESTATUS[@]}")
  mypy_status="${status[0]}"
  filter_status="${status[1]}"
  if [[ "$mypy_status" != "0" && "$mypy_status" != "1" ]]; then
    echo "::error::mypy exited with fatal status $mypy_status (expected 0=clean or 1=errors-found); treating as a crash, not a baseline diff."
    exit 1
  fi
  exit "$filter_status"
  ```
  The step must end with an explicit `exit "$filter_status"` — a `run:` step's reported exit code
  is its *last executed command's* status, not automatically tied to array contents.
  - Alternative considered: redirect mypy to a temp file, check `$?`, then `cat` the file into
    `filter`. Equivalent result, but adds a throwaway file and an extra step for no behavioral
    gain over `PIPESTATUS`, which the job's shell already supports.
  - Alternative considered: re-run `mypy` a second time without the pipe just to get its exit
    code. Doubles mypy's runtime for no benefit — `PIPESTATUS` gets the same information from the
    single invocation already happening.
- **Decision: treat exactly `{0, 1}` as "mypy ran"; anything else fails immediately.**
  mypy's documented convention is `0` = no errors, `1` = errors reported, and non-`0`/`1` (in
  practice `2` for fatal/usage errors, or `128+signal` e.g. `137` for a signal kill such as OOM)
  for crashes. Matching on the two expected codes (rather than denylisting `2` specifically) also
  catches signal kills and other unexpected exits without needing to enumerate them. Verified:
  `uv run` itself failing to start `mypy` (e.g. a broken env) also surfaces outside `{0, 1}`.

## Risks / Trade-offs

- **A legitimate future mypy exit code outside `{0, 1}` could false-positive.** Unlikely — this is
  mypy's own documented contract — but if it ever happens, the fix is a one-line change to the
  allowed set, not a redesign.
- **`PIPESTATUS` is bash-specific**, not POSIX `sh`. The step already declares `shell: bash`
  explicitly (see the existing step), so this is already assumed, not newly introduced.
- **GitHub Actions' `shell: bash` runs with `-e` (errexit) on by default, in addition to
  `pipefail`** (only `pipefail` is disabled here). When the pipeline's reported status (`filter`'s,
  since `pipefail` is off) is non-zero, `errexit` aborts the script right at the pipe line, before
  the guard's `status=(...)` capture or its `::error::` diagnostic ever runs. The job still ends
  up correctly red in that case (errexit propagates the same non-zero exit), so this is not a
  false-green risk — but it means the guard's distinguishing crash message can silently not
  appear whenever a crash also happens to make `filter` itself exit non-zero on garbled/partial
  input (as opposed to cleanly reporting `new: 0` on empty input). Accepted: the pass/fail outcome
  is still correct; only the diagnostic clarity is reduced in that sub-case.
- **Local reproduction commands in `docs/CONTRIBUTING.md` intentionally do not carry the guard.**
  A local terminal already shows a crash directly (stderr, non-zero `$?`), so the guard's value is
  CI-specific; the doc update should say so explicitly rather than leaving the two commands to
  silently diverge.

## Migration Plan

1. Add the `PIPESTATUS` check (atomic-capture form above) to the `type-check` job's mypy step.
2. Confirm the three existing cases (clean tree, baseline-covered errors, new untyped def) still
   pass/fail exactly as before — run the exact `run:` block locally against the real
   `.mypy-baseline.txt` for the first two, and against a scratch introduced untyped def for the
   third.
3. Confirm a simulated fatal exit fails the job even when piped through a `filter` that would
   otherwise report `new: 0`: point mypy at a bogus `--config-file` (forces exit `2`) and pipe
   through `filter` against an **empty scratch baseline file**, not the repo's real (non-empty)
   `.mypy-baseline.txt` — against the real baseline, `filter` already fails today regardless of
   this change (pre-existing errors read back as "fixed"), so that case doesn't exercise the
   false-green path this guard closes.
4. Document the guard in `docs/CONTRIBUTING.md`, noting local repro intentionally stays unguarded.
5. After pushing, confirm the real `type-check` job (and the unrelated lint/reproducibility/
   serialization jobs, which this change doesn't touch) go green via `gh pr checks`.

Rollback: revert the step to the current two-line `set +o pipefail` form.
