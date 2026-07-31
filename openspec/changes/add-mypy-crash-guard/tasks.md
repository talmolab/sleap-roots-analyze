## 1. CI guard

- [x] 1.1 In `.github/workflows/ci.yml`'s `type-check` job, capture the whole `PIPESTATUS` array
      in **one atomic statement** (`status=("${PIPESTATUS[@]}")`) immediately after the existing
      piped `run` command, then read `mypy_status="${status[0]}"` and
      `filter_status="${status[1]}"` from that copy — reading `PIPESTATUS[0]`/`[1]` as two
      separate statements loses the second value (bash resets `PIPESTATUS` on every subsequent
      simple command, including a bare assignment). See `design.md`'s Decisions section for the
      verified-safe snippet.
- [x] 1.2 Fail the step immediately (with a clear `::error::` log message) when `mypy_status` is
      not `0` or `1`; otherwise end the step with an explicit `exit "$filter_status"` — a `run:`
      step's reported exit is its last executed command's status, not automatically tied to
      array contents.

## 2. Verification (run locally, before opening the PR, against the exact `run:` block)

- [x] 2.1 Confirm the job still passes on a clean tree (mypy exit `0`, `filter` exit `0`).
      Verified locally against a trivial fully-typed scratch module + empty baseline:
      `mypy_status=0 filter_status=0 step_exit=0`.
- [x] 2.2 Confirm the job still passes when all reported errors match the real
      `.mypy-baseline.txt` (mypy exit `1`, `filter` exit `0`). Verified locally against the real
      tree: `mypy_status=1 filter_status=0 step_exit=0`.
- [x] 2.3 Confirm the job still fails when a new untyped def is introduced (mypy exit `1`,
      `filter` exit non-zero). Verified locally with a scratch untyped def added and removed:
      `mypy_status=1 filter_status=1 step_exit=1`.
- [x] 2.4 Confirm the job fails on a simulated fatal mypy exit (e.g. `--config-file` pointed at a
      nonexistent path, forcing exit `2`) piped through `filter` against an **empty scratch
      baseline file** (not the repo's real, non-empty `.mypy-baseline.txt` — against the real
      baseline, `filter` already fails today regardless of this change, so it wouldn't exercise
      the false-green path being closed). Verified locally: `mypy_status=2 filter_status=0
      step_exit=1` (crash-guard tripped) — confirms the false-green case is now closed, since
      `filter_status=0` alone would have passed before this change.
- [x] 2.5 Confirm the job still fails (for the pre-existing reason, not a new one) when a
      simulated fatal mypy exit is piped through `filter` against the **real, non-empty**
      baseline — i.e. the new guard doesn't interfere with or mask the already-correct behavior
      in this case. Verified locally: `mypy_status=2 filter_status=100 step_exit=1`.
- [ ] 2.6 After pushing, confirm via `gh pr checks` that the real `type-check` job goes green (or
      red, matching the local result) and that the unrelated lint/reproducibility/
      numerical-stability/serialization jobs are unaffected.

## 3. Documentation

- [x] 3.1 Add a short note to the "Type Checking (mypy ratchet)" section of
      `docs/CONTRIBUTING.md` explaining the crash-guard: a fatal mypy exit fails CI immediately,
      distinct from a normal baseline-diff failure. Clarify that the section's local-repro
      commands intentionally do NOT carry the guard (a local terminal already shows a crash
      directly), so they stay simpler than the CI step on purpose.

## 4. Validation

- [x] 4.1 Run `openspec validate add-mypy-crash-guard --strict` — must pass. Passes.
- [x] 4.2 Run `uv run black --check` and `uv run ruff check src/sleap_roots_analyze` — no
      regressions (this change only touches YAML/Markdown, but confirm nothing else drifted).
      Both pass (198 files unchanged, all checks passed) — expected, since no Python source
      changed.
