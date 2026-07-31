## 1. CI guard

- [ ] 1.1 In `.github/workflows/ci.yml`'s `type-check` job, capture `${PIPESTATUS[0]}` (mypy) and
      `${PIPESTATUS[1]}` (`mypy-baseline filter`) after the existing piped `run` command.
- [ ] 1.2 Fail the step immediately (with a clear `::error::` log message) when mypy's exit code
      is not `0` or `1`; otherwise exit with `filter`'s status, unchanged from today.

## 2. Verification

- [ ] 2.1 Confirm the job still passes on a clean tree (mypy exit `0`).
- [ ] 2.2 Confirm the job still passes when all reported errors match the baseline (mypy exit `1`,
      `filter` exit `0`).
- [ ] 2.3 Confirm the job still fails when a new untyped def is introduced (mypy exit `1`,
      `filter` exit non-zero).
- [ ] 2.4 Confirm the job fails on a simulated fatal mypy exit (e.g. an invalid `--config-file` or
      similar forced exit-`2` case) even when piped through a `filter` invocation that would
      report `new: 0` on empty input.

## 3. Documentation

- [ ] 3.1 Add a short note to the "Type Checking (mypy ratchet)" section of
      `docs/CONTRIBUTING.md` explaining the crash-guard: a fatal mypy exit fails CI immediately,
      distinct from a normal baseline-diff failure.

## 4. Validation

- [ ] 4.1 Run `openspec validate add-mypy-crash-guard --strict` — must pass.
- [ ] 4.2 Run `uv run black --check` and `uv run ruff check src/sleap_roots_analyze` — no
      regressions (this change only touches YAML/Markdown, but confirm nothing else drifted).
