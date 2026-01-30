# Tasks: Add and Update Claude Commands

## Update Existing Commands

- [ ] Fix `coverage.md` - Remove broken `scripts/cov_summary.py` reference, use CI-matching pytest coverage command
- [ ] Fix `lint.md` - Match CI workflow: `black --check` + `ruff check` (not `ruff format`)
- [ ] Update `review-pr.md` - Add `$ARGUMENTS` support for PR number, update examples
- [ ] Update `update-changelog.md` - Remove stale template, update to current project state

## Add New Commands

- [ ] Create `pre-merge-check.md` - Comprehensive pre-merge verification workflow
- [ ] Create `cleanup-merged.md` - Post-merge branch cleanup and OpenSpec archival
- [ ] Create `run-ci-locally.md` - Run full CI locally matching `.github/workflows/ci.yml`
- [ ] Create `verify-results.md` - Scientific accuracy verification for pipeline outputs
- [ ] Create `tdd.md` - Test-driven development workflow

## Validation

- [ ] Verify all commands reference correct paths (`src/sleap_roots_analyze`, not `sleap_roots_analyze`)
- [ ] Verify CI commands match `.github/workflows/ci.yml` exactly
- [ ] Test `/run-ci-locally` command runs successfully
