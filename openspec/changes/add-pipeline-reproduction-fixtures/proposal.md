## Why

The pipeline (`run-all` = QC → viz → cross-platform) has no shared, version-controlled
golden fixture set. Every reproduction, future bloom-mcp tool test, and generated test
re-reads ad-hoc data, so the published wheat-EDPIE numbers can silently drift and
failures are hard to localize. Issue [#120](https://github.com/talmolab/sleap-roots-analyze/issues/120)
asks for one reusable fixture set backing the **full** pipeline end-to-end, per platform,
with a documented tolerance + regenerate policy.

This change lands the **infrastructure plus a single vertical platform slice
(`turface_19`)** so the layout, loaders, README policy, harness, and per-stage tests are
proven end-to-end on real data. Follow-up changes add the remaining platforms
(`turface_150`, `cylinder`, `root_core`) against the same scaffold.

## What Changes

- Add `tests/fixtures/` reproduction layout: `README.md`, `harness/` (EDPIE
  `run_manifest.yaml` + `qc/` + `viz/` + `cross_platform/` configs), and
  `real/wheat_edpie/` (`inputs/` + `expected/`).
- Commit **curated** real wheat-EDPIE golden artifacts for `turface_19` only — the
  assertable CSV/JSON outputs (`10_final_data`, removed-trait/sample/outlier counts,
  heritability-filter results, PCA explained variance, UMAP, heritability H², ANOVA,
  trait statistics, cross-platform correlations). Non-assertable artifacts
  (`code_snapshot.tar.gz`, `pipeline.log`/`viz_pipeline.log`, oversized per-row
  `cross_platform_exp*_loaded.csv` intermediates) are **excluded**; the small
  `turface_19` `pipeline_summary.json` is kept because it is the practical carrier of
  the viz PCA/UMAP/ANOVA/heritability metrics. Committed slice ≈ 3 MB.
- Add `scope="session"` pytest loaders for the fixture tables and per-stage assertion
  tests (`@pytest.mark.parametrize` over stage) for `turface_19`, asserting `allclose`
  within a documented tolerance.
- Document tolerance + regenerate policy in `tests/fixtures/README.md`, reusing
  `docs/reproducibility.md` (#118): a method change requires reviewer approval + a
  paper-supplementary update, never a quiet bugfix drift.

This change adds **no dependency on `sleap-roots-contracts`** and imports it nowhere.
The reproduction harness is gated on nothing and merges on its own. Analysis-input
contract conformance — canonicalizing the post-QC fixture and validating it plus the
package's canonical examples — is deferred to a follow-up "analysis-input contract
conformance" change, opened once `sleap-roots-contracts[pandas]>=0.1.0a1` is released
(it adds a dev dependency + one test file; the post-QC fixture committed here is reused
there, canonicalizing a copy). The runtime wiring (validate in `run-all`, the
`validate_input` flag) is analyze#144, out of scope for both fixture changes.

## Impact

- Affected specs: `reproduction-fixtures` (new capability).
- Affected code: `tests/fixtures/**` (new fixture tree + README), `tests/fixtures.py`
  (new loaders), one new test module `tests/test_pipeline_reproduction.py`.
- No `src/` runtime code changes — this is test infrastructure only.
- Cross-platform expected outputs are the single source shared with #119.

## Non-Goals

- Platforms other than `turface_19` (follow-up changes reuse this scaffold).
- Analysis-input contract conformance (synthetic examples + `validate_analysis_input`)
  — deferred to a follow-up change gated on the `sleap-roots-contracts` release.
- A full `run-all` re-execution in CI (per-stage assertions against committed golden
  are the safer default; a reproduction harness is documented but not run in CI).
- Committing figures or the full 109 MB Box bundle verbatim (Git LFS not introduced).
