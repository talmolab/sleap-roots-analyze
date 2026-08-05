## Why

Fixes #204.

`QCPipelineConfig` (the class the issue calls `QCConfig`; `pipeline/config/qc_config.py:67`)
includes a `pca: PCAConfig` field. `QCPipeline`'s step list (`qc_pipeline.py:133-154`) has no
PCA-analysis step, no biplot step, and no feature-contribution-plot step — those steps
(`PCAAnalysisStep`, `GenerateStaticFiguresStep`, and the PCA figure in
`generate_summary_viz.py`) are wired only into `VizPipeline` (`viz_pipeline.py`). A direct grep
of `config\.pca\b` across `pipeline/steps/*.py` and `pipeline/pipelines/qc_pipeline.py` confirms
zero matches outside those three viz-only files. Every QC config setting
`pca.feature_selection_strategy`/`pca.n_top_features` (and often
`pca.n_components`/`pca.standardize`) is configuring a value nothing in a QC run reads — the
same shape as #94 and the pre-fix state of #142.

**What the investigation found beyond the issue text**: the field isn't just inert at the
pipeline-step level. `validate_qc_config()` and `validate_explicit_config()`
(`pipeline/config/utils.py`) both actively validate `config.pca.n_components`,
`config.pca.feature_selection_strategy`, and `config.pca.n_top_features` — and
`validate_explicit_config()` currently *requires* `pca.n_components` to be explicitly set. So
today a QC config can fail `validate_qc_config()` over a field that no QC pipeline step ever
reads. Removing `pca: PCAConfig` from `QCPipelineConfig` therefore also requires removing the
`config.pca.*` validation blocks from both functions — leaving them in place would raise
`AttributeError` the instant a caller validates a config without the field. This is a strictly
larger change than "delete one dataclass field"; `tests/test_pipeline_config.py` has 16 tests
built directly against `QCPipelineConfig.pca` and `validate_qc_config()`'s PCA-validation
branch, all of which must be removed or reworked in the same commit.

**Also found beyond the issue text**: two `.claude/commands/*.md` developer-tooling docs
actively author or inspect `pca.*` on QC configs and would generate or crash on broken output
after this change — `configure-run-all.md` (asks PCA questions and writes `pca.n_components`/
`feature_selection_strategy`/`n_top_features` into the **QC** config it authors, not just the
viz config) and `validate-config.md` (its example validation script prints
`config.pca.n_components` for a QC config). Both need fixing in this same change, not as a
follow-up — otherwise the very tooling used to author new QC configs keeps producing configs
that fail to load the moment this change merges.

**Config-loading constraint, verified empirically**: `load_qc_config()` uses
`OmegaConf.merge(OmegaConf.structured(QCPipelineConfig), yaml_conf)` — a strict structured
merge. Removing `pca` from the dataclass while a YAML file still has a top-level `pca:` key
raises `ConfigKeyError: Key 'pca' not in struct` at load time. There is no way to remove the
field but leave existing QC configs alone; every QC config with a `pca:` block must be edited in
the same change.

## What Changes

- **Remove `pca: PCAConfig` from `QCPipelineConfig`** (`pipeline/config/qc_config.py`), including
  its docstring line and the now-unused `PCAConfig` import in that module.
- **Remove the QC-only `config.pca.*` validation blocks** from `validate_explicit_config()` and
  `validate_qc_config()` in `pipeline/config/utils.py`. `validate_viz_config()`'s identical-
  looking PCA validation block is untouched — `VizPipelineConfig` keeps its own `pca: PCAConfig`
  field (still read by `PCAAnalysisStep`/`GenerateStaticFiguresStep`/`generate_summary_viz.py`),
  so its validation stays exactly as-is.
- **Strip the `pca:` block from every QC-only config file** — verified by an exhaustive repo
  grep (`^pca:` at top level, cross-checked against a looser `feature_selection_strategy|
  n_top_features` grep with an identical result set) to be exactly **59 files**: 4 committed
  test harness fixtures (`tests/fixtures/harness/qc/*.yaml`), 28 files under `configs/active/qc/`
  plus 3 flat pre-reorg duplicates directly under `configs/active/`, 4 under `configs/examples/`,
  15 flat files directly under `configs/`, 4 under `configs/templates/`, and
  `configs/test_nov30_reproduction.yaml`. Full removal, including already-used experiment
  configs under `configs/active/`, is intentional — a maintainer scope decision made in review:
  their provenance lives with the experiment's actual output directory (outside this repo, or
  in the run's own saved `config.yaml`), not in the committed input YAML staying byte-identical
  forever.
  - **Explicitly out of scope**: `configs/archive/**` and `configs/saved_backups/**` (frozen,
    timestamped historical snapshots that `/configure-run-all` itself moves aside when a config
    is replaced — analogous to the golden-fixture provenance argument above, these are already
    the frozen record and are not reloaded by any current tooling) and
    `tests/fixtures/real/wheat_edpie/expected/{qc,viz}/*/config.yaml` (the committed *output*
    config from the run that produced each golden fixture — a provenance artifact, not an input
    any test or loader reads; confirmed no test references these paths). Viz configs
    (`configs/**/viz/*.yaml`, `configs/**/viz_*.yaml`) are unaffected — `pca:` remains correct
    and required there.
- **Update QC-config tests**: remove or rework every `QCPipelineConfig.pca` /
  `validate_qc_config()` PCA-validation test in `tests/test_pipeline_config.py` (16 tests: field
  presence/round-trip/merge tests — including the override dict literal in
  `test_merge_configs`, not just its assertion — and the full `n_top_features`/
  `feature_selection_strategy` validation matrix), and strip the inline
  `pca:\n  n_components: 2\n` fixture string in
  `tests/test_replicate_optional.py::test_omit_replicate_in_yaml_disables_replicate`.
  `validate_viz_config()`'s equivalent tests in `tests/test_viz_pipeline_config.py` are
  untouched.
- **Add a one-time dynamic equivalence check** (not a permanently committed test — the field
  won't exist to set once removed): before removing the field, run
  `tests/test_qc_pipeline.py::TestQCPipelineIntegration::test_qc_pipeline_full_run`'s synthetic
  dataset through `QCPipeline.run()` twice — once with `config.pca` left at defaults, once with
  `config.pca.feature_selection_strategy` set to a non-default value (e.g. `"top_absolute"`) —
  and confirm byte-identical output. This is the actual dynamic (not just static-grep) proof
  that the field has zero effect on `QCPipeline` execution; record the result in the PR
  description. (Reviewer-identified gap: the pre-existing golden-fixture reproduction suite never
  calls `QCPipeline.run()` at all — see the Impact section below — so it alone cannot support
  this claim.)
- **Add a safety-net regression test** that calls `load_qc_config()` on the exact 59-file list
  from the sweep above (hardcoded, not a path/content heuristic, so it verifies precisely what
  was edited), asserting none raises `ConfigKeyError` — a concrete guard against a missed file,
  replacing "spot-check a sample." Written and confirmed *failing* against the pre-sweep files
  before the sweep, then confirmed passing after — the actual TDD red/green step for this
  section, not just a post-hoc check.
- **Docs**: rewrite `pipeline/README.md`'s illustrative example config (`README.md:119-123`),
  which currently shows a bare `pca:` block alongside `outlier_detection`/`visualization` with no
  pipeline context, implying every pipeline config needs one — remove the `pca:` block from that
  example. Fix `.claude/commands/configure-run-all.md` (drop the three now-QC-invalid `pca.*`
  lines from its "6.1 — QC Config" customized-parameters checklist; clarify its "3.8 — PCA
  settings" question block feeds only the viz config, matching the existing "(for viz config)"
  annotation already on its `pca_biplot_top_features` question). Fix
  `.claude/commands/validate-config.md` (drop the `config.pca.n_components` print line from its
  QC-config validation example script). Fix `docs/QC_PIPELINE_GUIDE.md` (remove its "PCA
  Configuration" subsection, currently listed under QC "Required Parameters"). Fix
  `configs/templates/README.md`'s "PCA Settings" section (re-scope to viz templates only,
  consistent with the existing "(viz template default)" annotation already on
  `pca_biplot_top_features` — QC templates have no PCA settings left after this change). Add a
  `docs/CHANGELOG.md` `[Unreleased]` entry.

## Impact

- Affected specs: `config-management` (the "PCA Feature Selection Config Validation"
  requirement, added by #206's fix, currently states validation applies to *both*
  `validate_qc_config()` and `validate_viz_config()`; this change narrows it to
  `validate_viz_config()` only, since `QCPipelineConfig` no longer has a `pca` field to
  validate).
- Affected code:
  - `src/sleap_roots_analyze/pipeline/config/qc_config.py` (`QCPipelineConfig`)
  - `src/sleap_roots_analyze/pipeline/config/utils.py` (`validate_explicit_config`,
    `validate_qc_config`, and the module-level `_WHOLE_NUMBER_TOLERANCE` comment, which currently
    mentions `validate_qc_config()` alongside `validate_viz_config()`)
  - `src/sleap_roots_analyze/pipeline/README.md`
  - `.claude/commands/configure-run-all.md`, `.claude/commands/validate-config.md`
  - `docs/QC_PIPELINE_GUIDE.md`, `configs/templates/README.md`
  - 59 QC config YAML files (listed above; see `tasks.md` for the exhaustive per-file list)
  - `tests/test_pipeline_config.py`, `tests/test_replicate_optional.py`
  - `tests/test_step_pca_analysis.py`, `tests/fixtures_visualization.py` (discovered during full
    local test verification, not by proposal review — both used `QCPipelineConfig` as a generic
    test double for the viz-only `PCAAnalysisStep`/`GenerateStaticFiguresStep`, a pre-existing
    mismatch this change's removal surfaced; fixed by switching to `VizPipelineConfig`, the
    correct type — see `tasks.md` section 6.0)
  - `docs/CHANGELOG.md` `[Unreleased]`
- **Validation becomes strictly more permissive on one axis, and this is safe**: today,
  `validate_explicit_config()` raises if a QC config has `pca.n_components` unset — after this
  change that check no longer exists (there's no field left to check). Every config in the
  59-file sweep already sets `pca.n_components` or relies on `PCAConfig()`'s non-`None` default,
  so no config that currently validates successfully stops validating after this change; the
  golden-fixture reproduction suite (`tests/test_pipeline_reproduction.py`) and the new
  safety-net test are the empirical proof.
- **No pipeline output changes**: `QCPipeline.run()` never read `config.pca` before this change.
  This rests on two distinct kinds of evidence, kept honestly separate: (1) a **static** grep
  showing zero `config.pca` reads inside any QC step file, and (2) the **one-time dynamic
  equivalence check** described above (`test_qc_pipeline_full_run`'s dataset run twice, default
  vs. non-default `pca.*`, byte-identical output). `tests/test_pipeline_reproduction.py` passing
  identically before and after (`tasks.md` section 1) is **not** evidence of unchanged pipeline
  *execution* — verified during review that this suite never calls `QCPipeline.run()` anywhere;
  its only config-touching test (`test_harness_qc_config_valid`) calls `load_qc_config()` +
  `validate_qc_config()` and stops, and every other test in the file reads pre-committed golden
  CSV/JSON directly. Re-running it is real evidence that the 4 harness configs still load/
  validate and that unrelated golden fixtures are undisturbed — not evidence about pipeline
  execution, which is what (1) and (2) above establish instead.
- **Implementation ordering constraint** (reviewer-identified): the YAML sweep (59 files) must
  land *before* the schema/validation removal, not after — `OmegaConf`'s strict merge only
  breaks in the direction of "YAML has a key the schema lacks," so removing `pca` from
  `QCPipelineConfig` while any of the 59 files still has a `pca:` block breaks every test that
  loads one of those files directly, independent of `tasks.md` section 1's own (narrower)
  baseline commands. Confirmed three such direct dependents:
  `tests/test_pipeline_reproduction.py` (harness configs), `tests/test_golden_templates.py`
  (`configs/templates/qc_template_{grouped,ungrouped}.yaml`), and
  `tests/test_cli.py::test_qc_with_real_config_dry_run` (`configs/qc_turface_150genotypes.yaml`).
  See `tasks.md` for the corrected section order.
- **bloommcp compatibility**: already verified safe — `bloommcp`
  (`Salk-Harnessing-Plants-Initiative/bloom`, `staging` branch) does not read the QC/Viz YAML
  config schema at all; it calls `sleap_roots_analyze` functions directly with its own Pydantic
  params. Unaffected by this change.
- Explicitly out of scope (tracked separately, not touched here): `PCAConfig` itself (still used
  by `VizPipelineConfig`, unrelated), `PCAOutlierConfig` (`components.py:445` — a separate,
  actually-used class for PCA-based outlier detection inside `OutlierDetectionConfig`, easy to
  confuse with `PCAConfig` by name alone), and any Viz-pipeline PCA behavior.
