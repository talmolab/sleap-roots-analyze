> **Atomicity note:** The type-tightening (§1) and the producer edits (§2–§3) MUST
> land in a single atomic commit/PR. `List` is invariant, so under the mypy
> baseline gate (PR #158) neither order is independently green: tighten-first adds
> `[list-item]` errors at still-`str(...)` producers AND strands resolved baseline
> entries; producers-first can add new invariance errors at not-yet-baselined files.
> See §5 for the merge-order coupling with PR #158.

## 1. Tighten the type (land with §2–§3 in one commit)

- [x] 1.1 Change `files_generated: List[str | Path]` → `List[Path]` in `pipeline/summary.py` (`StepSummary` field at line 36 **and** the `mark_step_success` signature at line 127)
- [x] 1.2 Change `files_generated: List[str | Path]` → `List[Path]` in `pipeline/core.py` (line 51)
- [x] 1.3 Change `files_generated: List[str | Path]` → `List[Path]` in `pipeline/task.py` (line 29)

## 2. Stop hand-stringifying at the producers — `files_generated`

- [x] 2.1 `steps/calculate_cross_platform_correlations.py:393` — `files_generated=[corr_output]`
- [x] 2.2 `steps/calculate_trait_enrichment.py:121,154` — `files_generated=[output_file]`
- [x] 2.3 `steps/reduce_trait_redundancy.py:244,266,277` — `.append(cluster_file / dendrogram_file / heatmap_file)`
- [x] 2.4 `steps/visualize_cross_platform.py:130,185,215,231` — `.append(<path>)`
- [x] 2.5 `steps/load_cross_platform_data.py:182,183,184` — `files_generated=[exp1_output, exp2_output, alignment_output]` (these are in `files_generated`, not `metadata`)

## 3. Stop hand-stringifying at the producers — `metadata`

- [x] 3.1 `steps/merge_all_traits.py:142,143` — `output_csv`, `metadata_json`
- [x] 3.2 `steps/visualize_depth_profiles.py:201,202,205` — `mean_plot`, `reps_plot`, `barplot`. **Keep the `if reps_plot_path else None` and `if barplot_path:` guards** — remove only the `str()` wrapper so the value stays `Path | None`.
- [x] 3.3 `steps/filter_heritability.py:327,328,329` — `diagnostic_csv`, `variance_plot`, `boxplot`
- [x] 3.4 `steps/generate_dashboards.py:107,110` — `dashboard_path` (`relative_to(run_dir)`), `run_directory`
- [x] 3.5 `steps/generate_summary_viz.py:56` — `run_directory`
- [x] 3.6 `steps/load_above_ground.py:101` — `csv_path`
- [x] 3.7 `steps/generate_interactive.py:169` — `files: [f.relative_to(run_dir) for f in generated_files]`
- [x] 3.8 `steps/generate_static_figures.py:284` — `files: [f.relative_to(run_dir) for f in generated_files]`

## 4. Tests

- [x] 4.1 Update the 4 existing tests that assert exact `str(...)` membership in `files_generated` (break once elements are `Path`): `test_step_calculate_cross_platform_correlations.py:258`, `test_step_visualize_cross_platform.py:165,592`, `test_step_load_cross_platform_data.py:513` — compare on `Path` (or `in [str(f) for f in ...]`)
- [x] 4.2 Add an OS-independent red-before-green test: assert the serialized manifest value equals `Path(p).as_posix()` and contains no `\` for a representative producer, using a backslash-bearing / `PureWindowsPath`-backed value so it fails against current `str(path)` producers on any OS and passes after
- [x] 4.3 Add a regression test for the `None`-path branch (`visualize_depth_profiles` `reps_plot`): when no replicate plot is produced, `metadata[...]["reps_plot"]` SHALL serialize to JSON `null`, not `"None"`
- [x] 4.4 Add a CI-enforced source guard (e.g. `tests/test_no_path_prestringify.py`) that scans `pipeline/steps/` and fails if `str(...)` co-occurs with a `files_generated`/`metadata` path assignment (AST- or multi-line-aware, so it catches `files_generated=[\n str(x)\n]`)

## 4b. Review follow-ups (PR #159 subagent review)

- [x] 4b.1 Fix Windows CI: `tests/test_no_path_prestringify.py` reads source with `encoding="utf-8"` (locale codec raised on emoji/`ρ`); also `rglob` + non-vacuous `>=13` assert + softened guard docstring
- [x] 4b.2 `output_directory` no longer pre-stringified: `base_pipeline.py` stores bare `Path`; `PipelineSummary.output_directory` typed `str | Path`
- [x] 4b.3 Third sink fixed: `generate_summary_viz._make_json_serializable` normalizes `PurePath` via `as_posix()` (was `str(obj)`) so the viz `summary.json` `run_directory` is POSIX
- [x] 4b.4 Unify the two serializer predicates: shared `data_utils.path_to_posix` (PurePath) used by `save_json`; add a `save_json` Path-normalization test
- [x] 4b.5 `PipelineSummary.load()` rehydrates `files_generated` to `List[Path]`; both readers use `encoding="utf-8"`
- [x] 4b.6 Regenerate the two committed goldens whose path fields were backslashed (`viz/cylinder/summary.json`, `cross_platform/root_core_vs_cylinder/pipeline_summary.json`)
- [x] 4b.7 Extend the cross-OS gate test to assert `output_directory` normalizes

## 5. Verify

- [x] 5.1 `grep -rn "str(" src/sleap_roots_analyze/pipeline/steps/ | grep -iE "files_generated|metadata|relative_to"` returns no path-stringification sites
- [x] 5.2 Run `pytest tests/test_result_serialization.py` (the cross-OS round-trip gate) green
- [x] 5.3 Run the #146 reproduction-fixture tests and inspect any committed `pipeline_summary.json` fixtures for path literals — **verify** (not just assert) the fixtures are undisturbed
- [x] 5.4 Full `pytest` + `ruff check` + `black --check` clean
- [x] 5.5 Update `docs/CHANGELOG.md` `[Unreleased] → Fixed`, `docs/reproducibility.md` serialization contract (producer-side "store `Path`" rule), the `summary.py:88-91` comment, and a one-line pointer in `openspec/project.md` (see proposal Impact)

## 6. Merge-order coupling with PR #158 (mypy baseline gate)

- [x] 6.1 Land this refactor **before** PR #158 if possible (then no mypy churn here; #158 rebases with 19 fewer baseline entries). If #158 lands first: fold the `.mypy-baseline.txt` resync into this PR's single commit (`mypy-baseline sync`) to drop the now-resolved `files_generated` invariance entries — CI runs `filter` without `--allow-unsynced`, so a stale baseline fails the gate
- [x] 6.2 Run `mypy src/` locally post-change to confirm no NEW downstream error beyond the expected baseline delta (e.g. verify `generate_interactive.py` `image_links` frozen entries at lines 204–206 are not perturbed)
