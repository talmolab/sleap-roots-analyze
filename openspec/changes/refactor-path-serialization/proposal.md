## Why

PR #156's cross-OS serialization round-trip gate caught a latent bug: pipeline steps hand-stringify paths with `str(path)` before they reach the central serializer, so on Windows a serialized path becomes `out\a.csv` instead of `out/a.csv`. #156 fixed only one site (`PipelineSummary.to_json` no longer pre-`str()`s `files_generated`).

The same `str(path)` anti-pattern is repeated across ~20 sites in the pipeline steps (issue #157). Each pre-stringification **defeats** `convert_to_json_serializable`'s `Path → as_posix()` normalization and bakes in `\` on Windows. The union type `files_generated: List[str | Path]` is what invites this divergence: it lets each producer choose to store a `str` or a `Path`. This is the per-run provenance manifest that rolls up into `pipeline_summary.json`, so the divergence is observable in shipped artifacts.

## What Changes

- Stop hand-stringifying paths at the producers across the pipeline steps. Store bare `Path` and let `convert_to_json_serializable` normalize once, centrally (no per-producer `.as_posix()` — that just relocates the choice). Affected sites:
  - `files_generated=[str(...)]` / `.append(str(...))`: `calculate_cross_platform_correlations.py`, `calculate_trait_enrichment.py`, `reduce_trait_redundancy.py` (×3), `visualize_cross_platform.py` (×4), `load_cross_platform_data.py` (×3)
  - `metadata[...] = str(path)` (and `[str(f.relative_to(run_dir)) …]`): `merge_all_traits.py`, `visualize_depth_profiles.py`, `filter_heritability.py`, `generate_dashboards.py`, `generate_summary_viz.py`, `load_above_ground.py`, `generate_interactive.py`, `generate_static_figures.py`
- **BREAKING (internal)** Tighten the field type `files_generated: List[str | Path]` → `List[Path]` in `pipeline/core.py`, `pipeline/summary.py`, `pipeline/task.py` so the union can't re-invite divergence. (Internal dataclass field; not a public API.) This MUST land in the same commit as the producer edits — `List` is invariant, so a split breaks the mypy baseline gate either way.
- Update 4 step tests that assert exact `str(...)` membership in `files_generated` (they break at runtime once elements are `Path`), and add an OS-independent regression test, a `None`-path test, and a CI-enforced source guard so the anti-pattern can't recur.
- Tighten the existing `cli-pipeline` "Summary config is JSON serializable" scenario from "converted to strings" → "converted to POSIX strings", so the spec no longer permits the backslash form this change removes.
- Verify against the cross-OS serialization round-trip gate from #156 (the `serialization-gate` CI job over ubuntu/windows/macos).

## Impact

- Affected specs: `cli-pipeline` (MODIFY config-serialization scenario; ADD provenance path serialization requirement)
- Affected code: `src/sleap_roots_analyze/pipeline/{core,summary,task}.py` (type tightening) and ~13 step files under `src/sleap_roots_analyze/pipeline/steps/`; plus, per PR #159 review: `pipelines/base_pipeline.py` (`output_directory` stores bare `Path`), `data_utils.py` (shared `path_to_posix` helper unifying the two serializer predicates), and `steps/generate_summary_viz.py` (third sink — its local viz-`summary.json` writer)
- Affected tests: 4 existing membership assertions updated, plus new regression/guard tests (`tests/test_result_serialization.py` incl. `output_directory`, a `save_json` POSIX test in `tests/test_pipeline_core.py`, `tests/test_no_path_prestringify.py`)
- Affected fixtures: two committed goldens whose path fields were Windows-backslashed (`viz/cylinder/summary.json` `run_directory`, `cross_platform/root_core_vs_cylinder/pipeline_summary.json` `output_directory`) regenerated to POSIX
- Affected docs: `docs/CHANGELOG.md` (Fixed entry — user-observable: Windows manifests flip `\`→`/`), `docs/reproducibility.md` (state the producer-side "store `Path`" rule once, as the gate's named source of truth), `openspec/project.md` (one-line pointer), and the now-understated comment at `pipeline/summary.py:88-91`
- Merge-order coupling: the mypy baseline gate is on the still-open PR #158, not `main`. Land this refactor before #158 if possible; otherwise resync `.mypy-baseline.txt` in this PR's single commit.
- Risk: low-to-medium. Producer edits are mechanical, but the type change requires the coordinated test updates above. No committed golden fixture asserts on serialized paths or carries backslash paths (to be verified, not just asserted — task 5.3), so the #146 reproduction fixtures are undisturbed. The cross-OS gate is the regression backstop.
