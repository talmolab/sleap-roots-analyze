## Why

The wheat EDPIE cross-platform paper reframes correlation as **predictability** (Wolfgang's
2026-04-03 ask). Tier 3 (`add-cross-platform-prediction`, merged
[#195](https://github.com/talmolab/sleap-roots-analyze/pull/195)) shipped the statistical machinery
— `logo_cv_predict()`, `fit_pca_on_fold()`, `CrossPlatformPredictionResult` — as plain Python-API
functions, deliberately with no pipeline wiring. This is Tier 3.5
(`add-prediction-pipeline-step`, tracking issue
[talmolab/sleap-roots-analyze#196](https://github.com/talmolab/sleap-roots-analyze/issues/196)) of
the program: it wires that machinery into the existing `CrossPlatformPipeline` so cross-platform
prediction runs as part of the same per-pair analysis config/command already used for correlation.

## What Changes

- **New `PredictionConfig` dataclass** (`pipeline/config/components.py`), nested as a new field —
  `prediction: PredictionConfig = field(default_factory=PredictionConfig)` — on the existing
  frozen `CrossPlatformConfig` (see `design.md` Decision 1 for why nested, not standalone). Fields:
  `enabled: bool = False`; `predictor_source: str = "blup"` (`{blup, genotype_means}`);
  `reduction_method: str = "pls_latent"` (primary, `{pls_latent, representatives, pc1}`);
  `comparison_methods: list[str]` (default `["representatives"]`, same 3-value set as
  `reduction_method`; must not duplicate the primary method — see Decision 7-adjacent validation in
  `tasks.md`); `representative_selection_metric: str = "variance"` (restricted to `"variance"` only
  for this tier — `"heritability"` deferred, see `design.md` Decision 7:
  `select_cluster_representatives` has no metric parameter to reuse, and this tier's Non-Goals
  already forbid changing it); `platform_pairs: list[dict]` (direction descriptor — which of
  `exp1_name`/`exp2_name` is predictor vs. predicted, narrowed from the roadmap's original
  multi-pair framing to a single, cardinality-validated entry per Decision 3/Decision 10);
  `blup_refit_per_fold: bool = False` (kept in the schema per the roadmap's settled field list, but
  currently inert — no `representative_selection_metric` value triggers it in this tier, see
  Decision 7); `source_blup_path`/`target_blup_path: Optional[str] = None` (required and
  existence-checked only when `predictor_source="blup"` — kept separate from
  `exp1_data_path`/`exp2_data_path`, which stay raw-per-sample-only for the unchanged correlation
  steps, per Decision 2).
- **Pre-flight validation**: `PredictionConfig.__post_init__` is a no-op when `enabled=False`
  (Decision 4, preserves backward compatibility for every existing config); when `enabled=True`, it
  validates enum fields, the `blup_refit_per_fold`/`heritability` interaction, and — when
  `predictor_source="blup"` — that `source_blup_path`/`target_blup_path` resolve on disk, raising
  plain `ValueError` at config-load time (Decision 5), before any pipeline step runs.
  `CrossPlatformConfig.__post_init__` additionally cross-checks `platform_pairs` against
  `exp1_name`/`exp2_name` (Decision 3), since `PredictionConfig` alone has no visibility into its
  parent's fields.
- **New `PredictCrossPlatformStep`** (`pipeline/steps/predict_cross_platform.py`), wired as an
  optional 6th task on `CrossPlatformPipeline` (`depends_on=["01_load_cross_platform_data",
  "05_visualize_cross_platform"]` — the first for data, the second for ordering only, see
  `design.md` Decision 15), entirely absent from `create_tasks()` when
  `config.prediction.enabled=False`. Consumes Tier 3's
  `logo_cv_predict`/`fit_pca_on_fold`/`CrossPlatformPredictionResult` unchanged. Target-set
  construction (which traits get predicted, how PC1-as-target differs from `fit_pca_on_fold`'s
  per-fold predictor-side use, and exactly which function computes it) is fully specified in
  `design.md` Decisions 6 and 12. `predictor_source="genotype_means"` selects task 1's
  already-`exclude_cols`-filtered trait columns before aggregating (Decision 13 — not a bare
  groupby-mean over the raw frame, which would crash or admit excluded columns), and `X`/every
  per-target `y`/`genotypes` are derived from one canonical, explicitly-indexed common-genotype list
  (Decision 14 — row-order alignment is not left to incidental DataFrame join behavior).
- **CLI**: no new command, no new flags. The existing `cross-platform` command's dry-run step list
  (`cli.py`) gains a conditional 6th entry when prediction is enabled; OmegaConf's structured-config
  merge picks up the new nested field automatically.
- **Backward compatibility**: `enabled=False` (default) means zero behavior change for every
  existing `CrossPlatformConfig` YAML — no validation runs, no task is added, analysis output
  (correlation CSVs, figures, `pipeline_summary.json`) is byte-identical to pre-Tier-3.5, mirroring
  the existing `enrichment_enabled` precedent. `config.yaml` itself gains a new `prediction: {...}`
  block reflecting the new field's existence regardless of `enabled` — an expected, harmless
  provenance-serialization side effect, not a behavior change (see `design.md` Decision 9 for why
  the backward-compat oracle is scoped to exclude it).

No changes to Tier 3's statistical machinery, to Tier 4 (permutation null/figures, separate future
change), or to `CrossPlatformSummaryGenerator` (tracked as follow-up
[#197](https://github.com/talmolab/sleap-roots-analyze/issues/197), since it doesn't yet surface
prediction results and this tier doesn't change that).

## Design decisions (resolved via brainstorming this session, then revised during `/review-openspec` round 1 — full rationale in `design.md`)

- `PredictionConfig` nests inside `CrossPlatformConfig` rather than becoming a standalone top-level
  config — reversed from an initial standalone choice after finding the backward-compat oracle is
  only meaningful against a real prior baseline (`design.md` Decision 1).
- `source_blup_path`/`target_blup_path` are new, `PredictionConfig`-only fields, distinct from
  `exp1_data_path`/`exp2_data_path` (Decision 2).
- `platform_pairs` narrows to a single-entry direction descriptor, cross-validated against
  `exp1_name`/`exp2_name` in `CrossPlatformConfig.__post_init__` (Decision 3).
- Validation is fully skipped when `enabled=False` (Decision 4) — required for backward
  compatibility, not just a convenience.
- Plain `ValueError`, not a new `ConfigValidationError` class (Decision 5) — no such class exists
  in the codebase today.
- Target-set construction: target platform's cluster-representative traits + one PC1-as-target
  entry (ground truth, via `pca.fit_pca()` directly with fixed hyperparameters, not
  `fit_pca_on_fold`, which remains the source-side per-fold predictor-reduction utility, unchanged
  from Tier 3) (Decision 6).
- **Added round 1:** `representative_selection_metric` restricted to `"variance"` for this tier —
  `select_cluster_representatives` has no metric parameter, so `"heritability"` has no
  implementation path yet; `blup_refit_per_fold` stays in the schema but is currently inert
  (Decision 7).
- **Added round 1:** task 6's `depends_on` includes `"01_load_cross_platform_data"` directly, not
  just `"05_visualize_cross_platform"` — the original single-dependency design relied on an
  undocumented data pass-through that silently breaks when `trait_reduction_method="clustering"` is
  also enabled (Decision 8).
- **Added round 1:** the backward-compat oracle excludes `config.yaml` from its byte-identical
  comparison, since the new field's presence alone changes that file regardless of `enabled`
  (Decision 9).
- **Added round 1:** `platform_pairs` cardinality (exactly one entry) is now explicitly validated,
  not just asserted in prose (Decision 10).
- **Added round 1:** target-trait *selection* (as opposed to the PC1 *value*) is documented as a
  selection-bias consideration distinct from fit-time leakage (Decision 11).
- **Added round 1:** PC1-as-target's exact computation is pinned to `pca.fit_pca()` with
  `StandardScaler` pre-applied and `random_state=42` fixed, explicitly not adding a new `PCAConfig`
  (Decision 12).
- **Added round 2 (a second, independent `/review-openspec` pass, run fresh with no memory of round
  1):** `predictor_source="genotype_means"` selects task 1's pre-filtered trait-name list, not
  every column in its raw DataFrame — round 1's own fix (Decision 8) had reintroduced a
  crash/data-pollution risk here, and its citation of the precedent it claimed to match was itself
  inaccurate (Decision 13). `X`, every per-target `y`, and `genotypes` are derived from one
  canonical, explicitly-indexed common-genotype list — row-order alignment was previously
  unenforced and untested, a silent-wrong-result risk (Decision 14). Task 6's second `depends_on`
  entry is for DAG ordering only, not data (Decision 15).
- **Added round 3 (a third, independent pass, run fresh with no memory of rounds 1-2 — every prior
  round's code citation independently re-verified and confirmed accurate):** any trait column with
  any `NaN` among the common-genotype set is dropped before building `X` or any target — real BLUP
  CSVs routinely contain NaN columns for failed model fits, previously unaddressed and untested
  (Decision 16). The BLUP CSV's genotype-column name is resolved via a fixed convention
  (`"Genotype"` then `"genotype"`, distinct from `exp1_genotype_col`/`exp2_genotype_col`) rather
  than left unspecified (Decision 17).

## Impact

### Affected specs

- `cross-platform-analysis` (MODIFIED) — `Cross-Platform Configuration` requirement gains the
  `prediction` field description; (ADDED) — new `Cross-Platform Prediction Configuration` and
  `Predict Cross-Platform Genotype Values Pipeline Step` requirements. **Note (found during
  `/review-openspec` round 1):** the MODIFIED requirement also backfills three fields
  (`enrichment_enabled`, `enrichment_p_value_column`, `validate_input`) that already shipped in code
  via earlier changes but were never added to this requirement's field list — pre-existing spec
  drift, not new behavior. Not lossy (all prior text is preserved), but disclosed here explicitly so
  the diff's size isn't a surprise.

### Affected code

- `src/sleap_roots_analyze/pipeline/config/components.py` — new `PredictionConfig` dataclass; new
  `prediction` field + extended `__post_init__` cross-check on `CrossPlatformConfig`.
- `src/sleap_roots_analyze/pipeline/steps/predict_cross_platform.py` (new) —
  `PredictCrossPlatformStep`.
- `src/sleap_roots_analyze/pipeline/pipelines/cross_platform_pipeline.py` — conditional 6th task.
- `src/sleap_roots_analyze/cli.py` — dry-run step list gains a conditional entry; docstring update.
- `tests/fixtures.py` / `tests/fixtures/` — new 2-platform synthetic BLUP fixture pair + harness
  YAML; a pre-tier golden-fixture snapshot for the backward-compat regression test if one doesn't
  already exist.
- `tests/test_predict_cross_platform.py` (new), `tests/test_cross_platform_config.py` (extended) —
  all oracle tests from issue #196's acceptance criteria.
- `docs/API.md`, `docs/CHANGELOG.md`, `docs/CROSS_PLATFORM_ANALYSIS.md` — new step/config entries.

### Explicitly out of scope

- Tier 4 (permutation null, figures) — separate future change, depends on Tier 3 only (Python API),
  not on this tier.
- `CrossPlatformSummaryGenerator`/`.claude/commands/cross-platform-summary.md` not surfacing
  prediction results — follow-up [#197](https://github.com/talmolab/sleap-roots-analyze/issues/197).
- `/configure-run-all`, `/dry-run`, `/validate-config` cross-platform/prediction coverage gaps —
  follow-up [#198](https://github.com/talmolab/sleap-roots-analyze/issues/198), pre-existing, not
  caused by this tier.
- Any change to `logo_cv_predict`, `fit_pca_on_fold`, `CrossPlatformPredictionResult`,
  `cluster_correlated_traits`, or `select_cluster_representatives` — all reused as-is.
