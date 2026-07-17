## Context

This is Tier 3.5 (`add-prediction-pipeline-step`, tracking issue
[talmolab/sleap-roots-analyze#196](https://github.com/talmolab/sleap-roots-analyze/issues/196)) of
the wheat EDPIE cross-platform genotype-prediction program. See the program roadmap and statistical
grounding at `c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md`
(external to this repo; referenced here for provenance only).

Tier 3 (`add-cross-platform-prediction`, merged
[#195](https://github.com/talmolab/sleap-roots-analyze/pull/195), archived
`2026-07-16-add-cross-platform-prediction`) shipped `logo_cv_predict()`, `fit_pca_on_fold()`, and
`CrossPlatformPredictionResult`/`TargetPrediction` as plain, stateless Python-API functions with no
pipeline wiring — deliberately out of scope for that tier (its own `design.md` Non-Goals: "the
`PredictionConfig` dataclass, `PredictCrossPlatformStep`, and any CLI/pipeline wiring"). This tier
wires that machinery into the existing `CrossPlatformPipeline` so prediction runs as part of the
same per-pair analysis config/command already used for correlation, rather than only being
reachable via direct Python calls. **No changes to Tier 3's statistical machinery are in scope
here** — `logo_cv_predict`, `fit_pca_on_fold`, and `CrossPlatformPredictionResult` are consumed
as-is.

Prior codebase investigation (see this change's brainstorm) established the following facts that
this design depends on:

- `CrossPlatformConfig` (`pipeline/config/components.py:835-1055`) is `frozen=True` and is its own
  standalone top-level config — not nested under `VizPipelineConfig`/`QCPipelineConfig` — loaded
  directly by `load_cross_platform_config()` and passed to `CrossPlatformPipeline`.
- `CrossPlatformPipeline` (`pipeline/pipelines/cross_platform_pipeline.py`) is built around a
  single `exp1`/`exp2` pair; its 5 existing tasks (`01_load_cross_platform_data` through
  `05_visualize_cross_platform`) all operate on that one pair's raw per-sample CSVs.
  `CrossPlatformConfig.enrichment_enabled` (a `bool = False` field added after the pipeline's
  initial ship, gating an optional step 4) is the direct precedent this tier follows.
- Tier 1's BLUP output (`08_blup_adjusted_means.csv`, one row per genotype) comes from a
  completely different pipeline (the QC/Viz statistics step, `StatisticalAnalysisStep` +
  `extract_blup_table()`) — one file per platform, produced by separate prior runs.
  `CrossPlatformPipeline` does not currently read this file shape at all.
- Elizabeth's existing cross-platform workflow already uses one `CrossPlatformConfig` YAML per
  directed pair (4 files for the 4 EDPIE pairs) — Tier 3's own manual validation (design.md
  Section 8) already exercised "all 4 directed pairs" as 4 separate single-pair pipeline
  invocations, not one multi-pair batch run.

## Goals / Non-Goals

- **Goals:** a new `PredictionConfig` dataclass, nested as a field on `CrossPlatformConfig`; a
  pre-flight existence check for BLUP CSV paths at config-load time; a new `PredictCrossPlatformStep`
  wired as an optional 6th task on `CrossPlatformPipeline`; a small CLI dry-run display update (no
  new command, no new flags); byte-identical backward compatibility when prediction is disabled
  (the default); a CI-safe synthetic wiring-correctness oracle; a manual, non-CI, sign-off-gated
  integration test against the 4 real EDPIE directed pairs.
- **Non-Goals:** any change to `logo_cv_predict`, `fit_pca_on_fold`, or `CrossPlatformPredictionResult`
  (Tier 3, unchanged); the permutation null or its figures (Tier 4); updating
  `CrossPlatformSummaryGenerator`/`.claude/commands/cross-platform-summary.md` to surface
  prediction results (tracked as follow-up
  [#197](https://github.com/talmolab/sleap-roots-analyze/issues/197)); extending
  `/configure-run-all`, `/dry-run`, or `/validate-config` to author/validate cross-platform or
  prediction configs (tracked as follow-up
  [#198](https://github.com/talmolab/sleap-roots-analyze/issues/198)); any change to
  `cluster_correlated_traits`/`select_cluster_representatives` (reused as-is, and — per Decision 7,
  added during `/review-openspec` round 1 — this rules out heritability-based representative
  selection for this tier, since that function has no metric parameter to reuse); a standalone
  prediction CLI command or pipeline class (considered and rejected — see Decision 1); a new
  `PCAConfig` on this pipeline (Decision 12 fixes PC1-as-target's hyperparameters instead).

## Decisions

### Decision 1: `PredictionConfig` nests inside `CrossPlatformConfig`, not standalone

**What:** `CrossPlatformConfig` gains a new field, `prediction: PredictionConfig =
field(default_factory=PredictionConfig)`. `PredictCrossPlatformStep` becomes task 6 on the existing
`CrossPlatformPipeline`, skipped entirely (not added to `create_tasks()`) when
`config.prediction.enabled` is `False`. No new CLI command, no new top-level pipeline class.

**Why:** Resolved with Elizabeth during this tier's brainstorm, after two rounds of
back-and-forth. The roadmap's own settled decision ("`PredictionConfig` is a new sub-dataclass,
**separate from** `CrossPlatformConfig`... this avoids a two-master problem") reads, on first pass,
like an argument for a fully standalone top-level config. Investigation showed the opposite is true
once "two-master" is examined concretely:

1. **The backward-compat oracle only means something against a real prior baseline.**
   `CrossPlatformConfig.enrichment_enabled` (`components.py:884-889`) is a directly-on-point,
   already-shipped precedent: a field added after the fact, gating a new step, `default=False`,
   with the explicit contract "existing runs are unchanged" — checkable against
   `CrossPlatformPipeline`'s real, already-committed golden fixtures. A brand-new standalone
   pipeline has no such baseline: "`enabled=False` produces identical output to pre-Tier-3.5" would
   be vacuously true (a pipeline that didn't exist before has nothing to preserve), not a
   meaningful regression test. Nesting makes issue #196's CI backward-compat oracle a real,
   checkable claim.
2. **The "two-master" problem doesn't actually recur here.** The roadmap's concern was about
   `platform_pairs` becoming a second, competing way to say "which pair" alongside
   `exp1_name`/`exp2_name` directly on `CrossPlatformConfig`. Keeping `platform_pairs` a field *on*
   `PredictionConfig` (not flattened onto `CrossPlatformConfig`'s own field list) already satisfies
   the literal settled text regardless of whether `PredictionConfig` itself is nested or standalone
   — nesting `CrossPlatformConfig.prediction: PredictionConfig` adds exactly one new field to
   `CrossPlatformConfig`, not a second pair-identification mechanism.
3. **Nesting matches Elizabeth's actual per-pair workflow.** Cross-platform correlation is already
   organized as one YAML per directed pair (4 files today). Turning on prediction for a pair means
   adding one `prediction:` block to that pair's existing YAML and rerunning the same
   `cross-platform` command — one command, one run directory, one report containing both
   correlation and predictability numbers for the same pair, matching the paper's actual framing
   (reframe correlation *as* predictability for the same pairs, not a separate analysis). A
   standalone config would need a second, differently-shaped file (a multi-pair batch descriptor)
   and a second command, duplicating pair-identification information that already lives in the 4
   existing YAMLs, with a real risk of the two silently drifting.

**Cost accepted:** `PredictionConfig` needs its own BLUP-path fields (Decision 2) since
`exp1_data_path`/`exp2_data_path` must stay raw-per-sample-only for the unchanged correlation
steps 1-5. `platform_pairs` (Decision 3) narrows from a general multi-pair batch descriptor to a
single-entry direction descriptor for the one pair `CrossPlatformConfig` already names.

**Alternatives considered:**
- **Standalone sibling config + new `PredictionPipeline` + new CLI command.** Initially chosen,
  then reversed after the `enrichment_enabled`/backward-compat-oracle analysis above. Rejected:
  vacuous backward-compat oracle, a second config shape and command to maintain and keep in sync
  with the 4 existing per-pair YAMLs, and no meaningful UX advantage once
  `platform_data_paths`-style path resolution turned out to need its own fields regardless of
  nesting.
- **New CLI command reusing `CrossPlatformPipeline` machinery internally.** Rejected: still needs a
  new BLUP-loading step (`LoadCrossPlatformDataStep` is raw-per-sample-only), so it saves no real
  implementation work over nesting while adding a second command surface.

**Capability placement (raised during `/review-openspec` round 3):** a separate `cross-platform-
prediction` OpenSpec capability already exists (created when Tier 3 was archived), housing the
LOGO-CV/per-fold-PCA statistical requirements this tier consumes unchanged. This tier's new
requirements (`PredictionConfig`, `PredictCrossPlatformStep`) were added to the `cross-platform-
analysis` capability instead, without this alternative ever being explicitly weighed — a real
diligence gap given every other decision in this document has an explicit alternatives-considered
note. Resolved: `cross-platform-analysis` is the correct home — `PredictionConfig` nests on
`CrossPlatformConfig` and the step is task 6 of `CrossPlatformPipeline`, both objects the
`cross-platform-analysis` capability already owns; splitting the new requirements into the
`cross-platform-prediction` capability would force a reader of `.prediction`'s field list to jump to
a different capability file than the one documenting the config object it lives on.
`cross-platform-prediction` remains correctly scoped to the reusable statistical functions
(`logo_cv_predict`, `fit_pca_on_fold`), independent of which pipeline calls them.

### Decision 2: `predictor_source` needs its own path fields, separate from `exp1_data_path`/`exp2_data_path`

**What:** `PredictionConfig` gains `source_blup_path: Optional[str] = None` and
`target_blup_path: Optional[str] = None`, required (and existence-checked) only when
`predictor_source == "blup"`. When `predictor_source == "genotype_means"`, `PredictCrossPlatformStep`
instead reuses the same raw per-sample data `LoadCrossPlatformDataStep` (task 1) already loaded and
aligned, aggregated via a plain `.groupby(genotype_col).mean()` — matching
`ReduceTraitRedundancyStep`'s existing convention (`reduce_trait_redundancy.py:208`) exactly, no
new path fields needed for this branch.

**Why:** `exp1_data_path`/`exp2_data_path` feed the raw-per-sample correlation steps (1-5), which
must keep working unchanged even when prediction is also enabled — they cannot be repointed at
BLUP tables (a different shape: one row per genotype, no replicate dimension) without breaking
correlation. BLUP tables are a genuinely different data source, produced by an earlier, separate
pipeline run (Tier 1's `StatisticalAnalysisStep`), so `PredictionConfig` needs its own way to name
where they live.

**Alternatives considered:**
- **Repoint `exp1_data_path`/`exp2_data_path` at BLUP CSVs when `predictor_source=blup`.** Rejected
  — breaks correlation steps 1-5, which need raw per-sample data regardless of whether prediction
  is also enabled in the same run.
- **A generic `platform_data_paths: dict[str, str]` lookup (considered during brainstorm for the
  standalone-config option).** Unnecessary once nesting was chosen — a single pair only ever needs
  two named paths, not a general name→path map.

### Decision 3: `platform_pairs` narrows to a single-entry direction descriptor, validated in `CrossPlatformConfig.__post_init__`

**What:** `PredictionConfig.platform_pairs: list[dict] = field(default_factory=list)` holds exactly
one entry, `[{"source": <exp1_name or exp2_name>, "target": <the other>}]`, describing which of the
pair's two platforms is the predictor and which is predicted (correlation, steps 1-5, is direction-
agnostic; prediction is inherently directional — Turface19→Cylinder is a different model from
Cylinder→Turface19). Because `PredictionConfig` is constructed independently (no visibility into
its parent), this cross-field consistency check — that `platform_pairs`' `{source, target}` set
equals `{exp1_name, exp2_name}` — is validated in `CrossPlatformConfig.__post_init__` (which already
exists and can see both `self.prediction` and `self.exp1_name`/`self.exp2_name`), not in
`PredictionConfig.__post_init__` in isolation.

**Why:** `source_blup_path`/`target_blup_path` alone already tell `PredictCrossPlatformStep` which
file is the predictor and which is the target; `platform_pairs` supplies the human-readable
platform *names* for labeling `CrossPlatformPredictionResult.source_platform`/`target_platform`
(Tier 3's existing fields) and for cross-checking that the caller didn't accidentally swap
`exp1`/`exp2` relative to their `source_blup_path`/`target_blup_path` choice.

**Alternatives considered:**
- **Drop `platform_pairs` entirely, infer direction from field order alone.** Considered; kept
  `platform_pairs` instead because the roadmap's settled decision explicitly names it as a
  `PredictionConfig` field, and it gives a validated, explicit direction statement rather than an
  implicit one inferred from which field happens to be named `source_blup_path`.

### Decision 4: `PredictionConfig.__post_init__` validation is fully skipped when `enabled=False`

**What:** `PredictionConfig.__post_init__` returns immediately, performing no validation at all,
when `self.enabled` is `False` (the default). All structural validation (valid
`predictor_source`/`reduction_method`/`comparison_methods`/`representative_selection_metric`
values, the `blup_refit_per_fold` auto-force-on-`heritability` check) and the pre-flight BLUP-path
existence check run only when `enabled=True`.

**Why:** Matches `enrichment_enabled`'s exact backward-compat contract (Decision 1): an existing
`CrossPlatformConfig` YAML that never mentions `prediction:` gets `PredictionConfig()`'s all-default
instance (`enabled=False`, `source_blup_path=None`, etc.) — if validation ran unconditionally, the
default `predictor_source="blup"` combined with `source_blup_path=None` would raise for every
existing cross-platform config in the repo that has never heard of this tier, which would violate
backward compatibility outright.

**Alternatives considered:**
- **Validate structure always, only skip the path-existence check when disabled.** Rejected: still
  breaks on `enabled=False` defaults as described above, since the default `predictor_source="blup"`
  would still fail a "path must be set" structural check even before reaching existence-on-disk.

### Decision 5: Pre-flight validation raises plain `ValueError`, not a new `ConfigValidationError`

**What:** `PredictionConfig.__post_init__` and `CrossPlatformConfig.__post_init__`'s new
cross-field check both raise plain `ValueError` on failure.

**Why:** Confirmed by repo-wide search: no `ConfigValidationError` class exists anywhere in this
codebase. Every existing config's own `__post_init__` (including `CrossPlatformConfig`'s current
body) raises plain `ValueError`. The roadmap's "`PredictionConfig.__post_init__` raises
`ConfigValidationError`" language is read here as describing the *behavior* (fails at config-load
time, before any pipeline step runs) rather than committing to a literal new exception class name —
introducing a new public exception type for one field, inconsistent with every other config in the
codebase, would be a larger and unrelated API-surface change smuggled into a wiring tier.

**Alternatives considered:**
- **Introduce `ConfigValidationError(ValueError)` as a new public exception, used only by
  `PredictionConfig`.** Rejected: no precedent, inconsistent with QC/Viz/`CrossPlatformConfig`'s own
  existing `ValueError` convention, and a genuine cross-cutting exception-type decision that (if
  wanted) deserves its own proposal rather than being introduced incidentally here.

### Decision 6: Target-set construction reuses Tier 3's existing shape; PC1-as-target is ground truth, not per-fold

**What:** For a given directed pair, `PredictCrossPlatformStep`:

1. Loads the source and target predictor matrices per Decision 2 (BLUP CSV or raw-data
   genotype-mean, depending on `predictor_source`), aligns to genotypes common to both.
2. Selects the **target** platform's cluster-representative traits (the prediction targets) via
   the existing, reused-as-is `cluster_correlated_traits`/`select_cluster_representatives`
   (`cross_experiment_analysis.py`), driven by `representative_selection_metric`.
3. Computes one additional target, `target_name="PC1"`: the **target** platform's own first
   principal component, via the existing **pipeline-level** `PCA` step (`pca.py`), fit once on the
   full common-genotype set. This is ground truth being predicted, not a predictor — there is no
   leakage concern in computing it from all genotypes, unlike `fit_pca_on_fold`
   (Tier 3), which remains reserved for reducing the **predictor** (source) matrix per-fold when
   `reduction_method="pc1"` is the chosen method. These are two distinct uses of "PC1" in this
   program and must not be conflated: one is a per-fold *predictor*-reduction utility (Tier 3,
   unchanged), the other is a whole-dataset *target* value (new in this tier, using the existing
   pipeline PCA step, not `fit_pca_on_fold`).
4. For each target trait (representatives + PC1) and each method (`config.reduction_method` plus
   each of `config.comparison_methods`), calls `logo_cv_predict(X=source_matrix, y=target_values,
   genotypes=common_genotypes, reduction_method=method, representative_names=...)` — where
   `representative_names`, when `method="representatives"`, is the **source** platform's own
   cluster-representative trait names (a separate application of the same clustering functions to
   the source's matrix, unsupervised and safe to fix pre-loop per `theory.md` §2.2).
5. Assembles one `CrossPlatformPredictionResult` per method via
   `CrossPlatformPredictionResult.from_logo_cv_results(...)` (Tier 3, unchanged), saving each to the
   run directory as JSON.

**Why:** This is the literal reading of Tier 3's own `design.md` Decision 5 ("each cluster-
representative trait in the target platform... plus one additional `TargetPrediction` with
`target_name="PC1"`"), which this tier must satisfy without modifying. Clarifying the
ground-truth-vs-per-fold-predictor distinction for PC1 up front avoids an implementer accidentally
using `fit_pca_on_fold` (a per-fold, training-only utility) to compute the *target* value, which
would be a category error (there is no "leakage" to prevent when computing a ground-truth label —
leakage is specifically about information flowing into the model's *inputs*).

**Alternatives considered:**
- **Compute target PC1 per-fold via `fit_pca_on_fold`, matching the predictor-side pattern.**
  Rejected: `fit_pca_on_fold`'s entire contract is "fit on training data, project held-out data" —
  appropriate for a value the model *uses as input*, not for the ground-truth value the model is
  being scored against. Using it here would need a wholly different call pattern (project the
  held-out genotype's *true* PC1 using loadings that exclude it) that doesn't match how any
  existing test or the Tier 3 result type is documented, and isn't what the roadmap's plain
  language describes.

### Decision 7: `representative_selection_metric` is restricted to `"variance"` for this tier — `"heritability"` deferred

**What:** `PredictionConfig.representative_selection_metric` accepts only `"variance"` in this tier
(not the `{variance, heritability}` pair the roadmap originally sketched). `blup_refit_per_fold`
stays in the schema (`bool = False`) per the roadmap's settled field list, but its
auto-force-on-`heritability` validation is removed, since there is currently no
`representative_selection_metric` value that would trigger it — it is documented as **inert for
this tier**, reserved for a future change that adds heritability-based representative selection.

**Why:** Found during `/review-openspec` round 1 (architecture reviewer): `select_cluster_representatives`
(`cross_experiment_analysis.py:1961-2014`) has no metric parameter at all — its docstring states
"Selection criterion: trait with highest variance across genotypes," hardcoded, no branch, no
`heritability` option anywhere in that file. This tier's own Non-Goals already rule out changing
`cluster_correlated_traits`/`select_cluster_representatives`, so shipping a `heritability` enum
value with no implementation path behind it would validate a behavior the codebase cannot produce —
exactly the kind of "looks done, isn't" gap the review caught. Restricting to `variance` now, and
filing heritability-based selection as a follow-up once `select_cluster_representatives` (or a new
sibling function) actually supports a metric parameter, keeps this tier's promise (config validates
what the pipeline can actually do) honest.

**Alternatives considered:**
- **Build heritability-based selection now, inside this tier.** Rejected: a genuine new statistical
  capability (which traits get selected based on H², not variance), out of place in a tier whose
  entire premise is "wiring only, no new statistics" — this is Tier-3-adjacent scope, not Tier 3.5's.
- **Ship the `heritability` enum value now, implement it later.** Rejected: this is the exact
  bug the review caught — a validated option with zero consumer, and (per `/review-openspec`'s
  finding) `blup_refit_per_fold`'s own auto-force logic would be validating a per-fold-refit
  contract that `logo_cv_predict`'s `representative_names` parameter (a single static list computed
  once outside the fold loop, per Tier 3's own contract) has no hook to honor even if a heritability
  metric existed.

### Decision 8: `depends_on` includes both `"01_load_cross_platform_data"` and `"05_visualize_cross_platform"`

**What:** Task 6 (`PredictCrossPlatformStep`) declares `depends_on=["01_load_cross_platform_data",
"05_visualize_cross_platform"]`, not `["05_visualize_cross_platform"]` alone. When
`predictor_source="genotype_means"`, the step reads `exp1_df`/`exp2_df` directly from task 1's
result (via the `dag`/`Task` mechanism's `kwargs["01_load_cross_platform_data"]`, confirmed at
`pipeline/dag.py:151-154` and `pipeline/task.py:126-131` — a task only receives `kwargs` for names
literally listed in its own `depends_on`), not from task 5's.

**Why:** Found during `/review-openspec` round 1: `depends_on=["05_visualize_cross_platform"]` alone
only "works" because `exp1_df`/`exp2_df` happen to be forwarded, untouched, through 3 intermediate
steps' `data` dicts — an undocumented, fragile reliance with no named mechanism anywhere in this
tier's original text. Worse, it silently breaks when the same `CrossPlatformConfig` also sets
`trait_reduction_method="clustering"` for the (unrelated) correlation steps:
`ReduceTraitRedundancyStep` (`reduce_trait_redundancy.py:280-281`) **replaces** `exp1_df`/`exp2_df`
with only the cluster-representative columns before forwarding them onward — so a user combining
clustering-based trait reduction with `predictor_source="genotype_means"` (documented, per
roadmap.md/theory.md, as "the full raw-data ablation") would silently get a reduced trait matrix
instead of the full one. Declaring both dependencies explicitly and reading from task 1's own result
removes both the undocumented reliance and the silent clustering interaction.

**Alternatives considered:**
- **Keep `depends_on=["05_visualize_cross_platform"]` only, document the pass-through reliance.**
  Rejected: doesn't fix the clustering-interaction bug, which is a real behavioral defect, not just
  an undocumented-but-harmless implementation detail.
- **Add a config-level guard rejecting `trait_reduction_method="clustering"` combined with
  `predictor_source="genotype_means"`.** Considered as a supplementary safeguard; not adopted as the
  *primary* fix, since reading from task 1 directly is a strictly better fix (removes the bug
  entirely rather than forbidding the combination that triggers it) — but see tasks.md for a
  regression test covering this exact interaction regardless.

### Decision 9: Backward-compat oracle excludes `config.yaml` from the byte-identical comparison

**What:** Task 5.3's "byte-identical output" claim is scoped to the run's **analysis** output (the 5
existing steps' generated files: correlation CSVs, alignment summary, figures, `pipeline_summary.json`)
— not `config.yaml`. `config.yaml` is expected to gain a new `prediction: {enabled: false, ...}`
block the moment the `prediction` field exists on `CrossPlatformConfig`, regardless of whether
prediction is enabled, since `BasePipeline._save_config()` (`base_pipeline.py:186-199,227-247`)
recursively serializes every field of the resolved config, including nested dataclasses at their
default values.

**Why:** Found during `/review-openspec` round 1: the proposal's original "byte-identical" claim, as
stated, is falsifiable the moment the new field is added — independent of any runtime-behavior
change — which would make task 5.3 fail on a technicality unrelated to the actual backward-compat
property being tested (does *behavior* change when disabled?). Scoping the oracle to analysis output
only, and explicitly documenting `config.yaml`'s expected (harmless) diff, keeps the test meaningful
without setting it up to fail on a config-serialization artifact.

**Alternatives considered:**
- **Special-case `config.yaml` serialization to omit default-valued fields.** Rejected: a
  general-purpose "omit defaults from provenance dumps" feature is a larger, unrelated change to
  `BasePipeline._save_config()`'s reproducibility contract (which intentionally serializes the
  *complete* resolved config for every run, per the `cli-pipeline` spec's "Pipeline Run Config
  Provenance" requirement) — not something to introduce incidentally in a wiring tier.

### Decision 10: `platform_pairs` cardinality is validated explicitly

**What:** `CrossPlatformConfig.__post_init__`'s Decision 3 cross-check additionally validates that
`prediction.platform_pairs` has **exactly one entry** (not zero, not more than one) whenever
`prediction.enabled=True`, before checking that entry's `{source, target}` names against
`exp1_name`/`exp2_name`.

**Why:** Found independently by three of five `/review-openspec` round-1 reviewers: the spec's own
"SHALL contain exactly one entry" language had no scenario or test enforcing it — a user enabling
prediction and forgetting `platform_pairs` (default empty list) would hit an undocumented
`IndexError`/`KeyError` deep in whichever code first indexes into the list, rather than the clean,
early `ValueError` Decision 5 promises.

**Alternatives considered:** None — this is a straightforward completion of Decision 3's existing
validation, not a new design choice.

### Decision 11: Target-trait *selection* uses full-outcome data — a selection-bias note, not a leakage fix

**What:** Step 2 of Decision 6's algorithm (target-side cluster-representative trait selection) is
computed from the full common-genotype **target** matrix, including whichever genotype a later LOGO
fold will hold out when scoring that trait. This is documented explicitly, in `PredictCrossPlatformStep`'s
docstring, as a **selection-bias** consideration distinct from **fit-time leakage**: no Ridge/PLS
coefficient ever sees a held-out genotype's target value (theory.md's CV-hygiene contract, which
concerns model *fitting*, is not violated), but the *choice of which traits are headline predictable
targets* is made using every genotype's own outcome data, including data a later fold will pretend
not to have. This differs from the *source*-side "representatives" predictor selection
(theory.md §2.2's "safe to fix pre-loop" case), which never touches `y` at all.

**Why:** Found during `/review-openspec` round 1 (architecture reviewer) as a real, if subtle,
distinction the original design conflated. Not a blocking defect — it does not inflate `logo_cv_predict`'s
reported R² the way theory.md's leakage regression test would detect, since it doesn't touch model
fitting — but worth an explicit docstring callout so a future reader doesn't assume "target selection
is unsupervised and safe" carries the same guarantee as "source predictor selection is unsupervised
and safe" (theory.md's own language for the latter, not the former).

**Alternatives considered:**
- **Select target traits from a held-out-genotype-excluding subset, per fold.** Rejected as
  overengineering for this tier: target-trait *identity* (which traits are the headline predictable
  set) is a reporting choice, not a parameter the model fits — refitting it per fold would multiply
  Section 6's oracle complexity for a concern this tier's own leakage regression test (inherited
  unchanged from Tier 3) does not measure.

### Decision 12: PC1-as-target uses `pca.fit_pca()` directly with fixed hyperparameters, not `PCAAnalysisStep`

**What:** The target's PC1 ground-truth value (Decision 6 step 3) is computed as:
`pca.fit_pca(StandardScaler().fit_transform(target_matrix), n_components=1, random_state=42)`
(`pca.py:180-199`) — called directly, with standardization applied first (matching every other
`logo_cv_predict` reduction method's `StandardScaler`-then-model convention) and `random_state=42`
fixed (matching this codebase's existing default, e.g. UMAP's `random_state=42`). This tier does
**not** add a `PCAConfig` to `CrossPlatformConfig`/`PredictionConfig`, and does **not** wire in the
existing `PCAAnalysisStep` (`pipeline/steps/pca_analysis.py`), which is config-driven
(`config.pca.standardize`, `pca_analysis.py:73`) and has never been part of `CrossPlatformPipeline`.

**Why:** Found during `/review-openspec` round 1: the original text ("the existing pipeline-level
`PCA` step (`pca.py`)") named a module, not a specific function or a config path, and `pca.py` itself
does not standardize before fitting (`fit_pca()`'s own body is a bare `PCA(...).fit_transform(X)` —
standardization is the caller's job, confirmed at `pca.py:197-198`). Since `CrossPlatformConfig`/
`PredictionConfig` has no `PCAConfig` field for `PCAAnalysisStep` to read settings from, wiring that
full step in would require adding one — a larger schema change than a wiring tier warrants for a
single, fixed ground-truth computation. Fixed hyperparameters (not user-configurable) match this
tier's overall shape: the new step is deliberately simple, and PC1-as-target is documented as such,
not exposed as a tunable.

**Alternatives considered:**
- **Wire in `PCAAnalysisStep` with a new `PCAConfig` field.** Rejected: introduces a new
  config surface (n_components, standardize, etc. as user-facing knobs) for a value this tier only
  needs once, as a fixed ground truth — disproportionate scope increase for a wiring tier.
- **Use `pca.py`'s other exported functions (`perform_pca_analysis`, `run_pca_and_export_artifacts`).**
  Rejected: both are higher-level orchestration functions bundling file I/O and metrics computation
  this tier doesn't need; `fit_pca()` is the minimal function that does exactly "fit PCA, return
  transformed values."

### Decision 13: `genotype_means` reads task 1's pre-filtered trait-name list, not the raw DataFrame's every column

**What:** When `predictor_source="genotype_means"`, `PredictCrossPlatformStep` selects columns using
`kwargs["01_load_cross_platform_data"].data.metadata["exp1_trait_names"]` /
`["exp2_trait_names"]` (`LoadCrossPlatformDataStep`'s own already-computed, `exclude_cols`-filtered
trait-name lists — via `get_trait_columns()`, `data_cleanup.py:80-125`) **before** grouping by
genotype and averaging — not a bare `.groupby(genotype_col).mean()` over task 1's entire
`exp1_df`/`exp2_df`.

**Why:** Found during `/review-openspec` round 2: task 1's raw `exp1_df`/`exp2_df`
(`load_cross_platform_data.py:158-162`) are the full aligned frame, still containing `genotype`,
`replicate`, and any metadata/date/notes columns — the trait-only column list is computed
separately and stashed only in `StepResult.metadata`, never applied back to the DataFrame itself.
Decision 2's original text ("matching `ReduceTraitRedundancyStep`'s existing convention,
`reduce_trait_redundancy.py:208`, exactly") also **mis-cited that precedent**: line 208 is actually
`df.groupby("genotype")[trait_names].mean()` — it explicitly subsets to a pre-filtered trait-name
list before aggregating, the same pattern as `calculate_genotype_means(df, trait_cols,
genotype_col)` (`cross_experiment_analysis.py:702-721`). A literal `.groupby(genotype_col).mean()`
over the raw frame, as Decision 8's fix could otherwise be read to imply, would either crash
(pandas ≥2.0 defaults `.mean()` to `numeric_only=False`, so a non-numeric metadata column raises)
or silently pollute the predictor matrix with non-trait columns and any `exp1_exclude_cols`/
`exp2_exclude_cols` the user explicitly asked excluded — defeating both the "full raw-data
ablation" contract this branch exists for and the exclude-cols contract every other step in this
pipeline honors.

**Alternatives considered:**
- **Have task 6 call `get_trait_columns()` itself, independently of task 1's already-computed
  list.** Rejected: redundant work and a second place the trait-selection logic could drift from
  task 1's own; reading task 1's already-computed, already-filtered metadata is simpler and
  guaranteed consistent with what steps 1-5 consider "the trait set" for this run.

### Decision 14: `X`, every per-target `y`, and `genotypes` are derived from one canonical, sorted common-genotype index

**What:** `PredictCrossPlatformStep` computes a single canonical, sorted list of genotypes common to
both source and target predictor matrices, once, and indexes `X` (via `.loc[canonical_genotypes]`)
and every per-target `y` vector from that same canonical list every time — never relying on two
DataFrames' incidental row order matching after independent loads/joins.

**Why:** Found during `/review-openspec` round 2: `logo_cv_predict`'s own docstring documents row-
order alignment between `X`, `y`, and `genotypes` as an **unenforced caller precondition** — nothing
inside the function can detect a systematic row-order mismatch, which would silently produce a
plausible-looking but wrong LOGO-CV result (a genotype-value swap), not a crash. This tier's step is
a higher-risk caller than a typical single-alignment consumer: it builds **multiple independent `y`
vectors** (one per target trait, including a separately-computed PC1) against one `X`, and — for
`predictor_source="blup"` — `X` and each `y` originate from two separately-loaded CSVs that must be
correctly joined every time. `tasks.md`'s existing 4.5 only tests *set*-membership correctness
(genotypes present in only one side are excluded), never *order*. Deriving everything from one
canonical, explicitly-indexed list removes the alignment risk structurally rather than trusting
incidental DataFrame join behavior.

**Alternatives considered:**
- **Trust `pandas` join/merge to preserve consistent order across `X` and every `y`.** Rejected:
  even if true in the common case, it is not a guaranteed contract across independently-loaded
  CSVs and multiple separate join operations (one per target trait) — a future refactor could
  silently break it with no test catching the regression, exactly the failure mode a canonical,
  explicitly-indexed approach avoids by construction.

### Decision 15: Task 6's second `depends_on` entry is for ordering only, not data

**What:** `PredictCrossPlatformStep`'s `depends_on=["01_load_cross_platform_data",
"05_visualize_cross_platform"]` (Decision 8) uses `kwargs["01_load_cross_platform_data"]` for actual
data (both `predictor_source` branches). `kwargs["05_visualize_cross_platform"]` is depended-upon
**only** to guarantee task ordering — i.e., that steps 1-5 completed successfully before prediction
runs, consistent with this being conceptually "step 6" of one coherent per-pair analysis — its
`data` payload is not otherwise read by `_run_predict_cross_platform`.

**Why:** Found during `/review-openspec` round 2: `BaseStep.execute(self, data, config, run_dir,
prev_result=None)` takes exactly one `data`/`prev_result` pair, and every existing
`CrossPlatformPipeline` runner method threads exactly one upstream result through it. Decision 8
gives task 6 two named dependencies without previously specifying how `_run_predict_cross_platform`
(tasks.md 5.4) reconciles two `kwargs` entries against that one-argument contract. This decision
resolves the ambiguity explicitly: task 5's result is consumed for DAG-ordering purposes only.

**Alternatives considered:**
- **Thread both dependencies' data into `PredictCrossPlatformStep.execute()` via a combined dict.**
  Rejected as unnecessary: task 6 never needs anything from steps 2-5's outputs (correlation
  results, visualizations) — only that they ran first and task 1's raw data. A single `data`
  argument (from task 1) plus the ordering-only second dependency keeps `BaseStep`'s existing
  single-`data`-argument contract intact for every step, including this one.

### Decision 16: NaN trait columns are dropped before building any predictor/target matrix

**What:** Before building the source predictor matrix `X`, or selecting/computing any target
value, any trait column containing **any** `NaN` value among the common-genotype set is dropped
entirely — a simple, conservative policy (no imputation), applied uniformly to both the source
side (predictor traits) and the target side (candidate representative traits, before
`select_cluster_representatives` runs). If dropping leaves the **source** matrix with zero trait
columns, the step SHALL raise a clear `ValueError` (a genuinely unusable predictor matrix — distinct
from, and stricter than, the target-side zero-representative-traits case already handled by task
4.3a, which still has PC1 to fall back on).

**Why:** Found during `/review-openspec` round 3: real `08_blup_adjusted_means.csv` files are
written directly from `extract_blup_table()`'s raw output
(`pipeline/steps/statistical_analysis.py:332-338`), **not** filtered through `BLUPResult`'s
finite-only-columns split. `extract_blup_table()`'s own documented contract (Tier 1,
`statistics.py:770-780`) is that a trait whose mixed-model fit failed gets an **entire NaN column**
— an expected, normal occurrence at EDPIE's `n≈19` scale, not an edge case. `logo_cv_predict`
(`cross_platform_prediction.py:243-244`) hard-rejects any `NaN` in `X` with no partial-tolerance
path. Nothing in this proposal, through 2 full review rounds, addressed this — and the CI synthetic
fixture (task 1.1) is fully finite by construction, so this gap would have shipped silently and
only surfaced as a crash during Section 8's manual real-EDPIE validation, a "looks done in CI,
breaks in production" pattern this program has flagged before (e.g. Tier 3's own round-1 CI-path
bug). The same root gap applies to `predictor_source="genotype_means"`, since
`.groupby(genotype_col).mean()` over raw per-sample data with missing values can likewise produce
NaN cells.

**Alternatives considered:**
- **Impute missing values (e.g. mean/median fill).** Rejected: a real statistical modeling choice
  with its own leakage/CV-hygiene implications (imputing from the full dataset before the fold loop
  would need the same per-fold-refit scrutiny as any other data-derived step) — out of place in a
  wiring-only tier. A future tier can revisit this if dropping NaN columns turns out to discard too
  much real signal (Section 8's manual validation will show this concretely).
- **Drop only genotypes with any NaN, keep all trait columns.** Rejected: a genotype missing one
  trait's value (common, per `extract_blup_table`'s documented cell-level NaN behavior) would drop
  that genotype from every target's LOGO-CV entirely, a much larger information loss than dropping
  just the affected trait column.

### Decision 17: BLUP CSV genotype-column name is resolved by a fixed convention, not a new config field

**What:** `source_blup_path`/`target_blup_path` are loaded by checking for a column named
`"Genotype"` first (the real, shipped convention — hardcoded in
`pipeline/steps/statistical_analysis.py:90`, `genotype_col = "Genotype"`, baked into
`08_blup_adjusted_means.csv` via `reset_index(names=genotype_col)`), falling back to `"genotype"`
(this pipeline's own dominant lowercase convention elsewhere, e.g.
`load_cross_platform_data.py:118`). Neither is `exp1_genotype_col`/`exp2_genotype_col` — those
govern the unrelated raw per-sample CSVs for steps 1-5, a different file and a different,
user-configurable convention entirely. If neither column name is present, the step SHALL raise a
clear `ValueError` naming both attempted column names, not a bare pandas `KeyError`.

**Why:** Found during `/review-openspec` round 3: no prior decision (including Decisions 2, 6, 13,
14) specified how the BLUP CSV's genotype column is identified at all — a real gap on exactly the
kind of caller-precondition surface Decision 14 was otherwise careful about (row order). Given this
codebase's own dominant lowercase `"genotype"` convention almost everywhere else prediction-adjacent
code lives, an implementer guessing `index_col="genotype"` against the real, capitalized
`08_blup_adjusted_means.csv` would hit a `KeyError` — loud, not silent, but an unnecessary landmine
with no visibility before implementation.

**Alternatives considered:**
- **Add a `PredictionConfig` field for the BLUP genotype-column name.** Rejected: the real column
  name is a fixed, code-level fact about Tier 1's own output contract (`statistical_analysis.py:90`
  hardcodes `"Genotype"`), not something that varies per user/config — making it configurable would
  invite a user to "fix" a mismatch that actually indicates their BLUP file wasn't produced by this
  pipeline's own Tier 1 step, which should be a clear error, not a silent workaround.

## Risks / Trade-offs

- **Duplicate genotype labels within a single loaded BLUP table** (a data-quality defect, distinct
  from Decision 14's shuffled-row-order case) are not explicitly guarded before `.loc[]`-reindexing
  — most likely surfaces as `logo_cv_predict`'s own `len(X) != len(genotypes)` crash rather than
  silent corruption (contained blast radius), but untested. Found during `/review-openspec` round 3;
  low-medium severity, not fixed with new code — flagged for implementation-time awareness.
- **`PredictionConfig` is a mutable dataclass nested inside frozen `CrossPlatformConfig`** — nothing
  prevents `config.prediction.enabled = True` (or any other field) being mutated in place after
  construction, bypassing every validation guarantee in Decisions 3/4/5/7/10/13 (all construction-
  time only). Investigated during `/review-openspec` round 2: this is not a new risk this tier
  introduces — every other `VizPipelineConfig`/`QCPipelineConfig` sub-config (`StatisticsConfig`,
  `HeritabilityConfig`, etc.) is likewise a plain, mutable dataclass with the same property, and
  this codebase has never guarded against post-construction mutation of nested config objects
  anywhere. Documented here as an accepted, pre-existing convention, not remedied specially for
  `PredictionConfig` alone.
- **`comparison_methods` and multiple target traits multiply `logo_cv_predict` calls.** For a
  target platform with, say, 15 cluster-representative traits, `comparison_methods=[
  "representatives"]`, and the default `reduction_method="pls_latent"`, one directed pair requires
  `16 targets × 2 methods = 32` `logo_cv_predict` calls (matching Tier 3 Section 8's own manual
  validation scale, `32` combinations for one pair) — cheap at this program's `n≈19` genotype
  scale (no permutation loop here; that's Tier 4), but worth noting as the reason `tasks.md`'s CI
  fixture stays small (a handful of traits, not EDPIE's real ~15-30).
- **Backward-compat golden-fixture regression test needs a real pre-existing `CrossPlatformPipeline`
  fixture to diff against.** A curated Tier-3 golden fixture already exists
  (`tests/fixtures/real/wheat_edpie/expected/cross_platform/<pairing>/`), but per Tier 3's own
  tasks.md (task 1.4, step 4) it is a deliberately curated subset — `config.yaml`,
  `cross_platform_alignment_summary.csv`, `cross_platform_correlations.csv`,
  `exp{1,2}_trait_clusters.csv`, `pipeline_summary.json` — excluding PNGs, logs, and step-1's loaded
  intermediate CSVs. Found during `/review-openspec` round 1 that this proposal's "byte-identical
  file list" claim can't be checked against that curated subset as-is; resolved by capturing a fresh,
  full-file-list snapshot from the small CI-fast synthetic fixture (tasks.md Section 1.1/1.2), not
  real EDPIE data, keeping the backward-compat regression test fast and decoupled from Section 8's
  manual real-data validation. See Decision 9 for the `config.yaml`-exclusion scoping this snapshot
  also needs.
- **`PredictCrossPlatformStep` introduces a third, independent application of
  `cluster_correlated_traits`/`select_cluster_representatives`** (target-side target selection,
  source-side representative-method predictor selection, in addition to Tier 3's own oracle use and
  the existing `ReduceTraitRedundancyStep` use) — worth a clear docstring cross-reference so a
  future reader doesn't assume these three call sites must always agree on the same trait set (they
  operate on different matrices: aligned raw/BLUP source data vs. aligned raw/BLUP target data vs.
  `ReduceTraitRedundancyStep`'s own genotype-mean matrices for the *correlation* steps).

## Migration Plan

Purely additive to existing behavior when disabled (the default). `CrossPlatformConfig` gains one
new field (`prediction`); `CrossPlatformPipeline` gains one new, conditionally-included task. No
existing function, CLI flag, or config field changes shape. No existing caller that doesn't set
`prediction: {enabled: true, ...}` is affected.

## Open Questions

None blocking after `/review-openspec` round 1's reconciliation (see below). Two related, explicitly
out-of-scope gaps were found during this tier's brainstorm and filed as separate follow-up issues
rather than folded in here: `CrossPlatformSummaryGenerator` not surfacing prediction results (#197),
and `/configure-run-all`/`/dry-run`/`/validate-config` not covering cross-platform (or now
prediction) configs at all (#198, pre-existing, not caused by this tier). A third, narrower gap
found during round 1 — heritability-based representative selection has no implementation path
(Decision 7) — is a candidate for a similar follow-up issue once `select_cluster_representatives`
supports a metric parameter; not yet filed, pending Elizabeth's go-ahead the same way #197/#198 were.

## Adversarial Review Reconciliation (round 1)

`/review-openspec` ran 5 parallel reviewers (spec quality, TDD/testing, pipeline architecture &
statistical correctness, documentation, git workflow) against this proposal before any
implementation began. 5 BLOCKING and roughly a dozen IMPORTANT findings, reconciled as follows:

- **BLOCKING — `representative_selection_metric="heritability"` had no implementation path.**
  `select_cluster_representatives` has no metric parameter (hardcoded to variance); this tier's own
  Non-Goals already forbid changing it. Fixed: Decision 7 restricts the field to `"variance"` for
  this tier; `blup_refit_per_fold` stays in the schema (roadmap's settled field list) but is
  documented as currently inert, with its auto-force validation removed since nothing triggers it.
- **BLOCKING — task 6's `depends_on=["05_visualize_cross_platform"]` relied on undocumented,
  fragile data pass-through, and silently broke under `trait_reduction_method="clustering"`.**
  Confirmed at the `dag.py`/`task.py` level that a task only receives `kwargs` for names in its own
  `depends_on`; `ReduceTraitRedundancyStep` replaces `exp1_df`/`exp2_df` with cluster-representative
  columns only, defeating `predictor_source="genotype_means"`'s "full raw-data ablation" promise
  when clustering is also enabled. Fixed: Decision 8 adds `"01_load_cross_platform_data"` to
  `depends_on` explicitly; tasks.md gains a regression test for the clustering interaction.
- **BLOCKING — "byte-identical when disabled" was false as stated.** `BasePipeline._save_config()`
  serializes every field, including new fields at their default values, into `config.yaml`
  regardless of runtime behavior. Fixed: Decision 9 scopes the backward-compat oracle to analysis
  output only, documenting `config.yaml`'s expected (harmless) diff.
- **BLOCKING — `platform_pairs` cardinality ("exactly one entry") was asserted but never enforced.**
  Found independently by 3 of 5 reviewers. Fixed: Decision 10; tasks.md gains explicit
  zero-entries/multiple-entries tests.
- **BLOCKING — tasks.md task 2.1 could not pass on Section 2's implementation alone**, since it
  asserted a `CrossPlatformConfig.prediction` field that isn't added until Section 3. Fixed: tasks.md
  2.1 trimmed to test `PredictionConfig()` standalone only; the nested-field assertion stays solely
  in task 3.1, restoring per-section commit atomicity.
- **IMPORTANT — `comparison_methods`'s valid-value set was referenced but never defined.** Fixed:
  spec.md now states it's drawn from the same 3-value set as `reduction_method`.
- **IMPORTANT — the new step's own 5-behavior list in spec.md had a matching `#### Scenario:` for
  only 1 of 5 (PC1), despite tasks.md already having tests for the other 4.** Fixed: spec.md gains 4
  additional scenarios (predictor-matrix construction ×2, `logo_cv_predict` call count, one-JSON-
  per-method).
- **IMPORTANT — Section 6's oracle asserted R² "matches exactly," inconsistent with this codebase's
  own `assert_allclose`/tolerance convention for pipeline-vs-direct-call reproduction claims
  (`tests/test_pipeline_reproduction.py`).** Fixed: reworded to `pytest.approx`.
- **IMPORTANT — task 1.3's golden-fixture baseline didn't account for the existing curated Tier-3
  fixture's documented exclusions (PNGs/logs/intermediate CSVs).** Fixed: reworded to capture a
  fresh, full-file-list snapshot from the CI-fast synthetic fixture, explicitly scoped per Decision 9.
- **IMPORTANT — `comparison_methods` naming the same method as `reduction_method` would silently
  overwrite one method's output JSON with the other's, with no validation.** Fixed: tasks.md gains a
  config-validation test rejecting this.
- **IMPORTANT — zero target-representative-traits and zero/near-zero common-genotype-overlap were
  untested edge cases**, previously left to surface as unclear, generic errors deep inside
  `logo_cv_predict`. Fixed: tasks.md gains tests for both, the latter requiring a clear, step-level
  error naming the pair and genotype counts rather than a bare pass-through.
- **IMPORTANT — the PC1-as-target mock-based test (4.4) only proved `fit_pca_on_fold` wasn't called,
  never that the computed values were correct.** Fixed: tasks.md gains a positive value-equality
  test against an independently-computed whole-dataset PCA.
- **IMPORTANT — "the existing pipeline-level PCA step" named a module, not a specific function or
  hyperparameters, and no `PCAConfig` exists on this pipeline.** Fixed: Decision 12 pins the exact
  call (`pca.fit_pca` with `StandardScaler` pre-applied, fixed `random_state=42`), explicitly
  rejecting a new `PCAConfig` as disproportionate scope for this tier.
- **IMPORTANT — doc task 9.1 misapplied API.md's `__all__`-driven pattern to a Config/Step pair that
  doesn't belong there** (Steps/Configs are documented in guide docs, never in API.md, confirmed
  against `__init__.py`'s `__all__`). Fixed: task 9.1 rewritten to drop the API.md target.
- **IMPORTANT — doc task 9.3 would have created a duplicate section, unaware that Tier 3 already
  shipped a `## Cross-Platform Genotype-Effect Prediction` section in
  `docs/CROSS_PLATFORM_ANALYSIS.md` with a now-stale forward-reference sentence.** Fixed: task 9.3
  rewritten to extend that existing section and correct its closing sentence.
- **IMPORTANT — task 8.2's wording foreclosed the possibility that a real EDPIE discrepancy might
  mean reopening a design decision (most likely Decision 6) rather than "just a wiring bug."** Fixed:
  reworded neutrally, with an explicit fallback task if findings are inconsistent with a pure
  wiring-bug explanation.
- **IMPORTANT — target-trait *selection* (as opposed to the PC1 *value*) is built from full-outcome
  target data, a materially different situation from the source-side "safe to fix pre-loop"
  representative selection theory.md §2.2 blesses.** Fixed: Decision 11 documents this explicitly as
  a selection-bias consideration, distinct from (and not remedied by) the leakage regression test
  inherited from Tier 3.
- **IMPORTANT — the MODIFIED "Cross-Platform Configuration" requirement silently backfilled
  pre-existing spec drift** (`enrichment_enabled`, `enrichment_p_value_column`, `validate_input` were
  already shipped in code but never added to the base spec by earlier changes). Not lossy, but
  undisclosed. Fixed: called out explicitly in `proposal.md`'s Impact section as a side-effect of
  this delta, not a new capability.
- **SUGGESTION — the change ID doesn't reference "cross-platform" the way the sibling Tier 3 change
  (`add-cross-platform-prediction`) does.** Not renamed (would require re-branching and re-filing
  the tracking issue for a purely stylistic gain) — noted here for future readers, not acted on.
- **SUGGESTION — task 9.4's docstring cross-reference location was unpinned.** Fixed: pinned to
  `PredictCrossPlatformStep`'s own docstring, not `cross_platform_summary.py` (keeping it clear of
  follow-up #197's territory even for a comment-only edit).

## Adversarial Review Reconciliation (round 2)

A second, independent round of the same 5-agent review (run fresh, with no memory of round 1,
specifically requested to catch anything round 1 missed) found round 1's investigative work held up
under direct re-verification (every code citation in Decisions 6/8/9/12 was independently confirmed
against source, the MODIFIED requirement was byte-diffed and confirmed lossless, and a hypothesized
PCA sign-flip gotcha was checked and ruled out — `sklearn`'s `svd_flip()` makes `PCA` deterministic
regardless of `random_state` at this scale). But it also found 2 new HIGH-severity issues — one a
direct consequence of round 1's own fix — plus several corroborated/new gaps:

- **HIGH (new) — Decision 8's fix (reading task 1's raw data directly) reintroduced a
  crash/data-pollution risk**, since task 1's raw `exp1_df`/`exp2_df` still contain non-trait
  columns and Decision 2's original precedent citation was itself inaccurate. Fixed: Decision 13 —
  select via task 1's already-computed, `exclude_cols`-filtered `exp{1,2}_trait_names` metadata
  before grouping; the mis-citation corrected.
- **HIGH (new) — genotype row-order alignment between `X`, every `y`, and `genotypes` was
  unenforced and untested**, a silent-wrong-result risk given `logo_cv_predict` treats order
  consistency as a caller precondition with no internal check. Fixed: Decision 14 — derive
  everything from one canonical, sorted, explicitly-indexed common-genotype list; `tasks.md` gains a
  deliberate-row-shuffle regression test.
- **MEDIUM (corroborated by 2 reviewers) — `blup_refit_per_fold`'s "inert" claim was asserted but
  never tested.** Fixed: `tasks.md` gains a regression test asserting `True` vs. `False` produce
  identical output.
- **MEDIUM (corroborated) — `comparison_methods` self-duplication** (e.g.
  `["representatives", "representatives"]`) wasn't rejected, the same silent-overwrite bug class as
  the cross-field case round 1 already fixed, just missed for the intra-list case. Fixed:
  `tasks.md`/`spec.md` gain a duplicate-entries-within-`comparison_methods` validation test.
- **MEDIUM (corroborated) — task 6.1's tolerance was wrong, not fixed.** `rel=1e-9` is ~1000x
  tighter than this codebase's actual documented cross-OS/BLAS tolerance convention
  (`rtol=1e-6, atol=1e-9`, `docs/reproducibility.md`), despite round 1 claiming it matched. Fixed:
  reworded to the actual convention.
- **MEDIUM (corroborated) — spec.md's scenario coverage was still incomplete.** Round 1 claimed 4
  new scenarios closed all 4 gaps in the step's 5-behavior list, but only 3 were actually added —
  target-side representative-trait *selection* had none. Fixed: added.
- **IMPORTANT (new) — task 6's two-`depends_on` wiring vs. `BaseStep`'s single-`data`-argument
  contract was unspecified.** Fixed: Decision 15 clarifies the second dependency is for ordering
  only.
- **IMPORTANT (new) — `PredictionConfig`'s mutability nested inside a frozen parent was an
  unaddressed gap.** Investigated and documented (Risks section) as a pre-existing, accepted
  codebase convention (every `VizPipelineConfig`/`QCPipelineConfig` sub-config has the same
  property), not remedied specially here.
- **IMPORTANT (new, documentation) — Section 9's doc plan wasn't sufficient for an actual end user**:
  no concrete YAML `prediction:` example, PC1's fixed hyperparameters not committed to any shipped
  doc, and task 9.4's cross-reference was docstring-only. Fixed: task 9.3 expanded to include a
  YAML example and the PC1 hyperparameter note; task 9.4 additionally cross-references
  `docs/CROSS_PLATFORM_ANALYSIS.md`.
- **SUGGESTION — task 1.3's "hard gate" has no tooling enforcement, only prose discipline.** Fixed:
  added a checkable tripwire to `tasks.md` (the snapshot commit must precede any `src/`-touching
  commit in git log, checkable at review time).
- **SUGGESTION — task 8.4 didn't specify commit mechanics for a rework loop.** Fixed: pinned to
  "append new `fix:` commit(s), do not amend," matching Tier 3 PR #195's own precedent for
  late-discovered findings.
- **SUGGESTION — task 4.5a's "mirrors `LoadCrossPlatformDataStep`'s own precedent" overstated the
  parallel** (that precedent doesn't actually name platform/pair identifiers either). Fixed: reworded
  to describe this as a stricter improvement over precedent, not a mirror of it.

Full re-validation (`openspec validate add-prediction-pipeline-step --strict`) passes after all
round-2 fixes.

## Adversarial Review Reconciliation (round 3)

A third, independent round of the same 5-agent review (run fresh, no memory of rounds 1-2)
re-verified every prior code citation (all held up, including Decisions 12-15's mechanisms traced
through the full `TaskResult`/`StepResult`/DAG-executor plumbing — one reviewer's own initial
suspicion that Decision 13's attribute path was wrong turned out, on full tracing, to be incorrect;
Decision 13 is accurate as written). This round found notably fewer and less severe issues than
rounds 1-2 — a converging signal — but two are real:

- **HIGH (new) — BLUP-table NaN handling was completely unaddressed.** Real
  `08_blup_adjusted_means.csv` files routinely contain NaN columns (failed model fits, Tier 1's own
  documented contract); `logo_cv_predict` hard-rejects any NaN. Fixed: Decision 16 — drop any trait
  column with any NaN before building `X` or any target, on both source and target sides.
- **MEDIUM-HIGH (new) — the BLUP CSV's genotype-column name was never specified anywhere.** The
  real, shipped column name (`"Genotype"`, capitalized) conflicts with this pipeline's own dominant
  lowercase `"genotype"` convention used almost everywhere else — a real implementer landmine.
  Fixed: Decision 17 — fixed-convention resolution (`"Genotype"` then `"genotype"`, distinct from
  `exp1_genotype_col`/`exp2_genotype_col`), clear error if neither matches.
- **LOW-MEDIUM (new) — duplicate genotype labels within a single BLUP table** aren't explicitly
  guarded before `.loc[]`-reindexing. Documented in Risks (likely surfaces as a contained crash via
  `logo_cv_predict`'s own length check, not silent corruption); not fixed with new code.
- **IMPORTANT (new) — Decision 15's "ordering-only, not data" claim had no enforcing scenario or
  test.** Fixed: `tasks.md` gains a spy-based test confirming task 6 never reads
  `kwargs["05_visualize_cross_platform"].data`.
- **IMPORTANT (new) — capability-placement alternative (the existing `cross-platform-prediction`
  capability) was never weighed, unlike every other decision in this document.** Fixed: added to
  Decision 1's alternatives, resolved in favor of the current placement (see above).
- **IMPORTANT (documentation, new) — Section 9's task 9.3 didn't match this `tasks.md`'s own
  one-assertion-per-checkbox convention, had no subheading plan for the doc extension, and missed a
  real user-facing gap**: silent genotype-set-intersection exclusion (above the hard-error
  threshold) is undocumented anywhere a user would see it. Fixed: task 9.3 split into lettered
  subtasks with an explicit subheading plan and a 4th documented behavior (silent exclusion) added;
  `blup_refit_per_fold`'s inert status also gets a one-line doc note.
- **IMPORTANT (git workflow, new) — task 1.3's suggested git tripwire command was empirically
  invalid.** `git log --follow -- src/` was verified (by actually running it) to behave identically
  to plain `git log -- src/` for a directory pathspec — `--follow` only applies to a single file.
  Fixed: replaced with an ancestry check (`git merge-base --is-ancestor`).
- **IMPORTANT (git workflow, new) — Section 10 had no post-implementation code-review gate**,
  unlike Tier 3's own two `/review-pr` passes (which caught its real implementation bugs — a CI path
  bug, input validation gaps, a mypy baseline violation). Fixed: added an explicit
  `/pre-merge-check`/`/review-pr` task to Section 10, before PR open.
- **SUGGESTION — task 8.4's "do not amend Sections 2-7" was an incomplete enumeration.** Fixed:
  broadened to "do not amend any prior section's commits."
- **SUGGESTION (testing) — task 4.2b's test name was disjunctive ("crashes_cleanly_or_excludes")
  when only one outcome is actually accepted.** Fixed: tightened.
- **SUGGESTION (testing) — task 4.5b's row-shuffle mechanism was unspecified**, and didn't state
  whether the PC1 target must be covered by the shuffle test (not just representative-trait
  targets). Fixed: pinned to a concrete mechanism, explicitly requiring PC1 coverage.
- **SUGGESTION (testing) — tasks 4.3a and 4.9a lacked spec.md scenarios**, unlike their sibling
  findings from the same review rounds. Fixed: added.
- **SUGGESTION (testing) — task 4.9a's placement after the section's "make green" implementation
  task silently contradicted Section 4's "(test-first)" label.** Fixed: one clarifying sentence
  added noting it's a post-implementation tripwire, not a pre-implementation red test.
- **SUGGESTION (spec quality) — spec.md's step-1 algorithm text had become overloaded**, stacking
  Decisions 2/13/14 into one dense numbered item unlike items 2-5's one-clean-sentence pattern.
  Fixed: split into two sub-items.

Full re-validation (`openspec validate add-prediction-pipeline-step --strict`) passes after all
round-3 fixes. Given the diminishing severity and volume across three rounds (round 1: 5 BLOCKING;
round 2: 2 new HIGH; round 3: 1 new HIGH + 1 new MEDIUM-HIGH, otherwise mostly process/polish), this
is a natural stopping point pending Elizabeth's review — a fourth round is available on request but
not proactively recommended.
