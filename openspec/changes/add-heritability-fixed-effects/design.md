## Context

`calculate_heritability_estimates` (`statistics.py:195-479`) fits
`smf.mixedlm("value ~ 1", model_data, groups=genotype)` per trait with REML,
genotype-only. Issue #114 (tracking this tier) documents a concrete case where
this inflates apparent H²: Mauricio Chiurazzi's combined alfalfa GWAS (35
accessions × 6 Bloom experiments) reports H² ≈ 0.83 for a trait whose naive
within-genotype ICC is 0.087–0.318, because genotypes are unbalanced across
experiments and a per-experiment scanner-calibration shift is confounded with
the genotype term.

Tier 1 (`add-blup-extraction`, merged) extracted `result.random_effects` into
a `blup`/`intercept` pair per successful trait and built `extract_blup_table()`
/ `BLUPResult` on top of it. Tier 1's proposal flagged the `R_j` (field
block/replicate) design question as tentatively resolved in favor of a fixed
effect, not a second random effect, for continuity into this tier — confirmed
during this session's brainstorm (see Decision 4).

## Goals / Non-Goals

- **Goals:** `fixed_effects: Optional[List[str]] = None` parameter on
  `calculate_heritability_estimates`; categorical (`C()`-wrapped) treatment for
  every fixed-effect column; missing-column validation consistent with the
  existing `genotype_col`/`replicate_col` pattern; a marginal (frequency-
  weighted) intercept so `extract_blup_table()`'s adjusted means stay
  report-ready rather than pinned to an arbitrary reference level;
  `StatisticsConfig.fixed_effects` threaded through `StatisticalAnalysisStep`;
  the roadmap's oracle (synthetic batch-confounded data: uncorrected H² >
  corrected H²; field BLUPs with block correction differ from genotype-only
  BLUPs).
- **Non-Goals:** changing the H² formula itself or genotype variance-component
  extraction; any LOGO-CV, ridge/PLS, or prediction machinery (Tier 3+);
  identifiability/collinearity pre-validation for `fixed_effects` (reuses the
  existing per-trait `try/except`); coupling `fixed_effects` to
  `replicate_col`; a new `blup.py`/formula-building module (stays inline in
  `statistics.py`, matching Tier 1's placement decision).

## Decisions

### Decision 1: Config lives on `StatisticsConfig`, not `HeritabilityConfig`

**What:** `fixed_effects: Optional[List[str]] = None` is added to
`StatisticsConfig` (`components.py:532`). Issue #114's illustrative YAML
(`heritability.fixed_effects`) is not followed literally.

**Why:** `HeritabilityConfig` (`components.py:216`) is documented as
"Heritability analysis and **filtering** configuration" and is consumed by a
later,
separate step (`FilterHeritabilityStep`, gated by `HeritabilityConfig.enabled`)
that has no relationship to how `calculate_heritability_estimates` itself is
invoked. `StatisticsConfig` is the dataclass `StatisticalAnalysisStep` actually
reads to build the `calculate_heritability_estimates(...)` call
(`statistical_analysis.py:172-179`) — it already owns `calculate_heritability`
and `generate_blup_table` (Tier 1), the two flags this new parameter is most
directly related to. Issue #114's YAML sketch was written for readability
before either config dataclass's actual wiring was inspected; no other issue
mandates the literal `heritability.*` placement.

**Alternatives considered:**
- **Add it to `HeritabilityConfig` and thread a new value through to
  `StatisticalAnalysisStep`.** Rejected: matches the issue's YAML literally,
  but requires new plumbing to move a value from a filtering-step config into
  a calculation-step call, and couples two config dataclasses that currently
  have no relationship — added complexity for no behavioral benefit.

### Decision 2: Every fixed-effect column is wrapped in `C()` unconditionally

**What:** The formula is built as `"value ~ " + " + ".join(f"C({fe})" for fe
in fixed_effects)`. No dtype inspection; every column is always coded as
categorical, regardless of whether its underlying dtype is numeric or object.

**Why:** Fixed effects in this context are metadata-style confounders
(experiment, wave, batch, scanner) — issue #114's own acceptance criteria
say so explicitly. A column like `wave_number` (values `1, 2, 3`) is
numeric-*looking* but categorical in meaning; without `C()`, patsy would treat
it as a continuous covariate (one linear-slope coefficient instead of one
coefficient per wave), silently changing what the model estimates. Since this
whole tier exists to stop a categorical confounder from leaking into the
genotype term, defaulting to the interpretation that actually removes the
confound — and making it unconditional rather than configurable — closes the
exact failure mode the issue describes rather than reintroducing a milder
version of it.

**Alternatives considered:**
- **Infer from dtype (`C()` only for object/category columns).** Rejected:
  more flexible (would allow a genuinely continuous covariate), but
  reintroduces the silent-misinterpretation risk for a numeric-looking
  metadata column, and adds a code path with its own edge cases (e.g. an
  integer-coded categorical that a naive dtype check would treat as
  continuous) for a flexibility this tier's use cases don't need.

### Decision 3: Empirical frequency-weighted intercept, not a reference-level pin

**What:** When `fixed_effects` is set, `intercept` is computed as a
frequency-weighted average across each fixed effect's *observed* levels — for
each fixed effect, each level's fitted contribution (`0` for the reference
level, its own coefficient in `result.fe_params` for every other level) is
weighted by that level's share of the fitted `model_data`, then summed across
all fixed effects and added to the base `Intercept` coefficient. When
`fixed_effects` is `None`, this is exactly `result.fe_params["Intercept"]` —
today's behavior, unchanged.

**Why:** Patsy's default treatment coding drops one level per fixed effect as
a reference (the first level in sorted order for a plain column, or the first
declared category for a pandas `Categorical` — not reliably "alphabetical," a
looser claim than an earlier draft of this decision made), so
`result.fe_params["Intercept"]` alone represents "the fitted value when every
fixed effect is at its reference level" — not a value with any particular
scientific meaning, since the reference level is a naming artifact, not a
chosen baseline. For any *relative* use of `extract_blup_table()`'s adjusted
means (Tier 3's ridge/PLS, rankings, correlations), the choice of intercept is
irrelevant — it is a single scalar added identically to every genotype (proof
in the Risks section below), so it cancels out of every comparison. It only
matters when the raw adjusted-mean numbers are read directly (e.g. in the
EDPIE paper's supplementary tables), where "value under the reference level's
specific condition" is a narrower, more surprising claim than "value under
typical/average conditions." The frequency-weighted average is cheap — it
reuses coefficients `fe_params` already contains, no additional model fit —
and removes this caveat entirely rather than documenting around it.

**Revised after adversarial review — naming precision matters here.** The
quantity computed is *not* a population-typical or EMM/lsmeans-style marginal
mean (the standard biometrics convention, e.g. R's `emmeans`, deliberately
uses *equal* weighting across levels precisely to avoid baking incidental
sample composition into a "typical" value). What's actually computed is an
**empirical, sample-frequency-weighted average over that specific trait's own
post-`dropna()` fitted rows** — an "observed-margins" quantity. Two concrete
consequences, now stated explicitly rather than left implicit: (1) if one
experiment has more scans purely for logistical reasons, the intercept skews
toward it regardless of biological representativeness; (2) since `model_data`
is trait-specific (each trait computes its own `dropna()`), two traits sharing
the same `fixed_effects` columns can get *different* level weights purely
from differing missing-data patterns — the intercept is not a single stable
per-dataset quantity. Documented in the docstring and the statistics-api spec
delta as "empirical frequency-weighted," not "population-typical," to avoid
misleading a future reader of raw `adjusted_mean`/`BLUPResult.intercepts`
values (e.g. in a paper table). An equally-weighted (true EMM-style) variant
is called out as a non-goal for a future tier, not conflated with what ships
here.

**Also revised — coefficient lookup must not reconstruct keys forward.** The
per-level coefficient is retrieved by parsing `result.fe_params`'s actual
fitted parameter index (regex `^C\({fe}\)\[T\.(.*)\]$` per fixed effect) to
recover the set of levels patsy actually fit non-reference coefficients for,
rather than building the expected key string forward from each observed
level's raw value (e.g. `f"C({fe})[T.{level}]"` for `level` enumerated from
`model_data[fe].unique()`). The forward-reconstruction approach was the
original draft of this decision; adversarial review found it has no way to
distinguish "this level genuinely is the reference level" from "the
reconstructed key string didn't match `fe_params`'s actual key due to a
dtype/formatting difference" (concretely: a `float64`-dtype fixed-effect
column, or a level whose `str()` representation doesn't exactly match
patsy's internal label). Both cases fall through the same `dict.get(key,
0.0)`-style default, silently misattributing a real, non-reference,
non-zero-coefficient level to the reference level's implicit contribution —
a silent-corruption risk with no error signal, which is a serious defect for
a change whose whole purpose is eliminating a different silent-corruption
risk (batch effects leaking into H²). Parsing the fitted index forward,
matching each recovered level string back to `model_data[fe]`'s values by
equality (not by positional pairing against a separately-sorted level list —
round-2 review found a `pandas.Categorical` with a non-default `categories=`
order would break a positional-pairing implementation without breaking a
count-only check), and asserting the recovered level count equals `n_levels -
1` (raising rather than silently defaulting otherwise) closes this gap.
**`n_levels` is `model_data[fe].nunique()` — that trait's own
post-`dropna()` fitted subset, not the raw input `df`** (clarified after
round-2 review found this was ambiguous): a level present in `df` can be
entirely absent from a specific trait's `model_data` due to that trait's own
missingness pattern, so using the raw `df` count would make an entirely
correct implementation raise a false `ValueError`.

**Alternatives considered:**
- **Keep the reference-level intercept, document the caveat.** Considered
  first; rejected after establishing the marginal version is a few lines over
  existing `fe_params`, not a new statistical method, so there is little cost
  to removing the caveat rather than living with it, given these BLUP tables
  may be read directly in paper-facing outputs.
- **True equal-weighted EMM/lsmeans-style intercept.** Considered during
  review; rejected for this tier as more statistical machinery than the
  oracle requires (the oracle only needs corrected vs. uncorrected H² to
  differ in the right direction, and BLUP tables to differ from
  genotype-only) — recorded as an explicit non-goal rather than silently
  approximated by the empirical-frequency version.

### Decision 4: `fixed_effects` and `replicate_col` are fully independent

**What:** `replicate_col` is unchanged — validated for presence (when
truthy), never read in the model. A block/replicate fixed effect (`R_j`) is
expressed by naming that column in `fixed_effects` (e.g.
`fixed_effects=["block"]`); no new validation or auto-inclusion links the two
parameters.

**Why:** This is Tier 1's tentative design note, confirmed here: `R_j` as a
fixed effect (this tier's mechanism) rather than a second random effect. Once
`fixed_effects` is a generic list of column names, "a block/replicate fixed
effect" is already fully expressible without any special-casing — adding
coupling (e.g. auto-including `replicate_col`, or validating it isn't
duplicated in `fixed_effects`) would add surface area for a scenario the
generic mechanism already covers.

**Alternatives considered:**
- **Special-case coupling between the two parameters.** Rejected: no
  behavior it would enable isn't already reachable by passing the column name
  into `fixed_effects` directly.

### Decision 5: Model-fit failures reuse the existing per-trait `try/except`, extended to capture `ConvergenceWarning`

**What:** No new upfront identifiability/collinearity pre-validation for
`fixed_effects`. A fixed effect fully confounded with genotype, or any other
convergence failure introduced by adding fixed effects, is caught by the
existing `except Exception as e:` block around the mixed-model fit
(`statistics.py:392-398`) and recorded as `{"error": "Mixed model failed:
...", "model_type": "mixed_model_failed"}` for that trait — identical handling
to today's non-convergence failures. **Revised after adversarial review:**
**only when `fixed_effects` is non-empty**, the fit call is additionally
wrapped in `warnings.catch_warnings(record=True)` with
`warnings.simplefilter("always")` called immediately inside that block, and
if a convergence warning is captured, that trait is treated as failed (same
error dict shape) even though `statsmodels` did not raise. **The
`fixed_effects`-non-empty gate on this specific behavior was caught during
implementation, not by any of the four review rounds**: earlier drafts of
this decision and the corresponding spec text described the warning-capture
without explicitly stating it was conditional, even though this decision's
own "byte-for-byte identical when `fixed_effects=None`" guarantee (shared
with every other behavior change in this tier) logically requires it — an
unconditional check would newly fail an existing, non-`fixed_effects` caller
whose fit happens to emit a convergence warning today and currently
succeeds. Fixed by making the gate explicit in the spec's requirement
prose and adding a dedicated regression test (tasks.md 1.9b). The
`simplefilter("always")` call is required, not optional (added after round-2
review): Python's default warning filter is once-per-source-location, and
`statsmodels` raises its `ConvergenceWarning` from the same internal source
line every time — without forcing "always," a *second* trait hitting the
same non-convergent code path in the same process could have its identical
warning silently dropped by the registry, reintroducing exactly the silent-
failure risk this decision exists to close, just for later traits. This bug
would not be caught by a test that observes only one warning occurrence in
isolation, since pytest's own warnings plugin already applies
`simplefilter("always")` around every test and would mask a missing call in
production code (see tasks.md 1.13, added specifically to exercise the
two-trait case). **Also required (added after round-3 review): the check
SHALL filter by warning *category* —
`issubclass(w.category, statsmodels.tools.sm_exceptions.ConvergenceWarning)`
— not by matching the warning's message text.** Several real `statsmodels`
convergence-related warning messages do not contain the word "convergence"
at all (e.g. "The MLE may be on the boundary of the parameter space." and
"The Hessian matrix at the estimated parameter values is not positive
definite."), which a message-substring implementation would silently miss —
defeating this decision's purpose on exactly the messages a near-confounded
fixed effect is most likely to trigger. Symmetrically, a warning of any
*other* category captured in the same block (a stray `RuntimeWarning` or
`FutureWarning` from an unrelated dependency) SHALL NOT be treated as a fit
failure — tasks.md 1.9a adds the false-positive counterpart test.

**Why:** The existing per-trait error path already tolerates model-fit
failure gracefully (records the error, continues to the next trait), so
reusing it for a *raised* exception needs no new machinery. But adversarial
review found a real gap: `statsmodels.MixedLM.fit()` does not reliably raise
on a fixed effect that is (near-)fully confounded with genotype — exactly the
scenario the batch-confounded oracle fixture (tasks.md 1.1) deliberately
constructs. It can instead emit a `ConvergenceWarning` via Python's `warnings`
machinery and still return a `result` with plausible-looking (but
potentially degenerate) `blup`/`intercept`/`heritability` values — no
exception, so the existing `try/except` alone would let it through silently.
Shipping a mechanism that can silently produce a plausible-but-wrong H² from
over-fitting/aliasing would reintroduce, via a different mechanism, the same
class of problem (silently-wrong H² from unmodeled structure) that issue #114
exists to fix. Capturing the warning and treating it as a failure closes this
without adding a new upfront validation layer — it still relies on
`statsmodels`' own signal, just observes both channels (exceptions and
warnings) instead of only exceptions.

**Alternatives considered:**
- **Pre-validate that no fixed effect is a deterministic function of
  genotype** (i.e. perfectly confounded), checked before fitting. Rejected:
  adds a new validation surface (what counts as "too confounded" is itself a
  judgment call) for a case `statsmodels`' own convergence diagnostics already
  signal, once that signal is actually observed (see revision above) — no
  oracle-driven need to distinguish "failed because of a fixed effect" from
  "failed for any other reason" in the returned error.
- **Rely on raised exceptions only (the original draft of this decision).**
  Rejected after adversarial review established this leaves the exact
  batch-confounded scenario this tier targets able to silently succeed with a
  degenerate fit.

## Risks / Trade-offs

- **Marginal intercept computation must handle multiple fixed effects
  correctly.** With `fixed_effects = ["experiment", "block"]`, `fe_params`
  contains additive main-effect offsets for both (patsy's `+` composes
  additively, not as an interaction) — the marginal-intercept helper must sum
  each effect's own frequency-weighted contribution independently and add
  both to the base `Intercept`, not conflate them. Re-derived independently
  during adversarial review by linearity of expectation:
  `mean_i[Intercept + β_exp[e_i] + β_block[b_i]] = Intercept +
  Σ_e freq_exp[e]·β_exp[e] + Σ_b freq_block[b]·β_block[b]` — this
  decomposition holds regardless of whether `experiment` and `block` are
  correlated/nested with each other in the data (no orthogonality assumption
  needed), so the "contribute independently" requirement is on solid ground.
  Covered by a dedicated multi-fixed-effect test (tasks.md §2).
- **Choice of intercept convention (reference-level vs. empirical
  frequency-weighted) shifts every genotype's `adjusted_mean` by the same
  additive constant, preserving Tier 1's ranking/shrinkage properties.**
  Re-derived during adversarial review: `blup[g]` comes solely from
  `result.random_effects`, fixed once the model is fit and independent of how
  `intercept` is reported afterward (pure post-hoc arithmetic on already-fit
  coefficients, touching no fitting step). So `adjusted_mean[g] = intercept +
  blup[g]` shifts by exactly `(I_marginal - I_reference)` for *every*
  genotype, regardless of which fixed-effect levels that genotype happened to
  appear in — rankings and pairwise differences are invariant. This confirms
  Decision 3 is safe without needing an explicit new regression scenario for
  this specific property, since it holds structurally rather than
  empirically. (This is a distinct question from Decision 3's own concern
  about the *level* of the intercept value being sample-composition-
  dependent — that concern is about the value's absolute interpretation, not
  about whether the shift is uniform across genotypes.)
- **`fixed_effects` changes which rows survive `dropna()`, identically for
  both the mixed-model and ANOVA-based (`force_method="anova_based"`)
  paths.** A row with a valid trait value and genotype but a missing
  fixed-effect value is now dropped from the model fit — a real behavior
  change from today, but only when `fixed_effects` is explicitly set; `None`
  (default) is unaffected. The subset computation happens once, before the
  `if use_mixed_model:` branch, so both paths see the same filtered rows;
  only the mixed-model path uses `fixed_effects` in its formula — the
  ANOVA-based path continues to ignore it for modeling purposes. Documented
  in the parameter's docstring and covered by a dedicated test, plus an
  explicit test that the `force_method="anova_based"` combination applies
  the same row-filtering without attempting to use `fixed_effects` in that
  branch's variance-component arithmetic (added after adversarial review
  flagged this combination as untested).
- **Silent convergence failure under genotype-confounded fixed effects** —
  see Decision 5's revision above. Mitigated by capturing
  `ConvergenceWarning` rather than relying solely on raised exceptions.
  Tasks.md adds one *organic* (non-mocked) test with a fixed effect that is a
  near-deterministic function of genotype, to observe and document actual
  `statsmodels` behavior on this fixture, in addition to the existing mocked
  test for the raised-exception path (`test_mixed_model_fit_failure_has_no_blup_keys`-style, from Tier 1's precedent).
- **Reusing Tier 1's shrinkage-property scenarios unmodified on a
  fixed-effects run risks an ambiguous reference point.** Tier 1's existing
  "BLUP Shrinkage and Balanced-Design Properties" requirement compares
  shrinkage against the naive, unconditional grand/raw mean
  (`df.groupby(genotype)[trait].mean()`). Once `fixed_effects` is set on a
  design unbalanced across fixed-effect levels (this tier's core use case), a
  genotype's naive raw mean is itself confounded by the fixed effect —
  shrinkage is properly toward the model's fixed-effects-adjusted intercept,
  not the naive unconditional grand mean. The shrinkage math itself isn't
  mechanically threatened by adding correctly-specified fixed effects (REML
  shrinkage of a random effect is a generic MixedLM property, independent of
  which fixed effects are in the formula), but no scenario re-verifies
  shrinkage-scales-with-`n_i` under `fixed_effects` using the *correct*
  reference point. Added as an explicit task (tasks.md §3) after adversarial
  review flagged the gap.

## Migration Plan

Purely additive — no existing caller changes required:
- `calculate_heritability_estimates(df, trait_cols, ...)` without
  `fixed_effects` (the default, `None`) produces byte-for-byte identical
  output to today, including the `intercept` value.
- `StatisticalAnalysisStep` gains one new optional config field
  (`fixed_effects`, default `None`); existing configs that don't set it are
  unaffected.
- `extract_blup_table()` and `BLUPResult` require no *code* changes — both
  already consume whatever `intercept` float
  `calculate_heritability_estimates` produces. `BLUPResult.intercepts`'s
  docstring (`result_types.py`) is updated to describe the new semantics of a
  value it already stores verbatim — a docs-only change, not a logic change
  (corrected after adversarial review found this docstring update, and the
  matching `docs/result-types.md` caveat, were listed in `proposal.md`'s
  Impact section but missing from `tasks.md` entirely; both are now tracked).

No rollback concerns beyond reverting the additive commits.

## Open Questions

None blocking Tier 2. Tier 3's design questions (PLS component count,
representative-clustering aggregation level, permutation runtime) are
unaffected by this tier and remain open there.

## Adversarial Review Reconciliation (round 1)

`/review-openspec` ran 5 parallel reviewers (spec quality, TDD/testing,
statistical correctness, documentation, git workflow). 3 BLOCKING and 7
IMPORTANT findings, all reconciled into this document, `proposal.md`, and
`tasks.md`:

- **BLOCKING** — Impact section promised a `result_types.py`/
  `docs/result-types.md` update with no corresponding task, plus a factual
  error (referenced a nonexistent `HeritabilityResult` per-trait `intercept`
  field). Fixed: proposal.md corrected, tasks.md gains explicit tasks
  (§2.7–2.8).
- **BLOCKING** — forward-reconstructed `fe_params` key lookup could silently
  misattribute a real level to the reference level on a dtype/formatting
  mismatch. Fixed: Decision 3 revised to parse `fe_params`'s actual fitted
  index instead.
- **BLOCKING** — `statsmodels.MixedLM.fit()` doesn't reliably raise on a
  fixed effect confounded with genotype; could silently ship a degenerate fit
  in exactly this tier's target scenario. Fixed: Decision 5 revised to
  capture `ConvergenceWarning` and treat it as a fit failure.
- **IMPORTANT** — "marginal intercept" mislabeled as population-typical;
  it's actually a per-trait, sample-frequency-weighted quantity. Fixed:
  Decision 3 and proposal.md reworded to "empirical frequency-weighted,"
  with the per-trait-instability caveat stated explicitly.
- **IMPORTANT** — no test for `fixed_effects` + `force_method="anova_based"`;
  no test for shrinkage-property regression under `fixed_effects`; no test
  for `BLUPResult.from_blup_table()` pass-through of fixed-effects-derived
  intercepts. Fixed: tasks.md gains 1.12, 3.2, 2.7 respectively (see
  tasks.md).
- **IMPORTANT** — fixture 1.1 lacked pinned numeric parameters (unlike Tier
  1's precedent), risking CI flakiness across the 3-OS matrix. Fixed: tasks.md
  1.1 now specifies concrete variance components, shift magnitude, seed, and
  confounding split.
- **IMPORTANT** — `calculate_heritability_estimates`'s `Returns:` docstring
  sentence describing `intercept` as unconditionally
  `result.fe_params["Intercept"]` would go stale. Fixed: folded into tasks.md
  2.6's implementation task.
- **IMPORTANT** — task 1.5's fallback plan ("if direct result access isn't
  practical...") was a live open decision. Fixed: resolved directly — the
  fitted `result` object is not exposed by
  `calculate_heritability_estimates`'s return value, so the test independently
  re-fits via `smf.mixedlm` on the same `model_data`, matching tasks 2.2-2.4's
  approach; tasks.md 1.5 updated to state this directly instead of offering
  two options.
- **SUGGESTION** — added non-goal bullets (continuous covariates,
  equal-weighted EMM variant) to proposal.md's "Explicitly out of scope."
- **SUGGESTION** — added an assertion that `fixed_effects=[]` behaves
  identically to `None`, folded into tasks.md 1.3.
- Not reconciled (deliberately, low severity per the documentation
  reviewer): `docs/QC_PIPELINE_GUIDE.md`'s `statistics:` block documentation
  gap pre-dates this tier (Tier 1's `generate_blup_table` isn't documented
  there either) — left as a candidate follow-up issue, not blocking this
  change.

## Adversarial Review Reconciliation (round 2)

A second, targeted round re-checked round 1's fixes specifically (not a full
re-review). Found the substantive fixes were correct, but surfaced 5 more
issues — all reconciled:

- **Real gap (statistical correctness)** — `warnings.catch_warnings(record=True)`
  alone doesn't force every occurrence of an identical warning to be
  recorded; Python's default once-per-location filter can silently drop a
  *second* trait's identical `ConvergenceWarning` in the same process, and no
  test could have caught it (pytest's own warnings plugin masks the gap in
  any single-occurrence test). Fixed: Decision 5 now requires
  `warnings.simplefilter("always")` inside the block; tasks.md gains 1.13
  (two-trait repeat-warning test).
- **Real gap (statistical correctness)** — the coefficient-lookup fix's
  "match recovered level to frequency" step was ambiguous about
  equality-matching vs. positional pairing, and `n_levels` was never defined
  as `model_data[fe]` (post-`dropna()`) vs. raw `df`. Fixed: Decision 3 and
  tasks.md 2.6 now state both explicitly; tasks.md gains 2.5a (non-sorted
  `pd.Categorical` test, the one case that would expose a positional-pairing
  bug that count-only or sorted-order-assuming code would pass).
- **Spec-quality gap** — task 1.10 (organic near-confounded test) is a
  characterization/regression-pinning test whose assertion content cannot be
  known before implementation exists, which doesn't fit "(test-first)"'s
  red-green framing and had no scenario documenting it. Fixed: added an
  explicit sequencing note to task 1.10 stating it's written after 1.11
  lands, and a note that it's intentionally spec-exempt (nothing to
  prescribe in advance, only to pin).
- **Mechanical** — task 1.12 (the `anova_based` combination test, itself a
  round-1 addition) was missing its "(New — added after review)" tag; this
  reconciliation section's own citations pointed at stale task numbers after
  tasks.md was renumbered mid-revision (`§2.6`→`§2.7–2.8`, `2.5`→`2.6`,
  `1.10`→`1.12` in the bullets above — already corrected in place rather than
  left as a visible diff, since the citations were simply wrong, not a
  design change). Fixed: 1.12 tagged; citations corrected.
- **Structural nit** — tasks.md's 5.3 was a checkbox that did nothing
  (pure cross-reference to 2.8). Fixed: converted to a blockquote note,
  consistent with the file's existing meta-commentary convention.

## Adversarial Review Reconciliation (round 3)

A third round did a fresh cold-read (assuming no memory of rounds 1–2) plus
targeted stress-tests of the newest round-2 material, including running an
actual `statsmodels` simulation rather than reasoning about fixture behavior
analytically. This surfaced one CRITICAL, previously-undetected defect and
several smaller gaps — all reconciled:

- **CRITICAL (statistical correctness) — task 1.1's batch-confounded fixture
  was verified, by direct simulation, to produce the OPPOSITE of the
  intended effect.** The pre-round-3 fixture (σ_G=3.0, shift=8.0, n=4 reps,
  a 16/4-genotype 3:1/1:3 partial mix) gave uncorrected H² *below* corrected
  H² by 0.36–0.85 across seeds — reliably wrong-signed, not merely
  occasionally short of the 0.05 margin. Root cause (confirmed by exact
  variance decomposition, then by simulation across dozens of parameter
  combinations): at n=4 reps, any genuinely partial per-genotype batch mix
  necessarily injects more *within*-genotype variance (from mixing a
  genotype's own reps across batches) than *between*-genotype variance
  (from the two genotype groups' differing batch composition) — a
  mathematical property of the 3:1/1:3 split at that rep count, not fixable
  by increasing the shift magnitude (the task's own stated remedy, now
  known to be wrong: both variance terms scale identically with shift², so
  the sign never flips). Fixed: task 1.1 rewritten with an empirically
  verified design (σ_G=0.4, σ_E=1.0, shift=+10.0, n=20 genotypes × n=10
  reps, a symmetric 10/10-genotype 90%/10% split — genuinely partial, not
  deterministic) that reliably clears the 0.05 threshold with more than 2x
  margin at the worst of 50 tested seeds under `np.random.default_rng`
  (min gap 0.109, mean 0.36). **Corrected during implementation**: this
  verification pass used `np.random.default_rng(seed)`, but the fixture
  file's actual convention is the legacy global `np.random.seed()` +
  `np.random.normal()` API, which produces a different stream for the same
  seed number — task 1.1's own text already flagged this exact risk
  ("an RNG-construction mismatch versus the verification script"). Re-run
  under the legacy API before writing the real fixture: still correctly
  and robustly signed (10-seed check: min gap 0.2138, mean 0.3756; at
  seed=42 specifically, gap≈0.2211), just different exact numbers — task
  1.1 now records the legacy-API numbers and specifies the exact draw order
  needed to reproduce them. The general lesson — a genuinely partial
  confound's sign depends on whether between-group variance exceeds
  within-genotype mixing variance, which
  requires an extreme, *symmetric* group split, not merely "genotypes
  unevenly distributed" — is now stated in task 1.1 itself so a future
  reader understands why these exact numbers were chosen, not just what
  they are.
- **Confirmed gap — task 3.2's shrinkage oracle had no rep-count variation
  to draw on.** Task 1.2, as literally written pre-round-3, specified
  uniform n=3 reps for all 15 genotypes; 3.2's "if 1.2 doesn't already vary
  rep count" hedge deferred rather than resolved this. Fixed at the time:
  task 1.2 pinned an explicit 7-low-rep(n=2)/8-high-rep(n=6) split and a
  previously-unspecified residual noise term. **Superseded by round 4
  below**: this fix resolved the "no rep-count variation" gap but did not
  catch a deeper problem in how 3.2's oracle compared values, which
  empirical verification (prompted by a follow-up question, not a formal
  review round) caught afterward.
- **Real gap (statistical correctness) — the `ConvergenceWarning` capture
  never specified filtering by warning *category*, only by "a convergence
  warning" in prose.** Several real `statsmodels` convergence-related
  warning messages don't contain the word "convergence" at all (e.g. "The
  MLE may be on the boundary of the parameter space."), so a
  message-text-matching implementation would silently miss exactly the
  warnings this decision exists to catch. Fixed: Decision 5, the
  statistics-api spec delta, and tasks.md 1.11 now require
  `issubclass(w.category, statsmodels.tools.sm_exceptions.ConvergenceWarning)`
  explicitly; tasks.md gains 1.9a (false-positive counterpart: an unrelated
  warning category must not fail a trait).
- **Minor (defensive, judgment call) — no guard against a `fixed_effects`
  column name containing a patsy formula operator** (e.g. `"rep*block"`,
  which combined with the coincidental presence of separate `"rep"`/`"block"`
  columns could silently misparse as elementwise multiplication rather than
  a literal column reference). Narrow tail case requiring two contrived
  conditions at once, but the guard is nearly free. Fixed: tasks.md gains
  1.4a (`fe.isidentifier()` validation, extending the existing
  missing-column check).
- **Mechanical fixes** — task 1.1's own text cited the wrong task number
  for "full determinism" (said 1.9, meant 1.10 — 1.9 mocks the fit directly
  and builds no confounded fixture); tasks.md 5.2's CHANGELOG-placement
  guidance said "last `### Added` bullet, mirroring Tier 1's placement,"
  which was backwards — Tier 1's actual rationale sentence sits on its
  *first* `### Added` bullet; this design.md's Decision 1 misquoted
  `HeritabilityConfig`'s docstring (dropped "analysis and"); tasks.md's top
  blockquote note and task 6.4 still referenced only "round 1" after round
  2 had already landed. All corrected in place.
- Confirmed **not** a defect (verified, no change needed): the
  multi-fixed-effect additive-composition proof (linearity of expectation,
  order-independent regardless of correlation between effects) — re-derived
  independently in round 3 and found correct; the uniform-additive-shift
  proof for intercept-convention choice — same, confirmed correct including
  its handling of a genotype with zero reference-level appearances.
- **Noted, not acted on (proportionality, not a defect):** the review
  process has produced a large design.md (400+ lines) and 35 tasks for a
  capability that is, in the end, one new parameter, one new config field,
  and one new helper function. Every addition traces to a specific defect
  or ambiguity an adversarial reviewer actually found and none are
  speculative scope growth — but the volume itself is worth naming to the
  user before approval as a reflection of how statistically subtle this
  feature turned out to be, not as feature creep. Some duplication between
  the Decision-level rationale and the three reconciliation sections'
  changelog-style bullets could be trimmed for future-reader maintainability
  (e.g. the `simplefilter`/equality-matching mechanism is now explained in
  full in at least two places each) — flagged as a documentation-quality
  nit, not requested as a required edit.

## Adversarial Review Reconciliation (round 4 — targeted empirical follow-up, not a formal panel)

After round 3, a direct question ("what's the minimum number of reps this
will work on?") prompted empirically re-verifying the one fixture round 3's
own numeric-verification pattern hadn't yet been applied to:
`heritability_data_field_block` (task 1.2), which backs both the
BLUP-difference oracle (3.1) and the shrinkage-regression oracle (3.2).
Round 3 had already pinned concrete numbers for it by analogy to Tier 1's
precedent, but — unlike task 1.1's fixture — had not run it. This surfaced
one further real, non-mechanical defect:

- **Real gap (statistical correctness), same root cause as round 3's
  headline finding — task 3.2's shrinkage oracle, as designed post-round-3,
  failed empirically 40% of the time.** Simulated the exact round-3 design
  (7-low-rep(n=2)/8-high-rep(n=6), block skew applied by genotype ID in a
  way that happened to correlate with the rep-count grouping): the
  BLUP-difference oracle (3.1) was robust (0/20 failures), but the
  shrinkage-scales-with-replication oracle (3.2) failed 8/20 seeds (40%).
  Root cause, confirmed by directly testing the stronger per-genotype
  assertion (Tier 1's own original shrinkage property — every genotype's
  adjusted gap smaller than its raw gap, not just a group-mean-ratio
  comparison): that per-genotype property failed **100% of the time**
  (50/50 seeds) under the naive raw-mean comparison. Diagnosis: the naive
  `df.groupby(genotype)[trait].mean()` used as 3.2's raw-mean reference
  point is itself contaminated by each genotype's own block composition —
  the exact systematic effect `fixed_effects=["block"]` corrects for — so
  comparing it directly against the corrected BLUP conflates two different
  phenomena (genuine mixed-model shrinkage of random noise, which Tier 1's
  original oracle tested on a design with no fixed-effect confound at all;
  and removal of a systematic per-genotype bias, which is what the fixed
  effect actually does). These aren't the same comparison, and nothing in
  round 1–3's design or spec text distinguished them. Fixed, verified by
  the same simulation: detrending the raw mean — subtracting each
  observation's fitted `C(block)` coefficient before averaging within
  genotype — makes both the per-genotype assertion and the group-mean-ratio
  assertion hold reliably (0/50 failures each, vs. 50/50 and 6+/30 failures
  respectively without detrending). Task 1.2's fixture numbers were also
  revised alongside this (n_high raised from 6 to 10 for a starker
  contrast, and block skew changed to apply orthogonally to the
  replicate-count grouping rather than correlating with it by genotype ID)
  — orthogonality is now required, not incidental, per an explicit note in
  task 1.2. Task 3.2 and the statistics-api spec's corresponding scenario
  now specify the detrending step precisely rather than leaving it
  implementation-discoverable.
- **Confirmed, not a defect:** the BLUP-difference oracle (3.1) needed no
  changes — it was robust (0/50 failures) under both the pre- and
  post-round-4 fixture designs, since it only asks whether *any* value
  differs, not a directional/magnitude comparison sensitive to the
  detrending issue above.
- **Process note, not a proposal defect:** this round was not a formal
  5-reviewer panel — it was one targeted empirical check, run because round
  3 had established that reasoning about this kind of fixture without
  simulating it is unreliable, and that lesson hadn't yet been applied to
  every fixture in the proposal. No further un-simulated fixtures remain in
  this change: both `heritability_data_batch_confounded` (1.1) and
  `heritability_data_field_block` (1.2) are now empirically verified against
  real `statsmodels`, not just reasoned about.

## Implementation-Time Correction (task 3.2 × task 2.6 interaction)

Writing task 3.2's test against the actual task 2.6 implementation (not
caught by any of the 4 prior rounds, since round 4's fixture verification
predated 2.6's existence and used the raw reference-level intercept
directly) surfaced one more real bug: task 3.2's spec text said to center
the shrinkage comparison on "the run's own returned `intercept`," but once
2.6 lands, that value is the *marginal* (frequency-weighted) intercept, not
the reference-level one the detrended raw mean is naturally expressed
relative to. Because `adjusted_mean = intercept + blup[g]`,
`|adjusted_mean - intercept|` always collapses to `|blup[g]|` regardless of
intercept convention (the two cancel structurally) — but
`raw_mean_detrended[g]` has no such dependency, so centering it on the
marginal intercept instead introduced a constant offset that broke the
per-genotype assertion for a subset of genotypes (confirmed by running the
test: it failed with a >2x gap violation). Fixed by comparing `blup[g]`
directly (not `extract_blup_table()`'s already-summed `adjusted_mean`)
against the raw-mean, both centered on the reference-level `Intercept` from
an independent re-fit — tasks.md 3.2 and the statistics-api spec's
corresponding scenario updated accordingly. Re-verified: both assertions
pass on the actual fixture and implementation.

A second, smaller implementation-time catch: task 1.4a's test
(`test_fixed_effect_column_name_with_patsy_metacharacter_rejected`) built a
fixed-effects name (`"rep_col*block_col"`) that was never actually a column
in the test DataFrame — so the pre-existing missing-column check intercepted
it before the new `isidentifier()` validation was ever reached, making the
test pass for the wrong reason (confirmed via coverage: the `isidentifier()`
branch showed as uncovered despite a passing test asserting on its error
message). Fixed by using a column that genuinely exists in the DataFrame
(pandas column names aren't required to be valid Python identifiers) but
still fails `isidentifier()` — coverage confirmed the branch is now
genuinely exercised.

## Pre-Merge Review (5-agent `/review-pr` team, pre-PR local diff)

A 5-subagent adversarial review of the complete implementation (Code
Quality, Testing, Statistical Rigor, Performance/Memory, Behavioural
Correctness) found no BLOCKING issues in code quality or performance, but
surfaced 4 more real, fixed issues:

- **Real bug (behavioural correctness) — `remove_low_h2=True` combined with
  `fixed_effects` could silently drop the fixed-effect column from
  `df_filtered`.** `calculate_heritability_estimates` never told
  `remove_low_heritability_traits` (via `additional_exclude`) that
  `fixed_effects` columns aren't candidate traits; `get_trait_columns`'s
  hardcoded metadata-name heuristic doesn't cover every plausible fixed-effect
  name (confirmed with `fixed_effects=["block"]`, the proposal's own running
  example). Not reachable via the current pipeline (`StatisticalAnalysisStep`
  always passes `remove_low_h2=False`), but directly reachable by any
  script/notebook caller of the public function. Fixed: `fixed_effects` is
  now unioned into `additional_exclude` before filtering.
- **Real gap (statistical rigor) — `_marginal_intercept`'s validation was a
  count check, not an identity check.** `len(level_coefficients) != n_levels
  - 1` can pass even when a real level's string silently failed to match its
  own coefficient (falling through to the `0.0` default) while an unrelated
  mismatch happened to keep the count the same. The reviewer found a genuine
  (if currently non-reachable with realistic pandas dtypes) theoretical
  vulnerability: patsy's level-labeling uses `repr()` for some value types,
  which can diverge from this code's `str()`-based matching for certain
  numpy scalar types. Fixed: the check now verifies every recovered
  coefficient key matches an actually-observed level string, and that
  exactly one observed level lacks a coefficient (the true reference) —
  closing the gap structurally rather than relying on incidental pandas
  boxing behavior to keep it safe.
- **Confirmed limitation (statistical rigor + behavioural correctness) — the
  `ConvergenceWarning` capture does not catch its own motivating scenario in
  all cases.** The reviewer reproduced a fixed effect confounded with
  genotype that converges cleanly with zero warnings, returning a normal
  success dict with an arbitrary, non-identifiable variance-component split
  — exactly the "plausible-but-degenerate fit with no error signal" Decision
  5 was written to prevent. This is not a code bug to fix (Decision 5
  already declared "no upfront identifiability pre-validation" a deliberate
  non-goal, and detecting this class of near-confounding programmatically
  would require design-matrix collinearity analysis well beyond this tier's
  scope) — it's a real, now-empirically-confirmed limit on what the
  warning-capture mechanism delivers. Fixed by making the docstring honest
  about this rather than implying the mechanism prevents silent degenerate
  fits categorically.
- **Test-quality gap — the organic characterization test
  (`test_near_fully_confounded_fixed_effect_organic_behavior`) pinned an
  exact heritability value (`abs=0.01`) that had never run on the
  Ubuntu/Windows/macOS CI matrix**, for a fit type (near-singular) most
  sensitive to BLAS/LAPACK differences across platforms. Fixed: the exact
  value is no longer pinned tightly — only that it's a valid probability in
  `[0, 1]`; the qualitative outcome (no `ConvergenceWarning`, successful
  fit) remains the pinned assertion, which is what this characterization
  test actually exists to catch drift on.
- **Minor, cheap fix — naming `genotype_col`/`replicate_col` inside
  `fixed_effects` produced a confusing pandas-internal error** ("Grouper for
  'geno' not 1-dimensional") deep inside the per-trait loop rather than a
  clear structural error. Fixed: rejected upfront with the same structural-
  error shape as the missing-column and `isidentifier()` checks.
- **Noted, not fixed (low priority, deferred):** a captured warning during a
  `fixed_effects` fit that is NOT a `ConvergenceWarning` is read and
  discarded rather than re-emitted via `warnings.warn` — a minor
  observability regression only for `fixed_effects` callers (an
  `fixed_effects=None` caller would still see such a warning). Not required
  by any spec scenario; left as a candidate follow-up rather than expanding
  this tier's scope further.
