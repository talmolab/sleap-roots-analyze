> Test fixtures: add two new fixtures to `tests/fixtures.py` — see 1.1 and 1.2.
> Section 1/2 tests import `from sleap_roots_analyze.statistics import
> calculate_heritability_estimates, extract_blup_table` directly (no new
> public API surface is added in this tier — `fixed_effects` is a parameter on
> an existing exported function).
>
> Reconciled after `/review-openspec` rounds 1–3 plus one targeted empirical
> follow-up (round 1: 5 parallel reviewers; round 2: a targeted re-check of
> round 1's fixes; round 3: a fresh cold-read plus an empirically-verified
> fixture correction; round 4: a direct simulation of the second fixture,
> prompted by a follow-up question, which found and fixed a related
> methodology gap in the shrinkage oracle — see `design.md`'s four
> "Adversarial Review Reconciliation" sections for the full synthesis).
> Changes from the pre-review draft are called out inline where they affect
> a specific task.

## 1. Fixtures + `fixed_effects` support in `calculate_heritability_estimates` (test-first)

- [x] 1.1 Add `heritability_data_batch_confounded` fixture to `tests/fixtures.py`
      (mirror `heritability_data_known_h2`'s column names — `geno`/`rep`/
      `Barcode`, seed 42). **Pin concrete numeric parameters, matching Tier
      1's `heritability_data_unbalanced_reps` precedent — verified by direct
      simulation against real `statsmodels` (round-3 review), not derived
      analytically or eyeballed.** An earlier draft of this fixture (σ_G=3.0,
      shift=8.0, n=4 reps, a 16/4-genotype 3:1/1:3 partial mix) was
      **confirmed by simulation to produce the opposite of the intended
      effect** (uncorrected H² *below* corrected H², by 0.36–0.85 across
      seeds) — with only n=4 reps, any genuinely partial (not fully
      deterministic) per-genotype batch mix necessarily injects more
      *within*-genotype variance than *between*-genotype variance for any
      shift magnitude, so "increase the shift" (that draft's own stated
      remedy) could not have fixed it; the confound *pattern* itself was
      unfixable at n=4. The verified replacement:

      One trait `trait_batch_confounded` with a genotype effect
      `np.random.normal(0, 0.4)` (σ_G), per-observation noise
      `np.random.normal(0, 1.0)` (σ_E), base value 50. 20 genotypes total,
      each with **n=10 reps (200 rows)**. Two synthetic `"experiment"`
      batches, split symmetrically into two **equal-size** genotype groups
      (this symmetry is what makes the between-genotype variance from the
      confound exceed the within-genotype variance from partial mixing —
      an uneven group split, like the rejected 16/4 draft, does not):
      genotypes `G01`-`G10` ("mostly-A") have exactly 1 of their 10 reps in
      `"Bloom_B"` (9 in `"Bloom_A"`); genotypes `G11`-`G20` ("mostly-B") have
      the reverse (9 of 10 reps in `"Bloom_B"`, 1 in `"Bloom_A"`) — genuinely
      partial per-genotype mixing (every genotype has at least one rep in
      each batch), not full determinism (which is exercised separately by
      task **1.10**, not 1.9 — 1.9 mocks the fit directly and builds no
      confounded fixture at all). Add a systematic per-batch shift of
      `+10.0` to every `"Bloom_B"` observation. Return `(df, meta)` where
      `meta = {"trait": "trait_batch_confounded", "batch_col": "experiment"}`.
      **Use `np.random.seed(42)` + `np.random.normal(...)` (this file's
      existing global-state RNG convention, matching every other fixture in
      this file) — NOT `np.random.default_rng(42)`.** The two APIs produce
      different streams for the same seed number; the verification below was
      re-run under the *legacy* API specifically because this distinction
      matters (an earlier verification pass, done under `default_rng`
      before this discrepancy was caught, gave different — still
      correctly-signed — numbers that do not apply to the actual fixture
      code). Draw each genotype's own effect (one `np.random.normal(0,
      0.4)` call) immediately before iterating its reps, genotype by
      genotype in the order `G01..G10` (mostly-A) then `G11..G20`
      (mostly-B); within each genotype, draw one `np.random.normal(0, 1.0)`
      noise call per rep, in `Bloom_A`-then-`Bloom_B` order (i.e. the
      `n_a` `"Bloom_A"` reps come first in the per-genotype rep loop, then
      the `n_b` `"Bloom_B"` reps) — this exact call order is what the
      verified numbers below assume; a different loop/draw order will
      produce different (though, per the 10-seed robustness check below,
      likely still correctly-signed) numbers. **Verified under this exact
      legacy-API construction, seed=42: H²_uncorrected≈0.9405,
      H²_corrected≈0.7194, gap≈0.2211.** Robustness re-checked across 10
      other legacy seeds (1, 2, 3, 7, 13, 21, 99, 123, 777, plus 42 itself):
      minimum gap 0.2138, mean 0.3756 — all comfortably clear the 0.05
      threshold task 1.7 asserts, with 4x+ margin at the worst seed. Do not
      re-derive these numbers from scratch — they are locked in from this
      verification; if implementation produces a different result on this
      exact fixture, treat that as a real discrepancy to investigate (a
      code bug, or a draw-order/RNG-API mismatch versus this
      specification), not a reason to re-tune the shift/σ_G values.
- [x] 1.2 Add `heritability_data_field_block` fixture to `tests/fixtures.py`
      (mirror `heritability_data_known_h2`'s column names, seed 42). **Pin
      concrete numeric parameters, empirically verified against real
      `statsmodels` (round-4 review — a direct follow-up empirical check
      after round 3, prompted by the same "don't trust eyeballed fixture
      design" lesson):** one trait with a genotype effect
      `np.random.normal(0, 2.0)` (σ_G), per-observation noise
      `np.random.normal(0, 1.0)` (σ_E, pinned explicitly — an earlier draft
      never specified a residual noise term at all, risking degenerate
      within-genotype variance for genotypes with all reps in one block),
      base value 50, and a systematic per-`"block"` shift of `+5.0` for
      `"block_2"` rows. Use `np.random.seed(42)` + `np.random.normal(...)`
      (this file's convention, not `default_rng`, per the same RNG-API
      distinction task 1.1 now calls out explicitly). 15 genotypes total,
      split into two **replicate-count** groups (needed for task 3.2's
      shrinkage oracle — an earlier draft specified uniform n=3 for all 15
      genotypes, which cannot support it at all): 7 "low-rep" genotypes
      (`G01`-`G07`) with **n=2** reps each, 8 "high-rep" genotypes
      (`G08`-`G15`) with **n=10** reps each (94 rows total). Block skew
      SHALL be applied as the same proportion within each replicate-count
      group, not correlated with which group a genotype is in — pinned
      concretely: `{G01, G02, G03, G04, G05}` (5 of the 7 low-rep genotypes)
      and `{G08, G09, G10, G11, G12, G13}` (6 of the 8 high-rep genotypes)
      get 80% of their reps in `"block_1"` / 20% in `"block_2"`; the
      remaining genotypes in each group (`{G06, G07}` and `{G14, G15}`) get
      the reverse (20%/80%) — a ~71% block-1-heavy ratio in both
      replicate-count groups. Draw each genotype's own effect (one
      `np.random.normal(0, 2.0)` call) immediately before iterating its
      reps, genotype by genotype in the order `G01..G07` (low-rep) then
      `G08..G15` (high-rep); within each genotype, draw one
      `np.random.normal(0, 1.0)` noise call per rep, in
      `"block_1"`-then-`"block_2"` order. **This orthogonality
      is required, not incidental**: an earlier draft skewed block
      assignment *by genotype ID* (`G01`-`G10` vs `G11`-`G15`), which
      happened to correlate with the replicate-count grouping (`G01`-`G07`
      is entirely inside the block-1-heavy range) — verified by simulation
      to produce an unreliable shrinkage oracle (40% failure rate on 3.2,
      §below) even though the simpler BLUP-difference oracle (3.1) was
      unaffected. Used by the field-BLUP oracle (3.1) — not required to show
      an H² gap, only a measurable BLUP-adjusted-mean difference between
      `fixed_effects=None` and `fixed_effects=["block"]`, **re-verified
      under the legacy RNG API and this exact concrete block assignment:
      0/10 seeds failed to show a difference (max diff 3.7–5.1 across
      seeds 1,2,3,7,13,21,42,99,123,777)** — and the shrinkage-regression
      oracle (3.2), **re-verified under the same conditions: 0/10 seeds
      failed, using the detrended comparison task 3.2 now specifies** (a
      naive, non-detrended comparison failed the majority of the time on an
      earlier draft of this fixture — see 3.2's own note for why, and do
      not skip the detrending step described there).
- [x] 1.3 Write failing test `test_fixed_effects_none_matches_current_behavior`
      (`tests/test_statistics.py`, new `TestFixedEffects` class): call
      `calculate_heritability_estimates` on `heritability_data_known_h2` three
      ways — without `fixed_effects`, with `fixed_effects=None`, and with
      `fixed_effects=[]` (empty list) — assert all three are identical to
      each other and to a hand-computed expectation matching pre-change
      behavior (formula `"value ~ 1"`, `intercept ==
      float(result.fe_params["Intercept"])`). Including the empty-list case
      closes a boundary-value gap flagged in review: `fixed_effects=[]` must
      behave identically to `None`, not attempt a formula with zero terms.
- [x] 1.4 Write failing test `test_missing_fixed_effect_column_returns_structural_error`:
      `calculate_heritability_estimates(df, trait_cols,
      fixed_effects=["nonexistent_col"])` returns `{"error": "Missing
      required columns: [...]"}` (top-level short-circuit, no per-trait
      entries), listing `"nonexistent_col"`.
- [x] 1.4a **(New — added after round-3 review)** Write failing test
      `test_fixed_effect_column_name_with_patsy_metacharacter_rejected`:
      `fixed_effects=["rep*block"]` (a column name containing a patsy
      formula operator) on a `df` that happens to also have separate
      `"rep"` and `"block"` columns; assert a clear, loud top-level error
      (e.g. extending the same `{"error": "Missing required columns:
      [...]"}` shape, or a distinct validation error — either is
      acceptable, just not a silent misinterpretation), NOT a silent
      formula misparse where `C(rep*block)` gets evaluated as elementwise
      multiplication of the two *different*, unintended columns. Guards a
      narrow but genuinely silent-corruption tail case round-3 review
      flagged: near-zero implementation cost (`fe.isidentifier()` check
      alongside the existing missing-column validation) closes it, even
      though it requires two contrived conditions simultaneously to trigger
      the silent-misparse form.
- [x] 1.5 Write failing test `test_fixed_effect_column_always_treated_as_categorical`:
      build a small fixture where a fixed-effect column has numeric-looking
      values (e.g. `wave_number` = 1, 2, 3) but a real per-level shift.
      **`calculate_heritability_estimates`'s return value never exposes the
      fitted `result` object** (only `blup`/`intercept` are extracted) — do
      not attempt to reach it from the public API. Assert indirectly instead:
      fit the same `model_data` through `smf.mixedlm` twice directly in the
      test (once with the production formula string `"value ~
      C(wave_number)"`, once with a deliberately continuous formula
      `"value ~ wave_number"`), and confirm the two fits' `fe_params` differ
      in shape (one coefficient per non-reference level vs. a single slope
      coefficient) — this is the same "fit independently in the test" pattern
      tasks 2.2/2.4 also use, resolved here rather than left as an open
      fallback decision (a pre-review draft of this task left the approach
      undecided; review flagged that as a stall risk).
- [x] 1.6 Write failing test `test_nan_in_fixed_effect_column_drops_row`:
      one row has a valid trait value and genotype but `NaN` in a named
      fixed-effect column; assert that row is excluded from the fitted
      `n_observations` when `fixed_effects=[...]` includes that column, and
      NOT excluded when `fixed_effects=None` on the same DataFrame.
- [x] 1.7 Write failing test `test_batch_confounded_uncorrected_h2_exceeds_corrected`
      using `heritability_data_batch_confounded` (1.1): assert
      `calculate_heritability_estimates(df, ["trait_batch_confounded"])["trait_batch_confounded"]["heritability"]`
      (no fixed effects) is greater than the same call with
      `fixed_effects=["experiment"]` by at least 0.05 (absolute) — the
      roadmap's core Tier 2 oracle, with an explicit margin rather than a
      bare direction-only assertion.
- [x] 1.8 Write failing test `test_mixed_model_failure_with_fixed_effects_recorded_as_error`
      (mirror Tier 1's `test_mixed_model_fit_failure_has_no_blup_keys`): mock
      `statsmodels.formula.api.mixedlm` (patch target confirmed correct —
      `statistics.py` does `import statsmodels.formula.api as smf` then calls
      `smf.mixedlm(...)`, so `patch("statsmodels.formula.api.mixedlm",
      side_effect=Exception("boom"))` is the right target, matching the
      existing precedent at `tests/test_statistics.py:451`) to raise when
      `fixed_effects` is set; assert the trait's dict is `{"error": "Mixed
      model failed: ...", "model_type": "mixed_model_failed"}`, no exception
      propagates, and the remaining traits in the same call are still
      processed.
- [x] 1.9 **(New — added after review)** Write failing test
      `test_convergence_warning_treated_as_failure`: using
      `warnings.catch_warnings()`, mock (or otherwise induce)
      `model.fit(reml=True)` to emit a `ConvergenceWarning` without raising
      (e.g. `patch.object` the fit method to call `warnings.warn(...,
      ConvergenceWarning)` and return a real-but-arbitrary fitted `result`);
      assert the trait's dict is classified as failed
      (`{"error": ..., "model_type": "mixed_model_failed"}`), NOT returned as
      a successful `mixed_model` result with `blup`/`intercept` values. This
      exercises the specific gap review found: `statsmodels.MixedLM.fit()`
      does not reliably *raise* on a fixed effect confounded with genotype —
      it can instead warn and still return a plausible-looking `result` — so
      1.8's raised-exception test alone does not cover this failure mode.
- [x] 1.9a **(New — added after round-3 review)** Write failing test
      `test_unrelated_warning_during_fit_does_not_fail_trait`: mock (or
      otherwise induce) `model.fit(reml=True)` to emit an unrelated warning
      of a *different* category (e.g. a plain `UserWarning` or `FutureWarning`,
      not `ConvergenceWarning`) without raising, on otherwise-normal data;
      assert the trait's dict is a normal successful `mixed_model` result
      (`blup`/`intercept`/`heritability` present, no `error` key) — the
      false-positive counterpart to 1.9, guarding against an implementation
      that treats "any captured warning" as a failure instead of checking
      specifically for `ConvergenceWarning`'s category.
- [x] 1.9b **(New — added during implementation)** Write failing test
      `test_convergence_warning_not_caught_without_fixed_effects`: same
      mocked-warning setup as 1.9, but call
      `calculate_heritability_estimates` with `fixed_effects=None` (or
      omitted); assert the trait's dict is a normal successful
      `mixed_model` result — the warning-capture behavior added by this
      tier must be gated on `fixed_effects` being non-empty, or it would
      silently break the "`fixed_effects=None` reproduces current behavior
      exactly" guarantee (a real ambiguity caught during implementation:
      the requirement text describing warning-capture did not explicitly
      say this was conditional, even though the earlier byte-for-byte
      guarantee logically required it — see the statistics-api spec's
      "Heritability Model Fixed Effects" requirement, now stated
      explicitly).
- [x] 1.10 **(New — added after review, organic/non-mocked)** Write a test
      `test_near_fully_confounded_fixed_effect_organic_behavior` that
      constructs a fixture where a fixed effect is a *near-deterministic*
      function of genotype (e.g. 18 of 20 genotypes appear in only one of two
      experiment batches, 2 genotypes split 50/50 as the only source of
      within-genotype batch variation) and calls
      `calculate_heritability_estimates` with that fixed effect, with no
      mocking. Document (via an assertion, not just a comment) whichever of
      the following `statsmodels` actually does on this fixture: raises
      (caught by 1.8's path), warns (caught by 1.9's path), or fits without
      warning (in which case assert on whatever heritability/intercept value
      results, so a future `statsmodels` version change that alters this
      behavior is caught as a test failure rather than silently drifting).
      **Sequencing note (round-2 review): unlike this section's other tasks,
      this test's assertion content cannot be predicted before 1.11 lands —
      it is a characterization/regression-pinning test, not a predictive
      red-green TDD test, despite its position in this "(test-first)"
      section. Write the test body itself after implementing 1.11: run it
      once to observe actual `statsmodels` behavior on this fixture, then pin
      that specific observed outcome as the assertion.** Not required to
      assert a specific one of the three outcomes in advance of running it —
      only required to observe and pin down the *actual* current behavior,
      since review found this is not guaranteed by `statsmodels`'
      documentation. Not represented by a normative spec scenario for the
      same reason (there is nothing to prescribe in advance — only to pin).
- [x] 1.11 Implement `fixed_effects: Optional[List[str]] = None` on
      `calculate_heritability_estimates` (`statistics.py:195-`): extend
      `required_cols` with `fixed_effects` (when truthy) for the top-level
      missing-column check; extend the per-trait `subset` to
      `df[[trait, genotype_col] + (fixed_effects or [])].dropna()` (computed
      once, before the `if use_mixed_model:` branch, so both the mixed-model
      and ANOVA-based paths see the same row-filtering — only the
      mixed-model path additionally uses `fixed_effects` in its formula);
      build the mixed-model formula as `"value ~ " + " + ".join(f"C({fe})"
      for fe in fixed_effects)` when `fixed_effects` is non-empty, else keep
      `"value ~ 1"`. **Only when `fixed_effects` is non-empty**, wrap
      `model.fit(reml=True)` in `warnings.catch_warnings(record=True)` and
      call `warnings.simplefilter("always")` as the first statement inside
      that block — **gating this on `fixed_effects` is required, not
      optional (caught during implementation): the requirement text
      describing this behavior did not originally say it was conditional,
      but the earlier "`fixed_effects=None` reproduces current behavior
      exactly" guarantee logically requires it — an unconditional
      warning-to-failure check would newly fail existing, non-`fixed_effects`
      callers whose fit happens to emit a convergence warning today and
      succeeds. See task 1.9b.** Without the `simplefilter("always")` call
      specifically, Python's
      default once-per-location warning filter can silently drop a repeat
      occurrence of the same `ConvergenceWarning` for a later trait in the
      same process, since `statsmodels` raises it from the same source
      line every time. `record=True` alone does not force every occurrence
      to be recorded; only `simplefilter("always")` does. This bug would NOT
      be caught by a test that checks only one warning occurrence in
      isolation, because pytest's own warnings plugin already wraps tests in
      `simplefilter("always")` by default and would mask a missing call
      inside production code — see task 1.13.** For each captured warning,
      check `issubclass(w.category,
      statsmodels.tools.sm_exceptions.ConvergenceWarning)` — **NOT** a
      substring match on the warning's message text (round-3 review found
      several real `statsmodels` convergence-related warning messages that
      do not contain the word "convergence" at all, e.g. "The MLE may be on
      the boundary of the parameter space." and "The Hessian matrix at the
      estimated parameter values is not positive definite." — a
      message-text filter would silently miss exactly the messages a
      near-confounded fixed effect is most likely to trigger). If any
      captured warning matches that category, treat the trait identically
      to a raised exception (same error dict, `model_type:
      "mixed_model_failed"`); a warning of any OTHER category captured in
      the same block SHALL NOT be treated as a fit failure — do not use "any
      captured warning" as the check.
      Also validate each `fixed_effects` name with `fe.isidentifier()`
      alongside the existing missing-column check, rejecting a name
      containing a patsy formula operator before it ever reaches the
      formula string (closes 1.4a's silent-misparse tail case). Reuse the
      existing per-trait `try/except` for raised exceptions
      unchanged. Make 1.3–1.9, 1.9a, 1.9b, 1.4a green (1.10, 1.12, and 1.13 all depend
      on this implementation too, but are written/verified immediately after
      — 1.10 per its own sequencing note above, since its assertion content
      isn't knowable before this implementation exists; 1.12/1.13 are
      ordinary post-implementation regression checks). Update the function's
      docstring:
      document the new parameter, the categorical-only (`C()`) treatment,
      the subset/dropna change (and that it applies identically regardless
      of `force_method`), the "metadata covariates only, not biological
      traits" convention (per issue #114), that `replicate_col` is
      independent of `fixed_effects`, and that convergence warnings are
      treated as failures. Do NOT yet update the `Returns:` section's
      `intercept` description — that's task 2.6, once the marginal-intercept
      logic exists to describe.
- [x] 1.12 **(New — added after review)** Write failing test
      `test_fixed_effects_with_anova_based_force_method`: call
      `calculate_heritability_estimates(df, trait_cols,
      fixed_effects=["experiment"], force_method="anova_based")` on
      `heritability_data_batch_confounded`; assert (a) the trait's
      `model_type == "anova_based"` (fixed effects do not change which method
      is used), (b) a row with `NaN` in `"experiment"` is still excluded from
      that trait's `n_observations` (same row-filtering as the mixed-model
      path), and (c) no `blup`/`intercept` keys are present (unchanged from
      today's ANOVA-based behavior). Should already be green once 1.11 lands
      — if it isn't, that's a real bug in how `fixed_effects` was scoped to
      the mixed-model branch.
- [x] 1.13 **(New — added after round-2 review)** Write failing test
      `test_repeat_convergence_warning_across_traits_both_fail`: mock the fit
      call so the *same* `ConvergenceWarning` (same message, same
      simulated source location) is emitted for **two different traits**
      within a single `calculate_heritability_estimates(df, trait_cols=[...])`
      call (not two separate calls/processes). Assert **both** traits'
      dicts are classified as failed. This specifically exercises the gap
      round-2 review found: Python's default warning filter is
      once-per-location, so without an explicit
      `warnings.simplefilter("always")` inside 1.11's `catch_warnings` block,
      the *second* trait's identical warning could be silently suppressed by
      the registry and that trait would incorrectly succeed. A test that
      checks only one warning occurrence in isolation (1.9) cannot expose
      this, because pytest's own warnings plugin already applies
      `simplefilter("always")` around every test and would mask a missing
      call in production code.

## 2. Empirical frequency-weighted intercept (test-first)

- [x] 2.1 Write failing test `test_marginal_intercept_none_equals_plain_intercept`:
      with `fixed_effects=None`, `intercept` equals
      `float(result.fe_params["Intercept"])` exactly (already covered by 1.3,
      but assert explicitly here as the anchor case for this section's
      helper).
- [x] 2.2 Write failing test `test_marginal_intercept_matches_hand_computed_weighted_average`:
      build a small fixture with one fixed effect (`"experiment"`) having two
      known, unequal observed-frequency levels; **independently re-fit the
      same `model_data` via a direct `smf.mixedlm(...).fit(reml=True)` call in
      the test** (the function's return value does not expose `result`, so
      this is the only way to obtain `fe_params` for a genuinely independent
      oracle — same pattern as 1.5), then compute the expected intercept by
      hand from that independently-fit `result.fe_params` and the fixture's
      known level frequencies (`fe_params["Intercept"] +
      freq[level] * offset[level]` summed over non-reference levels), and
      assert the production function's returned `intercept` matches within
      floating-point tolerance.
- [x] 2.3 Write failing test `test_marginal_intercept_differs_from_reference_level_when_unbalanced`:
      same fixture as 2.2 (unequal level frequencies); assert the returned
      `intercept` differs from the raw `result.fe_params["Intercept"]` (from
      the same independent re-fit) by more than floating-point tolerance —
      proves the weighting has an effect, not a silent no-op.
- [x] 2.4 Write failing test `test_marginal_intercept_multiple_fixed_effects_independent`:
      `fixed_effects=["experiment", "block"]`, each with its own known level
      frequencies; independently re-fit and assert the returned `intercept`
      equals the base `Intercept` plus `experiment`'s independently-computed
      weighted contribution plus `block`'s independently-computed weighted
      contribution — guards against conflating or double-counting the two
      effects.
- [x] 2.5 **(New — added after review)** Write failing test
      `test_marginal_intercept_float_dtype_fixed_effect_column`: build a
      fixture where the fixed-effect column is explicitly `float64`-typed
      (e.g. `pd.Series([1.0, 2.0, 3.0, ...])`, not `int` or `str`/`category`);
      assert the intercept computation correctly attributes each level's
      coefficient (matching an independent hand-computation as in 2.2), not a
      silently-defaulted `0.0` for a level whose reconstructed key failed to
      match `fe_params`'s actual key.
- [x] 2.5a **(New — added after round-2 review)** Write failing test
      `test_marginal_intercept_non_sorted_categorical_order`: build a fixture
      where the fixed-effect column is a `pd.Categorical` with an explicit,
      deliberately non-alphabetical/non-numeric `categories=[...]` order
      (so patsy's reference level is the first *declared* category, not the
      first in sorted order — none of 2.2/2.4/2.5's fixtures exercise this,
      since their levels' sorted order happens to coincide with patsy's
      fitted order). Assert the intercept computation still attributes each
      level's frequency to its correct coefficient (matching an independent
      hand-computation). This specifically catches a positional-pairing bug
      that count-only or sorted-order-assuming implementations would pass
      undetected on 2.2/2.4/2.5's fixtures alone.
- [x] 2.6 Implement a private helper (e.g. `_marginal_intercept(result,
      model_data, fixed_effects)`) in `statistics.py`. **Revised after
      review**: do NOT reconstruct the expected `fe_params` key forward from
      each observed level's raw value (the original draft's
      `f"C({fe})[T.{level}]"` pattern risks silently misattributing a level
      to the reference level on a dtype/formatting mismatch — e.g. a
      `float64` column). Instead: for each fixed effect, parse
      `result.fe_params`'s actual index with a regex
      (`^C\({fe}\)\[T\.(.*)\]$`) to recover the set of levels patsy actually
      fit non-reference coefficients for (the recovered string SHALL be
      matched back to `model_data[fe]`'s actual values by equality — NOT by
      positional pairing against a separately-sorted list of unique levels,
      which would silently mispair frequencies under a non-default category
      order; see 2.5a); compute each recovered level's frequency in
      `model_data[fe]` (that trait's own post-`dropna()` fitted subset —
      NOT the raw input `df`, since a level present in `df` can be entirely
      absent from a specific trait's `model_data` due to that trait's own
      missingness pattern, per round-2 review); assert the recovered
      non-reference coefficient count equals `model_data[fe].nunique() - 1`
      (raise `ValueError` if not — a silent mismatch is worse than a loud
      failure here); sum `frequency * coefficient` per fixed effect, sum
      across fixed effects, add `result.fe_params["Intercept"]`. Call this
      helper from the mixed-model branch instead of
      `float(result.fe_params["Intercept"])` whenever `fixed_effects` is
      non-empty; keep the plain `Intercept` lookup when `fixed_effects` is
      empty/`None`. Make 2.1–2.5a green. **Also**: update
      `calculate_heritability_estimates`'s docstring `Returns:` section —
      the current sentence ("intercept: ... from
      `result.fe_params["Intercept"]`") becomes inaccurate once this branch
      exists; describe the empirical frequency-weighted case explicitly,
      including the caveat that it is sample-composition-dependent and can
      differ trait-to-trait due to each trait's own `dropna()` (this was
      flagged in review as a docstring-staleness risk with no task covering
      it).
- [x] 2.7 **(New — added after review)** Write failing test
      `test_blupresult_intercepts_passthrough_fixed_effects`
      (`tests/test_blup_result.py`): build a `heritability_results`-shaped
      dict by hand where a trait's `intercept` is a known empirical
      frequency-weighted value (as if produced by `fixed_effects`), pass it
      through `extract_blup_table()` then `BLUPResult.from_blup_table(df,
      intercepts=...)`; assert `result.intercepts[trait]` equals that exact
      value, unchanged — confirms the pass-through the
      `serializable-result-types` delta spec's "intercepts values pass
      through unchanged when the source used fixed_effects" scenario
      requires. No implementation change expected (should already be green);
      this is a regression guard, not new logic.
- [x] 2.8 **(New — added after review)** Update
      `BLUPResult.intercepts`'s docstring `Attributes:` entry in
      `result_types.py` (currently ~line 832) to describe the empirical
      frequency-weighted semantics when the source
      `calculate_heritability_estimates` call used `fixed_effects` — a
      docs-only change (`BLUPResult`/`from_blup_table` already store/pass
      through whatever `intercept` float they're given, per 2.7). Also add a
      short caveat bullet to the `BLUPResult` row in `docs/result-types.md`,
      mirroring the existing per-type caveat bullets already in that file
      (e.g. the hierarchical-clustering "deterministic" caveat).

## 3. Field-block BLUP oracle (test-first, integration-level)

- [x] 3.1 Write failing test `test_field_block_fixed_effect_changes_blup_adjusted_means`
      using `heritability_data_field_block` (1.2): run
      `calculate_heritability_estimates` + `extract_blup_table` once with
      `fixed_effects=None` and once with `fixed_effects=["block"]` (note:
      independent of `replicate_col`, which stays at its default); assert at
      least one genotype/trait adjusted-mean value differs between the two
      runs beyond floating-point tolerance — the roadmap's second oracle half
      ("field BLUPs with block correction differ from genotype-only BLUPs").
      No implementation change expected here — this test should pass once
      Sections 1–2 are implemented; if it doesn't, that's a real integration
      bug to fix before moving on.
- [x] 3.2 **(New — added after review)** Write failing test
      `test_shrinkage_scales_with_replication_under_fixed_effects` using
      `heritability_data_field_block` (1.2, which provides the
      7-low-rep(n=2)/8-high-rep(n=10) split this oracle needs).
      Run `calculate_heritability_estimates(..., fixed_effects=["block"])`
      then `extract_blup_table()`. **The raw-mean reference point must be
      the block-detrended per-genotype mean, not the naive
      `df.groupby(genotype)[trait].mean()` (round-4 review — empirically
      verified this distinction is required, not stylistic): for each
      observation, subtract the fitted `C(block)` coefficient for that row's
      (non-reference) block level before averaging within genotype** — i.e.
      `raw_mean_detrended[g] = mean(trait_ij - block_coef[block_ij])` where
      `block_coef["block_1"] = 0` (the reference level) and
      `block_coef["block_2"]` is the fitted non-reference coefficient. The
      naive (non-detrended) raw mean is itself contaminated by each
      genotype's own block composition — the exact thing the fixed effect
      corrects for — so comparing it directly to the corrected BLUP is not a
      clean shrinkage test: **verified empirically, the naive comparison
      failed the per-genotype shrinkage assertion 50/50 times and the
      group-mean-ratio assertion 6+/30 times on this fixture; the detrended
      comparison failed 0/50 times on both.** **Revised during implementation
      (a 5th, code-level catch none of the 4 review rounds surfaced, since
      it only appears once task 2.6's marginal-intercept logic and this task
      run together): do NOT center on the run's own returned `intercept`.**
      Once 2.6 lands, `intercept` for a `fixed_effects` run is the *marginal*
      (frequency-weighted) value, not the reference-level one. Because
      `adjusted_mean = intercept + blup[g]`, `|adjusted_mean - intercept|`
      always simplifies to `|blup[g]|` regardless of which intercept
      convention is used (the two cancel) — but `raw_mean_detrended[g]` has
      no such dependency on intercept convention at all, so centering it on
      the *marginal* intercept (instead of the reference-level intercept
      the detrending itself is naturally expressed relative to) introduces a
      constant offset that does not cancel under the absolute-value
      comparison, and empirically breaks the assertion for a subset of
      genotypes. Use `blup[g]` directly (from
      `calculate_heritability_estimates`'s returned dict, not
      `extract_blup_table()`'s already-summed `adjusted_mean`) and the
      **reference-level** `Intercept` from an independent re-fit (same
      pattern as 2.2's fixture) as the shared center for both sides: assert
      (a) every genotype's `abs(blup[g])` is smaller than
      `|raw_mean_detrended[g] - reference_level_intercept|` (the stronger,
      per-genotype property — Tier 1's own original assertion shape, now
      confirmed to hold reliably once both detrended AND correctly
      centered), and (b) this shrinkage gap is larger for low-replicate
      genotypes than high-replicate genotypes — re-verifies Tier 1's
      existing shrinkage guarantee still holds with `fixed_effects` set.

## 4. Config + pipeline wiring (test-first)

- [x] 4.1 Write failing test `test_statistics_config_fixed_effects_default_none`
      (`tests/test_step_statistical_analysis.py`, mirroring Tier 1's
      `test_generate_blup_table_default_true`): `StatisticsConfig().fixed_effects
      is None`.
- [x] 4.2 Add `fixed_effects: Optional[List[str]] = None` to `StatisticsConfig`
      (`pipeline/config/components.py:532`), with a docstring `Attributes:`
      entry describing the parameter and cross-referencing
      `calculate_heritability_estimates`'s own docstring for the
      categorical-treatment, metadata-only, and empirical
      frequency-weighted-intercept conventions (canonical source stays
      `calculate_heritability_estimates`'s docstring; this field's docstring
      cross-references it rather than duplicating the prose). Make 4.1
      green.
- [x] 4.3 Write failing test `test_fixed_effects_threaded_into_heritability_call`
      (mirror Tier 1's `test_blup_csv_written_when_both_enabled`): run
      `StatisticalAnalysisStep.execute()` twice on the same fixture data
      (which must include a metadata column suitable as a fixed effect, e.g.
      an `"experiment"` column) — once with `statistics.fixed_effects=None`,
      once with `statistics.fixed_effects=["experiment"]` — assert the two
      runs' `08_heritability_results.csv` (and, if `generate_blup_table=True`,
      `08_blup_adjusted_means.csv`) differ for the affected trait.
- [x] 4.4 Write failing test `test_fixed_effects_qc_config_no_statistics_resolves_none`
      (mirror Tier 1's `test_blup_table_works_with_qc_config_no_statistics`):
      run `StatisticalAnalysisStep.execute()` with a bare `QCPipelineConfig`
      (no `statistics` field); assert no `AttributeError` and behavior
      identical to `fixed_effects=None`.
- [x] 4.5 Implement threading in `StatisticalAnalysisStep.execute()`
      (`pipeline/steps/statistical_analysis.py`): resolve `fixed_effects` via
      the same `getattr(config, "statistics", None)` guard already used for
      `calculate_heritability`/`generate_blup_table`
      (`statistical_analysis.py:154-169`), defaulting to `None` when
      `config.statistics` is absent; pass it into the existing
      `calculate_heritability_estimates(...)` call
      (`statistical_analysis.py:172-179`). Make 4.1, 4.3, 4.4 green.

## 5. Docs

- [x] 5.1 Update `docs/API.md`'s `## statistics Module` entry for
      `calculate_heritability_estimates` to include `fixed_effects` in its
      documented signature/defaults.
- [x] 5.2 Add a `docs/CHANGELOG.md` `[Unreleased]` entry: `### Added` —
      `calculate_heritability_estimates(fixed_effects=...)`,
      `StatisticsConfig.fixed_effects`; `### Changed` — when `fixed_effects`
      is used, the BLUP-adjusted-mean `intercept` becomes an empirical
      frequency-weighted value instead of the raw model intercept, and a
      captured convergence warning during fit is now treated as a trait-level
      failure (no change when `fixed_effects` is unset). One-line rationale
      on the **first** `### Added` bullet — round-3 review checked Tier 1's
      actual entry and found the rationale sentence there sits on the first
      `### Added` bullet (`extract_blup_table`), not the last; an earlier
      draft of this task cited "last," backwards from the actual precedent:
      Tier 2 of the cross-platform genotype-prediction program (#114).
> Note: `result_types.py`'s `BLUPResult.intercepts` docstring update and the
> `docs/result-types.md` caveat bullet are tracked as task 2.8, kept in
> Section 2 for TDD locality with 2.6/2.7's intercept work — not duplicated
> as a checkbox here.

## 6. Validation

- [x] 6.1 `openspec validate add-heritability-fixed-effects --strict` —
      resolve every reported issue before requesting review.
- [x] 6.2 `/lint` (black + ruff) on all changed files.
- [x] 6.3 Full `uv run pytest --cov --cov-branch` — confirm no regressions in
      `test_statistics.py`, `test_step_statistical_analysis.py`,
      `test_heritability_result.py`, `test_blup_result.py`, and confirm the
      new fixtures/tests pass, including the organic convergence-behavior
      test (1.10) on at least the CI platforms it actually runs on.
- [x] 6.4 `/review-openspec` — adversarial proposal review. Rounds 1–3 plus a
      targeted round-4 empirical follow-up are complete and reconciled (see
      `design.md`'s four "Adversarial Review Reconciliation" sections). This
      task is not satisfied until the user has reviewed and approved the
      reconciled proposal — required before implementation (Sections 1–5
      above) begins, per the roadmap's per-tier loop.

## 7. PR review follow-up (`/review-pr` on PR #193, after merge into main was not done — fixes landed on the open PR)

- [x] 7.1 Fix BLOCKING: `fixed_effects` element that isn't a `str` (e.g. an
      int-labeled CSV column) crashed with an uncaught `AttributeError` from
      `fe.isidentifier()`, contradicting the function's own documented
      "nothing propagates to the caller" contract. Fixed:
      `isinstance(fe, str)` checked first (short-circuiting) in the same
      validation line. Test:
      `test_fixed_effect_non_string_name_rejected_not_crashed`.
- [x] 7.2 Documented in the statistics-api spec delta: the non-str rejection
      (7.1) and the pre-existing genotype/replicate-collision rejection
      (already implemented and tested, but missing from the normative spec
      text per the review's spec-sync finding) both now have `#### Scenario:`
      entries.
- [x] 7.3 **(Fixed in this PR, on user request rather than deferred)**
      `fixed_effects` columns are excluded from the low-`H2`-filtering trait
      scan for direct API callers (`remove_low_h2=True`), but
      `StatisticalAnalysisStep` always calls with `remove_low_h2=False`, so
      the *pipeline's* upstream `trait_cols` (fixed once in
      `LoadDataAndImagesStep` via `get_trait_columns`) had no knowledge of
      `config.statistics.fixed_effects` and only excluded names matching a
      hardcoded substring list. Fixed via the auto-derive design (chosen
      over validate-disjoint-and-require-manual-sync, which would only make
      the trap loud instead of removing it): `VizPipelineConfig.__post_init__`
      (`viz_config.py`) now unions `statistics.fixed_effects` into
      `data.additional_exclude_cols` at config-construction time, deduped —
      no step-ordering change needed. `QCPipelineConfig` has no `statistics`
      field today, so this fix applies to `VizPipelineConfig`, the only
      config class composing both `data` and `statistics`. Tests:
      `test_viz_pipeline_config_auto_excludes_fixed_effects`,
      `test_viz_pipeline_config_auto_exclude_dedups_overlapping_names`,
      `test_viz_pipeline_config_no_fixed_effects_leaves_additional_exclude_unchanged`,
      and an integration test,
      `test_fixed_effect_column_excluded_from_pipeline_trait_cols`, proving
      `LoadDataAndImagesStep` no longer treats a `"block"`-named fixed effect
      as a trait.
- [x] 7.4 **(Fixed in this PR)** Added `StatisticsConfig.__post_init__`
      (`components.py`) rejecting `fixed_effects` that isn't `None` or a
      `list[str]` — catches the bare-string case (a `str` is iterable, so it
      would otherwise silently become a per-character `fixed_effects` list
      producing a misleading "Missing required columns" error). The
      nonexistent-column half of this task needed no new code:
      `calculate_heritability_estimates`'s existing missing-column check
      already covers it at runtime, since column names can't be validated
      until data loads. Added a commented `fixed_effects` example to both
      viz golden templates (`viz_template_with_images.yaml`,
      `viz_template_no_images.yaml`) and a note in `docs/API.md`. Tests:
      `test_statistics_config_fixed_effects_rejects_non_list`,
      `test_statistics_config_fixed_effects_rejects_non_str_elements`.
- [x] 7.5 **(Fixed in this PR)** The `ConvergenceWarning` heuristic's
      confirmed false-negative is now also surfaced as a `UserWarning` at
      call time (`statistics.py`, in the mixed-model success path, gated on
      `fixed_effects` being set — same gating convention as the
      `ConvergenceWarning`-capture block above it): after a successful fit
      with zero `ConvergenceWarning`s, for each fixed effect, if any
      genotype's observations are confined to a single level of a fixed
      effect that has more than one level overall, a `UserWarning`
      describing the possible confound is emitted. Purely diagnostic — does
      not change `model_type`/`blup`/`intercept` (unlike the
      `ConvergenceWarning`-as-failure path, a different, already-correct
      mechanism). Confirmed to fire correctly (not a false positive) on the
      existing `heritability_data_field_block` fixture (1.2): 7 of its
      15 genotypes have only n=2 reps at an 80/20 block skew, which rounds
      to 0 reps in one block for several of them — a real, previously-silent
      confound this heuristic now surfaces. Tests:
      `test_confounded_fixed_effect_emits_user_warning`,
      `test_unconfounded_fixed_effect_emits_no_user_warning`
      (`TestConfoundWarning` in `test_statistics.py`).
- [x] 7.6 **(Fixed in this PR)** Duplicate entries *within* `fixed_effects`
      itself are now rejected upfront with a structural
      `{"error": "Duplicate fixed_effects column name(s): [...]"}`, matching
      the existing missing-column/reused-name error shape, instead of
      degrading to an obscure `mixed_model_failed` error from patsy. Added
      the single-level (zero-variance) fixed-effect regression test —
      confirmed green with no implementation change needed (verified
      correct by hand-tracing in pre-merge review: `_marginal_intercept`'s
      "exactly one unmatched level" identity check holds even when
      `fixed_effects` has zero non-reference coefficients). Tests:
      `test_duplicate_fixed_effects_names_rejected`,
      `test_single_level_fixed_effect_succeeds`.
