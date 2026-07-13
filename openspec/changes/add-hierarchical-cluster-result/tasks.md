## 1. Result type + Optional random_state (test-first)

> Fixture note: the only producer of the labeled dict, `hierarchical_cluster_labels`,
> is not implemented until §2. So tasks 1.1 / 1.3 / 1.4 MUST construct an **inline
> labeled dict** (hand-built with the 11 output keys) — do **not** call the producer or
> reuse the dendrogram fixture `hierarchical_cluster_result` in `tests/fixtures.py`,
> which lacks the label/metric keys. This lets §1 go fully green at task 1.7,
> independently of §2.

- [x] 1.1 Write failing round-trip test: a `HierarchicalResult` built from an inline
      labeled dict serializes via `to_json()` / `json.dumps(dataclasses.asdict(...))`
      with `algorithm == "hierarchical"`, `cluster_labels` and `cluster_sizes` as
      `list[int]`, the three quality metrics plus `cophenetic_correlation` and
      `cut_height` as `float`, and `random_state` serializing to JSON `null` (under the
      default `allow_nan=False`).
- [x] 1.2 Write failing test: `ClusterResult(random_state=None)` and each subclass
      construct without error, and `dataclasses.asdict(result)["random_state"]` is `None`.
- [x] 1.3 Write failing test: `from_hierarchical_dict(d)` (with `d` an inline labeled
      dict) maps provenance keys with native casts (`linkage_method` `str`,
      `distance_metric` `str`, `cophenetic_correlation` `float`, `cut_height` `float`),
      stamps `random_state=None`, and does **not** mutate `d` (deep-copy compare — reuse
      the `_assert_dict_unchanged` pattern in `tests/test_cluster_result.py`).
- [x] 1.4 Write failing test: `HierarchicalResult.to_json()` raises `ValueError` on a
      non-finite `cophenetic_correlation` (mirror `test_to_json_rejects_non_finite_bic`).
- [x] 1.5 Make `ClusterResult.random_state` `Optional[int] = None` and update the base
      `ClusterResult` class docstring in lockstep — not just the `random_state`
      `Attributes:` line (`PCAResult` style: `None` when the algorithm is deterministic /
      no seed was supplied), but ALSO the three other spots that go stale once a third
      subclass exists: the class intro ("KMeans and GMM return…"), the `algorithm`
      `Attributes:` enumeration (add `ALGORITHM_HIERARCHICAL` / `"hierarchical"`), and the
      "Build via `from_kmeans_dict` / `from_gmm_dict`" line (add `from_hierarchical_dict`).
      The `check_public_api_docs` audit checks field-name *presence* only, so it will NOT
      catch this prose drift — it is a human-review fix. Keep it consistent with the
      `docs/result-types.md` discriminator note updated in task 3.3.
- [x] 1.6 Add `ALGORITHM_HIERARCHICAL = "hierarchical"` (append to
      `result_types.__all__` beside `ALGORITHM_KMEANS`/`ALGORITHM_GMM`) and
      `HierarchicalResult(ClusterResult)` with hierarchical-specific fields
      `linkage_method`, `distance_metric`, `cophenetic_correlation`, `cut_height` —
      **each with a default** (required now that base `random_state` has a default);
      no `cluster_centers`. The class docstring MUST enumerate **every** field in an
      `Attributes:` block (the `check_public_api_docs` audit requires it).
- [x] 1.7 Add `ClusterResult.from_hierarchical_dict(d)` adapter (no `random_state`),
      non-mutating, stamping `random_state=None`; make 1.1–1.4 green.

## 2. Public producer entry point (test-first)

- [x] 2.1 Write failing regression test: `perform_hierarchical_clustering(data)` still
      returns its **exact** dendrogram key set (`method`, `linkage_matrix`,
      `linkage_method`, `distance_metric`, `cophenetic_correlation`, `data_indices`,
      `feature_names`, `data_processed`) with **no** `cluster_labels` key.
- [x] 2.2 Write failing test: `hierarchical_cluster_labels(df)` returns a dict with
      `cluster_labels`, `n_clusters`, `cluster_sizes`, `silhouette_score`,
      `davies_bouldin_score`, `calinski_harabasz_score`, `feature_names`, and the
      provenance keys; assert `len(cluster_labels)` equals the number of clustered
      samples, `n_clusters == len(set(cluster_labels))`, and
      `2 <= n_clusters <= max_clusters` (structural — never a specific `k`).
- [x] 2.2b Write failing delegation test (S1's operative guarantee): on the same input,
      `hierarchical_cluster_labels(df)["n_clusters"]` equals
      `calculate_optimal_clusters_hierarchical(perform_hierarchical_clustering(df),
      method="silhouette")["optimal_n_clusters"]` (in-process equality — platform-safe;
      pins the auto-`k` delegation, which 2.2's structural bounds do not cover).
- [x] 2.3 Write failing test: `hierarchical_cluster_labels(df, n_clusters=3)` yields
      `n_clusters == 3`, `len(cluster_sizes) == 3`, and
      `sum(cluster_sizes) == len(cluster_labels)`.
- [x] 2.4 Write failing determinism test: two calls on identical input produce
      identical `cluster_labels` (same-process exact equality — **not** a committed
      golden; see design.md).
- [x] 2.5 Write failing error-path tests: `hierarchical_cluster_labels` propagates
      `ValueError` for (a) `method="ward"` with `metric="manhattan"`, (b) fewer than 2
      valid rows, (c) all-NaN input (each raised **before** the internal `try`, so it
      propagates unwrapped as `ValueError`).
- [x] 2.5b Write failing test for the `optimization_method` surface: each accepted value
      `"silhouette"` / `"calinski"` / `"davies_bouldin"` returns a valid dict with
      `2 <= n_clusters <= max_clusters`; and the metric-key-name footgun
      `optimization_method="silhouette_score"` raises `ValueError` — the producer
      validates `optimization_method` up front (against the accepted set) so the footgun
      surfaces as a clean `ValueError`, not the optimizer's re-wrapped `RuntimeError`.
- [x] 2.6 Write failing degenerate test: `n_clusters=1` yields `n_clusters == 1`,
      `cluster_sizes` of length 1, and the three quality metrics each `0.0` (finite);
      `from_hierarchical_dict(...).to_json()` still succeeds under `allow_nan=False`.
- [x] 2.6b Write failing boundary error-contract test: argument errors surface as a
      single `ValueError` type. `hierarchical_cluster_labels(df, n_clusters=n_samples)`
      raises `ValueError` (the producer rejects an out-of-range `n_clusters` — must be in
      `[1, n_samples - 1]` — up front, before `cut_dendrogram`), and a 2-row df with
      `n_clusters=None` raises `ValueError` (optimizer `max_clusters < 2`). A caller
      catches one exception type for every argument error.
- [x] 2.7 Implement `hierarchical_cluster_labels()` composing
      `perform_hierarchical_clustering` → `calculate_optimal_clusters_hierarchical`
      (when `n_clusters is None`, using `optimization_method`) → `cut_dendrogram`;
      full Google docstring with all params annotated and `Args:`/`Returns:`/`Raises:`;
      make 2.2–2.6b green.
- [x] 2.8 Write end-to-end test then confirm green:
      `ClusterResult.from_hierarchical_dict(hierarchical_cluster_labels(df))` yields a
      `HierarchicalResult` with `algorithm == "hierarchical"` and populated
      `cluster_labels`, `cluster_sizes`, `n_clusters`, and the three quality scores.

## 3. Exports + docs

- [x] 3.1 Write failing import test:
      `from sleap_roots_analyze import HierarchicalResult, hierarchical_cluster_labels`
      succeeds; both appear in `sleap_roots_analyze.__all__` with no duplicates; and
      `ALGORITHM_HIERARCHICAL` imports from `sleap_roots_analyze.result_types` and is
      in `result_types.__all__`.
- [x] 3.2 Add `HierarchicalResult` and `hierarchical_cluster_labels` (only — **not**
      the constant) to `__init__.py` and `__all__`; make 3.1 green.
- [x] 3.3 Update `docs/result-types.md`: add the hierarchical row to the types table
      (Built from `hierarchical_cluster_labels()`; adapter
      `ClusterResult.from_hierarchical_dict(d)` with **no** `random_state` kwarg) and
      add `ALGORITHM_HIERARCHICAL` to the discriminator-constants note. This
      discriminator note MUST agree with the base `ClusterResult` docstring enumeration
      updated in task 1.5 (same three constants) — they are DRY-duplicated and must not
      drift.
- [x] 3.4 Add a `docs/CHANGELOG.md` `[Unreleased]` entry: `### Added`
      (`hierarchical_cluster_labels`, `HierarchicalResult`, `from_hierarchical_dict`,
      `ALGORITHM_HIERARCHICAL`) and `### Changed` (`ClusterResult.random_state` →
      `Optional[int]`). Do **not** bump `pyproject.toml` or add a dated version
      heading — `0.1.0a5` is cut in a separate release PR (per `#176`).

## 4. Validation

- [x] 4.1 `openspec validate add-hierarchical-cluster-result --strict` (CLI not
      currently installed — if unavailable, hand-check format against `openspec/AGENTS.md`).
- [x] 4.2 `/lint` + full pytest + coverage via `/pre-merge-check`.

## 5. PR #182 review follow-ups (eberrigan)

- [x] 5.1 Honor the "every argument error → `ValueError`" contract fully: validate
      `method` (scipy linkage set), `metric` (scipy `pdist` set), and integer
      `n_clusters` up front, so a bogus `method`/`metric` or a float `n_clusters` no
      longer leaks a wrapped `RuntimeError`. Hoist the accepted-value sets to
      module-level constants shared with the producer's validation.
- [x] 5.2 Carry `data_indices` in the labeled dict (was silently dropped, unlike the
      sibling producers) so labels map back to source rows after NaN-row dropping; add a
      partial-NaN test asserting the mapping. `HierarchicalResult` still does not add the
      field (no `ClusterResult` subclass carries it).
- [x] 5.3 Validate `optimization_method` unconditionally (not only when `n_clusters is
      None`); document the `davies_bouldin_score == 0.0` single-cluster caveat, the
      possible `NaN` `cophenetic_correlation`, and the O(n^2) memory ceiling in the
      docstring.
- [x] 5.4 Reword the propagated error-path tests to assert exception *type* (not
      internal messages owned by the composed functions); add tests for unknown
      `method`/`metric`, non-integer `n_clusters`, `optimization_method` when
      `n_clusters` is set, empty-DataFrame, single-row auto-k, and the `cluster_labels`
      numpy-array type.

## 6. PR #182 review-round-2 follow-ups (subagent team, incl. eberrigan)

- [x] 6.1 **Blocking:** `method="centroid"`/`"median"` + a non-euclidean metric still
      leaked `RuntimeError` — scipy requires euclidean for centroid/median, not only
      ward, and neither `perform_hierarchical_clustering`'s pre-`try` check nor the
      producer's set-membership checks caught the *combination*. Extended
      `perform_hierarchical_clustering`'s check to `method in {"ward", "centroid",
      "median"}`; added regression tests at both the `perform_hierarchical_clustering`
      and `hierarchical_cluster_labels` levels.
- [x] 6.2 **Important:** the `random_state: Optional[int] = None` widening silently
      removed the pre-PR guardrail that rejected a missing seed on `KMeansResult`/
      `GMMResult` (both are always seeded by their producers). Added a
      `__post_init__` to each rejecting `random_state=None` (`TypeError`); only
      `HierarchicalResult` may omit it. Updated the test that previously asserted all
      three subclasses accept `None`; added a `GMMResult.random_state is int` type
      assertion (was untested, unlike the `KMeansResult` equivalent).
- [x] 6.3 **Important:** qualified the "hierarchical clustering is deterministic"
      claim everywhere it shipped unqualified (the `HierarchicalResult` class
      docstring, `hierarchical_cluster_labels`'s docstring, `docs/result-types.md`) —
      no RNG in the composed call path means same-process determinism, not
      cross-platform reproducibility (scipy tie-breaking is BLAS/platform-sensitive).
- [x] 6.4 **Important:** eliminated the redundant `cut_dendrogram` re-execution on the
      auto-`k` path (~11% extra O(n^2) silhouette pass with the default
      `max_clusters=10`). `calculate_optimal_clusters_hierarchical` now returns the
      winning candidate's `cut_result` (additive key) from its scan; the producer
      reuses it instead of re-cutting for the same `k`.
- [x] 6.5 Documented (not code-changed, per the review's own triage) that a
      non-numeric column reaching `linkage` with `standardize=False` is a *data*
      failure, not an *argument* error, and intentionally still raises `RuntimeError`
      — added a regression test locking this in. Documented the `feature_names`
      pre-standardization footgun and the `data_indices`-not-on-typed-view gap in
      `docs/result-types.md` (both pre-existing/epic-wide, per the review).
- [x] 6.6 Added tests for `n_clusters` edge cases the round-1 pass missed: `bool`,
      `numpy.float64`, and `<= 0` (only the upper `n_clusters == n_samples` bound was
      covered); added an end-to-end (non-synthetic) test driving a real `NaN`
      `cophenetic_correlation` through the full pipeline via all-identical input.

> Notes: `hierarchical_cluster_labels` takes no `random_state`, so the reproducibility
> determinism coverage guard does **not** require a `CASES` registry entry — do not add
> one. No hierarchical golden/pinned artifact is committed (scipy `linkage`/`fcluster`
> tie-breaking is BLAS/platform-sensitive).
