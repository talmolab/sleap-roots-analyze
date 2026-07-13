## Context

`ClusterResult` (issue #129, in `result_types.py`) is a frozen, JSON-serializable
view of a clustering run, with `KMeansResult` / `GMMResult` subclasses built by the
`from_kmeans_dict` / `from_gmm_dict` adapters. The adapters follow a strict
convention: **the producer function computes and returns a plain dict; the adapter
only maps that dict into the typed view (no computation, no mutation).**

Hierarchical clustering breaks this pattern because its producer,
`perform_hierarchical_clustering`, returns a *dendrogram* (linkage matrix,
cophenetic correlation) — not labels. Labels only appear after a separate
`cut_dendrogram` step, and choosing `k` needs `calculate_optimal_clusters_hierarchical`.
The issue offers three API shapes; this change picks one.

## Goals / Non-Goals

- **Goals:** one public call returns a `HierarchicalResult` with populated
  `cluster_labels`, `cluster_sizes`, `n_clusters`, and three quality scores;
  `ClusterResult(random_state=None)` is valid; identical input → identical labels.
- **Non-Goals:** the `GMMResult` constructor gap (separate issue); changing
  `perform_hierarchical_clustering`'s return shape; adding `cluster_centers` to the
  hierarchical view (agglomerative clustering has no centroids).

## Decisions

- **Decision: new producer `hierarchical_cluster_labels()` + `from_hierarchical_dict()`
  adapter + `HierarchicalResult` subclass.** This mirrors the existing
  producer → dict → adapter split exactly and keeps
  `perform_hierarchical_clustering` (dendrogram-only) unchanged, so the existing
  `cut_dendrogram` / `calculate_optimal_clusters_hierarchical` consumers keep their
  contract.
  - *Alternative (a): modify `perform_hierarchical_clustering(..., n_clusters=None)`
    to conditionally return labels.* Rejected — overloads one function with two
    return shapes and risks every downstream caller that expects the dendrogram dict.
  - *Alternative (b): put the labeling logic inside
    `from_hierarchical_dict(dendrogram_dict, n_clusters=...)`.* Rejected — the adapter
    would run clustering (cut/optimize) rather than map a dict, breaking the
    "adapter maps, does not compute" convention shared by every other result type.

- **Decision: `HierarchicalResult` fields.** Common base fields
  (`cluster_labels`, `cluster_sizes`, `n_clusters`, the three quality scores,
  `feature_names`, `random_state`) plus hierarchical provenance: `linkage_method`,
  `distance_metric`, `cophenetic_correlation`, `cut_height`. No `cluster_centers`.
  `algorithm = ALGORITHM_HIERARCHICAL = "hierarchical"`.

- **Decision: `random_state: Optional[int] = None` on the base `ClusterResult`.**
  `from_hierarchical_dict` stamps `None`; `from_kmeans_dict` / `from_gmm_dict` keep
  stamping the `int` seed. Giving the base field a default is safe because all
  existing subclass fields already have defaults — but the **new** `HierarchicalResult`
  fields (`linkage_method`, `distance_metric`, `cophenetic_correlation`, `cut_height`)
  MUST also be given defaults, or the dataclass raises `TypeError` at import and the
  whole test suite fails to collect.

- **Decision: export placement.** Root-export `HierarchicalResult` (class) and
  `hierarchical_cluster_labels` (function). Keep `ALGORITHM_HIERARCHICAL` in
  `result_types.__all__` only, matching `ALGORITHM_KMEANS`/`ALGORITHM_GMM`. The root
  public-API audit (`scripts/check_public_api_docs.py::run_audit`) flags any root
  `__all__` entry that is not a class or callable, so a bare `str` constant at the
  root would break `tests/test_public_api_docs.py`.

- **Decision: `optimization_method` accepted values.** When `n_clusters is None`,
  `optimization_method` is passed straight to `calculate_optimal_clusters_hierarchical`,
  which accepts only `"silhouette"` / `"calinski"` / `"davies_bouldin"`. These differ
  from the metric-key names (`silhouette_score`, `calinski_harabasz_score`,
  `davies_bouldin_score`), a consumer footgun — the accepted set is pinned in the spec.

- **Decision: `hierarchical_cluster_labels()` parameter surface.** Mirror the
  underlying functions: `data`, `n_clusters=None`, `method="ward"`,
  `metric="euclidean"`, `standardize=True`, `optimization_method="silhouette"`,
  `max_clusters=10`. When `n_clusters is None`, `k` comes from
  `calculate_optimal_clusters_hierarchical(..., method=optimization_method)`.

- **Decision: uniform `ValueError` for argument errors (up-front validation).** Every
  invalid argument surfaces as `ValueError`, so a consumer (bloom-mcp) catches one
  exception type. The producer validates *all* arguments that would otherwise leak a
  `RuntimeError` *before* calling the composed functions: `method` (against the scipy
  linkage set), `metric` (against the scipy `pdist` set — a bogus name otherwise reaches
  `linkage`/`pdist` inside `perform_hierarchical_clustering`'s try/except and re-wraps as
  `RuntimeError`), `optimization_method` (validated unconditionally, even when unused),
  and `n_clusters` (integer, and in `[1, n_samples - 1]`). The three accepted-value sets
  live in module-level constants (`_HIERARCHICAL_LINKAGE_METHODS` /
  `_HIERARCHICAL_DISTANCE_METRICS` / `_HIERARCHICAL_OPTIMIZATION_METHODS`) so the
  producer's check cannot drift from what the composed calls accept. Validating up front
  — rather than catching-and-rewrapping the composed functions' `RuntimeError` — keeps
  error messages clear and does **not** mask a *genuine* runtime failure (e.g. non-finite
  values that survive NaN-dropping still raise `RuntimeError`).
  - *Alternative: propagate the composed functions' exceptions faithfully (mixed
    `ValueError` / `RuntimeError`).* Rejected — semantically identical "bad argument"
    errors surfaced as two types depending only on where validation happened, forcing a
    consumer to catch both.

- **Decision: the producer dict carries `data_indices`; the typed view does not.**
  `hierarchical_cluster_labels` returns `cut_dendrogram`'s `data_indices` (the original
  row labels aligned to `cluster_labels`) so labels map back to source rows after
  NaN-row dropping — scientifically load-bearing for root phenotyping (label → plant /
  genotype ID). This matches the sibling producers (`perform_kmeans_clustering` /
  `perform_gmm_clustering` also return `data_indices`). `HierarchicalResult` does **not**
  add a `data_indices` field: no `ClusterResult` subclass carries it, so keeping it a
  producer-dict-only concern preserves the typed family's shape.

- **Decision: the euclidean-only linkage constraint covers `ward`, `centroid`, and
  `median`.** scipy's `linkage()` requires euclidean distance for all three methods,
  not only `ward`. Fixed at the `perform_hierarchical_clustering` level (not
  duplicated as a separate check in `hierarchical_cluster_labels`) so every caller —
  the new producer, `detect_outliers_hierarchical`, `create_dendrogram` — gets the
  same clear `ValueError` instead of a wrapped `RuntimeError`. Single source of truth
  avoids the exact drift risk already flagged for the accepted-method/metric sets.

- **Decision: `KMeansResult`/`GMMResult` reject `random_state=None` via
  `__post_init__`; only `HierarchicalResult` may omit it.** Widening the base
  `random_state` to `Optional[int] = None` (to let `HierarchicalResult` omit it)
  otherwise silently removes the pre-widening guarantee that a `KMeansResult`/
  `GMMResult` always carries a real seed (previously enforced for free by the
  required-field `TypeError` on missing construction args). Both algorithms are
  always seeded by their producers, so `random_state=None` on those two subclasses is
  a real gap, not a valid deterministic run — restore the guardrail explicitly.

- **Decision: `calculate_optimal_clusters_hierarchical` additively returns the
  winning `cut_result`.** The auto-`k` scan already computes `cut_dendrogram` for
  every candidate `k`, including the optimal one; `hierarchical_cluster_labels`
  previously discarded that and re-cut for the same `k`, an avoidable ~1/(max_clusters
  − 1) extra O(n^2) quality-metric pass. Threading the winning `cut_result` out as an
  additive dict key (no existing caller asserts an exact key set) lets the producer
  reuse it — and also benefits the existing `detect_outliers_hierarchical` caller,
  which has the identical redundant-cut pattern (not changed in this PR — out of
  scope, flagged as a follow-up opportunity).

## Risks / Trade-offs

- **`random_state` widening across the JSON boundary** → a strict consumer that
  assumed `random_state` is always an `int` will now sometimes see `null`.
  Mitigation: matches the precedent already set by `PCAResult.random_state`; the
  bloom-mcp schema is updated downstream (bloom#422). Not breaking for *producers* —
  KMeans/GMM still emit an int.
- **`cut_dendrogram` degenerate cases** (e.g. `n_clusters=1`) produce zero-valued
  quality metrics rather than raising. Mitigation: the adapter carries the numbers as
  produced; `to_json`'s `allow_nan=False` still guards non-finite values.
- **`davies_bouldin_score == 0.0` for a single cluster is misleading** — lower is better
  for DB, so `0.0` reads as the *best* possible score for an undefined metric.
  Mitigation: documented in the producer docstring (do not rank a degenerate
  `n_clusters=1` run against real runs by DB). Reaffirmed on review (round 2, item #7):
  the auto-`k` path never selects `k=1`, so the footgun only reaches a consumer that
  manually constructs/ranks `n_clusters=1` results — not changed to `NaN` (unlike
  `cophenetic_correlation` below) because the design chose `0.0` for `to_json`
  serializability under `allow_nan=False`, and this metric (unlike a genuinely
  degenerate scipy computation) is a deliberate sentinel, not incidental fallout.
- **`cophenetic_correlation` can be `NaN`** for degenerate inputs (a 2-row input, or
  identical points) and is accepted silently by the producer and adapter — it only
  raises at `ClusterResult.to_json()` (`allow_nan=False`), not at production. Mitigation:
  documented in the producer docstring so a caller using the dict/dataclass directly
  knows to check.

## Migration Plan

Additive; no existing caller changes required — the new function, subclass, adapter,
and constant are all opt-in. This PR carries only a `docs/CHANGELOG.md` `[Unreleased]`
entry; the `0.1.0a5` version is cut in a **separate `chore` release PR** via
`uv version` (which updates `pyproject.toml` and `uv.lock` together), per the `#176`
precedent — a hand-edit of `pyproject.toml` alone desyncs `uv.lock` and fails the
`type-check` CI job's `uv sync --frozen`.

## Open Questions

- Confirm the default optimization metric (`silhouette`) and the exact keyword
  surface of `hierarchical_cluster_labels()` are what bloom-mcp's contract wrapper
  expects to pass through.
