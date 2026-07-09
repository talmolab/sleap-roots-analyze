## Why

The `bloom-mcp` Tier 5 contract-wrapped clustering tool delegates to the public
`perform_*_clustering` functions and constructs a `ClusterResult`. KMeans and GMM
work end-to-end, but **hierarchical clustering has no public call that yields
cluster labels**: `perform_hierarchical_clustering` returns only a dendrogram
(linkage matrix + cophenetic correlation), so a `ClusterResult` — with
`cluster_labels` / `n_clusters` / `cluster_sizes` / silhouette — cannot be built
from it. Separately, `ClusterResult.random_state` is a required `int`, but
hierarchical (agglomerative) clustering is deterministic with no seed, so there is
no valid value to stamp. Follow-up to #129 (which added the result types for
KMeans/GMM only); unblocks downstream consumer bloom#422.

## What Changes

- Add a public producer `hierarchical_cluster_labels()` in `clustering.py` that
  **composes existing pieces** — `perform_hierarchical_clustering` →
  `calculate_optimal_clusters_hierarchical` (when `n_clusters` is omitted) →
  `cut_dendrogram` — into a single labeled dict with `cluster_labels`,
  `n_clusters`, `cluster_sizes`, the three quality metrics, and hierarchical
  provenance (`linkage_method`, `distance_metric`, `cophenetic_correlation`,
  `cut_height`).
- Add `HierarchicalResult(ClusterResult)` plus the adapter
  `ClusterResult.from_hierarchical_dict(d)` (no `random_state` argument —
  hierarchical is deterministic) returning a fully populated typed view; add the
  `ALGORITHM_HIERARCHICAL = "hierarchical"` discriminator constant.
- Make `ClusterResult.random_state` **`Optional[int]` (default `None`)**, aligning
  with the existing `PCAResult.random_state`. This is source-compatible for
  *producers* (KMeans/GMM adapters keep stamping the `int` seed), but it is a
  **BREAKING** change for *readers*: a consumer that typed `random_state` as always-`int`
  will now sometimes observe `null` and must handle it (the downstream bloom-mcp schema
  is updated in bloom#422). See design.md Risks.
- Root-export **`HierarchicalResult`** and **`hierarchical_cluster_labels`** from the
  package namespace and `__all__`. Keep **`ALGORITHM_HIERARCHICAL` in
  `result_types.__all__` only** (alongside `ALGORITHM_KMEANS` / `ALGORITHM_GMM`),
  **not** the package root — the public-API docstring audit
  (`scripts/check_public_api_docs.py`) requires every root `__all__` entry to be a
  class or callable, so a bare `str` constant at the root would fail it.
- Document the new surface in `docs/result-types.md` (types table + the
  discriminator-constants note) and add a `docs/CHANGELOG.md` `[Unreleased]` entry.
  **No `pyproject.toml` version bump in this PR** — per repo convention (`#176`),
  `0.1.0a5` is cut in a separate `chore` release PR via `uv version` (which keeps
  `pyproject.toml` and `uv.lock` in sync; a hand-edit desyncs `uv.lock` and fails
  the `type-check` CI job's `uv sync --frozen`).

## Impact

- Affected specs: `serializable-result-types` (clustering requirements modified + one
  new producer requirement).
- Affected code:
  - `src/sleap_roots_analyze/clustering.py` — new `hierarchical_cluster_labels()`
  - `src/sleap_roots_analyze/result_types.py` — `HierarchicalResult`,
    `from_hierarchical_dict`, `random_state: Optional[int]`, `ALGORITHM_HIERARCHICAL`
    (added to `result_types.__all__`), and the base `ClusterResult` class docstring
    (the `random_state` line **plus** the intro, the `algorithm` discriminator
    enumeration, and the `Build via from_kmeans_dict/from_gmm_dict` line — all stale
    once a third subclass exists; the public-API audit does not catch this)
  - `src/sleap_roots_analyze/__init__.py` — root exports + `__all__`
    (`HierarchicalResult`, `hierarchical_cluster_labels`)
  - `tests/test_cluster_result.py` — `HierarchicalResult` round-trip, `Optional`
    `random_state`, `from_hierarchical_dict` adapter, `to_json` finite guard
  - `tests/test_clustering.py` — `hierarchical_cluster_labels` producer, auto/explicit
    `k`, determinism, degenerate + error paths
  - `tests/test_public_api.py`, `tests/test_public_api_docs.py` — new root exports pass
    the import + docstring audit
  - `docs/result-types.md`, `docs/CHANGELOG.md` (`[Unreleased]`)
- Explicitly out of scope: `pyproject.toml` / `uv.lock` version bump (separate release
  PR); `docs/API.md` and `docs/public_api_audit_2026.md` (clustering + result types
  are documented in `docs/result-types.md`; the audit doc is a historical #117
  snapshot); the sibling `GMMResult` constructor gap noted in the issue.
- Downstream: unblocks bloom#422.
