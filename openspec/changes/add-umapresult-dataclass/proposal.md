## Why

The serializable-result-types epic #130 added JSON-serializable dataclasses for the
other analytical functions — `PCAResult` (#127), `HeritabilityResult` (#128),
`ClusterResult`/`KMeansResult`/`GMMResult` (#129) — but **UMAP was skipped** (#180).
`perform_umap_analysis` still returns a plain dict
`{embedding, reducer, scaler, n_neighbors, min_dist}` where `reducer` (a fitted
`umap.UMAP`) and `scaler` (a fitted `StandardScaler`) are **non-serializable**, and the
dict does not carry `feature_names`, `n_components`, or `random_state`. A consumer that
needs a serializable, self-describing result therefore has nothing to build one from —
specifically the bloom-mcp `umap_analysis` granular tool
(Salk-Harnessing-Plants-Initiative/bloom#425) which, like its `pca_analysis` sibling
consuming `PCAResult`, must return a typed result and persist it rather than re-shape a
raw dict downstream (thin-delegation). Follow-up to epic #130; sibling to #127
(`PCAResult`).

## What Changes

- Add a `UMAPResult` frozen stdlib `@dataclass` (sibling to `PCAResult`) in
  `result_types.py` holding only the JSON-serializable science of a UMAP run —
  `embedding`, `n_neighbors`, `min_dist`, `n_components`, `feature_names`, `n_samples`,
  `standardized`, and `random_state` — and **excluding** the fitted `reducer`/`scaler`
  (matching `PCAResult`, which does not serialize the fitted `PCA`/`StandardScaler`). The
  class provides `to_dict()` and `to_json()` (the finite-floats `allow_nan=False`
  contract). `n_samples` and `n_components` are stored fields (not properties) because the
  bloom#425 consumer and the existing pipeline payload
  ([`umap_analysis.py`](../../../src/sleap_roots_analyze/pipeline/steps/umap_analysis.py)
  serializes `n_samples`) contract on them being present in the serialized form — a
  deliberate, documented exception to the epic's "derivations are properties" rule
  (see design.md).
- Add the classmethod adapter `UMAPResult.from_umap_dict(d, *, random_state=None)` —
  reads `embedding`, `n_neighbors`, `min_dist`, and `feature_names` from the dict;
  derives `n_components`/`n_samples` from the embedding shape and `standardized` from
  `d.get("scaler") is not None`; and resolves `random_state` from the explicit argument,
  falling back to `d.get("random_state")` (see below). Non-mutating, mirroring
  `PCAResult.from_pca_dict(d, *, random_state=None)`.
- **Additively** enrich `perform_umap_analysis` to also return `feature_names` (the
  `feature_cols` it was given) **and** `random_state` (the seed it used) in its dict.
  `feature_names` is the one field the adapter cannot otherwise recover; echoing
  `random_state` lets the adapter record the *actual* seed rather than one stamped on
  trust — material because UMAP's seed is load-bearing (it governs the stochastic
  embedding), unlike PCA's seed-insensitive full-SVD path. This is a non-breaking key
  addition: `embedding`, `reducer`, `scaler`, `n_neighbors`, and `min_dist` are all
  preserved, so every current caller (the pipeline `UMAPAnalysisStep`,
  `interactive_visualization`, the reproducibility sweep, the golden-embedding recompute)
  keeps working unchanged.
- Make `UMAPResult.random_state` **`Optional[int]` (default `None`)**, matching the
  existing `PCAResult.random_state` (`Optional[int]` today) — one consistent seed typing
  with the PCA sibling (and, pending #179, the clustering result types). A real UMAP run
  always stamps an `int`; `None` only appears if a caller builds a `UMAPResult` from a
  dict with no seed.
- Root-export **`UMAPResult`** from the package namespace and `__all__` (and list it in
  `result_types.__all__`); update the `result_types.py` module docstring enumeration to
  include it. No new module-level `str` constant is introduced.
- Document the new surface in `docs/result-types.md` (UMAP row in the types table) and add
  a `docs/CHANGELOG.md` `[Unreleased]` entry. **No version bump here** — `0.1.0a5` is cut
  in a separate `chore` release PR via `uv version` (per #176), bundling this with #179.

## Impact

- Affected specs:
  - `serializable-result-types` — four new ADDED requirements (UMAP result type, adapter,
    public export, non-breaking return shape).
  - `umap-analysis` — **unaffected**: its `UMAP Results Structure` requirement uses
    non-exclusive "SHALL contain", so the additive `feature_names`/`random_state` keys are
    compatible; no delta needed.
- Affected code:
  - `src/sleap_roots_analyze/umap.py` — additive `feature_names` + `random_state` keys in
    the return dict + `Returns:` docstring
  - `src/sleap_roots_analyze/result_types.py` — `UMAPResult`, `from_umap_dict`, added to
    `result_types.__all__`, module-docstring enumeration
  - `src/sleap_roots_analyze/__init__.py` — root export + `__all__` (`UMAPResult`)
  - `docs/result-types.md` (types table row), `docs/CHANGELOG.md` (`[Unreleased]`)
  - `tests/test_umap_result.py` — new module (mirrors `tests/test_cluster_result.py`)
- Explicitly out of scope: changing `perform_umap_analysis`'s return **type** (the typed
  view stays opt-in via the adapter); a `standardize=False` code path for UMAP (today it
  always standardizes); the `pyproject.toml` / `uv.lock` version bump (separate `0.1.0a5`
  release PR); `docs/API.md` / `docs/public_api_audit_2026.md` (result types are
  documented in `docs/result-types.md`).
- Downstream: unblocks bloom#425.
