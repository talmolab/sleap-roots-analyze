# Serializable result types

Most analytical functions in `sleap_roots_analyze` return a `dict` (or tuple) that
mixes JSON-hostile members — `numpy.ndarray`, sklearn objects (`PCA`,
`StandardScaler`, `UMAP`), `pandas.DataFrame`. Those results cannot cross a JSON
boundary without bespoke conversion, which blocks exposing the functions as MCP tools
(bloom-mcp), caching results as artifacts, and schema-checkable outputs (FAIR:
interoperable + reusable).

The **result types** in [`result_types.py`](../src/sleap_roots_analyze/result_types.py)
solve this: flat stdlib `@dataclass` views that hold **only JSON-serializable science**.
This is the same house pattern already used in
[`summary/cross_platform_summary.py`](../src/sleap_roots_analyze/summary/cross_platform_summary.py)
(`TraitReductionStats`, `PowerStats`, …), extended to the analytical functions under the
serializable-result-types epic (#130).

## The pattern

A result type is a `@dataclass(frozen=True)` that satisfies these rules:

1. **stdlib `dataclasses`, not Pydantic.** The package stays dependency-light and
   MCP-agnostic; bloom-mcp wraps these plain dicts in its own Pydantic models at its
   boundary.
2. **Every field is JSON-native** — Python scalars, `str`, `bool`, and (nested) `list`s
   of those. Never `numpy` arrays, `pandas` frames, or sklearn objects. The conversion
   to native types happens once, in the adapter (`np.asarray(...).tolist()`,
   `float(...)`, `int(...)`).
3. **Science only.** Fitted sklearn objects (`PCA`, `StandardScaler`) stay out of the
   clean view — they remain available via the legacy dict for in-process callers.
4. **Derivations are `@property`, not stored fields**, so the serialized state has no
   redundant/derivable members (e.g. `PCAResult.cumulative_variance`,
   `HeritabilityResult.mean_h2`).
5. **A `from_*_dict()` classmethod adapter** builds the dataclass from the legacy dict
   and must **not mutate** the input dict.

Because every field is JSON-native, the invariant that anchors the reproducibility CI
gate holds with no custom serializer:

```python
import json, dataclasses
from sleap_roots_analyze import PCAResult

result = PCAResult.from_pca_dict(perform_pca_analysis(df), random_state=42)
restored = json.loads(json.dumps(dataclasses.asdict(result)))   # round-trips, no encoder
```

Each type also provides convenience helpers:

- **`to_dict()`** — the plain `dict` view (`dataclasses.asdict(self)`), for callers that
  don't want to import `dataclasses`.
- **`to_json(**kwargs)`** — a strict-JSON string with `allow_nan=False` by default, so a
  non-finite value (a degenerate `bic`/`aic`/`h2`, an all-zero-loadings PCA `NaN`
  fractional contribution) raises a `ValueError` at serialization rather than emitting the
  non-standard `NaN`/`Infinity` tokens a strict consumer (bloom-mcp) rejects. This enforces
  the **finite-floats contract** of the JSON boundary; extra kwargs forward to `json.dumps`.

```python
result.to_json(indent=2)   # strict JSON; raises on NaN/Inf instead of emitting invalid JSON
```

For clustering, the `ClusterResult.algorithm` discriminator a consumer branches on is
exported as the constants `ALGORITHM_KMEANS` / `ALGORITHM_GMM` / `ALGORITHM_HIERARCHICAL`
(in `result_types`) — use those rather than re-spelling the string literals.

## The types

| Type | Built from | Adapter |
| --- | --- | --- |
| `PCAResult` (+ `FeatureContribution`) | `perform_pca_analysis` dict | `PCAResult.from_pca_dict(d, *, random_state=None, explained_variance_threshold=None)` |
| `HeritabilityResult` (+ `TraitHeritability`) | `calculate_heritability_estimates` dict | `HeritabilityResult.from_heritability_dict(d, threshold)` |
| `KMeansResult` / `GMMResult` (subclasses of `ClusterResult`) | `perform_kmeans_clustering` / `perform_gmm_clustering` dict | `ClusterResult.from_kmeans_dict(d, *, random_state)` / `from_gmm_dict(...)` |
| `HierarchicalResult` (subclass of `ClusterResult`) | `hierarchical_cluster_labels` dict | `ClusterResult.from_hierarchical_dict(d)` (no `random_state` — hierarchical is deterministic) |

All are exported from the top-level `sleap_roots_analyze` namespace and `__all__`.

**`random_state` guardrail.** The base `ClusterResult.random_state` defaults to `None`
so `HierarchicalResult` can omit it, but `KMeansResult`/`GMMResult` reject
`random_state=None` at construction (`TypeError`) — both algorithms are always seeded
by their producers, so a `None` there is a real gap, not a valid deterministic run.

**Hierarchical "deterministic" caveat.** No RNG is used in the composed call path, so
identical input yields identical labels **within one process**. scipy's `linkage`/
`fcluster` tie-breaking is BLAS/platform-sensitive, so this is not a promise of
byte-for-byte reproducibility across machines — no golden artifact is committed for it.

**Two known footguns, not specific to this epic:**
- `feature_names` (on `KMeansResult`/`GMMResult`/`HierarchicalResult`) is captured
  before `standardize_data` drops non-numeric/zero-variance columns, so it can list
  more names than were actually clustered.
- `data_indices` (the row-label mapping back to source rows after NaN-dropping) is
  available on every clustering **producer dict** but is **not** carried onto any
  `ClusterResult` subclass — use the producer dict directly if you need that mapping.

## Backwards compatibility

**Additive only.** Existing callers (`result["loadings"]`, the wheat EDPIE paper, the
pipeline) keep working:

1. Add the dataclass + `from_*_dict()` adapter; the analytical functions keep returning
   the legacy dict → MINOR bump. The dataclass is an *additional* typed view, opt-in via
   the adapter.
2. Later (separate, deprecation-windowed) the functions may return the dataclass with a
   `__getitem__` shim → MAJOR bump.

## Adding a new result type (test-first)

Follow the epic's pairing loop:

1. Write the `json.dumps(dataclasses.asdict(...))` **round-trip test first** (plus a
   golden / determinism test where the function feeds the #120 reproduction suite), and
   watch it fail.
2. Add the `@dataclass(frozen=True)` to `result_types.py` with JSON-native fields,
   `@property` derivations, a non-mutating `from_*_dict()` adapter, and Google-style
   docstrings.
3. Export it from `result_types.__all__` and the top-level `__init__`.
4. Add a unit test module `tests/test_<name>_result.py` covering the adapter and the
   round-trip; where the function is re-run in the #120 golden suite
   (`tests/test_pipeline_reproduction.py`), assert the golden numbers against the typed
   view too.
