# Add PCAResult Serializable Dataclass Return Type (issue #127)

## Why

`perform_pca_analysis()` (`pca.py`) returns a `Dict` that mixes four
serializability classes in one bag: clean scalars/lists
(`n_components_selected`, `feature_names`), `np.ndarray` fields
(`loadings`, `transformed_data`, `explained_variance_ratio`, `eigenvalues`),
`pd.DataFrame` fields (`feature_contributions`, `feature_metrics_df`), and
non-serializable sklearn objects (`pca`, `scaler`). This makes the function
impossible to expose over any JSON boundary — bloom-mcp (Metcalf 2026), an API,
or a cached artifact — without bespoke conversion.

The package already has JSON-safe stdlib `@dataclass` result types in
`summary/cross_platform_summary.py` (`TraitReductionStats`, `PowerStats`,
`CorrelationStats`, `TopCorrelation`, …) and the `EnrichmentResult` pattern
(flat, JSON-native, `to_dict()`). This change extends that pattern to PCA as the
**detailed exemplar** for the serializable-result-types epic (#130), with #128
(`HeritabilityResult`) and #129 (`ClusterResult`) following the identical
convention.

## What Changes

- **New module `src/sleap_roots_analyze/result_types.py`** — the shared home for
  the epic's serializable result types (PCA now; heritability/cluster later). It
  imports nothing from `pca.py`, keeping dependencies one-way.
- **`PCAResult` and `FeatureContribution` stdlib dataclasses** exposing **only
  serializable science** — sklearn `pca`/`scaler` objects stay out of the clean
  view (still available via the legacy dict for in-process callers).
  `frozen=True` and a `to_dict()` convenience (`asdict(self)`), matching the
  `CrossPlatformPCResult`/`EnrichmentResult` precedent. Every scalar is cast to
  a native Python `int`/`float`/`bool` (numpy scalars like `np.int64` are **not**
  reliably JSON-serializable), so `json.dumps(dataclasses.asdict(result))`
  round-trips with no custom serializer.
- **`PCAResult.from_pca_dict(d, *, random_state=None, explained_variance_threshold=None)`
  adapter** mapping the existing `perform_pca_analysis` return dict into the
  dataclass:
  - `scores` is built from `d["transformed_data"]` (the dict has no `scores`
    key); arrays are copied via `.tolist()`.
  - `feature_contributions` is built from the DataFrame **index** (the source is
    sorted by `total_contribution` descending, indexed by feature name), so the
    name↔value correspondence and ordering are preserved — not a positional zip
    against `feature_names`.
  - `standardized` is `d["scaler"] is not None` (equivalent to the requested
    `standardize` flag given the function's guarantees).
  - `cumulative_variance_ratio` is carried from `d["cumulative_variance_ratio"]`
    (per-component cumsum), not discarded.
  - `random_state` / `explained_variance_threshold` are stamped from the adapter
    arguments (they are not present in the dict) for reproducibility provenance.
  - The per-PC `PC{k}_variance_contrib` columns and the redundant
    `feature_metrics_df` are intentionally omitted from the clean view.
- **`@property cumulative_variance`** returning the summed retained
  explained-variance ratio as a native `float`.
- **Public exports.** `PCAResult` and `FeatureContribution` added to the package
  `__all__` (in `__init__.py`) with full type hints + Google-style docstrings
  whose `Attributes:` block documents every field — this exemplar sets the
  convention #128/#129 copy.
- **Additive / non-breaking.** `perform_pca_analysis` keeps returning its dict
  unchanged (MINOR bump); `from_pca_dict` does not mutate its input. No existing
  key, shape, or caller (`result["loadings"]`, the wheat EDPIE paper, the
  pipeline) changes.
- **Tests.** A `json.dumps(asdict(...))` round-trip that asserts **native Python
  types after `json.loads`** (feeds the reproducibility CI gate), plus adapter
  tests over real `perform_pca_analysis` runs: array shapes equal `n_components`,
  the `standardized` flag for both `standardize` branches, determinism for a
  fixed `random_state` (epic #118), `n_components=1` shape preservation,
  `feature_contributions` ordering/name fidelity, no sklearn object in the view,
  and a dict-unchanged/non-mutating guard.

## Out of Scope (deferred to the epic)

- The epic-wide `docs/result-types.md` pattern doc (#130 acceptance) is deferred
  to the epic-closing child so it can describe the convention across all three
  result types rather than PCA alone; #127 seeds the runnable exemplar in code.
- Golden numeric validation of `PCAResult` fields against the wheat EDPIE
  reference values is deferred to the epic's verification milestone (#120/#130);
  #127 covers structure, types, serialization, provenance, and determinism only.

## Impact

- Affected specs: **new** `serializable-result-types` capability (the durable
  home for the epic's pattern; #128/#129 will add requirements here).
- Affected code: **new** `src/sleap_roots_analyze/result_types.py` (dataclasses +
  adapter); `src/sleap_roots_analyze/__init__.py` (`__all__` exports);
  **new** `tests/test_pca_result.py`. The two new `__all__` entries must satisfy
  the existing `tests/test_public_api_docs.py` introspection contract (type
  hints + Google docstrings); there is no hardcoded count to update.
- No breaking changes; purely additive public API (MINOR version bump).
