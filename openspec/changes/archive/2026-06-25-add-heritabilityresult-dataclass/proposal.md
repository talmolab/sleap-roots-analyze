# Add HeritabilityResult Serializable Dataclass Return Type (issue #128)

## Why

`calculate_heritability_estimates()` (`statistics.py`, exposed publicly via
#116) returns a `Dict` keyed by trait name, each value a per-trait dict of
broad-sense heritability plus variance components and a special
`__calculation_metadata__` entry. Like the PCA dict, this nested, metadata-mixed
shape needs a clean serializable view to cross a JSON boundary (bloom-mcp,
caching, an API) and to anchor the wheat EDPIE golden tests.

This is the Phase-1 anchor of the serializable-result-types epic (#130). It
follows the exact convention established by `PCAResult` (#127): a frozen stdlib
`@dataclass` holding only JSON-serializable science, a `from_*_dict()` adapter,
`frozen=True` + `to_dict()`, and native-Python scalar casts.

## What Changes

- **`HeritabilityResult` and `TraitHeritability` frozen dataclasses** added to
  the shared `result_types.py` module (created in #127). They expose only
  serializable science — per-trait H², variance components, and counts — never
  sklearn/statsmodels objects.
- **`HeritabilityResult.from_heritability_dict(d, threshold)` adapter** mapping
  the `calculate_heritability_estimates` return dict (the
  `remove_low_h2=False` form, or the first element of the
  `remove_low_h2=True` tuple):
  - `method` is read from `d["__calculation_metadata__"]["method_used_for_all_traits"]`.
  - The `__calculation_metadata__` key is skipped; trait entries carrying an
    `"error"` (or lacking a `"heritability"` value) are collected into
    `failed_traits` rather than `per_trait`.
  - Each successful trait becomes a `TraitHeritability` with `passed_threshold =
    h2 >= threshold`; all scalars cast to native `float`/`int`.
- **`@property mean_h2`** (mean H² over successful traits) and
  **`@property n_above_threshold`** (count with `passed_threshold`).
- **Public exports.** `HeritabilityResult` and `TraitHeritability` added to the
  package `__all__` with full type hints + Google-style docstrings (every field
  documented in an `Attributes:` block), satisfying the
  `test_public_api_docs.py` introspection contract.
- **Additive / non-breaking.** `calculate_heritability_estimates` keeps its dict
  / tuple returns unchanged (MINOR bump); the adapter does not mutate its input.
- **Tests.** A native-type JSON round-trip (reproducibility CI gate) plus
  adapter tests over a real `calculate_heritability_estimates` run on the
  `heritability_data_known_h2` fixture (known H² ≈ 0.8 / 0.5 / 0.09): threshold
  classification, `mean_h2` / `n_above_threshold`, `failed_traits` capture,
  determinism, exports, and a dict-unchanged / non-mutating guard.

## Out of Scope (deferred to the epic)

- The wheat EDPIE Turface-19 golden numbers (mean H²=0.77; 8 traits ≥0.60) are
  deferred to the epic's verification milestone (#120/#130), consistent with
  #127. This change anchors numeric correctness on the synthetic
  `heritability_data_known_h2` fixture (known variance components) instead.
- The shared `docs/result-types.md` pattern doc remains an epic-close
  deliverable.

## Impact

- Affected specs: the `serializable-result-types` capability gains heritability
  requirements (same capability introduced by #127).
- Affected code: `src/sleap_roots_analyze/result_types.py` (new dataclasses +
  adapter); `src/sleap_roots_analyze/__init__.py` (`__all__` exports);
  **new** `tests/test_heritability_result.py`.
- No breaking changes; purely additive public API (MINOR version bump).
- Stacked on #127 (`pcaresult-dataclass-127`), which introduces
  `result_types.py`.
