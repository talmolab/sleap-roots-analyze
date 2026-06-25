# Tasks — Add PCAResult Serializable Dataclass (issue #127)

> Commit strategy: a single `feat: add PCAResult serializable dataclass (#127)`
> commit (the repo does not commit red tests separately; CI must be green at
> HEAD). The export edit and the introspection-guard bump MUST land in the same
> commit as the dataclass definitions. OpenSpec archive happens post-merge as a
> separate `chore:` commit.

## 1. Failing tests first (red) — `tests/test_pca_result.py`
- [x] 1.1 `test_pcaresult_json_roundtrip_native_types`: run `perform_pca_analysis`
      on a small synthetic frame, build `PCAResult.from_pca_dict(d)`, assert
      `json.dumps(dataclasses.asdict(result))` does not raise, then `json.loads`
      it and assert `n_components` is `int`, `standardized` is `bool`, and every
      element of `explained_variance_ratio`/`eigenvalues`/
      `cumulative_variance_ratio`/`loadings`/`scores` is native `float`. Assert
      `"pca"`, `"scaler"`, `"feature_metrics_df"` are absent from `asdict`.
- [x] 1.2 `test_adapter_maps_core_fields`: assert `n_components ==
      int(d["n_components_selected"])`; `explained_variance_ratio`/`eigenvalues`/
      `cumulative_variance_ratio` are `list[float]` of length `n_components`;
      `scores` is sourced from `d["transformed_data"]` with shape
      `(n_samples, n_components)`; `loadings` shape `(n_features, n_components)`.
      Use a fixture where retained `n_components < n_features` so the shape
      checks are non-trivial.
- [x] 1.3 `test_feature_contributions_order_and_names`: assert the list is
      ordered by `total_contribution` descending and each
      `FeatureContribution.feature` is the DataFrame index label with values
      equal to `feature_contributions.loc[feature]` (catches a positional-zip
      mis-alignment against `feature_names`).
- [x] 1.4 `test_standardized_flag` parametrized over `standardize in (True,
      False)`: `result.standardized is standardize`, and `json.dumps(asdict)`
      succeeds in both branches.
- [x] 1.5 `test_pcaresult_deterministic`: same `random_state` → identical
      `asdict(r1) == asdict(r2)` and identical `json.dumps` output (epic #118).
- [x] 1.6 `test_n_components_one_shapes`: a run retaining a single component
      keeps `loadings` `(n_features, 1)` and `scores` `(n_samples, 1)` nested
      (no flatten/squeeze).
- [x] 1.7 `test_cumulative_variance_property`: `pytest.approx(sum(evr))`, value
      in `(0, 1]`, native `float`.
- [x] 1.8 `test_provenance_args`: `from_pca_dict(d, random_state=42,
      explained_variance_threshold=0.95)` stamps both fields; omitting them
      yields `None`.
- [x] 1.9 `test_exports_and_all`: `from sleap_roots_analyze import PCAResult,
      FeatureContribution` succeeds; both in `__all__`, no dupes.
- [x] 1.10 `test_perform_pca_analysis_dict_unchanged`: returned dict still
      contains the canonical keys; `from_pca_dict(d)` does not mutate `d`.

## 2. Implement to green
- [x] 2.1 Create `src/sleap_roots_analyze/result_types.py` with `from __future__
      import annotations`, module docstring, and the `FeatureContribution` +
      frozen `PCAResult` dataclasses. Google-style class docstrings with an
      `Attributes:` block documenting **every** field (type, meaning, and
      shape/units, e.g. `loadings: (n_features, n_components) nested list`).
      Required fields first, then `random_state`/`explained_variance_threshold`
      (`= None`), then `feature_contributions = field(default_factory=list)`.
- [x] 2.2 Implement `PCAResult.from_pca_dict()`: `scores` from
      `transformed_data`; `cumulative_variance_ratio` from the dict; explicit
      `int()`/`float()` scalar casts; `feature_contributions` built from the
      DataFrame index (preserving sort order); `standardized` from `scaler is not
      None`; provenance args stamped; no mutation of `d`.
- [x] 2.3 Implement `@property cumulative_variance` (native `float`) and
      `to_dict()` returning `dataclasses.asdict(self)`.
- [x] 2.4 Export `PCAResult`, `FeatureContribution` from `__init__.py` (import +
      add to `__all__`). Note: `pca.py` has no `__all__`; exports live only in
      `__init__.py`.
- [x] 2.5 Ensure the two new names satisfy the `__all__` introspection contract
      (`tests/test_public_api_docs.py` checks every `__all__` entry for complete
      type hints + parsable Google docstrings): no hardcoded count to bump, but
      `PCAResult`/`FeatureContribution` must have resolvable annotations and
      Google-style docstrings so the contract test stays green.

## 3. Verify non-breaking
- [x] 3.1 Confirm `perform_pca_analysis()` return dict is unchanged and existing
      `tests/test_pca*.py` pass (covered concretely by task 1.10).

## 4. Pre-merge
- [x] 4.1 `black` + `ruff` + full `pytest` + coverage green; `openspec validate
      add-pcaresult-dataclass --strict` passes.
