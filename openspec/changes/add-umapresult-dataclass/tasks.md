> Test inputs: build from the existing `traits_summary_sample` fixture and its feature
> column list (see `tests/test_umap.py:22-27`) — no new UMAP fixture is needed. Section-1
> tests MUST import `from sleap_roots_analyze.result_types import UMAPResult` (the package-
> root export does not land until §3); the root-import assertion lives only in task 3.1.

## 1. Result type + adapter (test-first)

- [x] 1.1 Create `tests/test_umap_result.py` (mirror `tests/test_cluster_result.py`,
      reusing the `_assert_dict_unchanged` pattern). Write a failing round-trip test: a
      `UMAPResult` built from `perform_umap_analysis(traits_summary_sample, feature_cols,
      random_state=42)` serializes via `to_json()` / `json.dumps(dataclasses.asdict(...))`,
      parses back with `embedding` as `list[list[float]]`, `n_neighbors`/`n_components`/
      `n_samples` as `int`, `min_dist` as `float`, `standardized` as `bool`,
      `feature_names` as `list[str]`, and the round-tripped `embedding` equal to
      `dataclasses.asdict(result)["embedding"]` within `np.allclose` tolerance.
- [x] 1.1a Write failing test `test_fields_are_native_types_pre_serialization` (mirror
      `test_cluster_result.py`): assert on the **dataclass fields directly, before any
      JSON** — `type(result.n_neighbors) is int`, `type(result.n_components) is int`,
      `type(result.n_samples) is int`, `type(result.min_dist) is float`,
      `type(result.standardized) is bool`, `all(type(v) is float for row in
      result.embedding for v in row)`, `all(type(v) is str for v in
      result.feature_names)`. Reproduce the sibling's comment that a JSON round-trip hides
      an `np.float64` leak, so the assertion MUST be pre-serialization.
- [x] 1.2 Write failing test: `dataclasses.asdict(result)` contains **no** `reducer` or
      `scaler` key and no fitted `umap.UMAP`/`StandardScaler` object.
- [x] 1.3 Write failing test: `result.n_samples == len(result.embedding)` and
      `"n_samples"` **is** a key in `dataclasses.asdict(result)` (a materialized field,
      not a property); likewise `n_components == len(result.embedding[0])`.
- [x] 1.4 Write failing test: `UMAPResult.to_json()` raises `ValueError` on a non-finite
      `embedding` value (mirror `test_to_json_rejects_non_finite_bic`); and a positive
      test that `json.loads(result.to_json()) == json.loads(json.dumps(result.to_dict()))`
      on finite data.
- [x] 1.5 Write failing adapter tests: `from_umap_dict(d)` on a real
      `perform_umap_analysis()` dict maps `embedding` to shape `(n_samples, n_components)`;
      asserts **by value** `n_neighbors == int(d["n_neighbors"])`, `min_dist ==
      pytest.approx(float(d["min_dist"]))`, `feature_names == [str(c) for c in
      feature_cols]` (order/identity), and `np.allclose(result.embedding, d["embedding"])`;
      sets `standardized` `True`; and does **not** mutate `d` (deep-copy compare via
      `_assert_dict_unchanged`).
- [x] 1.5a Write failing test `test_standardized_false_when_scaler_none`:
      `from_umap_dict({"embedding": <real 2-col list>, "n_neighbors": ..., "min_dist": ...,
      "feature_names": [...], "scaler": None}).standardized is False`, and `to_json()`
      succeeds. Pairs with 1.5's `True` branch so the `d.get("scaler") is not None`
      derivation is exercised both directions (the only guard against a hardcoded
      `standardized=True`, since the real producer always standardizes).
- [x] 1.6 Write failing tests: `n_components == 1` preserves `(n_samples, 1)` nested shape
      (each inner row a one-element list); `random_state` resolution — explicit arg wins
      (`from_umap_dict(d, random_state=7).random_state == 7`), else falls back to
      `d["random_state"]` when present, else `None` (serializing to JSON `null`).
- [x] 1.7 Add `UMAPResult(frozen=True)` to `result_types.py` with fields `embedding`,
      `n_neighbors`, `min_dist`, `n_components`, `feature_names`, `n_samples`,
      `standardized`, `random_state: Optional[int] = None` (only `random_state` defaulted,
      so field order is valid); `to_dict()` and `to_json()` (the `allow_nan=False`
      finite-floats contract). The class docstring MUST enumerate **every** field in an
      `Attributes:` block (the `check_public_api_docs` audit requires each field name to
      appear), and — mirroring `PCAResult` — state that `reducer`/`scaler` are
      intentionally excluded and carry the shallow-`frozen=True` read-only caveat.
- [x] 1.8 Add the `UMAPResult.from_umap_dict(d, *, random_state=None)` classmethod
      (Google docstring), non-mutating: `np.asarray(d["embedding"]).tolist()`, derive
      `n_components`/`n_samples` from the embedding shape, `standardized` from the scaler,
      and resolve `random_state` as `random_state if random_state is not None else
      d.get("random_state")`; make 1.1–1.6 green. Append `"UMAPResult"` to
      `result_types.__all__`.

## 2. Producer enrichment (test-first)

- [x] 2.1 Write failing test: `perform_umap_analysis(traits_summary_sample, feature_cols,
      random_state=42)` returns a dict that still carries `embedding`, `reducer`,
      `scaler`, `n_neighbors`, `min_dist` **and** now `feature_names == feature_cols`
      (order preserved) and `random_state == 42`. Assert the pre-existing keys are still
      present (membership, non-breaking).
- [x] 2.2 Add `feature_names` (the `feature_cols` used) and `random_state` (the seed used)
      keys to the `perform_umap_analysis` return dict and update its `Returns:` docstring;
      make 2.1 green.
- [x] 2.3 Write end-to-end test then confirm green:
      `UMAPResult.from_umap_dict(perform_umap_analysis(traits_summary_sample, feature_cols,
      random_state=42))` (no explicit `random_state` arg) yields a `UMAPResult` with
      populated `embedding`, `feature_names`, `n_neighbors`, `min_dist`, `n_components`,
      `n_samples`, and `random_state == 42` (resolved from the echoed dict key), and
      `to_json()` succeeds.
- [x] 2.4 Write test `test_n_neighbors_clamped_value_reflected`: on a small frame, call
      `perform_umap_analysis(df, feature_cols, n_neighbors=len(df)+50, random_state=42)`,
      build the `UMAPResult`, and assert `result.n_neighbors == len(df) - 1` (the clamped
      effective value from `umap.py:81-82`) and `type(result.n_neighbors) is int`.

## 3. Exports + docs

- [x] 3.1 Write failing import test: `from sleap_roots_analyze import UMAPResult` succeeds
      and `"UMAPResult"` appears in `sleap_roots_analyze.__all__` with no duplicates;
      `UMAPResult` imports from `sleap_roots_analyze.result_types` and is in
      `result_types.__all__`.
- [x] 3.2 Add `UMAPResult` to `__init__.py` (the `result_types` import block) and
      `__all__` (in the "Serializable result types (#130)" group); make 3.1 green. (This
      is the commit where the public-API docstring audit fires — 1.7's complete
      `Attributes:` block MUST already be in place.)
- [x] 3.3 Update the `result_types.py` **module docstring** enumeration (the "`PCAResult`
      … `HeritabilityResult` (#128) and `ClusterResult` (#129) follow" sentence, ~lines
      20-22) to include `UMAPResult`.
- [x] 3.4 Update `docs/result-types.md`: add the UMAP row to the types table (Built from
      `perform_umap_analysis` dict; adapter `UMAPResult.from_umap_dict(d, *,
      random_state=None)`).
- [x] 3.5 Add a `docs/CHANGELOG.md` `[Unreleased]` entry: `### Added` (`UMAPResult`,
      `UMAPResult.from_umap_dict`, and the additive `feature_names`/`random_state` keys on
      `perform_umap_analysis`). Do **not** bump `pyproject.toml` or add a dated version
      heading — `0.1.0a5` is cut in a separate release PR (per #176), bundling this with
      #179.

## 4. Validation

- [x] 4.1 `openspec validate add-umapresult-dataclass --strict` (CLI not currently
      installed — if unavailable, hand-check format against `openspec/AGENTS.md`).
- [x] 4.2 `/lint` + full pytest via `/pre-merge-check`. Lint ✅ (black + ruff clean on all
      changed files); full suite ✅ **2433 passed, 31 skipped, 0 failed** (run in three
      letter-range batches on a fresh WSL VM; coverage plugin disabled for the batched run —
      Codecov upload is not gating). New `test_umap_result.py`: 21 passed.

> Notes: `perform_umap_analysis` is already in the reproducibility `CASES` registry
> (`tests/reproducibility_cases.py`), so the determinism sweep and its coverage guard
> already cover the seed — **do not** add a new `CASES` entry. The round-trip gate
> (`tests/test_result_serialization.py`) auto-skips dict-returning functions, so
> `UMAPResult` is exercised in the dedicated `tests/test_umap_result.py` (no edit to the
> gate needed). No new UMAP golden artifact is committed — the existing
> `golden_umap_embedding.csv` numerical-stability check is unchanged.
