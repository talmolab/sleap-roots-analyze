## ADDED Requirements

### Requirement: Serializable UMAP Result Type

The package SHALL provide a frozen stdlib `@dataclass` `UMAPResult` in
`result_types.py` that captures only the JSON-serializable science of a
`perform_umap_analysis()` run, excluding the fitted `umap.UMAP` `reducer` and
`StandardScaler` `scaler` objects. Every scalar field SHALL be a native Python type
(`int`, `float`, `str`, `bool`) — not a numpy scalar — and every array field a list
thereof, so that `json.dumps(dataclasses.asdict(result))` succeeds without a custom
serializer. The type SHALL expose the fields `embedding` (an `(n_samples,
n_components)` nested `list[list[float]]`), `n_neighbors` (`int`), `min_dist`
(`float`), `n_components` (`int`), `feature_names` (`list[str]`), `n_samples` (`int`),
`standardized` (`bool`), and `random_state` (`Optional[int]`, default `None`).
`n_samples` and `n_components` are stored fields (materialized into the serialized
payload for the JSON consumer), not derived properties. `UMAPResult` SHALL provide a
`to_dict()` method returning `dataclasses.asdict(self)` and a `to_json(**kwargs)` method
that defaults to `allow_nan=False`.

#### Scenario: UMAPResult round-trips through JSON as native types

- **WHEN** a `UMAPResult` built from a `perform_umap_analysis()` return dict is passed
  to `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing the string back with `json.loads` SHALL yield `n_neighbors`,
  `n_components`, and `n_samples` as Python `int`, `min_dist` as a Python `float`,
  `standardized` as a Python `bool`, every element of `embedding` as a Python `float`,
  and every element of `feature_names` as a Python `str` (no `np.int64`/`np.float64`)
- **AND** the round-tripped `embedding` values SHALL equal the input values within
  floating-point tolerance

#### Scenario: Fields are native Python types before serialization

- **WHEN** the `UMAPResult` dataclass fields are inspected directly, before any JSON
  serialization
- **THEN** `n_neighbors`, `n_components`, and `n_samples` SHALL each be a native `int`,
  `min_dist` a native `float`, `standardized` a native `bool`, every `embedding` element
  a native `float`, and every `feature_names` element a native `str` — not a numpy
  scalar (a JSON round-trip would silently cast an `np.float64` leak to `float`, so the
  check SHALL be pre-serialization)

#### Scenario: No sklearn or UMAP object is present in the clean view

- **WHEN** `dataclasses.asdict(result)` is inspected
- **THEN** it SHALL contain no fitted `umap.UMAP` or `StandardScaler` object
- **AND** SHALL contain no `reducer` or `scaler` key

#### Scenario: n_samples equals the embedding row count

- **WHEN** `result.n_samples` is read
- **THEN** it SHALL be a native `int` equal to the number of rows of `result.embedding`
- **AND** `n_samples` SHALL appear as a key in `dataclasses.asdict(result)` (a
  materialized field, not a runtime-only property)

#### Scenario: to_json rejects a non-finite embedding value

- **WHEN** `to_json()` is called on a `UMAPResult` whose `embedding` contains a
  non-finite value (`NaN` or `Infinity`)
- **THEN** a `ValueError` SHALL be raised (under the default `allow_nan=False`) rather
  than emitting the non-standard `NaN`/`Infinity` tokens a strict JSON consumer rejects

### Requirement: UMAPResult Adapter From Legacy Dict

The package SHALL provide `UMAPResult.from_umap_dict(d, *, random_state=None)` that maps
the canonical `perform_umap_analysis()` return dict into a `UMAPResult`. The adapter
SHALL read `embedding`, `n_neighbors`, `min_dist`, and `feature_names` from `d`; SHALL
derive `n_components` from the width and `n_samples` from the row count of `embedding`;
SHALL set `standardized` to `d.get("scaler") is not None`; and SHALL resolve
`random_state` as the explicit argument when supplied, otherwise falling back to
`d.get("random_state")`. The adapter SHALL NOT mutate `d`. Behavior on a partial dict
(missing required keys) is unspecified.

#### Scenario: Adapter maps the core fields from a real run

- **WHEN** `UMAPResult.from_umap_dict(d)` is called on the dict returned by
  `perform_umap_analysis()`
- **THEN** `embedding` SHALL be a nested `list[list[float]]` of shape `(n_samples,
  n_components)`
- **AND** `n_components` SHALL equal the number of columns and `n_samples` the number of
  rows of `d["embedding"]`
- **AND** `n_neighbors` SHALL be an `int` equal to `d["n_neighbors"]` (the effective
  value the run used, after any clamping to `n_samples - 1`), `min_dist` a `float` equal
  to `d["min_dist"]`, and `feature_names` the list of feature-column names carried in `d`
  (order preserved)

#### Scenario: Nested shape is preserved at n_components == 1

- **WHEN** a run with `n_components=1` is passed to `from_umap_dict`
- **THEN** `embedding` SHALL be `(n_samples, 1)` — each inner row a one-element list,
  not a flattened scalar — and `n_components` SHALL be `1`

#### Scenario: standardized reflects whether a scaler was fitted

- **WHEN** `from_umap_dict` is applied to a `perform_umap_analysis()` dict
- **THEN** `standardized` SHALL be `True` (the dict carries a fitted `scaler`)
- **WHEN** `from_umap_dict` is applied to a dict whose `scaler` is `None`
- **THEN** `standardized` SHALL be `False`

#### Scenario: random_state resolves from the argument or the dict

- **WHEN** `from_umap_dict(d, random_state=7)` is called
- **THEN** `result.random_state` SHALL be `7` (the explicit argument wins)
- **WHEN** `from_umap_dict(d)` is called with no argument on a dict that carries
  `random_state`
- **THEN** `result.random_state` SHALL equal `d["random_state"]` (the echoed seed)
- **WHEN** `from_umap_dict(d)` is called with no argument on a dict lacking
  `random_state`
- **THEN** `result.random_state` SHALL be `None`, serializing to JSON `null`

#### Scenario: Adapter does not mutate the source dict

- **WHEN** `from_umap_dict(d)` is called
- **THEN** `d`'s keys and values (including the fitted `reducer`/`scaler`) SHALL be
  unchanged after the call

#### Scenario: Same seed yields an identical embedding via the typed view

- **WHEN** `perform_umap_analysis` is run twice with the same `random_state` and each
  dict is passed through `from_umap_dict`
- **THEN** the two results' `embedding` values SHALL be identical

### Requirement: UMAPResult Public Export

The package SHALL export `UMAPResult` from the top-level `sleap_roots_analyze`
namespace and list it in `__all__`, and SHALL list it in `result_types.__all__`.

#### Scenario: UMAP result type importable from package root

- **WHEN** a consumer runs `from sleap_roots_analyze import UMAPResult`
- **THEN** the import SHALL succeed
- **AND** `"UMAPResult"` SHALL appear in `sleap_roots_analyze.__all__` with no duplicate
  entries
- **AND** `UMAPResult` SHALL be importable from `sleap_roots_analyze.result_types` and
  listed in `result_types.__all__`

### Requirement: Non-Breaking UMAP Return Shape

Adding `UMAPResult` SHALL NOT change the return **type** of `perform_umap_analysis()`;
the function SHALL keep returning its existing dict so all current callers (the pipeline
`UMAPAnalysisStep`, `interactive_visualization`, the reproducibility sweep, the golden
embedding recompute) continue to work unchanged. The function MAY additively add
`feature_names` and `random_state` keys to that dict; it SHALL keep returning the
existing `embedding`, `reducer`, `scaler`, `n_neighbors`, and `min_dist` keys.

#### Scenario: Existing dict return is preserved and additively enriched

- **WHEN** `perform_umap_analysis()` is called
- **THEN** it SHALL return a `dict` containing the existing keys `embedding`,
  `reducer`, `scaler`, `n_neighbors`, and `min_dist`
- **AND** it SHALL also carry a `feature_names` key equal to the `feature_cols` used and
  a `random_state` key equal to the seed used
- **AND** the `UMAPResult` view SHALL be opt-in, built only via
  `UMAPResult.from_umap_dict()`
