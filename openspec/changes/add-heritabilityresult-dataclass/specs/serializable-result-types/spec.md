# serializable-result-types Specification (delta)

## ADDED Requirements

### Requirement: Serializable Heritability Result Type

The package SHALL provide a frozen stdlib `@dataclass` `HeritabilityResult`
(with a nested `TraitHeritability` dataclass) in `result_types.py` that captures
only the JSON-serializable science of a `calculate_heritability_estimates()`
run. Every scalar field SHALL be a native Python type (`int`, `float`, `str`,
`bool`), so that `json.dumps(dataclasses.asdict(result))` succeeds without a
custom serializer. `HeritabilityResult` SHALL provide a `to_dict()` method
returning `dataclasses.asdict(self)`.

#### Scenario: HeritabilityResult round-trips through JSON as native types

- **WHEN** a `HeritabilityResult` built from a
  `calculate_heritability_estimates()` dict is passed to
  `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing the string back with `json.loads` SHALL yield `threshold` as a
  Python `float` and, for every entry in `per_trait`, `h2` as a Python `float`,
  `passed_threshold` as a Python `bool`, and the count fields as Python `int`
- **AND** the result SHALL contain no statsmodels model object

#### Scenario: mean_h2 and n_above_threshold derive from successful traits

- **WHEN** `result.mean_h2` and `result.n_above_threshold` are accessed
- **THEN** `mean_h2` SHALL be the mean of `h2` over `per_trait` (a native
  `float`, `0.0` when `per_trait` is empty)
- **AND** `n_above_threshold` SHALL be the count of `per_trait` entries whose
  `passed_threshold` is `True`

### Requirement: HeritabilityResult Adapter From Legacy Dict

The package SHALL provide
`HeritabilityResult.from_heritability_dict(d, threshold)` that maps the
`calculate_heritability_estimates()` return dict (the `remove_low_h2=False`
form, or the first element of the `remove_low_h2=True` tuple) into a
`HeritabilityResult`, without mutating `d`.

#### Scenario: Adapter classifies traits by threshold

- **WHEN** `HeritabilityResult.from_heritability_dict(d, threshold)` is called
- **THEN** `method` SHALL equal
  `d["__calculation_metadata__"]["method_used_for_all_traits"]`
- **AND** the `__calculation_metadata__` entry SHALL NOT appear as a trait
- **AND** each trait carrying a `"heritability"` value SHALL become a
  `TraitHeritability` with `h2` cast to `float` and `passed_threshold` equal to
  `h2 >= threshold`

#### Scenario: Failed traits are separated from successful ones

- **WHEN** the dict contains trait entries carrying an `"error"` (or lacking a
  `"heritability"` value)
- **THEN** those trait names SHALL be collected into `failed_traits`
- **AND** SHALL NOT appear in `per_trait`

### Requirement: HeritabilityResult Public Export

The package SHALL export `HeritabilityResult` and `TraitHeritability` from the
top-level `sleap_roots_analyze` namespace and list them in `__all__`.

#### Scenario: Heritability result types importable from package root

- **WHEN** a consumer runs
  `from sleap_roots_analyze import HeritabilityResult, TraitHeritability`
- **THEN** the import SHALL succeed
- **AND** both names SHALL appear in `sleap_roots_analyze.__all__` with no
  duplicate entries

### Requirement: Non-Breaking Heritability Return Shape

Adding `HeritabilityResult` SHALL NOT change the return shape of
`calculate_heritability_estimates()`; the function SHALL keep returning its
existing dict / tuple so all current callers continue to work unchanged.

#### Scenario: Existing heritability return is preserved and not mutated

- **WHEN** `calculate_heritability_estimates()` is called with
  `remove_low_h2=False`
- **THEN** it SHALL return the existing trait-keyed dict (including the
  `__calculation_metadata__` entry)
- **AND** calling `HeritabilityResult.from_heritability_dict(d, threshold)` SHALL
  leave `d`'s keys and values unchanged
- **AND** the `HeritabilityResult` view SHALL be opt-in, built only via
  `HeritabilityResult.from_heritability_dict()`
