# Tasks — Add HeritabilityResult Serializable Dataclass (issue #128)

> Single `feat: ... (#128)` commit (CI green at HEAD); exports land with the
> dataclass. Stacked on #127. OpenSpec archive post-merge.

## 1. Failing tests first (red) — `tests/test_heritability_result.py`
- [x] 1.1 `test_json_roundtrip_native_types`: run
      `calculate_heritability_estimates` on `heritability_data_known_h2`, build
      `HeritabilityResult.from_heritability_dict(d, threshold=0.3)`, assert
      `json.dumps(asdict(...))` succeeds, then `json.loads` and assert
      `threshold` is `float`, each `per_trait` `h2` is `float`,
      `passed_threshold` is `bool`, counts are `int`.
- [x] 1.2 `test_adapter_classifies_and_reads_method`: `method == "mixed_model"`;
      `__calculation_metadata__` not present as a trait; known H² (~0.8/0.5/0.09)
      mapped; `passed_threshold == (h2 >= threshold)`.
- [x] 1.3 `test_mean_h2_and_n_above_threshold`: `mean_h2` ≈ mean of per-trait
      H² (native float); `n_above_threshold` counts passing traits at a chosen
      threshold (e.g. 0.3 → high+moderate pass, low fails).
- [x] 1.4 `test_failed_traits_separated`: feed a dict with an error/`missing`
      trait entry; it lands in `failed_traits`, not `per_trait`.
- [x] 1.5 `test_deterministic`: same input → identical `asdict`/json.
- [x] 1.6 `test_exports_and_all`: import from package root; both names in
      `__all__`, no dupes.
- [x] 1.7 `test_dict_unchanged_and_nonmutating`: keys preserved (incl.
      `__calculation_metadata__`); adapter does not mutate `d`.

## 2. Implement to green
- [x] 2.1 Add `TraitHeritability` + frozen `HeritabilityResult` to
      `result_types.py` (Google docstrings with `Attributes:` for every field).
- [x] 2.2 Implement `from_heritability_dict(d, threshold)`: read `method`; skip
      `__calculation_metadata__`; split success vs `failed_traits`; native casts;
      no mutation of `d`.
- [x] 2.3 Implement `@property mean_h2`, `@property n_above_threshold`, and
      `to_dict()`.
- [x] 2.4 Export `HeritabilityResult`, `TraitHeritability` from `__init__.py`
      (`__all__`).

## 3. Verify non-breaking
- [x] 3.1 Existing `tests/test_statistics*.py` pass; return shape unchanged
      (covered by task 1.7).

## 4. Pre-merge
- [x] 4.1 `black` + `ruff` + full `pytest` green; `openspec validate
      add-heritabilityresult-dataclass --strict` passes.
