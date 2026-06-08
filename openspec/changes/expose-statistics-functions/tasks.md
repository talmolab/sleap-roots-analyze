# Tasks: Expose `statistics.py` Functions in the Public API

> **Deferred:** fully passing `mypy --strict src/sleap_roots_analyze/statistics.py`
> (bare `Dict` params, narrow mixed-value return types, `list`-vs-`ndarray`,
> third-party stubs) is out of scope and tracked for a follow-up change. Only the
> missing `Any` import is fixed here (task 2.1) because it breaks `get_type_hints()`
> at runtime.

## 1. Public API export + tests

- [ ] 1.1 Write `tests/test_public_api.py` (new) asserting, for each of the 8
  statistics functions: (a) `hasattr(sleap_roots_analyze, name)`; (b)
  `getattr(sra, name) is getattr(sra.statistics, name)` (identity); (c)
  `name in sra.__all__`; (d) `from sleap_roots_analyze import *` (via
  `exec` into a namespace) binds each name. Confirm the test FAILS against `main`
  (no exports) to demonstrate red → green.
- [ ] 1.2 Add `__all__` hygiene asserts to the same test: no duplicates
  (`len(__all__) == len(set(__all__))`) and every `__all__` name resolves.
- [x] 1.3 Add `from sleap_roots_analyze.statistics import (...)` block to
  `__init__.py` for all 8 functions
- [x] 1.4 Add a `# Statistics / heritability functions` section listing all 8
  names in `__all__`
- [ ] 1.5 Run `uv run black src/sleap_roots_analyze/__init__.py` to remove the
  trailing whitespace (Black, not Ruff, flags it) — fold into the export commit so
  the first commit keeps CI lint green

## 2. Type-hint correctness

- [ ] 2.1 Add `Any` to the `from typing import ...` line in `statistics.py`
- [ ] 2.2 Add a test asserting `typing.get_type_hints(fn)` succeeds for each of the
  8 functions and that no parameter/return annotation is empty

## 3. Docstrings & module scope

- [ ] 3.1 Audit each of the 8 functions for complete Google-style Args/Returns/
  Raises and accurate type hints; enumerate returned dict keys for
  `calculate_trait_statistics` and `perform_anova_by_genotype` (currently vague)
- [ ] 3.2 Expand the `statistics.py` module docstring to describe scope vs
  `cross_experiment_analysis.py` (name it explicitly)
- [ ] 3.3 Add a test asserting each function's docstring contains `Args:` and
  `Returns:`, and that `statistics.__doc__` mentions `cross_experiment_analysis`
- [ ] 3.4 Run `uv run ruff check` (pydocstyle) to confirm docstrings pass

## 4. Documentation

- [ ] 4.1 In `docs/API.md` `## statistics Module`: add the **3** missing entries
  (`analyze_trait_variance`, `diagnose_heritability_issues`,
  `compare_trait_heritabilities`) following the existing entry format, AND reconcile
  the drifted existing entries (`perform_anova_by_genotype` missing `alpha` + stale
  return keys; `identify_high_heritability_traits` default `0.3` → code `0.5`)
- [ ] 4.2 Add a test asserting `docs/API.md` contains each of the 8 function names
- [ ] 4.3 Add an `### Added` entry under `[Unreleased]` in `docs/CHANGELOG.md`
  noting the 8 functions are now importable from `sleap_roots_analyze`

## 5. Validation

- [ ] 5.1 `uv run pytest tests/test_public_api.py tests/test_statistics.py` pass
- [ ] 5.2 `uv run black --check` + `uv run ruff check` clean
- [ ] 5.3 `openspec validate expose-statistics-functions --strict` passes
