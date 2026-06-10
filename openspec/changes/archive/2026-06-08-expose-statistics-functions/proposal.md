# Proposal: Expose `statistics.py` Functions in the Public API

## Why

`src/sleap_roots_analyze/statistics.py` contains 8 heritability/ANOVA/variance
functions that are **not** exported through the package's `__init__.py`. Today
`__init__.py` imports only `cross_experiment_analysis.calculate_genotype_statistics`;
`statistics.py` is missing entirely.

As a result, downstream code reaches into the internal module
(`from sleap_roots_analyze.statistics import ...`) or, worse, maintains a
duplicate copy — `bloommcp/source/trait_statistics.py` is an 810-line
re-implementation of these same functions. This is fragile and unmaintainable.

This is a **blocker** for bloom-mcp MCP tool wrapping and the Metcalf 2026 intern's
Phase 1 work (reproducing the wheat EDPIE paper heritability/PCA analyses). The
public function signatures are the source-of-truth for downstream tool schemas,
so they must be importable, documented, and have type hints that resolve.

Tracked by issue #116. This change is **Part of #116** — it does not close it,
because the issue's `mypy --strict` acceptance item is deferred (see below).

## What Changes

1. **Export all 8 functions** from `__init__.py` and list each in `__all__`:
   `calculate_trait_statistics`, `perform_anova_by_genotype`,
   `calculate_heritability_estimates`, `identify_high_heritability_traits`,
   `analyze_heritability_thresholds`, `analyze_trait_variance`,
   `diagnose_heritability_issues`, `compare_trait_heritabilities`.

2. **Fix the missing `Any` import** in `statistics.py`. `Any` is used in
   `Dict[str, Any]` annotations on three of the now-public functions but is never
   imported, so `typing.get_type_hints()` raises `NameError` on them — breaking the
   exact downstream tool-schema path this change enables. Adding
   `from typing import Any` is a one-line correctness fix and is in scope.

3. **Audit docstrings and type hints** for all 8 functions so each has complete
   Google-style Args/Returns (and Raises where it raises) and accurate type hints,
   enumerating returned dict keys where the docstring is currently vague
   (`calculate_trait_statistics`, `perform_anova_by_genotype`).

4. **Expand the `statistics.py` module docstring** to describe its scope
   (single-experiment heritability, ANOVA, trait-variance) and how it differs from
   `cross_experiment_analysis.py` (cross-experiment alignment and correlation).

5. **Update `docs/API.md`**: add the **3** functions currently missing from the
   existing `## statistics Module` section (`analyze_trait_variance`,
   `diagnose_heritability_issues`, `compare_trait_heritabilities`) and reconcile the
   5 existing entries that have drifted from the code (`perform_anova_by_genotype`
   missing `alpha` + stale return keys; `identify_high_heritability_traits` default
   documented as `0.3` but code is `0.5`).

6. **Add a `docs/CHANGELOG.md` `[Unreleased]` entry** recording the newly-public
   statistics API.

> **Deferred (out of scope for this change):** making
> `mypy --strict src/sleap_roots_analyze/statistics.py` fully pass. The remaining
> strict cleanup (bare `Dict` type parameters, narrow mixed-value return-type
> annotations, `list`-vs-`ndarray` assignment, third-party stub handling for
> statsmodels/scipy) is tracked for a follow-up change. Only the single missing
> `Any` import (item 2) is fixed here, because it breaks `get_type_hints()` at
> runtime, not merely strict type-checking.

## Impact

- Affected specs: **statistics-api** (new capability).
- Affected code:
  - `src/sleap_roots_analyze/__init__.py` (exports + `__all__`)
  - `src/sleap_roots_analyze/statistics.py` (`Any` import, docstrings, module docstring)
  - `docs/API.md` (3 new entries + 5 reconciled)
  - `docs/CHANGELOG.md` (`[Unreleased]` entry)
  - a new public-API import test under `tests/`
- **No behavior change** to the functions themselves — this is an API-surface +
  documentation + one type-hint-import change. Existing imports from
  `sleap_roots_analyze.statistics` continue to work.
