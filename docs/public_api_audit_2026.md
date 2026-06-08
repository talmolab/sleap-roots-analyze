# Public API Introspection Audit (2026)

**Issue:** [#117](https://github.com/talmolab/sleap-roots-analyze/issues/117) —
*Audit `__all__` exports for introspection-readiness (type hints + docstrings)*

**Blocker for:** bloom-mcp autopop generator (Phase 3 of the Metcalf 2026 project),
which targets a ≥80% auto-generation rate for MCP tools.

## Why

bloom-mcp builds MCP tool descriptors by reading `sleap_roots_analyze.__all__` and,
for each name, calling `inspect.signature()`, `typing.get_type_hints()`, and
`__doc__` to derive parameters, types, defaults, and descriptions. Any public symbol
with a missing annotation, an unresolvable type hint, or an unparsable docstring
drops out of auto-generation and falls back to a hand-written wrapper. This audit
brings every `__all__` entry up to a machine-introspectable bar and makes that bar a
permanent, enforced contract.

## Methodology

The audit is implemented as a committed script,
[`scripts/check_public_api_docs.py`](../scripts/check_public_api_docs.py), and is
enforced on every test run via
[`tests/test_public_api_docs.py`](../tests/test_public_api_docs.py). It iterates
`sleap_roots_analyze.__all__` (112 entries: 110 functions + 2 classes) and applies
the following criteria.

For each public **function**:

1. Every parameter (excluding `*args`/`**kwargs`) has a type annotation.
2. The return value has a type annotation.
3. `typing.get_type_hints()` resolves without raising.
4. A docstring exists with a `Returns:` section, plus an `Args:` section when the
   function takes parameters.
5. Every parameter name appears in the docstring body.
6. A `Raises:` section is present when the function body raises a non-trivial
   exception (a `raise SomeError(...)` in its own body — bare re-raises and raises
   inside nested functions are ignored).

For each public **class** (`VizPipeline`, `VizPipelineConfig`):

7. The class has a docstring.
8. Every `__init__` parameter beyond `self` is annotated and named in the class or
   `__init__` docstring.

## Findings (before the change)

18 of the 112 `__all__` entries failed at least one criterion. The 94 others already
passed.

### Unresolvable type hints (`get_type_hints()` → `NameError: 'Any'`)

`visualization.py` used `Any` in annotations but never imported it. Because the
module uses `from __future__ import annotations`, the annotations are stringized and
the failure only surfaces when `get_type_hints()` evaluates them — exactly the
bloom-mcp path. (Same class of bug fixed for `statistics.py` in #116.)

| Function | Module |
| --- | --- |
| `create_trait_boxplots_by_genotype` | `visualization` |
| `create_exploratory_summary_plots` | `visualization` |
| `create_trait_by_genotype_boxplots` | `visualization` |

### Missing return annotation / `Returns:` section

| Function | Module | Gap |
| --- | --- | --- |
| `main` | `cli` | no return annotation **and** no `Returns:` |
| `save_viz_config` | `pipeline.config.utils` | no `Returns:` |
| `validate_viz_config` | `pipeline.config.utils` | no `Returns:` |
| `create_publication_figure` | `visualization` | no `Returns:` |

### Raises non-trivially but no `Raises:` section

All raise `ValueError` for input validation (one also `RuntimeError`).

| Function | Module |
| --- | --- |
| `calculate_figure_size` | `viz_utils` |
| `calculate_barplot_size` | `viz_utils` |
| `link_rhizovision_images_to_samples` | `data_utils` |
| `link_cylinder_images_from_scan_path` | `data_utils` |
| `perform_pca_analysis` | `pca` |
| `calculate_optimal_clusters_hierarchical` | `clustering` (also `RuntimeError`) |
| `detect_outliers_pca` | `outlier_detection` |
| `create_pca_biplot` | `visualization` |
| `identify_extreme_genotypes_by_pc` | `visualization` |
| `create_pc_genotype_boxplots` | `visualization` |
| `create_feature_contribution_heatmap` | `visualization` |

## Changes made

All changes are **type-hint and docstring only — no behavior change**. No public
signatures or return values were altered.

- **`visualization.py`** — added `from typing import Any`; added `Returns:` to
  `create_publication_figure`; added `Raises:` to `create_pca_biplot`,
  `identify_extreme_genotypes_by_pc`, `create_pc_genotype_boxplots`, and
  `create_feature_contribution_heatmap`.
- **`cli.py`** — added a `-> None` return annotation and a `Returns:` section to
  `main`.
- **`pipeline/config/utils.py`** — added `Returns:` sections to `save_viz_config`
  and `validate_viz_config`.
- **`viz_utils.py`** — added `Raises:` to `calculate_figure_size` and
  `calculate_barplot_size`.
- **`data_utils.py`** — added `Raises:` to `link_rhizovision_images_to_samples` and
  `link_cylinder_images_from_scan_path`.
- **`pca.py`** — added `Raises:` to `perform_pca_analysis`.
- **`clustering.py`** — added `Raises:` to `calculate_optimal_clusters_hierarchical`.
- **`outlier_detection.py`** — added `Raises:` to `detect_outliers_pca`.

## Result (after the change)

```
Public API introspection audit: 112 __all__ entries
  passing: 112
  failing: 0
```

All 110 functions and 2 classes are introspection-ready. The contract is now
enforced by `tests/test_public_api_docs.py`, so any new public symbol that regresses
the bar will fail the test suite. The script can also be wired into CI directly via
`python scripts/check_public_api_docs.py` (exit code 0 = clean).
