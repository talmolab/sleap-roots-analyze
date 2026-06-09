# Proposal: Audit `__all__` Exports for Introspection-Readiness

## Why

bloom-mcp's Phase 3 autopop generator builds MCP tool descriptors by reading
`sleap_roots_analyze.__all__` and, for each name, calling `inspect.signature()` +
`typing.get_type_hints()` + `__doc__` to derive parameters, types, defaults, and
descriptions. Every public symbol with a missing annotation, an unresolvable type
hint, or an unparsable docstring drops out of auto-generation and falls back to a
hand-written wrapper. The target auto-generation rate is **≥80%**, so each gap
directly expands the manual fallback surface and delays Phase 3.

A current audit of the 112 `__all__` entries (110 functions + 2 classes) finds
**18 functions** that fail an introspection check, in three categories:

- **Unresolvable type hints (3)** — `create_trait_boxplots_by_genotype`,
  `create_exploratory_summary_plots`, `create_trait_by_genotype_boxplots` (in
  `visualization.py`): `get_type_hints()` raises `NameError: name 'Any' is not
  defined` because `Any` is used in annotations but never imported (the module uses
  `from __future__ import annotations`, so the failure surfaces at
  `get_type_hints()` time, exactly on the bloom-mcp path). Same class of bug fixed
  for `statistics.py` in #116.
- **Missing return annotation / `Returns:` (4)** — `cli.main` (no return annotation
  *and* no `Returns:`); `save_viz_config`, `validate_viz_config`,
  `create_publication_figure` (no `Returns:`).
- **Raises but undocumented (11)** — `calculate_figure_size`,
  `calculate_barplot_size` (`viz_utils`); `link_rhizovision_images_to_samples`,
  `link_cylinder_images_from_scan_path` (`data_utils`); `perform_pca_analysis`
  (`pca`); `calculate_optimal_clusters_hierarchical` (`clustering`);
  `detect_outliers_pca` (`outlier_detection`); `create_pca_biplot`,
  `identify_extreme_genotypes_by_pc`, `create_pc_genotype_boxplots`,
  `create_feature_contribution_heatmap` (`visualization`): each raises a non-trivial
  exception (`ValueError`, and `RuntimeError` for the clustering one) with no
  `Raises:` section.

Beyond fixing these, the gap will silently reappear as new public functions are
added unless the contract is **enforced**. This change adds a committed audit
script that fails on any violation, making introspection-readiness a permanent,
testable contract.

Tracked by issue #117.

## What Changes

1. **Define the introspection-readiness contract** as a new capability
   (`public-api-introspection`): for every public function in `__all__` —
   annotations on every parameter and the return; `get_type_hints()` resolves;
   a Google-style docstring with `Args:` and `Returns:`; every parameter named in
   the docstring; a `Raises:` section where the function raises non-trivial
   exceptions. Public classes must carry a class docstring and annotate/document
   any non-trivial `__init__` parameters.

2. **Fix the 18 failing functions** (type-hint + docstring only, no behavior
   change):
   - Add `from typing import Any` to `visualization.py` (fixes the 3 `get_type_hints`
     failures).
   - Add a return annotation + `Returns:` to `cli.main`; add `Returns:` sections to
     `save_viz_config`, `validate_viz_config`, `create_publication_figure`.
   - Add `Raises:` sections to the 11 functions that raise non-trivial exceptions
     (in `viz_utils`, `data_utils`, `pca`, `clustering`, `outlier_detection`, and
     `visualization`), each derived from the function's actual raised type and
     condition.

3. **Commit `scripts/check_public_api_docs.py`** — iterates `__all__`, applies the
   contract, prints a per-symbol report, and exits non-zero on any violation. It is
   runnable standalone (CI) and is exercised by a pytest test so the contract is
   enforced in the existing test job.

4. **Commit `docs/public_api_audit_2026.md`** — the audit report: methodology, the
   pre-change findings (the 18 failures above), what was changed, and the
   post-change result (0 violations across all 112 entries).

## Impact

- Affected specs: **public-api-introspection** (new capability).
- Affected code:
  - `src/sleap_roots_analyze/visualization.py` (`Any` import; `Returns:` for
    `create_publication_figure`; `Raises:` for `create_pca_biplot`,
    `identify_extreme_genotypes_by_pc`, `create_pc_genotype_boxplots`,
    `create_feature_contribution_heatmap`)
  - `src/sleap_roots_analyze/cli.py` (`main` return annotation + docstring)
  - `src/sleap_roots_analyze/pipeline/config/utils.py` (`save_viz_config`,
    `validate_viz_config` docstrings)
  - `src/sleap_roots_analyze/viz_utils.py` (`Raises:` for `calculate_figure_size`,
    `calculate_barplot_size`)
  - `src/sleap_roots_analyze/data_utils.py` (`Raises:` for
    `link_rhizovision_images_to_samples`, `link_cylinder_images_from_scan_path`)
  - `src/sleap_roots_analyze/pca.py` (`Raises:` for `perform_pca_analysis`)
  - `src/sleap_roots_analyze/clustering.py` (`Raises:` for
    `calculate_optimal_clusters_hierarchical`)
  - `src/sleap_roots_analyze/outlier_detection.py` (`Raises:` for
    `detect_outliers_pca`)
  - `scripts/check_public_api_docs.py` (new)
  - `docs/public_api_audit_2026.md` (new)
  - a new test wiring the audit script into pytest
- **No behavior change** — annotations, docstrings, a new check script, and a
  report only. No public signatures or return values change.
