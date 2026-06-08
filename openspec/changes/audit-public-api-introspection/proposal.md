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
**7 functions** that fail an introspection check:

- `create_trait_boxplots_by_genotype`, `create_exploratory_summary_plots`,
  `create_trait_by_genotype_boxplots` (in `visualization.py`) — `get_type_hints()`
  raises `NameError: name 'Any' is not defined` because `Any` is used in
  annotations but never imported (the module uses `from __future__ import
  annotations`, so the failure surfaces at `get_type_hints()` time, exactly on the
  bloom-mcp path). This is the same class of bug fixed for `statistics.py` in #116.
- `cli.main` — no return annotation and no `Returns:` section.
- `save_viz_config`, `validate_viz_config`, `create_publication_figure` — no
  `Returns:` section.

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

2. **Fix the 7 failing functions** (type-hint + docstring only, no behavior change):
   - Add `from typing import Any` to `visualization.py`.
   - Add a return annotation + `Returns:` to `cli.main`.
   - Add `Returns:` sections to `save_viz_config`, `validate_viz_config`,
     `create_publication_figure`.

3. **Commit `scripts/check_public_api_docs.py`** — iterates `__all__`, applies the
   contract, prints a per-symbol report, and exits non-zero on any violation. It is
   runnable standalone (CI) and is exercised by a pytest test so the contract is
   enforced in the existing test job.

4. **Commit `docs/public_api_audit_2026.md`** — the audit report: methodology, the
   pre-change findings (the 7 failures above), what was changed, and the
   post-change result (0 violations across all 112 entries).

## Impact

- Affected specs: **public-api-introspection** (new capability).
- Affected code:
  - `src/sleap_roots_analyze/visualization.py` (`Any` import)
  - `src/sleap_roots_analyze/cli.py` (`main` return annotation + docstring)
  - `src/sleap_roots_analyze/pipeline/config/utils.py` (`save_viz_config`,
    `validate_viz_config` docstrings)
  - `create_publication_figure` docstring (in its defining module)
  - `scripts/check_public_api_docs.py` (new)
  - `docs/public_api_audit_2026.md` (new)
  - a new test wiring the audit script into pytest
- **No behavior change** — annotations, docstrings, a new check script, and a
  report only. No public signatures or return values change.
