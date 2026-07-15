"""Tests for the public statistics API surface exposed by ``sleap_roots_analyze``.

These tests guard the contract from the ``statistics-api`` OpenSpec change:
the nine ``statistics.py`` functions must be importable from the package root,
listed in ``__all__``, have resolvable type hints, carry Google-style docstrings,
and stay in sync with the hand-maintained docs.
"""

import typing
from pathlib import Path

import pytest

import sleap_roots_analyze as sra
import sleap_roots_analyze.statistics as stats_module

# The nine functions this change exposes.
STATISTICS_FUNCTIONS = [
    "calculate_trait_statistics",
    "perform_anova_by_genotype",
    "calculate_heritability_estimates",
    "identify_high_heritability_traits",
    "analyze_heritability_thresholds",
    "analyze_trait_variance",
    "diagnose_heritability_issues",
    "compare_trait_heritabilities",
    "extract_blup_table",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPublicImportSurface:
    """The eight functions are importable from the package root."""

    @pytest.mark.parametrize("name", STATISTICS_FUNCTIONS)
    def test_attribute_present(self, name):
        """Each function is accessible as a package attribute."""
        assert hasattr(sra, name), f"{name} is not exposed on sleap_roots_analyze"

    @pytest.mark.parametrize("name", STATISTICS_FUNCTIONS)
    def test_identity_with_statistics_module(self, name):
        """The exposed object is the same function defined in statistics.py."""
        assert getattr(sra, name) is getattr(stats_module, name)

    @pytest.mark.parametrize("name", STATISTICS_FUNCTIONS)
    def test_listed_in_all(self, name):
        """Each function is advertised in __all__."""
        assert name in sra.__all__, f"{name} missing from __all__"

    @pytest.mark.parametrize("name", STATISTICS_FUNCTIONS)
    def test_bound_by_star_import(self, name):
        """`from sleap_roots_analyze import *` binds each function name."""
        namespace = {}
        exec("from sleap_roots_analyze import *", namespace)
        assert name in namespace, f"{name} not bound by star import"


class TestAllHygiene:
    """`__all__` is internally consistent."""

    def test_no_duplicate_entries(self):
        """__all__ contains no duplicate names."""
        assert len(sra.__all__) == len(set(sra.__all__))

    def test_every_name_resolves(self):
        """Every name in __all__ resolves to a real attribute."""
        unresolved = [name for name in sra.__all__ if not hasattr(sra, name)]
        assert unresolved == [], f"__all__ names do not resolve: {unresolved}"


class TestResolvableTypeHints:
    """Downstream tool-schema generation relies on get_type_hints() succeeding."""

    @pytest.mark.parametrize("name", STATISTICS_FUNCTIONS)
    def test_get_type_hints_succeeds(self, name):
        """typing.get_type_hints() resolves without raising (e.g. NameError on Any)."""
        fn = getattr(sra, name)
        hints = typing.get_type_hints(fn)
        # Every parameter and the return value should carry an annotation.
        import inspect

        sig = inspect.signature(fn)
        for param in sig.parameters.values():
            assert param.name in hints, f"{name}: parameter {param.name} has no hint"
        assert "return" in hints, f"{name}: missing return annotation"


class TestDocstrings:
    """Each public function and the module are documented Google-style."""

    @pytest.mark.parametrize("name", STATISTICS_FUNCTIONS)
    def test_has_args_and_returns(self, name):
        """Each function's docstring has populated Args and Returns sections."""
        doc = getattr(sra, name).__doc__
        assert doc is not None, f"{name} has no docstring"
        assert "Args:" in doc, f"{name} docstring missing Args:"
        assert "Returns:" in doc, f"{name} docstring missing Returns:"
        # Guard against empty section headings: there must be indented body text
        # after each header, not just the bare label.
        for section in ("Args:", "Returns:"):
            body = doc.split(section, 1)[1]
            first_line = body.splitlines()[1] if len(body.splitlines()) > 1 else ""
            assert first_line.strip(), f"{name} has an empty {section} section"

    def test_module_docstring_distinguishes_cross_experiment(self):
        """The module docstring names cross_experiment_analysis to clarify scope."""
        doc = stats_module.__doc__
        assert doc is not None
        assert "cross_experiment_analysis" in doc


class TestDocsInSync:
    """Hand-maintained docs reference every public function."""

    @pytest.mark.parametrize("name", STATISTICS_FUNCTIONS)
    def test_api_md_lists_function(self, name):
        """docs/API.md documents each of the nine functions."""
        api_md = (REPO_ROOT / "docs" / "API.md").read_text(encoding="utf-8")
        assert name in api_md, f"{name} not documented in docs/API.md"

    @pytest.mark.parametrize(
        "snippet",
        [
            # Defaults reconciled in this PR — guard against doc drift.
            "threshold: float = 0.5",  # identify_high_heritability_traits
            "alpha: float = 0.05",  # perform_anova_by_genotype
        ],
    )
    def test_api_md_documents_signature_defaults(self, snippet):
        """docs/API.md documents the corrected signature defaults verbatim."""
        api_md = (REPO_ROOT / "docs" / "API.md").read_text(encoding="utf-8")
        assert snippet in api_md, f"docs/API.md missing documented default: {snippet!r}"

    def test_changelog_records_public_api(self):
        """docs/CHANGELOG.md documents the public statistics API permanently.

        Each statistics function must be recorded *somewhere* in the changelog.
        They were documented once, in the release that exposed them (#116), and
        a Keep-a-Changelog roll moves that entry from ``[Unreleased]`` into a
        dated version section every release. A windowed check (``[Unreleased]``
        + the top-most version section) therefore breaks on the next release,
        when the entry scrolls out of the window — that only *defers* the
        failure. Documentation is permanent, so the invariant is too: scan the
        whole file.
        """
        changelog = (REPO_ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "sleap_roots_analyze" in changelog
        for name in STATISTICS_FUNCTIONS:
            assert name in changelog, f"{name} not documented anywhere in CHANGELOG"
