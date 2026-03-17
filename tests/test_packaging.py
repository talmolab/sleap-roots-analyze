"""Tests for package metadata and dynamic versioning."""

import re
from importlib.metadata import version


def test_version_is_valid_pep440():
    """Package __version__ must be a valid PEP 440 version string."""
    import sleap_roots_analyze

    # PEP 440 pattern: N.N.N[{a|b|rc}N][.postN][.devN]
    pep440_pattern = r"^\d+\.\d+\.\d+(a\d+|b\d+|rc\d+)?(\.post\d+)?(\.dev\d+)?$"
    assert re.match(
        pep440_pattern, sleap_roots_analyze.__version__
    ), f"Version '{sleap_roots_analyze.__version__}' is not valid PEP 440"


def test_version_is_not_unknown():
    """Package __version__ should not be 'unknown' in a normal install."""
    import sleap_roots_analyze

    assert sleap_roots_analyze.__version__ != "unknown"


def test_version_matches_metadata():
    """Package __version__ must match importlib.metadata.version()."""
    import sleap_roots_analyze

    metadata_version = version("sleap-roots-analyze")
    assert sleap_roots_analyze.__version__ == metadata_version
