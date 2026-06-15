"""Analysis-input contract conformance for the EDPIE fixtures (#147).

Proves the committed post-QC fixtures and the packaged canonical examples conform to the
``sleap-roots-contracts`` analysis-input contract (``validate_analysis_input``). The real
post-QC frame is canonicalized on a **copy** only — the frame that feeds
QC/viz/cross-platform is never mutated (the #146/#120 reproduction goldens are the guard).
This previews the #144 load-boundary transform; it does not implement the runtime wiring.

Depends on the ``sleap-roots-contracts[pandas]`` dev dependency; with it installed these
tests run unskipped (no ``importorskip``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from sleap_roots_analyze import get_trait_columns
from sleap_roots_contracts import canonicalize_role_dtypes, validate_analysis_input
from sleap_roots_contracts.examples import (
    analysis_input_example_names,
    load_analysis_input_example,
)

PLATFORMS = ["turface_19", "turface_150", "cylinder", "root_core"]

# Native post-QC role column -> canonical contract role. All four EDPIE platforms carry
# Barcode/Genotype/Replicate; root_core additionally carries Plot (dropped by
# get_trait_columns), and turface_150's Salk_geno is non-numeric (dropped by the trait
# filter).
ROLE_RENAME = {"Genotype": "genotype", "Barcode": "sample_id", "Replicate": "replicate"}
CANONICAL_ROLES = ["genotype", "sample_id", "replicate", "image_path"]


def _build_analysis_input(df: pd.DataFrame) -> pd.DataFrame:
    """Build a contract-shaped analysis-input frame from a post-QC fixture.

    Renames native roles to canonical names, drops non-trait metadata via
    ``get_trait_columns`` — called with the *renamed* role kwargs so the numeric
    ``replicate`` column is excluded instead of leaking in as a duplicate trait — then
    casts role columns to string via ``canonicalize_role_dtypes``. Operates on copies and
    never mutates ``df``.

    This recipe duplicates, in test code, the canonicalization that #144/#153 wire into
    ``validation/input_contract.py`` at the QC load boundary. Once #153 merges, this should
    call that production transform so the two recipes cannot drift; until then it previews
    the transform without implementing the runtime wiring.

    Note: ``get_trait_columns`` only drops the *role* columns it is told about. The contract
    treats every remaining numeric non-role column as a trait, so platform-specific numeric
    metadata (root_core's ``Cid``/``Sid``/``GID``/``Ent``/``Sub``, turface_150's
    ``Entry``/``GID``) flows into the validated "trait" set. Conformance therefore proves the
    frame is contract-*shaped*, not that every trait column is a real measured trait.

    Args:
        df: A post-QC fixture frame with native Barcode/Genotype/Replicate columns.

    Returns:
        A canonicalized analysis-input frame: role columns plus numeric traits.
    """
    renamed = df.rename(columns=ROLE_RENAME)
    roles = [c for c in CANONICAL_ROLES if c in renamed.columns]
    traits = get_trait_columns(
        renamed,
        barcode_col="sample_id",
        genotype_col="genotype",
        replicate_col="replicate",
    )
    return canonicalize_role_dtypes(renamed[roles + traits].copy())


@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
@pytest.mark.parametrize("platform", PLATFORMS)
def test_post_qc_fixture_conforms(final_data_by_platform, platform, strict):
    """Each post-QC fixture validates after canonicalization on a copy.

    Run under both ``strict=False`` and ``strict=True``: all four EDPIE platforms carry the
    recommended ``sample_id`` role, so they satisfy the stronger strict contract too.
    Parametrizing ``strict`` locks that in — a regression that only breaks strict would
    otherwise pass silently.
    """
    check = _build_analysis_input(final_data_by_platform[platform].copy())

    # Rename + trait selection are non-vacuous: roles renamed, native names gone, and no
    # role column leaked back in as a duplicate trait.
    assert {"genotype", "sample_id"}.issubset(check.columns)
    assert "Genotype" not in check.columns and "Barcode" not in check.columns
    assert check.columns.is_unique
    trait_cols = [c for c in check.columns if c not in CANONICAL_ROLES]
    assert trait_cols, "expected at least one numeric trait column"
    # Explicit guard for the get_trait_columns role kwargs (matches the spec wording): a
    # wrong call returns the numeric ``replicate`` as a trait. is_unique above catches the
    # resulting duplicate column; this asserts the underlying intent directly.
    assert "replicate" not in trait_cols

    validate_analysis_input(check, strict=strict).raise_for_status()


@pytest.mark.parametrize("platform", PLATFORMS)
def test_canonicalization_does_not_mutate_fixture(final_data_by_platform, platform):
    """``_build_analysis_input`` is pure — it never mutates its input frame.

    Passes the *raw* shared session frame (deliberately no defensive ``.copy()``) and
    asserts it is byte-identical afterwards. This proves the real protection: the build is
    safe even for a caller that forgets to copy, so it cannot corrupt the session fixture
    the QC/viz/cross-platform reproduction tests also read.
    """
    raw = final_data_by_platform[platform]
    before = raw.copy(deep=True)

    _build_analysis_input(raw)

    pd.testing.assert_frame_equal(raw, before)


def test_canonical_examples_conform():
    """Every packaged canonical example validates as-is (the contract's own truth)."""
    names = analysis_input_example_names()
    assert names, "contract package exposed no canonical examples"
    for name in names:
        example = load_analysis_input_example(name)
        validate_analysis_input(example).raise_for_status()


def test_canonical_examples_conform_strict():
    """Strict mode is a strictly stronger contract, exercised per example.

    Strict promotes the recommended ``sample_id`` role from warning to error, so any example
    carrying ``sample_id`` must still validate, while the genotype-aggregated
    ``genotype_means`` example (no per-sample rows, no ``sample_id``) must fail. Pinning both
    directions keeps the coverage non-vacuous and documents *why* strict and lenient diverge.
    """
    names = analysis_input_example_names()
    assert names, "contract package exposed no canonical examples"
    checked_strict_pass = checked_strict_fail = False
    for name in names:
        example = load_analysis_input_example(name)
        result = validate_analysis_input(example, strict=True)
        if "sample_id" in example.columns:
            result.raise_for_status()
            checked_strict_pass = True
        else:
            assert not result.ok, f"{name} unexpectedly passed strict without sample_id"
            checked_strict_fail = True

    # Both branches must have run, or the test silently checked only one side.
    assert (
        checked_strict_pass
    ), "no sample_id-bearing example exercised the strict-pass path"
    assert checked_strict_fail, "no aggregated example exercised the strict-fail path"


def test_negative_control_validation_can_fail(final_data_by_platform):
    """A frame missing the genotype role fails validation (asserts are non-vacuous)."""
    check = _build_analysis_input(final_data_by_platform["turface_19"].copy())
    bad = check.drop(columns=["genotype"])

    result = validate_analysis_input(bad)

    assert not result.ok
    # Assert the failure is *for the reason we removed* — not some incidental future
    # validation change — so this stays a real negative control.
    with pytest.raises(ValueError, match="genotype"):
        result.raise_for_status()


def test_no_stale_validation_json(repro_fixtures_dir):
    """No ``*_validation.json`` expected files remain in the fixture tree (#147).

    They were removed in 73583f9 — their ``summary`` shape never matched
    ``ValidationResult``; the contract is asserted live, never against stored JSON.
    """
    stale = list(repro_fixtures_dir.rglob("*_validation.json"))
    assert not stale, f"unexpected validation JSON fixtures: {stale}"
