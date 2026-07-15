"""Tests for the serializable ``BLUPResult`` dataclass and adapter (#109).

Covers the serializable-result-types contract for BLUP-adjusted genotype
means: native-type JSON round-trips (the reproducibility CI gate), adapter
partition logic (succeeded vs. failed trait columns, including the
cell-level-NaN and zero-succeeded-traits edge cases), the finite-floats
contract, and the non-mutating guarantee. Sibling to
``tests/test_heritability_result.py`` (#128) and ``tests/test_umap_result.py``
(#180).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.result_types import BLUPResult


def _blup_df():
    """A hand-built ``extract_blup_table()``-shaped DataFrame.

    Two succeeded traits (``trait_a``, ``trait_b``), one failed trait
    (``trait_failed``, entirely NaN), two genotypes.
    """
    return pd.DataFrame(
        {
            "trait_a": [10.5, 9.5],
            "trait_b": [21.0, 22.0],
            "trait_failed": [np.nan, np.nan],
        },
        index=["G01", "G02"],
    )


class TestBLUPResultJSON:
    """BLUPResult clean view serializes to native Python types."""

    def test_fields_are_native_types_pre_serialization(self):
        """Fields are native types on the dataclass, not laundered by JSON.

        ``np.float64`` is a subclass of ``float``, so a JSON round-trip
        silently casts a leak to native float before any assertion — assert
        on the fields directly.
        """
        result = BLUPResult.from_blup_table(
            _blup_df(), intercepts={"trait_a": 10.0, "trait_b": 20.0}
        )

        assert all(type(v) is str for v in result.genotype_names)
        assert all(type(v) is str for v in result.trait_names)
        assert all(type(v) is str for v in result.failed_traits)
        assert all(type(v) is float for row in result.adjusted_means for v in row)
        assert all(type(v) is float for v in result.intercepts.values())

    def test_json_roundtrip_native_types(self):
        """BLUPResult round-trips to native types with values preserved."""
        result = BLUPResult.from_blup_table(
            _blup_df(), intercepts={"trait_a": 10.0, "trait_b": 20.0}
        )
        parsed = json.loads(json.dumps(dataclasses.asdict(result)))

        assert all(type(v) is str for v in parsed["genotype_names"])
        assert all(type(v) is str for v in parsed["trait_names"])
        assert all(type(v) is str for v in parsed["failed_traits"])
        assert all(type(v) is float for row in parsed["adjusted_means"] for v in row)
        np.testing.assert_allclose(
            np.asarray(parsed["adjusted_means"]),
            np.asarray(dataclasses.asdict(result)["adjusted_means"]),
        )

    def test_failed_trait_excluded_from_matrix_not_nan(self):
        """A failed (all-NaN) column is excluded from the matrix, not NaN within it."""
        result = BLUPResult.from_blup_table(_blup_df())

        assert "trait_failed" in result.failed_traits
        assert "trait_failed" not in result.trait_names
        assert all(np.isfinite(v) for row in result.adjusted_means for v in row)
        result.to_json()  # succeeds despite the source table having a NaN column

    def test_cell_level_nan_column_classified_as_failed(self):
        """A single cell-level NaN in an otherwise-finite column also fails it."""
        df = pd.DataFrame(
            {
                "trait_a": [10.5, 9.5, np.nan],
                "trait_b": [21.0, 22.0, 11.0],
            },
            index=["G01", "G02", "G03"],
        )

        result = BLUPResult.from_blup_table(df)

        assert "trait_a" in result.failed_traits
        assert "trait_a" not in result.trait_names
        assert result.trait_names == ["trait_b"]

    def test_zero_succeeded_traits_not_misclassified(self):
        """A zero-row DataFrame's columns are all failed, not vacuously finite.

        ``pd.Series([], dtype=float).notna().all()`` is ``True`` in pandas —
        the adapter must special-case a zero-row column as failed rather than
        relying on ``.notna().all()`` alone.
        """
        df = pd.DataFrame({"trait_a": [], "trait_b": []}).astype(float)

        result = BLUPResult.from_blup_table(df)

        assert result.genotype_names == []
        assert result.trait_names == []
        assert set(result.failed_traits) == {"trait_a", "trait_b"}
        result.to_json()  # succeeds

    def test_to_json_rejects_non_finite_adjusted_mean(self):
        """A non-finite adjusted_means value raises at to_json, not to_dict."""
        result = BLUPResult(
            genotype_names=["G01"],
            trait_names=["trait_a"],
            adjusted_means=[[float("nan")]],
        )

        with pytest.raises(ValueError):
            result.to_json()
        result.to_dict()  # does not raise


class TestBLUPResultAdapter:
    """from_blup_table maps the extract_blup_table() DataFrame faithfully."""

    def test_adapter_splits_succeeded_and_failed_columns(self):
        """Finite columns become trait_names; all-NaN columns become failed_traits."""
        result = BLUPResult.from_blup_table(_blup_df())

        assert result.trait_names == ["trait_a", "trait_b"]
        assert result.failed_traits == ["trait_failed"]
        assert len(result.adjusted_means) == 2
        assert all(len(row) == 2 for row in result.adjusted_means)

    def test_genotype_names_preserves_row_order(self):
        """genotype_names equals the DataFrame's row order."""
        result = BLUPResult.from_blup_table(_blup_df())
        assert result.genotype_names == ["G01", "G02"]

    def test_intercepts_covers_exactly_succeeded_traits(self):
        """Intercepts has one entry per trait_names name, none for failed_traits."""
        result = BLUPResult.from_blup_table(
            _blup_df(),
            intercepts={"trait_a": 10.0, "trait_b": 20.0, "trait_failed": 0.0},
        )
        assert result.intercepts == {"trait_a": 10.0, "trait_b": 20.0}

    def test_intercepts_defaults_to_empty_when_not_supplied(self):
        """Intercepts defaults to {} when the caller omits it."""
        result = BLUPResult.from_blup_table(_blup_df())
        assert result.intercepts == {}

    def test_adapter_does_not_mutate_input_dataframe(self):
        """from_blup_table does not mutate the input DataFrame."""
        df = _blup_df()
        before = df.copy(deep=True)

        BLUPResult.from_blup_table(df, intercepts={"trait_a": 10.0, "trait_b": 20.0})

        pd.testing.assert_frame_equal(df, before)
