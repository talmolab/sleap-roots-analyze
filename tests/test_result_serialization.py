"""Result-object JSON round-trip gate (issue #133).

The FAIR "interoperability" guarantee: every analytical result object must
serialize cleanly to JSON and round-trip without loss. This gate is **opt-in by
construction** — it calls each *registered* analytical function once and only
asserts when the return is a dataclass result object; functions still returning
plain dicts are skipped, so they begin asserting the moment they adopt a result
dataclass (with no test edits).

Scope caveat (not fully auto-extending): the registered set is the determinism
registry's ``CASES`` (covered by that module's whole-package coverage guard) plus a
hardcoded ``_STATS_RT_CASES`` list. A brand-new result type returned by a function
in *neither* list is not covered until it is added here — there is no package-walking
membership guard for "functions that return a dataclass" the way there is for
``random_state`` functions. Add new result-returning functions to ``RT_CASES``.

"Lossless" is defined on the JSON-native projection produced by
``convert_to_json_serializable`` (which is deliberately asymmetric: ``ndarray`` →
list, unknown objects → ``"<TypeName>"`` string). The gate asserts the projection
survives a ``json.dumps``/``loads`` round-trip unchanged (NaN-aware), that no field
silently degraded to a ``"<TypeName>"`` placeholder, and — when the result type
exposes ``from_dict`` — that it reconstructs an equal object.

NaN/Inf scope: this gate uses default ``json.dumps`` (``allow_nan=True``) because
several result objects legitimately carry NaN (e.g. heritability ``h2``,
reconstruction errors). Strict finite-JSON for the boundary (``allow_nan=False``) is
the individual result type's own contract via its ``to_json`` method, covered by
``test_pca_result`` / ``test_heritability_result`` / ``test_cluster_result``.

See ``docs/reproducibility.md`` for the serialization contract.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

from sleap_roots_analyze import statistics as st
from sleap_roots_analyze.data_utils import convert_to_json_serializable
from sleap_roots_analyze.pipeline.summary import PipelineSummary, StepSummary
from tests.reproducibility_cases import CASES, _silence, make_context

_PLACEHOLDER = re.compile(r"^<[A-Za-z_][A-Za-z0-9_]*>$")


def _find_placeholders(obj, path="root"):
    """Collect ``(path, value)`` for every ``"<TypeName>"`` placeholder string.

    Args:
        obj: A JSON-native structure (dict/list/scalars).
        path: Dotted path accumulated for error messages.

    Returns:
        List of ``(path, value)`` tuples for any lossy-stringified field.
    """
    bad: List = []
    if isinstance(obj, str):
        if _PLACEHOLDER.match(obj):
            bad.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _find_placeholders(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += _find_placeholders(v, f"{path}[{i}]")
    return bad


def _json_equal(a, b) -> bool:
    """Recursively compare two JSON-native structures, NaN-aware.

    Args:
        a: First JSON-native value.
        b: Second JSON-native value.

    Returns:
        True if equal, treating ``NaN == NaN`` as True.
    """
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def assert_round_trips(result) -> dict:
    """Assert a dataclass result object round-trips losslessly through JSON.

    Asserts the JSON-native projection (a) survives ``json.dumps``/``loads``
    unchanged, (b) holds no ``"<TypeName>"`` placeholder (lossy stringification),
    and (c) — when the result type exposes a ``from_dict`` classmethod — that
    ``from_dict`` reconstructs an object whose own projection equals the original
    (the spec's "from_dict reconstructs an equal object" contract).

    Args:
        result: A dataclass instance.

    Returns:
        The decoded JSON-native projection, for optional ``from_dict`` checks.
    """
    projected = convert_to_json_serializable(asdict(result))
    loaded = json.loads(json.dumps(projected))  # must not raise
    placeholders = _find_placeholders(loaded)
    assert not placeholders, f"lossy stringification of fields: {placeholders}"
    # Meaningful stability check: the projection itself must be unchanged by a JSON
    # round-trip (catches any non-JSON-native value json would coerce). Comparing
    # `projected` to `loaded` — not re-encoding `loaded`, which is vacuously equal.
    assert _json_equal(projected, loaded), "projection is not stable through JSON"

    # When the contract's from_dict is present, assert it reconstructs an equal
    # object (otherwise the "from_dict reconstructs an equal object" promise ships
    # dead). Compare on the projection so numpy/Path fields compare cleanly.
    from_dict = getattr(type(result), "from_dict", None)
    if callable(from_dict):
        rebuilt = from_dict(loaded)
        assert _json_equal(
            convert_to_json_serializable(asdict(rebuilt)), loaded
        ), "from_dict did not reconstruct an equal object"
    return loaded


# --- synthetic teeth (exercise the harness before any result type lands) ------


@dataclass
class _SyntheticResult:
    """A stand-in result object covering the field kinds real ones will hold."""

    name: str
    count: int
    score: float
    nan_value: float
    vector: np.ndarray
    nested: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict) -> "_SyntheticResult":
        """Reconstruct from a JSON-native dict (the serialization contract)."""
        return cls(
            name=d["name"],
            count=d["count"],
            score=d["score"],
            nan_value=d["nan_value"],
            vector=np.asarray(d["vector"], dtype=float),
            nested=dict(d["nested"]),
        )


def test_synthetic_result_round_trips_and_reconstructs():
    """The harness round-trips numpy/NaN fields and `from_dict` rebuilds an equal object."""
    obj = _SyntheticResult(
        name="pca",
        count=np.int64(3),
        score=np.float64(1.5),
        nan_value=float("nan"),
        vector=np.array([1.0, 2.0, np.nan]),
        nested={"ratios": [np.float64(0.1), np.float64(0.9)], "n": np.int64(4)},
    )
    loaded = assert_round_trips(obj)

    rebuilt = _SyntheticResult.from_dict(loaded)
    assert rebuilt.name == "pca"
    assert rebuilt.count == 3
    assert rebuilt.score == 1.5
    assert math.isnan(rebuilt.nan_value)
    assert np.allclose(rebuilt.vector, obj.vector, equal_nan=True)
    assert rebuilt.nested["n"] == 4
    assert np.allclose(rebuilt.nested["ratios"], [0.1, 0.9])


def test_lossy_stringification_fails_the_gate():
    """A field the serializer can only stringify must fail, not pass vacuously."""

    class _Unserializable:
        pass

    @dataclass
    class _BadResult:
        model: object

    bad = _BadResult(model=_Unserializable())
    # Sanity: the serializer degrades it to a placeholder string.
    assert convert_to_json_serializable(bad.model) == "<_Unserializable>"
    with pytest.raises(AssertionError, match="lossy stringification"):
        assert_round_trips(bad)


def test_pipeline_summary_round_trips():
    """A shipped serializable dataclass round-trips today (real-object teeth)."""
    step = StepSummary(
        name="load",
        status="success",
        elapsed_time=1.2,
        files_generated=[Path("out/a.csv")],
        metadata={"n_rows": np.int64(10), "frac": np.float64(0.5)},
    )
    summary = PipelineSummary(
        pipeline_name="qc",
        steps=[step],
        config={"seed": np.int64(42)},
        output_directory="out",
    )
    loaded = json.loads(summary.to_json())
    assert not _find_placeholders(loaded)
    # Path and numpy values survived as JSON-native.
    assert loaded["steps"][0]["files_generated"] == ["out/a.csv"]
    assert loaded["steps"][0]["metadata"]["n_rows"] == 10
    assert loaded["config"]["seed"] == 42


def test_windows_paths_normalize_to_posix_on_any_os():
    r"""Path normalization is OS-independent: a Windows-style path serializes to POSIX.

    Regression for #157. On Linux/macOS a producer's old ``str(Path("out/a.csv"))``
    already yields ``"out/a.csv"``, so the manifest looks fine and only Windows CI
    catches the backslash bug. ``PureWindowsPath`` lets us exercise the Windows
    separator on any host: ``str(...)`` would bake in ``out\\a.csv``, while the
    serializer must emit ``out/a.csv`` via ``as_posix()``. The producer-side fix is
    to store the path object (here a ``PureWindowsPath``) and never ``str()`` it.
    """
    win = PureWindowsPath("out", "sub", "a.csv")
    assert str(win) == "out\\sub\\a.csv"  # the bug, if a producer pre-stringifies

    step = StepSummary(
        name="viz",
        status="success",
        files_generated=[win],
        metadata={"dashboard_path": win, "reps_plot": None},
    )
    summary = PipelineSummary(
        pipeline_name="viz",
        steps=[step],
        output_directory=PureWindowsPath("pipeline_runs", "viz_01"),
    )
    loaded = json.loads(summary.to_json())

    assert loaded["steps"][0]["files_generated"] == ["out/sub/a.csv"]
    assert loaded["steps"][0]["metadata"]["dashboard_path"] == "out/sub/a.csv"
    # output_directory normalizes too (regression for #157 review item 2).
    assert loaded["output_directory"] == "pipeline_runs/viz_01"
    # An optional path that was None survives as JSON null, not "None".
    assert loaded["steps"][0]["metadata"]["reps_plot"] is None
    # No backslash leaked anywhere in the serialized manifest.
    assert "\\" not in summary.to_json()


# --- the gate over the registered analytical surface --------------------------


@dataclass(frozen=True)
class RTCase:
    """An analytical function exercised by the round-trip gate.

    Attributes:
        name: Display/id for the case.
        produce: ``produce(env)`` returns the function's result on shared inputs.
    """

    name: str
    produce: Callable


@dataclass
class _Env:
    """Shared inputs for the round-trip gate."""

    ctx: Any
    stats_df: Any
    traits: List[str]


@pytest.fixture(scope="module")
def rt_env() -> _Env:
    """Build the shared stochastic + statistics inputs for the gate."""
    ctx = make_context()
    rng = np.random.RandomState(1)
    geno = np.repeat([f"g{i}" for i in range(6)], 4)
    rep = np.tile([1, 2, 3, 4], 6)
    n = len(geno)
    import pandas as pd

    stats_df = pd.DataFrame(
        {
            "geno": geno,
            "rep": rep,
            "trait_a": np.repeat(rng.randn(6) * 2, 4) + rng.randn(n),
            "trait_b": np.repeat(rng.randn(6) * 2, 4) + rng.randn(n),
        }
    )
    return _Env(ctx=ctx, stats_df=stats_df, traits=["trait_a", "trait_b"])


def _stochastic_rt_cases() -> List[RTCase]:
    """One round-trip case per stochastic function in the shared registry."""
    return [RTCase(c.label, (lambda env, c=c: c.run(env.ctx, 42))) for c in CASES]


# Statistics/heritability functions — the home of #128 `HeritabilityResult`. They
# return dicts today (so they skip), but listing them here means the gate begins
# asserting the moment they adopt a dataclass, with no test edits.
_STATS_RT_CASES: List[RTCase] = [
    RTCase(
        "calculate_trait_statistics",
        lambda env: _silence(
            lambda: st.calculate_trait_statistics(env.stats_df, env.traits)
        ),
    ),
    RTCase(
        "perform_anova_by_genotype",
        lambda env: _silence(
            lambda: st.perform_anova_by_genotype(env.stats_df, env.traits)
        ),
    ),
    RTCase(
        "calculate_heritability_estimates",
        lambda env: _silence(
            lambda: st.calculate_heritability_estimates(env.stats_df, env.traits)
        ),
    ),
]

RT_CASES: List[RTCase] = _stochastic_rt_cases() + _STATS_RT_CASES


@pytest.mark.parametrize("case", RT_CASES, ids=[c.name for c in RT_CASES])
def test_result_round_trip_gate(case, rt_env):
    """Each analytical function's dataclass result round-trips; dicts skip (opt-in)."""
    result = case.produce(rt_env)
    if not is_dataclass(result) or isinstance(result, type):
        pytest.skip(
            f"{case.name} returns {type(result).__name__}; no result object yet"
        )
    assert_round_trips(result)
