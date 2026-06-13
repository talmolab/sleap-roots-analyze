"""Result-object JSON round-trip gate (issue #133).

The FAIR "interoperability" guarantee: every analytical result object must
serialize cleanly to JSON and round-trip without loss. This gate is **opt-in by
construction** — it calls each analytical function once and only asserts when the
return is a dataclass result object; functions still returning plain dicts are
skipped. It therefore extends automatically as the #127/#128/#129 result types
land, with no test edits.

"Lossless" is defined on the JSON-native projection produced by
``convert_to_json_serializable`` (which is deliberately asymmetric: ``ndarray`` →
list, unknown objects → ``"<TypeName>"`` string). The gate asserts the projection
survives a ``json.dumps``/``loads`` round-trip unchanged (NaN-aware) and that no
field silently degraded to a ``"<TypeName>"`` placeholder — so a result holding an
unserializable object fails loudly instead of passing vacuously.

See ``docs/reproducibility.md`` for the serialization contract.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

from sleap_roots_analyze import statistics as st
from sleap_roots_analyze.data_utils import convert_to_json_serializable
from sleap_roots_analyze.pipeline.summary import PipelineSummary, StepSummary
from tests.reproducibility_cases import CASES, make_context

_PLACEHOLDER = re.compile(r"^<[A-Za-z_][A-Za-z0-9_]*>$")


def _silence(fn):
    """Run ``fn`` with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


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

    Args:
        result: A dataclass instance.

    Returns:
        The decoded JSON-native projection, for optional ``from_dict`` checks.
    """
    projected = convert_to_json_serializable(asdict(result))
    loaded = json.loads(json.dumps(projected))  # must not raise
    placeholders = _find_placeholders(loaded)
    assert not placeholders, f"lossy stringification of fields: {placeholders}"
    assert _json_equal(
        json.loads(json.dumps(loaded)), loaded
    ), "projection is not stable across a JSON round-trip"
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
    assert _json_equal(json.loads(json.dumps(loaded)), loaded)
    # Path and numpy values survived as JSON-native.
    assert loaded["steps"][0]["files_generated"] == ["out/a.csv"]
    assert loaded["steps"][0]["metadata"]["n_rows"] == 10
    assert loaded["config"]["seed"] == 42


# --- the auto-extending gate over the analytical surface ----------------------


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
