"""Serializable stdlib dataclass result types for analytical functions.

This module is the shared home for the serializable-result-types epic (#130):
flat ``@dataclass`` views that hold **only JSON-serializable science** — native
Python scalars and lists, never ``numpy`` arrays, ``pandas`` frames, or sklearn
objects. Because every field is JSON-native, ``json.dumps(asdict(result))``
round-trips with no custom serializer, which is what lets these results cross a
JSON boundary (bloom-mcp, a cached artifact, an API) and anchor the
reproducibility CI gate.

**Finite-floats contract.** The JSON boundary is *strict* JSON: floats must be
finite. ``json.dumps`` will happily emit the non-standard ``NaN``/``Infinity``
tokens (which a strict consumer such as bloom-mcp rejects), so each result type
exposes a ``to_json`` method that serializes with ``allow_nan=False`` — a
degenerate science value (e.g. an all-zero-loadings PCA producing a ``NaN``
fractional contribution, a non-finite ``h2``/``bic``) raises a ``ValueError`` at
serialization rather than silently producing invalid JSON. Callers crossing the
boundary should use ``to_json`` (or pass ``allow_nan=False`` themselves).

The first type is :class:`PCAResult` (issue #127, the detailed exemplar);
``HeritabilityResult`` (#128) and ``ClusterResult`` (#129) follow the same
convention here. This module imports nothing from the analytical modules
(``pca.py`` etc.) so dependencies stay one-way.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = [
    "FeatureContribution",
    "PCAResult",
    "TraitHeritability",
    "HeritabilityResult",
]

# Key the heritability dict reserves for calculation metadata (not a trait).
_HERITABILITY_METADATA_KEY = "__calculation_metadata__"


@dataclass(frozen=True)
class FeatureContribution:
    """A single feature's contribution to the retained PCA components.

    Attributes:
        feature: Feature (column) name the contribution belongs to.
        total_contribution: Sum over retained components of
            ``loading**2 * eigenvalue`` for this feature (variance contributed).
        fractional_contribution: ``total_contribution`` normalized to sum to 1
            across all features.
    """

    feature: str
    total_contribution: float
    fractional_contribution: float


@dataclass(frozen=True)
class PCAResult:
    """JSON-serializable view of a PCA run (science only, no sklearn objects).

    Built from the legacy ``perform_pca_analysis`` dict via
    :meth:`from_pca_dict`. Component-indexed fields cover only the retained
    components; the fitted ``PCA``/``StandardScaler`` objects are intentionally
    excluded (still available via the legacy dict for in-process callers), as are
    the secondary per-feature variance breakdowns
    (``explained_variance_ratio_per_feature``, ``feature_variances``) — this view
    keeps only the primary component-level science.

    ``frozen=True`` is shallow: the dataclass fields cannot be rebound, but the
    nested ``list`` fields (``loadings``, ``scores``, ``feature_contributions``)
    are still mutable in place. Treat the result as read-only. Float fields must be
    finite to satisfy the JSON boundary — use :meth:`to_json` to enforce it.

    Attributes:
        n_components: Number of retained principal components.
        feature_names: Feature (column) names in their original order; row order
            of ``loadings``.
        explained_variance_ratio: Per-component fraction of total variance
            explained, length ``n_components``.
        eigenvalues: Per-component eigenvalues, length ``n_components``.
        cumulative_variance_ratio: Per-component cumulative sum of
            ``explained_variance_ratio``, length ``n_components``.
        loadings: ``(n_features, n_components)`` nested list of component
            loadings.
        scores: ``(n_samples, n_components)`` nested list of transformed sample
            coordinates.
        standardized: Whether the data was standardized (a ``StandardScaler``
            was fitted) before PCA.
        random_state: Random state used for the run, stamped for reproducibility
            provenance; ``None`` if not supplied to the adapter.
        explained_variance_threshold: Cumulative-variance threshold used for
            component selection, stamped for provenance; ``None`` if not
            supplied to the adapter.
        feature_contributions: Per-feature contributions, ordered by
            ``total_contribution`` descending.
    """

    n_components: int
    feature_names: list[str]
    explained_variance_ratio: list[float]
    eigenvalues: list[float]
    cumulative_variance_ratio: list[float]
    loadings: list[list[float]]
    scores: list[list[float]]
    standardized: bool
    random_state: Optional[int] = None
    explained_variance_threshold: Optional[float] = None
    feature_contributions: list[FeatureContribution] = field(default_factory=list)

    @property
    def cumulative_variance(self) -> float:
        """Total fraction of variance explained by the retained components."""
        return float(sum(self.explained_variance_ratio))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` view via :func:`dataclasses.asdict`."""
        return dataclasses.asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to a strict-JSON string, enforcing the finite-floats contract.

        Defaults to ``allow_nan=False`` so a non-finite value (``NaN``/``Infinity``
        from a degenerate run) raises a ``ValueError`` here rather than emitting the
        non-standard tokens that a strict JSON consumer (e.g. bloom-mcp) rejects.
        Extra keyword arguments are forwarded to :func:`json.dumps`.

        Raises:
            ValueError: If any float field is non-finite (under the default
                ``allow_nan=False``).
        """
        kwargs.setdefault("allow_nan", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_pca_dict(
        cls,
        d: dict,
        *,
        random_state: Optional[int] = None,
        explained_variance_threshold: Optional[float] = None,
    ) -> "PCAResult":
        """Build a :class:`PCAResult` from a ``perform_pca_analysis`` dict.

        Assumes the canonical key set returned by ``perform_pca_analysis``
        (``n_components_selected``, ``feature_names``, ``loadings``,
        ``eigenvalues``, ``explained_variance_ratio``,
        ``cumulative_variance_ratio``, ``transformed_data``, ``scaler``,
        ``feature_contributions``). Does not mutate ``d``.

        Provenance caveat: ``random_state`` and ``explained_variance_threshold``
        are *stamped as supplied* — the source dict does not carry them, so they
        are recorded on trust, not cross-checked. Pass the **same** values that
        produced ``d`` (e.g. the ``random_state`` handed to ``perform_pca_analysis``);
        a mismatched seed creates a false-but-authoritative reproducibility record.

        Args:
            d: The dict returned by ``perform_pca_analysis``.
            random_state: Random state to stamp into the result for provenance.
                Must match the seed used to produce ``d`` (see caveat above).
            explained_variance_threshold: Threshold to stamp into the result for
                provenance. Must match the threshold used to produce ``d``.

        Returns:
            A frozen :class:`PCAResult` holding only JSON-serializable science.
        """
        n = int(d["n_components_selected"])

        fc_df = d.get("feature_contributions")
        feature_contributions: list[FeatureContribution] = []
        if fc_df is not None:
            for idx, row in fc_df.iterrows():
                feature_contributions.append(
                    FeatureContribution(
                        feature=str(idx),
                        total_contribution=float(row["total_contribution"]),
                        fractional_contribution=float(row["fractional_contribution"]),
                    )
                )

        return cls(
            n_components=n,
            feature_names=[str(name) for name in d["feature_names"]],
            explained_variance_ratio=np.asarray(d["explained_variance_ratio"])[
                :n
            ].tolist(),
            eigenvalues=np.asarray(d["eigenvalues"])[:n].tolist(),
            cumulative_variance_ratio=np.asarray(d["cumulative_variance_ratio"])[
                :n
            ].tolist(),
            loadings=np.asarray(d["loadings"])[:, :n].tolist(),
            scores=np.asarray(d["transformed_data"])[:, :n].tolist(),
            standardized=d.get("scaler") is not None,
            random_state=random_state,
            explained_variance_threshold=explained_variance_threshold,
            feature_contributions=feature_contributions,
        )


@dataclass(frozen=True)
class TraitHeritability:
    """A single trait's heritability estimate.

    The source ``calculate_heritability_estimates`` computes
    ``H² = var_G / (var_G + var_E / mean_n_reps)`` — a genotype-mean
    (repeatability) heritability, **not** the textbook broad-sense
    ``var_G / (var_G + var_E)``. Because ``var_E`` is divided by the mean number
    of replicates, ``h2`` runs higher than broad-sense H² but preserves the
    genetic-variance ordering across traits.

    Attributes:
        trait: Trait (column) name.
        h2: Genotype-mean heritability estimate in ``[0, 1]`` (see above — not
            textbook broad-sense).
        passed_threshold: Whether ``h2`` meets the result's threshold.
        var_genetic: Genetic variance component (σ²_G).
        var_residual: Residual/environmental variance component (σ²_E).
        n_genotypes: Number of genotypes used for this trait.
        n_observations: Number of observations used for this trait.
        model_type: Model used for this trait (e.g. ``"mixed_model"``,
            ``"anova_based"``, ``"no_variance"``).
    """

    trait: str
    h2: float
    passed_threshold: bool
    var_genetic: float
    var_residual: float
    n_genotypes: int
    n_observations: int
    model_type: str


@dataclass(frozen=True)
class HeritabilityResult:
    """JSON-serializable view of a heritability run (science only).

    Built from the legacy ``calculate_heritability_estimates`` dict via
    :meth:`from_heritability_dict`. Traits that failed estimation are listed in
    ``failed_traits`` rather than ``per_trait``; no statsmodels model object is
    retained.

    ``frozen=True`` is shallow: the fields cannot be rebound, but the nested
    ``per_trait`` / ``failed_traits`` lists are still mutable in place — treat the
    result as read-only. Float fields must be finite for the JSON boundary; use
    :meth:`to_json` to enforce it.

    Attributes:
        method: Estimation method applied to all traits (e.g.
            ``"mixed_model"``).
        threshold: H² threshold used to classify ``passed_threshold``.
        per_trait: Per-trait heritability estimates that succeeded.
        failed_traits: Names of traits whose estimation errored or produced no
            heritability value.
        error: Run-level error message when the whole calculation short-circuited
            (e.g. missing required columns) and produced no per-trait results;
            ``None`` for a normal run. Distinct from ``failed_traits``, which holds
            individual traits that failed within an otherwise-successful run.
    """

    method: str
    threshold: float
    per_trait: list[TraitHeritability] = field(default_factory=list)
    failed_traits: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def mean_h2(self) -> float:
        """Mean H² over successful traits (``0.0`` when there are none)."""
        if not self.per_trait:
            return 0.0
        return float(sum(t.h2 for t in self.per_trait) / len(self.per_trait))

    @property
    def n_above_threshold(self) -> int:
        """Number of successful traits whose H² meets the threshold."""
        return sum(1 for t in self.per_trait if t.passed_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` view via :func:`dataclasses.asdict`."""
        return dataclasses.asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to a strict-JSON string, enforcing the finite-floats contract.

        Defaults to ``allow_nan=False`` so a non-finite value (e.g. a degenerate
        ``h2``) raises a ``ValueError`` here rather than emitting the non-standard
        ``NaN``/``Infinity`` tokens a strict consumer (bloom-mcp) rejects. Extra
        keyword arguments are forwarded to :func:`json.dumps`.

        Raises:
            ValueError: If any float field is non-finite (under the default
                ``allow_nan=False``).
        """
        kwargs.setdefault("allow_nan", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_heritability_dict(cls, d: dict, threshold: float) -> "HeritabilityResult":
        """Build a :class:`HeritabilityResult` from a heritability dict.

        Accepts the ``calculate_heritability_estimates`` return dict (the
        ``remove_low_h2=False`` form, or the first element of the
        ``remove_low_h2=True`` tuple). Does not mutate ``d``.

        A run-level short-circuit — ``{"error": "..."}`` with no per-trait entries,
        returned when the calculation aborts before processing any trait (e.g.
        missing required columns) — is surfaced on ``error``, not mistaken for a
        trait named ``"error"``. The string value distinguishes it from a per-trait
        failure (whose value is always a dict).

        Provenance caveat: ``threshold`` is applied *as supplied* — the source dict
        does not carry it, so it is recorded on trust, not cross-checked. Pass the
        threshold you intend the ``passed_threshold`` classification to reflect.

        Args:
            d: Trait-keyed heritability dict, optionally including the
                ``__calculation_metadata__`` entry.
            threshold: H² threshold used to set each trait's
                ``passed_threshold`` (see provenance caveat above).

        Returns:
            A frozen :class:`HeritabilityResult` holding only serializable
            science.
        """
        metadata = d.get(_HERITABILITY_METADATA_KEY) or {}
        method = str(metadata.get("method_used_for_all_traits", "unknown"))
        threshold = float(threshold)

        # Run-level short-circuit: a lone {"error": "<str>"} (no metadata, no
        # per-trait entries). The string value distinguishes it from a per-trait
        # failure (always a dict), so a trait legitimately named "error" is unaffected.
        run_error = d.get("error")
        if isinstance(run_error, str):
            return cls(method=method, threshold=threshold, error=run_error)

        per_trait: list[TraitHeritability] = []
        failed_traits: list[str] = []
        for trait, entry in d.items():
            if trait == _HERITABILITY_METADATA_KEY:
                continue
            if not isinstance(entry, dict) or "heritability" not in entry:
                failed_traits.append(str(trait))
                continue
            h2 = float(entry["heritability"])
            per_trait.append(
                TraitHeritability(
                    trait=str(trait),
                    h2=h2,
                    passed_threshold=h2 >= threshold,
                    var_genetic=float(entry.get("var_genetic", 0.0)),
                    var_residual=float(entry.get("var_residual", 0.0)),
                    n_genotypes=int(entry.get("n_genotypes", 0)),
                    n_observations=int(entry.get("n_observations", 0)),
                    model_type=str(entry.get("model_type", "unknown")),
                )
            )

        return cls(
            method=method,
            threshold=threshold,
            per_trait=per_trait,
            failed_traits=failed_traits,
        )
