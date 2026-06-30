"""Public outlier-removal entry point for analysis-ready trait tables.

This module adds a single tested entry point, :func:`remove_outlier_samples`,
that **detects and removes outlier samples** from a clean (NaN-free) trait table
so PCA/UMAP/clustering can optionally run on outlier-trimmed data. It is the
quality-step follow-up to the minimal-QC entry point
:func:`sleap_roots_analyze.data_cleanup.clean_traits_for_analysis` (#164): that
function makes a frame *runnable* (no NaNs, >=2 samples, >=1 varying trait), and
this one makes it *higher quality* by trimming outliers that distort
principal-component directions even though they are not NaNs.

The function introduces **no new detection or removal algorithm**. It imports and
composes the already-public primitives — ``detect_outliers_mahalanobis`` /
``detect_outliers_isolation_forest`` for detection and ``remove_outliers_from_data``
for row removal (the same functions the QC pipeline's outlier steps use) — so the
detection and index-alignment semantics cannot drift from the pipeline. See the
OpenSpec change ``add-outlier-removal-entry-point`` (issue #165).
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sleap_roots_analyze.data_cleanup import (
    MIN_SAMPLES_FOR_ANALYSIS,
    get_trait_columns,
    validate_clean_traits,
)
from sleap_roots_analyze.data_utils import convert_to_json_serializable

# Imported as module-level names (not called via the ``outlier_detection`` module
# object) so a test can monkeypatch ``outlier_removal.detect_outliers_*`` /
# ``remove_outliers_from_data`` to assert the entry point composes them.
from sleap_roots_analyze.outlier_detection import (
    detect_outliers_isolation_forest,
    detect_outliers_mahalanobis,
    remove_outliers_from_data,
)

# Dispatch table: the public ``method`` keys -> the detector each selects. The
# report records the *key* (e.g. ``"isolation_forest"``), not the detector's own
# internal ``method`` label (``"IsolationForest"``).
_DETECTORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "mahalanobis": detect_outliers_mahalanobis,
    "isolation_forest": detect_outliers_isolation_forest,
}

# Removing more than this fraction of samples almost always signals a mis-set
# contamination/threshold rather than that many real outliers -> UserWarning.
_OVER_REMOVAL_GUARD = 0.5

# Below this sample count the Mahalanobis path's chi-squared tail / covariance
# estimate is statistically fragile, so the default trim deserves a UserWarning.
_SMALL_N_MAHALANOBIS = 30


class OutlierRemovalError(ValueError):
    """Raised when the outlier-trimmed frame is not analysis-ready.

    A subclass of :class:`ValueError` (so callers can keep catching ``ValueError``)
    that additionally carries the assembled :attr:`outlier_report`, so a caller who
    hit an over-aggressive trim can still inspect *what would have been removed*
    even though the result failed a readiness gate.

    Attributes:
        outlier_report: The report dict the entry point assembled before the
            readiness gate failed.
    """

    def __init__(self, message: str, outlier_report: Dict[str, Any]) -> None:
        """Initialize the error with a message and the assembled report.

        Args:
            message: The actionable error message.
            outlier_report: The report dict to attach for caller inspection.
        """
        super().__init__(message)
        self.outlier_report = outlier_report


def _assert_output_analysis_ready(
    trimmed_df: pd.DataFrame,
    trait_cols: List[str],
    outlier_report: Dict[str, Any],
) -> Tuple[int, int]:
    """Re-apply the #164 analysis-readiness gates to the trimmed frame.

    Factored out as a white-box helper so the constant-trait branch — which is not
    reliably reachable through real detection on a frame that varied *before*
    trimming — can be unit-tested directly. Checks run in a fixed order (sample
    count first, then non-constant trait) for a deterministic message.

    Args:
        trimmed_df: The frame after outlier rows were removed.
        trait_cols: The trait column names to gate on.
        outlier_report: The assembled report, attached to any raised error.

    Returns:
        Tuple ``(n_samples, n_nonconstant_traits)`` when both gates pass.

    Raises:
        OutlierRemovalError: If fewer than ``MIN_SAMPLES_FOR_ANALYSIS`` samples
            survive, or if no surviving numeric trait has ``var(ddof=0) > 0``. The
            error carries ``outlier_report``.
    """
    n_samples = len(trimmed_df)
    if n_samples < MIN_SAMPLES_FOR_ANALYSIS:
        raise OutlierRemovalError(
            f"remove_outlier_samples: only {n_samples} sample(s) remain after "
            f"removing {outlier_report['n_outliers']} outlier(s); need at least "
            f"{MIN_SAMPLES_FOR_ANALYSIS} for analysis. Loosen the detection "
            "threshold (e.g. raise chi2_percentile or lower contamination).",
            outlier_report,
        )

    numeric = trimmed_df[trait_cols].select_dtypes(include=[np.number])
    n_nonconstant = int((numeric.var(ddof=0) > 0).sum()) if not numeric.empty else 0
    if n_nonconstant < 1:
        raise OutlierRemovalError(
            "remove_outlier_samples: no non-constant numeric trait remains after "
            "outlier removal; PCA/UMAP/clustering require at least one varying "
            "trait. Loosen the detection threshold.",
            outlier_report,
        )
    return n_samples, n_nonconstant


def remove_outlier_samples(
    clean_df: pd.DataFrame,
    trait_cols: Optional[List[str]] = None,
    *,
    method: str = "mahalanobis",
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    random_state: int = 42,
    **detect_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect and remove outlier samples from a clean, analysis-ready trait table.

    Quality-step follow-up to
    :func:`~sleap_roots_analyze.data_cleanup.clean_traits_for_analysis` (#164): it
    takes the clean (NaN-free) frame that entry point produces and trims outlier
    samples that distort PCA/UMAP/clustering even though they are not NaNs. It
    **composes the existing public primitives** —
    :func:`~sleap_roots_analyze.outlier_detection.detect_outliers_mahalanobis` /
    :func:`~sleap_roots_analyze.outlier_detection.detect_outliers_isolation_forest`
    for detection and
    :func:`~sleap_roots_analyze.outlier_detection.remove_outliers_from_data` for
    removal — and defines no detection or removal logic of its own, so its
    semantics cannot drift from the QC pipeline's outlier steps.

    The function is the opt-in: trimming runs only when a consumer calls it. The
    intended chain is ``clean_traits_for_analysis`` -> (optional)
    ``remove_outlier_samples`` -> ``perform_pca_analysis`` / UMAP / clustering.

    Preconditions (both enforced before any detector runs). The input trait columns
    must be **NaN-free** and the input index must be **unique**. Both are
    correctness guards, not mere defense: the detectors run PCA which silently
    ``dropna()``s rows and reports outlier indices against the post-``dropna``
    index, and removal is label-based — so a NaN-carrying or duplicate-labelled
    input would misalign the indices handed to ``remove_outliers_from_data`` and
    mis-drop rows. Together they guarantee detection labels align one-to-one with
    the input's rows.

    After removal the function re-applies the #164 readiness gates (at least
    ``MIN_SAMPLES_FOR_ANALYSIS`` surviving samples, at least one non-constant
    ``var(ddof=0) > 0`` trait) so the trimmed frame is still safe to hand straight
    to PCA/UMAP/clustering; if a gate fails it raises
    :class:`OutlierRemovalError` (a ``ValueError``) **carrying the report** so the
    caller can still inspect what would have been removed. It emits a
    ``UserWarning`` when the removed fraction is large (> 0.5, before the readiness
    gates — the signature of a mis-set ``contamination``/threshold) and another in
    the ``p > n`` regime (surviving traits outnumber samples), matching
    ``clean_traits_for_analysis``. On the Mahalanobis path it additionally warns when
    the sample count is small (< 30 — fragile chi-squared tail / covariance) and when
    the detector's chi-squared goodness-of-fit reports the distributional assumption
    is violated (so the percentile threshold's meaning is questionable).

    Determinism: ``random_state`` is threaded into the detector and recorded in the
    report; the same ``clean_df`` + ``method`` + ``random_state`` + per-method
    params yield identical ``outlier_indices`` and ``trimmed_df``. The seed is only
    load-bearing on stochastic paths (``method="isolation_forest"``,
    ``robust_covariance=True``, or large-n randomized SVD); the default exact-SVD
    Mahalanobis path is identical regardless of seed.

    Note:
        The default ``method="mahalanobis"`` with ``chi2_percentile=97.5`` trims
        roughly the top 2.5% of samples *by construction* on a well-fit chi-squared
        tail — i.e. it removes ~2.5% even on clean data. Tighten ``chi2_percentile``
        or switch to ``method="isolation_forest"`` with an explicit
        ``contamination`` to control this; the report's ``goodness_of_fit`` surfaces
        when the chi-squared assumption (and thus the percentile's meaning) is
        violated.

    Args:
        clean_df: A clean (NaN-free in the trait columns), unique-indexed wide trait
            table, e.g. the first element returned by ``clean_traits_for_analysis``.
        trait_cols: Trait column names to score. If ``None``, resolved via
            :func:`~sleap_roots_analyze.data_cleanup.get_trait_columns`.
        method: Detection method, either ``"mahalanobis"`` (default — Mahalanobis
            distance on PCA-transformed data with a chi-squared threshold) or
            ``"isolation_forest"``. An unknown value raises ``ValueError``.
        barcode_col: Barcode/plant-ID column. Excluded from inferred traits, and the
            column read to populate ``outlier_barcodes``.
        genotype_col: Genotype column to exclude when inferring traits.
        replicate_col: Replicate column to exclude if present (``None`` if absent).
        random_state: Seed forwarded to the detector for reproducibility.
        **detect_kwargs: Per-method parameters forwarded to the chosen detector
            unchanged — ``contamination`` for ``"isolation_forest"``;
            ``chi2_percentile``, ``variance_threshold``, ``use_chi_squared``,
            ``distance_threshold``, ``robust_covariance`` for ``"mahalanobis"``.
            Detector defaults apply for any not passed.

    Returns:
        Tuple ``(trimmed_df, outlier_report)`` where ``trimmed_df`` is ``clean_df``
        with the flagged outlier rows removed (all columns preserved — outlier
        removal drops rows, never columns) and ``outlier_report`` is an auditable,
        JSON-serializable dict with keys: ``method``, ``method_params``,
        ``random_state``, ``n_input_samples``, ``n_outliers``, ``n_output_samples``,
        ``removal_fraction``, ``outlier_indices``, ``outlier_barcodes``,
        ``threshold_type``, ``threshold_value``, ``n_components``,
        ``variance_threshold``, and ``goodness_of_fit``. The four method-dependent
        keys (``threshold_type``, ``threshold_value``, ``n_components``,
        ``goodness_of_fit``) are populated for Mahalanobis and ``None`` for
        isolation forest (whose control is ``contamination``, in ``method_params``).

    Raises:
        ValueError: On empty input; duplicate column names; explicit ``trait_cols``
            missing from ``clean_df`` or non-numeric; an unknown ``method``; unknown
            or cross-method ``**detect_kwargs`` for the chosen detector (the message
            names the unrecognized keys and the supported set); a non-unique index;
            NaN in the trait columns (the message points to
            ``clean_traits_for_analysis``); or a detector failure (rather than
            silently reporting zero outliers).
        OutlierRemovalError: A ``ValueError`` subclass, raised when trimming leaves
            fewer than ``MIN_SAMPLES_FOR_ANALYSIS`` samples or no non-constant
            trait. It carries the ``outlier_report`` as an attribute.
    """
    # (1) Empty input -> our own message before delegating.
    if clean_df is None or clean_df.shape[0] == 0:
        raise ValueError(
            "remove_outlier_samples: input table has no rows; nothing to trim."
        )

    # (2) Duplicate column names make trait selection ambiguous.
    if clean_df.columns.duplicated().any():
        dups = sorted(set(clean_df.columns[clean_df.columns.duplicated()]))
        raise ValueError(
            f"remove_outlier_samples: duplicate column names in input: {dups}. "
            "Deduplicate columns before trimming."
        )

    # (3) Validate explicit trait_cols up front (actionable, not a bare KeyError).
    if trait_cols is not None:
        missing = [c for c in trait_cols if c not in clean_df.columns]
        if missing:
            raise ValueError(
                f"remove_outlier_samples: trait_cols not found in dataframe: "
                f"{missing}."
            )
        non_numeric = [
            c for c in trait_cols if not pd.api.types.is_numeric_dtype(clean_df[c])
        ]
        if non_numeric:
            raise ValueError(
                "remove_outlier_samples: trait_cols must be numeric; non-numeric "
                f"columns passed: {non_numeric}."
            )

    # (4) Unknown method -> fail fast, naming the supported set.
    if method not in _DETECTORS:
        raise ValueError(
            f"remove_outlier_samples: unknown method {method!r}. Supported methods: "
            f"{sorted(_DETECTORS)}."
        )

    # (5) Unique index -> label-based removal would otherwise mis-drop rows.
    if not clean_df.index.is_unique:
        raise ValueError(
            "remove_outlier_samples: input index must be unique (label-based outlier "
            "removal would otherwise mis-drop rows sharing a flagged label). Reset "
            "the index before trimming."
        )

    # (6) Resolve trait columns if not supplied (same ergonomics as the sibling).
    if trait_cols is None:
        trait_cols = get_trait_columns(
            clean_df,
            barcode_col=barcode_col,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
        )
    if not trait_cols:
        raise ValueError(
            "remove_outlier_samples: no trait columns found to trim. Pass trait_cols "
            "explicitly or check the metadata column names."
        )

    # (7) Clean-input precondition: trait columns must be NaN-free. Wrap the shared
    # validator's message (which names the traits but not this entry point) to add
    # the pointer to clean_traits_for_analysis.
    try:
        validate_clean_traits(clean_df, trait_cols)
    except ValueError as exc:
        raise ValueError(
            f"{exc} Run clean_traits_for_analysis first to produce a NaN-free, "
            "analysis-ready frame before remove_outlier_samples."
        ) from exc

    # (8) Detect with the single selected method. Record the effective per-method
    # params from the detector's (import-bound) signature: caller override or the
    # detector's own default. The actual call goes through the module-level names
    # (not the _DETECTORS dict) so a test can monkeypatch
    # ``outlier_removal.detect_outliers_*`` to exercise the composition / failure path.
    method_params = {
        name: detect_kwargs.get(name, param.default)
        for name, param in inspect.signature(_DETECTORS[method]).parameters.items()
        if name not in ("data", "random_state")
        and param.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    # Reject unknown / cross-method detect_kwargs up front with an actionable error,
    # mirroring the unknown-method guard. The detectors take fixed signatures (no
    # **kwargs), so an unrecognized key (a typo, or an isolation-forest knob passed
    # to the default Mahalanobis method) would otherwise be silently absent from
    # method_params and then raise a bare TypeError leaking the detector's name.
    unknown_kwargs = set(detect_kwargs) - set(method_params)
    if unknown_kwargs:
        raise ValueError(
            f"remove_outlier_samples: unknown detect_kwargs for method {method!r}: "
            f"{sorted(unknown_kwargs)}. Supported for this method: "
            f"{sorted(method_params)}."
        )
    if method == "mahalanobis":
        detect_result = detect_outliers_mahalanobis(
            clean_df[trait_cols], random_state=random_state, **detect_kwargs
        )
    else:  # "isolation_forest" — the only other key validated above.
        detect_result = detect_outliers_isolation_forest(
            clean_df[trait_cols], random_state=random_state, **detect_kwargs
        )

    # Surface a detector failure rather than treating "no indices" as "no outliers".
    if "error" in detect_result or "outlier_indices" not in detect_result:
        raise ValueError(
            "remove_outlier_samples: outlier detection failed: "
            f"{detect_result.get('error', 'detector returned no outlier_indices')}"
        )
    outlier_indices = detect_result["outlier_indices"]

    # (9) Remove the flagged rows (keep_metadata=True preserves Barcode/geno/rep).
    trimmed_df, outlier_df = remove_outliers_from_data(
        clean_df, outlier_indices, keep_metadata=True, return_outliers=True
    )

    # (10) Assemble the auditable report. Method-dependent keys are absent from the
    # isolation-forest detector dict, so ``.get`` yields ``None`` for it.
    n_input = len(clean_df)
    n_output = len(trimmed_df)
    # Derive the count from the trimmed-frame delta (what remove_outliers_from_data
    # actually dropped after its own valid-index filter) rather than trusting
    # len(outlier_indices). The NaN-free + unique-index preconditions guarantee the
    # two agree, so assert it to keep the invariant self-checking.
    n_outliers = n_input - n_output
    assert n_outliers == len(outlier_indices), (
        f"outlier count mismatch: trimmed {n_outliers} rows but detector flagged "
        f"{len(outlier_indices)} labels (precondition violated?)"
    )
    outlier_barcodes = (
        outlier_df[barcode_col].tolist() if barcode_col in clean_df.columns else None
    )
    outlier_report: Dict[str, Any] = {
        "method": method,
        "method_params": method_params,
        "random_state": random_state,
        "n_input_samples": n_input,
        "n_outliers": n_outliers,
        "n_output_samples": n_output,
        "removal_fraction": (n_outliers / n_input) if n_input else 0.0,
        "outlier_indices": outlier_indices,
        "outlier_barcodes": outlier_barcodes,
        "threshold_type": detect_result.get("threshold_type"),
        "threshold_value": detect_result.get("threshold_value"),
        "n_components": detect_result.get("n_components"),
        "variance_threshold": detect_result.get("variance_threshold"),
        "goodness_of_fit": detect_result.get("goodness_of_fit"),
    }
    # Cast every value to plain Python types (numpy scalars/arrays -> int/float/list,
    # nested goodness_of_fit included) so json.dumps(outlier_report) succeeds. Large
    # per-sample arrays were never copied in, so the report stays compact.
    outlier_report = convert_to_json_serializable(outlier_report)

    # (11) Over-removal rail BEFORE the readiness gates, so the warning that points
    # at the over-removal is observable even when the same trim then fails a gate.
    if outlier_report["removal_fraction"] > _OVER_REMOVAL_GUARD:
        warnings.warn(
            f"remove_outlier_samples: removed {n_outliers}/{n_input} samples "
            f"({outlier_report['removal_fraction']:.1%}); a large removal fraction "
            "usually means a mis-set contamination/threshold rather than that many "
            "real outliers. Check the detection parameters.",
            stacklevel=2,
        )

    # (11b) Mahalanobis quality signals (also before the readiness gates, so they
    # are observable even when an aggressive trim then fails a gate). The percentile
    # threshold's meaning rests on the chi-squared fit, which degrades on small n and
    # when the squared-distance distribution does not match chi-squared; the default
    # ~2.5% trim of genuinely-clean data otherwise passes with no signal.
    if method == "mahalanobis":
        if n_input < _SMALL_N_MAHALANOBIS:
            warnings.warn(
                f"remove_outlier_samples: only {n_input} samples on the Mahalanobis "
                f"path (< {_SMALL_N_MAHALANOBIS}); the chi-squared tail and covariance "
                "estimate are statistically fragile at this size, so the flagged "
                "outliers may be unreliable. Inspect the result or use a larger sample.",
                stacklevel=2,
            )
        gof = outlier_report.get("goodness_of_fit")
        if (
            isinstance(gof, dict)
            and gof.get("distributional_assumption_valid") is False
        ):
            warnings.warn(
                "remove_outlier_samples: the squared Mahalanobis distances do not "
                f"fit the chi-squared assumption (goodness-of-fit: "
                f"{gof.get('fit_quality')!r}), so the chi2_percentile threshold's "
                "meaning is questionable; treat the flagged outliers with caution or "
                "use method='isolation_forest' with an explicit contamination.",
                stacklevel=2,
            )

    # (12) Output readiness gates (raise OutlierRemovalError carrying the report).
    _assert_output_analysis_ready(trimmed_df, trait_cols, outlier_report)

    # (13) p > n warning: trimming shrinks n, so an n > p frame can become p > n.
    if n_output < len(trait_cols):
        warnings.warn(
            f"remove_outlier_samples: {n_output} samples < {len(trait_cols)} traits "
            "(p > n) after trimming; multivariate analysis results will be "
            "statistically unreliable. Consider more samples or fewer traits.",
            stacklevel=2,
        )

    return trimmed_df, outlier_report
