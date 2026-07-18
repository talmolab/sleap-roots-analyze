"""Cross-platform genotype-effect prediction: LOGO-CV ridge/PLS machinery.

Tier 3 of the wheat EDPIE cross-platform genotype-prediction program (#194).
Reframes the cross-platform result from *correlation* to *predictability*:
given genotype BLUPs (or raw genotype means) estimated within one platform,
this module tests whether they predict genotype effects in another platform
via ridge regression / Partial Least Squares (PLS) with leave-one-genotype-out
(LOGO) cross-validation.

See the program's grounding document (``theory.md``, referenced from
``openspec/changes/add-cross-platform-prediction/design.md``) for the
CV-hygiene contract this module implements against.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_VALID_REDUCTION_METHODS = ("pls_latent", "representatives", "pc1")


def fit_pca_on_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """Fit PCA on training data only, project held-out data onto it.

    A fresh ``sklearn.decomposition.PCA`` is fit on ``X_train`` only; the
    resulting components are used to project ``X_test``. This is
    deliberately distinct from the pipeline-level ``PCA`` step in ``pca.py``:
    that step is fit on every genotype before a leave-one-genotype-out (LOGO)
    fold loop runs, so reusing it here would leak the held-out genotype's
    position into the component loadings.

    Args:
        X_train: Training-fold data, shape ``(n_train, n_traits)``.
        X_test: Held-out data to project, shape ``(n_test, n_traits)``.
        n_components: Number of principal components to fit/project onto.

    Returns:
        ``X_test`` projected onto the ``n_components`` components fit from
        ``X_train``, shape ``(n_test, n_components)``.

    Raises:
        ValueError: If ``X_train`` has fewer traits (columns) than
            ``n_components``.
    """
    if X_train.shape[1] < n_components:
        raise ValueError(
            f"X_train has {X_train.shape[1]} traits, fewer than "
            f"n_components={n_components}."
        )
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    return pca.transform(X_test)


@dataclass
class LOGOCVResult:
    """Leave-one-genotype-out cross-validation result for one prediction target.

    Attributes:
        genotypes: Genotype labels, in the same order as ``y_true``/``y_pred``.
        y_true: Observed target values, one per genotype.
        y_pred: Leave-one-genotype-out predicted values, one per genotype
            (``y_pred[i]`` is the prediction for genotype ``i`` when genotype
            ``i`` was held out).
        r2: Aggregate R^2 over the concatenated leave-one-out predictions.
        rmse: Aggregate Root Mean Squared Error over the same predictions.
            Not comparable across ``LOGOCVResult`` instances built from
            differently-scaled traits.
        spearman_rho: Aggregate Spearman rank correlation over the same
            predictions.
        spearman_p: ``spearman_rho``'s p-value. An asymptotic approximation
            (``scipy.stats.spearmanr``'s default); scipy's own documentation
            states this p-value "is only accurate for very large samples
            (>500 observations)" -- descriptive, not hypothesis-test-grade,
            at this program's n~=19.
    """

    genotypes: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    r2: float
    rmse: float
    spearman_rho: float
    spearman_p: float


def logo_cv_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    genotypes: Sequence[str],
    reduction_method: str = "pls_latent",
    representative_names: Optional[Sequence[str]] = None,
) -> LOGOCVResult:
    """Predict each genotype's target value via leave-one-genotype-out CV.

    Implements the CV-hygiene contract: a fresh ``sklearn.pipeline.Pipeline``
    is instantiated and fit inside each fold, so no step ever sees the
    held-out genotype during fit. Aggregate R^2, RMSE, and Spearman rho are
    computed once, over the concatenated leave-one-out predictions across all
    folds (not as a separate score per single held-out genotype, which is
    statistically undefined for R^2/rho at n_test=1).

    Precondition (not verifiable from ``X`` alone): ``X``'s columns must never
    include the target trait's own values -- callers are responsible for
    excluding the prediction target from the predictor matrix.

    Args:
        X: Predictor matrix, shape ``(n_genotypes, n_traits)``, columns named
            by trait (no duplicate column names), index by genotype label.
        y: Target values, one per genotype, same order as ``X``'s rows.
        genotypes: Genotype labels, same order as ``X``'s rows. Must not
            contain duplicates -- see ``Raises`` below.
        reduction_method: One of ``"pls_latent"`` (default) -- a
            ``StandardScaler`` + ``PLSRegression(n_components=1)`` pipeline
            fit directly on the full trait matrix; ``"representatives"`` --
            ``X`` reduced to ``representative_names`` before a
            ``StandardScaler`` + ``Ridge()`` (default ``alpha=1.0``, an
            accepted but undiscussed choice) pipeline; ``"pc1"`` -- ``X``
            reduced to a single principal-component score computed per fold
            via :func:`fit_pca_on_fold`, before a ``StandardScaler`` +
            ``Ridge()`` pipeline.
        representative_names: Trait (column) names to select when
            ``reduction_method="representatives"``. Required, must be
            non-empty and duplicate-free, for that method, and every name
            must be present in ``X``'s columns; ignored otherwise.

    Returns:
        A :class:`LOGOCVResult` with per-genotype predictions and aggregate
        R^2/RMSE/Spearman rho.

    Note:
        ``n_genotypes=3`` (the minimum this function accepts) is a
        degenerate/saturated regime, not merely a noisy one: each LOGO fold
        then trains on exactly 2 genotypes, and 2 points give
        ``PLSRegression(n_components=1)`` zero residual degrees of freedom --
        it exactly reproduces both training targets every fold. Results at
        or near this boundary should not be trusted quantitatively; the
        `n>=3` guard only guarantees the function runs, not that its output
        is statistically meaningful at the boundary.

    Raises:
        ValueError: If ``X`` is not a ``pandas.DataFrame``; if
            ``reduction_method`` is not one of the three valid values; if
            ``X``, ``y``, and ``genotypes`` have mismatched lengths; if
            ``genotypes`` contains duplicate labels (this would silently
            leave a held-out genotype's other row in its own training fold,
            defeating the leave-one-genotype-out contract entirely); if
            fewer than 3 genotypes are provided (each LOGO-CV fold's training
            set needs at least 2 genotypes -- ``PLSRegression``'s own
            minimum -- so 2 total genotypes is not enough, even though it
            looks superficially sufficient to form one fold); if
            ``reduction_method="representatives"`` and
            ``representative_names`` is ``None``/empty, contains duplicates,
            or contains a name absent from ``X``'s columns; if ``X`` contains
            duplicate column names; if ``X`` contains a non-numeric column;
            or if ``X`` or ``y`` contains any ``NaN`` value.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pandas.DataFrame, got {type(X).__name__}")
    if reduction_method not in _VALID_REDUCTION_METHODS:
        raise ValueError(
            f"reduction_method must be one of {_VALID_REDUCTION_METHODS}, "
            f"got {reduction_method!r}"
        )
    if len(X) != len(y) or len(X) != len(genotypes):
        raise ValueError(
            "X, y, and genotypes must have the same length: "
            f"got {len(X)}, {len(y)}, {len(genotypes)}"
        )
    genotype_counts = Counter(genotypes)
    duplicate_genotypes = sorted(g for g, count in genotype_counts.items() if count > 1)
    if duplicate_genotypes:
        raise ValueError(
            "genotypes contains duplicate labels, which would silently "
            "defeat leave-one-genotype-out cross-validation (a held-out "
            f"genotype's other row would remain in its own training fold): "
            f"{duplicate_genotypes}"
        )
    if len(genotypes) < 3:
        raise ValueError(
            "logo_cv_predict requires at least 3 genotypes for "
            "leave-one-genotype-out cross-validation: each fold's training "
            "set must have at least 2 genotypes (PLSRegression's own minimum), "
            "so 2 total genotypes (1 per training fold) is not enough"
        )
    if reduction_method == "representatives":
        if not representative_names:
            raise ValueError(
                "representative_names is required and must be non-empty when "
                "reduction_method='representatives'"
            )
        rep_name_counts = Counter(representative_names)
        duplicate_rep_names = sorted(
            name for name, count in rep_name_counts.items() if count > 1
        )
        if duplicate_rep_names:
            raise ValueError(
                f"representative_names contains duplicate entries: "
                f"{duplicate_rep_names}"
            )
        unknown_names = [name for name in representative_names if name not in X.columns]
        if unknown_names:
            raise ValueError(
                f"representative_names contains names not present in X's "
                f"columns: {unknown_names}"
            )
        # `representative_names` is confirmed non-None/non-empty above; compute
        # here (not in a second, separate `if reduction_method ==
        # "representatives":` block below) so static type checkers can narrow
        # it from `Optional[Sequence[str]]` to `Sequence[str]` within this one
        # scope.
        rep_names: list[str] = list(representative_names)

    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"X contains duplicate column name(s): {duplicate_cols}")

    non_numeric_cols = [
        col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
    ]
    if non_numeric_cols:
        raise ValueError(f"X contains non-numeric column(s): {non_numeric_cols}")

    X_values = X.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isnan(X_values).any():
        raise ValueError("X contains NaN values")
    if np.isnan(y).any():
        raise ValueError("y contains NaN values")

    n = len(genotypes)
    y_pred = np.full(n, np.nan)
    loo = LeaveOneOut()

    if reduction_method == "representatives":
        rep_values = X[rep_names].to_numpy(dtype=float)

    for train_idx, test_idx in loo.split(X_values):
        y_train = y[train_idx]

        if reduction_method == "pls_latent":
            X_train, X_test = X_values[train_idx], X_values[test_idx]
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression(n_components=1)),
                ]
            )
            pipe.fit(X_train, y_train)
            y_pred[test_idx] = pipe.predict(X_test).ravel()
        elif reduction_method == "representatives":
            X_train, X_test = rep_values[train_idx], rep_values[test_idx]
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            pipe.fit(X_train, y_train)
            y_pred[test_idx] = pipe.predict(X_test).ravel()
        else:  # pc1
            X_train_full, X_test_full = X_values[train_idx], X_values[test_idx]
            X_train_reduced = fit_pca_on_fold(
                X_train_full, X_train_full, n_components=1
            )
            X_test_reduced = fit_pca_on_fold(X_train_full, X_test_full, n_components=1)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            pipe.fit(X_train_reduced, y_train)
            y_pred[test_idx] = pipe.predict(X_test_reduced).ravel()

    r2 = float(r2_score(y, y_pred))
    rmse = float(root_mean_squared_error(y, y_pred))
    spearman_rho, spearman_p = spearmanr(y, y_pred)

    return LOGOCVResult(
        genotypes=list(genotypes),
        y_true=y,
        y_pred=y_pred,
        r2=r2,
        rmse=rmse,
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
    )


def top_quartile_recovery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: Optional[int] = None,
) -> float:
    """Fraction of the true top-``q`` genotypes recovered in the predicted top-``2q``.

    Tier 4 (#200) metric: ranks genotypes by ``y_true`` and by ``y_pred``
    independently, then measures what fraction of the true top-``q`` set
    (by ``y_true``) also appears in the *predicted* top-``2q`` set (by
    ``y_pred``) -- a wider predicted window than the true one, since a
    ranking-based recovery metric at n~=19 is too noisy to expect exact
    top-``q``-for-top-``q`` agreement.

    Chance-level (random ``y_pred``) expected recovery is ``2*q/n`` by
    linearity of expectation, not a fixed "25%" -- see design.md Decision 11
    for the derivation. Used both for the observed value (real ``y``, real
    LOGO-CV predictions) and, once per permutation inside
    :func:`permutation_test`, for the null distribution.

    Args:
        y_true: Observed target values, one per genotype.
        y_pred: Predicted target values, same order as ``y_true``.
        q: Size of the true top-quartile set. Defaults to
            ``max(1, round(len(y_true) / 4))`` -- clamped to at least 1 so a
            small ``n`` never produces a vacuous, zero-size window. An
            explicitly-supplied ``q`` is validated strictly (must be
            positive and satisfy ``2 * q <= len(y_true)``); the computed
            default is never invalid at this program's real n>=3 scale (or
            any ``n>=2``) but is only well-defined for ``n>=2`` -- see
            ``Raises`` below.

    Returns:
        The fraction (in ``[0, 1]``) of the true top-``q`` genotypes present
        in the predicted top-``2q`` genotypes.

    Raises:
        ValueError: If an explicitly-supplied ``q`` is not positive, or if
            ``2 * q`` exceeds ``len(y_true)``; or if ``len(y_true) < 2``
            (below this, even the default ``q`` would violate
            ``2 * q <= len(y_true)``).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if q is None:
        if n < 2:
            raise ValueError(
                f"len(y_true)={n} is too small for a default q: even q=1 "
                "would violate 2*q <= len(y_true); provide an explicit "
                "valid q (impossible below n=2) or use a larger input"
            )
        q = max(1, round(n / 4))
    elif q <= 0 or 2 * q > n:
        raise ValueError(
            f"q must be a positive integer with 2*q <= len(y_true) ({n}), " f"got q={q}"
        )

    top_q_true = set(np.argsort(-y_true, kind="stable")[:q])
    top_2q_pred = set(np.argsort(-y_pred, kind="stable")[: 2 * q])
    return len(top_q_true & top_2q_pred) / q


@dataclass
class PermutationTestResult:
    """Permutation-null significance test result for one prediction target.

    Returned by :func:`permutation_test`. Deliberately does not carry a
    ``target_name`` field -- this type mirrors :class:`LOGOCVResult`'s role
    as the analytical module's own "raw" result, with no knowledge of which
    named target it was computed for (that context is attached only by
    ``result_types.CrossPlatformPermutationResult.from_permutation_test_results``,
    exactly as ``TargetPrediction`` attaches ``target_name`` to a plain
    ``LOGOCVResult``).

    Attributes:
        observed_r2: Aggregate R^2 from one ``logo_cv_predict()`` call on the
            real (unshuffled) ``y``.
        observed_rmse: Aggregate RMSE from the same observed call.
        observed_spearman_rho: Aggregate Spearman rho from the same observed
            call.
        observed_top_quartile_recovery: :func:`top_quartile_recovery`
            computed from the same observed call's predictions.
        null_r2: R^2 for each of ``n_permutations`` shuffled-``y`` calls.
        null_rmse: RMSE for each shuffled-``y`` call.
        null_spearman_rho: Spearman rho for each shuffled-``y`` call.
        null_top_quartile_recovery: :func:`top_quartile_recovery` for each
            shuffled-``y`` call, using that permutation's own shuffled ``y``
            as ground truth (not the original, unshuffled ``y``).
        p_value_r2: One-sided p-value, right-tailed (higher R^2 is better).
        p_value_rmse: One-sided p-value, **left-tailed** (lower RMSE is
            better -- the opposite convention from R^2/rho). Do not read a
            low ``p_value_rmse`` as indicating a bad fit.
        p_value_spearman_rho: One-sided p-value, right-tailed, same
            direction as ``p_value_r2``.
        n_permutations: Number of permutations run (length of every
            ``null_*`` array).
    """

    observed_r2: float
    observed_rmse: float
    observed_spearman_rho: float
    observed_top_quartile_recovery: float
    null_r2: np.ndarray
    null_rmse: np.ndarray
    null_spearman_rho: np.ndarray
    null_top_quartile_recovery: np.ndarray
    p_value_r2: float
    p_value_rmse: float
    p_value_spearman_rho: float
    n_permutations: int


def permutation_test(
    X: pd.DataFrame,
    y: np.ndarray,
    genotypes: Sequence[str],
    reduction_method: str = "pls_latent",
    representative_names: Optional[Sequence[str]] = None,
    n_permutations: int = 1000,
    random_state: Union[int, np.random.SeedSequence, np.random.Generator] = 42,
) -> PermutationTestResult:
    """Permutation-null significance test for a cross-platform LOGO-CV prediction.

    Self-contained: first calls :func:`logo_cv_predict` once on the real
    (unshuffled) ``y`` to populate the observed R^2/RMSE/Spearman rho/
    top-quartile-recovery, then draws ``n_permutations`` independent
    shuffles of ``y`` relative to ``genotypes`` (``X`` and ``genotypes`` are
    never shuffled) and calls :func:`logo_cv_predict` once per shuffle to
    build the null distributions and one-sided p-values.

    Args:
        X: Predictor matrix, forwarded unchanged to every ``logo_cv_predict``
            call -- see that function's own contract.
        y: Target values, one per genotype. The real, unshuffled labeling;
            each permutation iteration shuffles a copy of this array.
        genotypes: Genotype labels, forwarded unchanged to every
            ``logo_cv_predict`` call.
        reduction_method: Forwarded to every ``logo_cv_predict`` call.
        representative_names: Forwarded to every ``logo_cv_predict`` call.
        n_permutations: Number of shuffled-``y`` calls to run. Must be
            positive.
        random_state: Seed for the permutation draws, built into an RNG via
            ``numpy.random.default_rng`` (not ``numpy.random.Generator``
            directly, which requires a ``BitGenerator`` instance and rejects
            a bare ``int``). Accepts ``int``/``numpy.random.SeedSequence``
            reproducibly (the same input always reproduces the same null
            draws). A passed-in ``numpy.random.Generator`` instance is
            stateful -- reusing the *same* ``Generator`` instance across two
            calls will **not** reproduce identical results, since its
            internal state has advanced between calls.

    Returns:
        A :class:`PermutationTestResult` with the observed values, full null
        distributions, and one-sided p-values.

    Raises:
        ValueError: If ``n_permutations`` is not positive (checked before
            any ``logo_cv_predict`` call); if the observed-value
            ``logo_cv_predict`` call itself raises (e.g. an invalid
            ``reduction_method`` or mismatched-length inputs) -- surfaced
            unchanged, before any permutation is drawn; if a constant
            (zero-variance) ``y`` produces a non-finite observed
            ``spearman_rho``/``top_quartile_recovery`` (legal per
            ``logo_cv_predict``'s own contract, but unusable here) -- raised
            immediately after the observed call, before any permutation is
            drawn; if any permutation produces a non-finite null value --
            raised only after all ``n_permutations`` calls complete (a
            genuinely non-finite-producing bug is expected to affect many
            permutations within one target, not a rare one-off, so failing
            fast saves little wall-clock time while complicating
            which-permutations-ran accounting), naming both the offending
            metric(s) and permutation index(es).
    """
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}")

    observed = logo_cv_predict(X, y, genotypes, reduction_method, representative_names)
    observed_top_quartile_recovery = top_quartile_recovery(
        observed.y_true, observed.y_pred
    )

    observed_metrics = {
        "r2": observed.r2,
        "rmse": observed.rmse,
        "spearman_rho": observed.spearman_rho,
        "top_quartile_recovery": observed_top_quartile_recovery,
    }
    non_finite_observed = sorted(
        name for name, value in observed_metrics.items() if not math.isfinite(value)
    )
    if non_finite_observed:
        raise ValueError(
            f"Observed value(s) for metric(s) {non_finite_observed} are "
            "non-finite (e.g. a constant y produces a non-finite "
            "spearman_rho) -- refusing to run n_permutations shuffled calls "
            "on data already known to be unusable"
        )

    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype=float)

    null_r2 = np.empty(n_permutations)
    null_rmse = np.empty(n_permutations)
    null_spearman_rho = np.empty(n_permutations)
    null_top_quartile_recovery = np.empty(n_permutations)

    for i in range(n_permutations):
        y_shuffled = rng.permutation(y_arr)
        shuffled_result = logo_cv_predict(
            X, y_shuffled, genotypes, reduction_method, representative_names
        )
        null_r2[i] = shuffled_result.r2
        null_rmse[i] = shuffled_result.rmse
        null_spearman_rho[i] = shuffled_result.spearman_rho
        null_top_quartile_recovery[i] = top_quartile_recovery(
            y_shuffled, shuffled_result.y_pred
        )

    non_finite_entries = [
        (name, int(idx))
        for name, arr in (
            ("r2", null_r2),
            ("rmse", null_rmse),
            ("spearman_rho", null_spearman_rho),
            ("top_quartile_recovery", null_top_quartile_recovery),
        )
        for idx in np.where(~np.isfinite(arr))[0]
    ]
    if non_finite_entries:
        raise ValueError(
            f"Non-finite null value(s) found (metric, permutation index): "
            f"{non_finite_entries}"
        )

    p_value_r2 = (np.sum(null_r2 >= observed.r2) + 1) / (n_permutations + 1)
    p_value_rmse = (np.sum(null_rmse <= observed.rmse) + 1) / (n_permutations + 1)
    p_value_spearman_rho = (np.sum(null_spearman_rho >= observed.spearman_rho) + 1) / (
        n_permutations + 1
    )

    return PermutationTestResult(
        observed_r2=observed.r2,
        observed_rmse=observed.rmse,
        observed_spearman_rho=observed.spearman_rho,
        observed_top_quartile_recovery=observed_top_quartile_recovery,
        null_r2=null_r2,
        null_rmse=null_rmse,
        null_spearman_rho=null_spearman_rho,
        null_top_quartile_recovery=null_top_quartile_recovery,
        p_value_r2=float(p_value_r2),
        p_value_rmse=float(p_value_rmse),
        p_value_spearman_rho=float(p_value_spearman_rho),
        n_permutations=n_permutations,
    )
