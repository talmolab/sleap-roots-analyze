"""Tests for cross-platform genotype-effect prediction (Tier 3, #194).

See openspec/changes/add-cross-platform-prediction/ for the full design and
acceptance-criteria oracles this test suite implements against.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from sleap_roots_analyze.cross_platform_prediction import (
    fit_pca_on_fold,
    logo_cv_predict,
)


class TestFitPcaOnFold:
    """Tests for fit_pca_on_fold (theory.md Section 5's contract)."""

    def test_fit_pca_on_fold_fits_on_train_only(self):
        """Projection depends only on X_train, not X_test."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((18, 5))
        X_test_a = rng.standard_normal((1, 5))
        X_test_b = rng.standard_normal((1, 5))

        out_a = fit_pca_on_fold(X_train, X_test_a, n_components=1)
        out_b = fit_pca_on_fold(X_train, X_test_b, n_components=1)

        expected_pca = PCA(n_components=1).fit(X_train)
        np.testing.assert_allclose(out_a, expected_pca.transform(X_test_a))
        np.testing.assert_allclose(out_b, expected_pca.transform(X_test_b))

    def test_fit_pca_on_fold_output_shape(self):
        """Output shape is (n_test, n_components)."""
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((18, 5))
        X_test = rng.standard_normal((3, 5))

        for n_components in (1, 2):
            out = fit_pca_on_fold(X_train, X_test, n_components=n_components)
            assert out.shape == (3, n_components)

    def test_fit_pca_on_fold_raises_when_n_traits_less_than_n_components(self):
        """Raises ValueError before calling sklearn when n_traits < n_components."""
        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((18, 2))
        X_test = rng.standard_normal((1, 2))

        with patch("sleap_roots_analyze.cross_platform_prediction.PCA") as mock_pca:
            with pytest.raises(ValueError):
                fit_pca_on_fold(X_train, X_test, n_components=3)
            mock_pca.assert_not_called()

    def test_fit_pca_on_fold_deterministic(self):
        """Calling twice with identical inputs returns identical output."""
        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((18, 5))
        X_test = rng.standard_normal((1, 5))

        out1 = fit_pca_on_fold(X_train, X_test, n_components=1)
        out2 = fit_pca_on_fold(X_train, X_test, n_components=1)

        np.testing.assert_array_equal(out1, out2)

    def test_fit_pca_on_fold_does_not_mutate_inputs(self):
        """X_train and X_test are unchanged after the call."""
        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((18, 5))
        X_test = rng.standard_normal((1, 5))
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()

        fit_pca_on_fold(X_train, X_test, n_components=1)

        np.testing.assert_array_equal(X_train, X_train_copy)
        np.testing.assert_array_equal(X_test, X_test_copy)


def _build_simple_dataset(n_genotypes=6, n_traits=4, seed=0):
    """Small synthetic (X, y, genotypes, representative_names) for structural tests.

    Not statistically meaningful -- only used to check CV-hygiene call
    patterns (which instance was fit on which data), not recovery/oracle
    properties (those use the fixtures in tests/fixtures.py).
    """
    rng = np.random.default_rng(seed)
    genotypes = [f"g{i}" for i in range(n_genotypes)]
    trait_names = [f"trait_{j}" for j in range(n_traits)]
    X = pd.DataFrame(
        rng.standard_normal((n_genotypes, n_traits)),
        index=genotypes,
        columns=trait_names,
    )
    y = rng.standard_normal(n_genotypes)
    representative_names = trait_names[:2]
    return X, y, genotypes, representative_names


class TestLogoCvPredictCvHygiene:
    """CV-hygiene structural tests: fresh Pipeline per fold, no leakage."""

    def test_logo_cv_predict_pipeline_instantiated_inside_fold(self):
        """A fresh Ridge instance is fit exactly once per fold (representatives)."""
        X, y, genotypes, rep_names = _build_simple_dataset(n_genotypes=6)
        # Keep strong references to every fit-ed instance: id() is only a
        # reliable uniqueness key while the objects stay alive -- CPython can
        # (and does, depending on what else is running) reuse a garbage
        # collected Ridge's memory address for the next fold's instance.
        calls = []
        original_fit = Ridge.fit

        def spy_fit(self, X_train, y_train, *args, **kwargs):
            calls.append((self, X_train.shape[0]))
            return original_fit(self, X_train, y_train, *args, **kwargs)

        with patch.object(Ridge, "fit", spy_fit):
            logo_cv_predict(
                X,
                y,
                genotypes,
                reduction_method="representatives",
                representative_names=rep_names,
            )

        assert len(calls) == len(genotypes)
        assert len({id(instance) for instance, _ in calls}) == len(genotypes)
        assert all(n_train == len(genotypes) - 1 for _, n_train in calls)

    def test_logo_cv_predict_representatives_scaler_never_sees_held_out_genotype(
        self,
    ):
        """StandardScaler's fit data excludes the held-out genotype's row."""
        X, y, genotypes, rep_names = _build_simple_dataset(n_genotypes=6)
        rep_values = X[rep_names].to_numpy()
        fit_calls = []
        original_fit = StandardScaler.fit

        def spy_fit(self, X_train, *args, **kwargs):
            fit_calls.append(X_train.copy())
            return original_fit(self, X_train, *args, **kwargs)

        with patch.object(StandardScaler, "fit", spy_fit):
            logo_cv_predict(
                X,
                y,
                genotypes,
                reduction_method="representatives",
                representative_names=rep_names,
            )

        assert len(fit_calls) == len(genotypes)
        for held_out_idx, fit_data in enumerate(fit_calls):
            held_out_row = rep_values[held_out_idx]
            assert not any(
                np.array_equal(held_out_row, row) for row in fit_data
            ), f"fold {held_out_idx}'s scaler saw the held-out genotype's row"

    def test_logo_cv_predict_pls_latent_uses_fixed_n_components_1(self):
        """Every fold's PLSRegression instance has n_components == 1.

        Spies on ``fit`` (not ``__init__``): patching ``__init__`` with a
        wrapper signature containing ``*args``/``**kwargs`` breaks sklearn's
        own parameter-name introspection (``_get_param_names`` inspects
        ``cls.__init__``'s signature and rejects varargs), corrupting
        ``PLSRegression`` for every subsequent test in the same process.
        """
        X, y, genotypes, _ = _build_simple_dataset(n_genotypes=6)
        observed_n_components = []
        original_fit = PLSRegression.fit

        def spy_fit(self, X_train, y_train, *args, **kwargs):
            observed_n_components.append(self.n_components)
            return original_fit(self, X_train, y_train, *args, **kwargs)

        with patch.object(PLSRegression, "fit", spy_fit):
            logo_cv_predict(X, y, genotypes, reduction_method="pls_latent")

        assert len(observed_n_components) == len(genotypes)
        assert all(n == 1 for n in observed_n_components)

    def test_logo_cv_predict_pls_latent_never_sees_held_out_genotype_y(self):
        """PLSRegression.fit's y argument excludes the held-out genotype's target."""
        X, y, genotypes, _ = _build_simple_dataset(n_genotypes=6)
        fit_y_calls = []
        original_fit = PLSRegression.fit

        def spy_fit(self, X_train, y_train, *args, **kwargs):
            fit_y_calls.append(np.asarray(y_train).copy())
            return original_fit(self, X_train, y_train, *args, **kwargs)

        with patch.object(PLSRegression, "fit", spy_fit):
            logo_cv_predict(X, y, genotypes, reduction_method="pls_latent")

        assert len(fit_y_calls) == len(genotypes)
        for held_out_idx, y_train in enumerate(fit_y_calls):
            held_out_y = y[held_out_idx]
            assert held_out_y not in y_train

    def test_logo_cv_predict_representative_names_fixed_pre_loop(self):
        """The same representative_names reduce X in every fold."""
        X, y, genotypes, rep_names = _build_simple_dataset(n_genotypes=6)
        rep_values = X[rep_names].to_numpy()
        fit_calls = []
        original_fit = Ridge.fit

        def spy_fit(self, X_train, y_train, *args, **kwargs):
            fit_calls.append(X_train.copy())
            return original_fit(self, X_train, y_train, *args, **kwargs)

        with patch.object(Ridge, "fit", spy_fit):
            logo_cv_predict(
                X,
                y,
                genotypes,
                reduction_method="representatives",
                representative_names=rep_names,
            )

        loo = LeaveOneOut()
        for (train_idx, _), fit_data in zip(loo.split(rep_values), fit_calls):
            # fit_data is StandardScaler-transformed; compare column count and
            # row count only (transformed values differ, shape must match the
            # same fixed representative_names column selection every fold).
            assert fit_data.shape == (len(train_idx), len(rep_names))

    def test_logo_cv_predict_pc1_calls_fit_pca_on_fold_per_fold(self):
        """fit_pca_on_fold is called with only that fold's data, twice per fold.

        Matches theory.md Section 3.1's documented pattern: one call to reduce
        X_train (X_test=X_train, since fit_pca_on_fold only returns the
        transformed second argument), one call to reduce the held-out X_test.
        """
        X, y, genotypes, _ = _build_simple_dataset(n_genotypes=6)
        calls = []

        def spy(X_train, X_test, n_components=1):
            calls.append((X_train.copy(), X_test.copy()))
            return fit_pca_on_fold(X_train, X_test, n_components=n_components)

        with patch(
            "sleap_roots_analyze.cross_platform_prediction.fit_pca_on_fold",
            side_effect=spy,
        ):
            logo_cv_predict(X, y, genotypes, reduction_method="pc1")

        assert len(calls) == 2 * len(genotypes)
        X_values = X.to_numpy()
        loo = LeaveOneOut()
        fold_boundaries = list(loo.split(X_values))
        for fold_idx, (train_idx, test_idx) in enumerate(fold_boundaries):
            call_train_reduce, call_test_reduce = calls[2 * fold_idx : 2 * fold_idx + 2]
            expected_X_train = X_values[train_idx]
            np.testing.assert_array_equal(call_train_reduce[0], expected_X_train)
            np.testing.assert_array_equal(call_test_reduce[0], expected_X_train)
            np.testing.assert_array_equal(call_test_reduce[1], X_values[test_idx])

    def test_logo_cv_predict_returns_one_prediction_per_genotype(self):
        """Output has len(genotypes) predictions, in input order."""
        X, y, genotypes, rep_names = _build_simple_dataset(n_genotypes=7)
        result = logo_cv_predict(
            X,
            y,
            genotypes,
            reduction_method="representatives",
            representative_names=rep_names,
        )
        assert len(result.y_pred) == len(genotypes)
        assert list(result.genotypes) == list(genotypes)

    def test_logo_cv_predict_computes_rmse_and_spearman_rho(self):
        """RMSE and Spearman rho (with p-value) are reported alongside R2."""
        X, y, genotypes, rep_names = _build_simple_dataset(n_genotypes=8)
        result = logo_cv_predict(
            X,
            y,
            genotypes,
            reduction_method="representatives",
            representative_names=rep_names,
        )
        assert isinstance(result.rmse, float)
        assert result.rmse >= 0
        assert isinstance(result.spearman_rho, float)
        assert isinstance(result.spearman_p, float)


class TestLogoCvPredictOracles:
    """Planted-signal / pure-noise / non-EDPIE recovery oracles (design.md Decision 6)."""

    @staticmethod
    def _mean_r2(realizations, **kwargs):
        r2s = []
        for X, y, genotypes in realizations:
            result = logo_cv_predict(X, y, genotypes, **kwargs)
            r2s.append(result.r2)
        return float(np.mean(r2s))

    @pytest.mark.parametrize("reduction_method", ["pls_latent", "representatives"])
    def test_logo_cv_predict_planted_signal_recovers_expected_r2(
        self, cross_platform_planted_signal_fixture, reduction_method
    ):
        """Mean R2 across 20 realizations is comfortably positive."""
        kwargs = {"reduction_method": reduction_method}
        if reduction_method == "representatives":
            trait_names = list(cross_platform_planted_signal_fixture[0][0].columns)
            kwargs["representative_names"] = trait_names
        mean_r2 = self._mean_r2(cross_platform_planted_signal_fixture, **kwargs)
        assert 0.4 <= mean_r2 <= 0.95

    @pytest.mark.parametrize("reduction_method", ["pls_latent", "representatives"])
    def test_logo_cv_predict_pure_noise_r2_clearly_below_signal(
        self,
        cross_platform_planted_signal_fixture,
        cross_platform_pure_noise_fixture,
        reduction_method,
    ):
        """Mean noise R2 is well below mean signal R2 (design.md Decision 6)."""
        kwargs = {"reduction_method": reduction_method}
        if reduction_method == "representatives":
            trait_names = list(cross_platform_planted_signal_fixture[0][0].columns)
            kwargs["representative_names"] = trait_names
        signal_mean = self._mean_r2(cross_platform_planted_signal_fixture, **kwargs)
        noise_mean = self._mean_r2(cross_platform_pure_noise_fixture, **kwargs)
        assert signal_mean - noise_mean > 0.5

    @pytest.mark.parametrize("reduction_method", ["pls_latent", "representatives"])
    def test_logo_cv_predict_synthetic_non_edpie_fixture_generalizes(
        self, cross_platform_synthetic_non_edpie_fixture, reduction_method
    ):
        """The non-EDPIE-shaped fixture recovers a comparably positive mean R2."""
        kwargs = {"reduction_method": reduction_method}
        if reduction_method == "representatives":
            trait_names = list(cross_platform_synthetic_non_edpie_fixture[0][0].columns)
            kwargs["representative_names"] = trait_names
        mean_r2 = self._mean_r2(cross_platform_synthetic_non_edpie_fixture, **kwargs)
        assert 0.4 <= mean_r2 <= 0.95


class TestLogoCvPredictInputValidation:
    """Upfront input validation (added after /review-openspec round 1)."""

    def test_logo_cv_predict_rejects_mismatched_lengths(self):
        """Mismatched X/y/genotypes lengths raise ValueError."""
        X, y, genotypes, _ = _build_simple_dataset(n_genotypes=6)
        with pytest.raises(ValueError):
            logo_cv_predict(X, y[:-1], genotypes, reduction_method="pls_latent")
        with pytest.raises(ValueError):
            logo_cv_predict(X, y, genotypes[:-1], reduction_method="pls_latent")

    def test_logo_cv_predict_rejects_invalid_reduction_method(self):
        """An unrecognized reduction_method raises ValueError."""
        X, y, genotypes, _ = _build_simple_dataset(n_genotypes=6)
        with pytest.raises(ValueError):
            logo_cv_predict(X, y, genotypes, reduction_method="not_a_real_method")

    def test_logo_cv_predict_representatives_requires_representative_names(self):
        """representative_names=None with reduction_method='representatives' raises."""
        X, y, genotypes, _ = _build_simple_dataset(n_genotypes=6)
        with pytest.raises(ValueError):
            logo_cv_predict(
                X,
                y,
                genotypes,
                reduction_method="representatives",
                representative_names=None,
            )

    def test_logo_cv_predict_rejects_too_few_genotypes(self):
        """Fewer than 2 genotypes raises ValueError (LOGO-CV needs a train fold)."""
        X, y, genotypes, rep_names = _build_simple_dataset(n_genotypes=1)
        with pytest.raises(ValueError):
            logo_cv_predict(
                X,
                y,
                genotypes,
                reduction_method="representatives",
                representative_names=rep_names,
            )

    def test_logo_cv_predict_constant_y_does_not_crash(self):
        """Zero-variance y does not raise (R2/rho may be degenerate)."""
        X, _, genotypes, rep_names = _build_simple_dataset(n_genotypes=6)
        constant_y = np.ones(len(genotypes))
        result = logo_cv_predict(
            X,
            constant_y,
            genotypes,
            reduction_method="representatives",
            representative_names=rep_names,
        )
        assert len(result.y_pred) == len(genotypes)

    def test_logo_cv_predict_rejects_nan_in_X(self):
        """A NaN value anywhere in X raises ValueError."""
        X, y, genotypes, rep_names = _build_simple_dataset(n_genotypes=6)
        X_with_nan = X.copy()
        X_with_nan.iloc[0, 0] = np.nan
        with pytest.raises(ValueError):
            logo_cv_predict(
                X_with_nan,
                y,
                genotypes,
                reduction_method="representatives",
                representative_names=rep_names,
            )


def _logo_cv_r2_leakage_check(X, y, fit_inside_fold=True):
    """Reference LOGO-CV R2 with an explicit leakage toggle (theory.md Section 4.3).

    ``fit_inside_fold=True`` mirrors ``logo_cv_predict``'s correct hygiene
    (scaler and Ridge fit on training-fold data only). ``fit_inside_fold=False``
    deliberately leaks: both are fit on the FULL dataset (including the
    held-out genotype) once, before the fold loop. Test-only -- production
    ``logo_cv_predict`` has no code path equivalent to the leaked branch
    (structurally asserted by 3.1/3.2/3.4's mock/spy tests).
    """
    from sklearn.metrics import r2_score

    loo = LeaveOneOut()
    y_pred = np.full(len(y), np.nan)
    if not fit_inside_fold:
        scaler_global = StandardScaler().fit(X)
        model_global = Ridge().fit(scaler_global.transform(X), y)
    for train_idx, test_idx in loo.split(X):
        if fit_inside_fold:
            scaler = StandardScaler().fit(X[train_idx])
            model = Ridge().fit(scaler.transform(X[train_idx]), y[train_idx])
        else:
            scaler = scaler_global
            model = model_global
        y_pred[test_idx] = model.predict(scaler.transform(X[test_idx])).ravel()
    return float(r2_score(y, y_pred))


class TestLeakageRegression:
    """Explicit leakage regression test (theory.md Section 4)."""

    def test_leakage_detectable_ratio_at_least_1_10(
        self, cross_platform_planted_signal_fixture
    ):
        """Mean outside-fold-fit R2 is inflated >= 1.10x vs mean inside-fold-fit R2."""
        r2_inside = [
            _logo_cv_r2_leakage_check(X.to_numpy(), y, fit_inside_fold=True)
            for X, y, _ in cross_platform_planted_signal_fixture
        ]
        r2_outside = [
            _logo_cv_r2_leakage_check(X.to_numpy(), y, fit_inside_fold=False)
            for X, y, _ in cross_platform_planted_signal_fixture
        ]
        mean_inside = float(np.mean(r2_inside))
        mean_outside = float(np.mean(r2_outside))
        ratio = mean_outside / max(mean_inside, 1e-6)
        assert ratio >= 1.10, (
            f"Leakage not detectable: ratio={ratio:.3f} < 1.10. "
            f"mean_inside={mean_inside:.3f}, mean_outside={mean_outside:.3f}"
        )

    def test_leakage_inflates_r2_even_on_pure_noise(
        self, cross_platform_pure_noise_fixture
    ):
        """Outside-fold-fit R2 exceeds inside-fold-fit R2 even with no real signal.

        Confirms the leakage mechanism (fitting on data that includes the
        held-out genotype) inflates R2 generally -- not only when a real
        planted signal exists to be "over-recovered" -- so 4.1's ratio isn't
        specific to the signal fixture's parameters. Uses a plain mean
        comparison rather than the 1.10 ratio (empirically, pure-noise
        inside-fit R2 is negative, so a multiplicative ratio against a
        near-zero/negative baseline is not a meaningful threshold).
        """
        r2_inside = [
            _logo_cv_r2_leakage_check(X.to_numpy(), y, fit_inside_fold=True)
            for X, y, _ in cross_platform_pure_noise_fixture
        ]
        r2_outside = [
            _logo_cv_r2_leakage_check(X.to_numpy(), y, fit_inside_fold=False)
            for X, y, _ in cross_platform_pure_noise_fixture
        ]
        assert float(np.mean(r2_outside)) > float(np.mean(r2_inside))


class TestTraitSetIdentityOracle:
    """Reproduces the wheat EDPIE paper's Section 3.4 trait-set identity result.

    Resolved via a 2026-07-16 handoff investigation (design.md Decision 2):
    the real mechanism is cluster each platform independently -> correlate
    every representative pair -> filter to |rho|>=0.55 -> count DISTINCT
    traits per side among the surviving pairs. NOT raw per-platform
    representative counts (that was the original, empirically-wrong design).
    Runs the real production code (LoadCrossPlatformDataStep,
    cluster_correlated_traits/select_cluster_representatives,
    CalculateCrossPlatformCorrelationsStep) against the regenerated
    root_core_vs_cylinder fixture (paper-vintage data) -- not a hardcoded
    lookup of the committed CSV.
    """

    @pytest.fixture(scope="class")
    def loaded_root_core_vs_cylinder(self, tmp_path_factory):
        """Load + align the paper-vintage root_core_vs_cylinder fixture data."""
        from sleap_roots_analyze.pipeline.config.utils import (
            load_cross_platform_config,
        )
        from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
            LoadCrossPlatformDataStep,
        )

        config_path = (
            Path(__file__).parent
            / "fixtures"
            / "harness"
            / "cross_platform"
            / "cross_platform_rootcore_vs_cylinder_paper_vintage.yaml"
        )
        cfg = load_cross_platform_config(str(config_path))
        run_dir = tmp_path_factory.mktemp("root_core_vs_cylinder_load")
        result = LoadCrossPlatformDataStep().execute(
            data=None, config=cfg, run_dir=run_dir, prev_result=None
        )
        return cfg, result

    @pytest.fixture(scope="class")
    def cluster_representatives(self, loaded_root_core_vs_cylinder):
        """Cluster each platform's traits independently (threshold=0.8)."""
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
            select_cluster_representatives,
        )

        cfg, result = loaded_root_core_vs_cylinder
        exp1_df = result.data["exp1_df"]
        exp2_df = result.data["exp2_df"]
        exp1_trait_names = list(result.metadata["exp1_trait_names"])
        exp2_trait_names = list(result.metadata["exp2_trait_names"])

        trait_data1 = exp1_df.groupby("genotype")[exp1_trait_names].mean()
        trait_data2 = exp2_df.groupby("genotype")[exp2_trait_names].mean()

        clusters1 = cluster_correlated_traits(
            trait_data1,
            threshold=cfg.trait_clustering_threshold,
            linkage=cfg.trait_clustering_linkage,
        )
        clusters2 = cluster_correlated_traits(
            trait_data2,
            threshold=cfg.trait_clustering_threshold,
            linkage=cfg.trait_clustering_linkage,
        )
        reps1 = select_cluster_representatives(trait_data1, clusters1)
        reps2 = select_cluster_representatives(trait_data2, clusters2)
        return exp1_df, exp2_df, reps1, reps2

    def test_cluster_and_correlate_reproduces_section_3_4_representative_counts(
        self, cluster_representatives
    ):
        """Clustering each platform independently gives 22 field / 129 cylinder reps."""
        _, _, reps1, reps2 = cluster_representatives
        assert len(reps1) == 22
        assert len(reps2) == 129

    def test_cross_platform_correlation_filter_reproduces_section_3_4_trait_set(
        self, loaded_root_core_vs_cylinder, cluster_representatives
    ):
        """Correlating representative pairs at |rho|>=0.55 reproduces 14/28 distinct traits."""
        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
            CalculateCrossPlatformCorrelationsStep,
        )

        cfg, load_result = loaded_root_core_vs_cylinder
        exp1_df, exp2_df, reps1, reps2 = cluster_representatives

        exp1_df_reduced = exp1_df[["genotype"] + reps1].copy()
        exp2_df_reduced = exp2_df[["genotype"] + reps2].copy()

        prev_result = StepResult(
            data=load_result.data,
            metadata={
                **load_result.metadata,
                "exp1_trait_names": reps1,
                "exp2_trait_names": reps2,
            },
        )
        corr_result = CalculateCrossPlatformCorrelationsStep().execute(
            data={
                "exp1_df": exp1_df_reduced,
                "exp2_df": exp2_df_reduced,
                "common_genotypes": load_result.data["common_genotypes"],
            },
            config=cfg,
            run_dir=(
                Path(load_result.files_generated[0]).parent
                if load_result.files_generated
                else Path(".")
            ),
            prev_result=prev_result,
        )
        correlation_df = corr_result.data["correlation_df"]

        assert len(correlation_df) == 2838

        hits = correlation_df[correlation_df["spearman_r"].abs() >= 0.55]
        assert len(hits) == 36
        assert hits["exp1_trait"].nunique() == 14
        assert hits["exp2_trait"].nunique() == 28

    def test_cluster_representatives_deterministic_given_same_input(
        self, loaded_root_core_vs_cylinder
    ):
        """cluster_correlated_traits/select_cluster_representatives are deterministic."""
        from sleap_roots_analyze.cross_experiment_analysis import (
            cluster_correlated_traits,
            select_cluster_representatives,
        )

        _, result = loaded_root_core_vs_cylinder
        exp1_df = result.data["exp1_df"]
        exp1_trait_names = list(result.metadata["exp1_trait_names"])
        trait_data = exp1_df.groupby("genotype")[exp1_trait_names].mean()

        clusters_a = cluster_correlated_traits(trait_data, threshold=0.8)
        reps_a = select_cluster_representatives(trait_data, clusters_a)
        clusters_b = cluster_correlated_traits(trait_data, threshold=0.8)
        reps_b = select_cluster_representatives(trait_data, clusters_b)

        assert clusters_a == clusters_b
        assert reps_a == reps_b
