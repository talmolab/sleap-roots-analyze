"""Step: Predict cross-platform genotype values via LOGO-CV (Tier 3.5, #196)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sleap_roots_analyze.cross_experiment_analysis import (
    cluster_correlated_traits,
    select_cluster_representatives,
)
from sleap_roots_analyze.cross_platform_prediction import (
    fit_pca_on_fold,  # noqa: F401 -- imported for the docstring cross-reference below,
    # never called directly: this step computes the PC1 *target* via pca.fit_pca()
    # (whole-dataset, ground truth), while fit_pca_on_fold remains reserved for
    # reducing the *source* predictor matrix per-fold inside logo_cv_predict's own
    # "pc1" reduction_method branch (design.md Decision 6/12).
    logo_cv_predict,
)
from sleap_roots_analyze.pca import fit_pca
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.result_types import CrossPlatformPredictionResult

_BLUP_GENOTYPE_COLUMN_CANDIDATES = ("Genotype", "genotype")
_MIN_COMMON_GENOTYPES = 3


class PredictCrossPlatformStep(BaseStep):
    """Predict cross-platform genotype values via leave-one-genotype-out CV.

    Optional 6th step on ``CrossPlatformPipeline``, entirely absent from
    ``create_tasks()`` when ``config.prediction.enabled=False``. Consumes
    Tier 3's ``logo_cv_predict``/``CrossPlatformPredictionResult`` unchanged.

    Selection-bias note (Decision 11): target-side cluster-representative
    trait *selection* (which traits become headline predictable targets) is
    computed from the full common-genotype target matrix, including
    genotypes a later LOGO fold will hold out -- a selection-bias
    consideration distinct from fit-time leakage. No Ridge/PLS coefficient
    ever sees a held-out genotype's target value; only the *choice* of which
    traits are reported uses every genotype's own outcome data. This differs
    from the *source*-side "representatives" predictor selection, which
    never touches ``y`` at all and is unconditionally safe to fix pre-loop.
    """

    def __init__(self):
        """Initialize PredictCrossPlatformStep."""
        super().__init__(
            step_name="PredictCrossPlatform",
            description="Predict cross-platform genotype values via LOGO-CV",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the cross-platform prediction step.

        Args:
            data: Task 1's (``LoadCrossPlatformDataStep``) result data dict
                (``exp1_df``, ``exp2_df``, ``common_genotypes``). Only read
                when ``config.prediction.predictor_source="genotype_means"``.
            config: CrossPlatformConfig with a populated ``prediction`` field.
            run_dir: Directory to save one JSON file per method.
            prev_result: Task 1's StepResult, whose ``metadata`` supplies
                ``exp1_trait_names``/``exp2_trait_names`` (already
                ``exclude_cols``-filtered by ``get_trait_columns()``). Only
                read for ``predictor_source="genotype_means"``.

        Returns:
            StepResult with one ``CrossPlatformPredictionResult`` (as a
            dict) per method, saved as JSON to ``run_dir``.
        """
        pcfg = config.prediction
        pair = pcfg.platform_pairs[0]
        source_platform = pair["source"]
        target_platform = pair["target"]

        if pcfg.predictor_source == "blup":
            source_raw = self._load_blup_table(pcfg.source_blup_path)
            target_raw = self._load_blup_table(pcfg.target_blup_path)
        else:
            source_is_exp1 = source_platform == config.exp1_name
            source_df = data["exp1_df"] if source_is_exp1 else data["exp2_df"]
            target_df = data["exp2_df"] if source_is_exp1 else data["exp1_df"]
            source_trait_names = list(
                prev_result.metadata[
                    "exp1_trait_names" if source_is_exp1 else "exp2_trait_names"
                ]
            )
            target_trait_names = list(
                prev_result.metadata[
                    "exp2_trait_names" if source_is_exp1 else "exp1_trait_names"
                ]
            )
            source_raw = source_df.groupby("genotype")[source_trait_names].mean()
            target_raw = target_df.groupby("genotype")[target_trait_names].mean()

        # Decision 14: derive X, every per-target y, and genotypes from one
        # canonical, sorted, explicitly-indexed common-genotype list -- never
        # from incidental row-order agreement between independently-loaded
        # DataFrames.
        common_genotypes = sorted(
            set(source_raw.index.astype(str)) & set(target_raw.index.astype(str))
        )
        if len(common_genotypes) < _MIN_COMMON_GENOTYPES:
            raise ValueError(
                f"Only {len(common_genotypes)} genotype(s) are common between "
                f"'{source_platform}' (source) and '{target_platform}' (target), "
                f"fewer than the minimum of {_MIN_COMMON_GENOTYPES} required for "
                "leave-one-genotype-out cross-validation"
            )
        source_raw.index = source_raw.index.astype(str)
        target_raw.index = target_raw.index.astype(str)
        source_aligned = source_raw.loc[common_genotypes]
        target_aligned = target_raw.loc[common_genotypes]

        # Decision 16: drop any trait column with any NaN among the
        # common-genotype set, on both sides, before further use.
        source_clean = source_aligned.dropna(axis=1, how="any")
        target_clean = target_aligned.dropna(axis=1, how="any")
        if source_clean.shape[1] == 0:
            raise ValueError(
                f"Every trait column in the source ('{source_platform}') predictor "
                "matrix contains at least one NaN value among the common "
                "genotypes -- no usable predictor traits remain"
            )

        # Target-side cluster-representative trait selection (Decision 6/11).
        target_clusters = cluster_correlated_traits(
            target_clean,
            threshold=config.trait_clustering_threshold,
            linkage=config.trait_clustering_linkage,
        )
        target_representatives = select_cluster_representatives(
            target_clean, target_clusters
        )

        # PC1-as-target: whole-dataset ground truth, not a per-fold predictor
        # reduction (Decision 6/12).
        _, pc1_transformed = fit_pca(
            StandardScaler().fit_transform(target_clean.to_numpy()),
            n_components=1,
            random_state=42,
        )
        pc1_values = pc1_transformed.ravel()

        target_names = list(target_representatives) + ["PC1"]
        target_y = {
            name: target_clean[name].to_numpy() for name in target_representatives
        }
        target_y["PC1"] = pc1_values

        # Source-side "representatives" predictor selection (a separate,
        # unsupervised application of the same clustering functions -- safe
        # to fix pre-loop per theory.md Section 2.2, never touches y).
        source_clusters = cluster_correlated_traits(
            source_clean,
            threshold=config.trait_clustering_threshold,
            linkage=config.trait_clustering_linkage,
        )
        source_representative_names = select_cluster_representatives(
            source_clean, source_clusters
        )

        methods = [pcfg.reduction_method] + list(pcfg.comparison_methods)
        genotypes = common_genotypes
        files_generated: List[Path] = []
        results_by_method = {}

        for method in methods:
            logo_cv_results = {}
            for target_name in target_names:
                y = target_y[target_name]
                logo_cv_results[target_name] = logo_cv_predict(
                    X=source_clean,
                    y=y,
                    genotypes=genotypes,
                    reduction_method=method,
                    representative_names=(
                        source_representative_names
                        if method == "representatives"
                        else None
                    ),
                )
            cp_result = CrossPlatformPredictionResult.from_logo_cv_results(
                source_platform=source_platform,
                target_platform=target_platform,
                predictor_source=pcfg.predictor_source,
                reduction_method=method,
                logo_cv_results=logo_cv_results,
            )
            output_path = run_dir / f"06_prediction_{method}.json"
            output_path.write_text(cp_result.to_json(indent=2))
            files_generated.append(output_path)
            results_by_method[method] = cp_result.to_dict()

        metadata = {
            "source_platform": source_platform,
            "target_platform": target_platform,
            "predictor_source": pcfg.predictor_source,
            "methods": methods,
            "target_names": target_names,
            "common_genotypes": common_genotypes,
            "n_common_genotypes": len(common_genotypes),
            "source_trait_columns": list(source_clean.columns),
            "target_candidate_columns": list(target_clean.columns),
            "target_representative_traits": list(target_representatives),
        }

        return StepResult(
            data=results_by_method,
            metadata=metadata,
            files_generated=files_generated,
        )

    @staticmethod
    def _load_blup_table(path: str) -> pd.DataFrame:
        """Load a BLUP-adjusted-means CSV, indexed by its genotype column.

        Resolves the genotype column via a fixed convention -- ``"Genotype"``
        then ``"genotype"`` (Decision 17) -- distinct from
        ``exp1_genotype_col``/``exp2_genotype_col``, which govern the
        unrelated raw per-sample CSVs for steps 1-5.

        Raises:
            ValueError: If neither ``"Genotype"`` nor ``"genotype"`` is a
                column of the loaded CSV.
        """
        df = pd.read_csv(path)
        for candidate in _BLUP_GENOTYPE_COLUMN_CANDIDATES:
            if candidate in df.columns:
                return df.set_index(candidate)
        raise ValueError(
            f"BLUP CSV '{path}' has neither of the expected genotype columns "
            f"{_BLUP_GENOTYPE_COLUMN_CANDIDATES!r} (got columns: "
            f"{list(df.columns)})"
        )
