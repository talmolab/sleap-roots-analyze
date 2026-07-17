"""CI wiring-correctness oracle for cross-platform prediction (tasks.md Section 6, #196).

Requires Sections 1-5 fully implemented and green -- it exercises the
complete pipeline path (config, validation, step, task wiring), not any
single unit in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.cross_platform_prediction import logo_cv_predict
from sleap_roots_analyze.pipeline.config.utils import load_cross_platform_config
from sleap_roots_analyze.pipeline.pipelines.cross_platform_pipeline import (
    CrossPlatformPipeline,
)

HARNESS_DIR = Path(__file__).parent / "fixtures" / "harness" / "cross_platform"


def test_predict_cross_platform_pipeline_matches_direct_logo_cv_predict_call(
    tmp_path,
):
    """Pipeline R^2 for a target matches a direct logo_cv_predict() call (tasks.md 6.1).

    Not a statistical signal-recovery claim -- a wiring-correctness oracle:
    the pipeline and a hand-rolled direct call must agree exactly (within
    float tolerance) on the same data, reduction method, and target.
    """
    config = load_cross_platform_config(
        HARNESS_DIR / "cross_platform_prediction_wiring.yaml"
    )
    pipeline = CrossPlatformPipeline(config=config, output_dir=tmp_path)
    pipeline.run()

    saved = json.loads((pipeline.run_dir / "06_prediction_pls_latent.json").read_text())
    representative_targets = [
        p for p in saved["predictions"] if p["target_name"] != "PC1"
    ]
    assert representative_targets, "expected at least one representative-trait target"
    pipeline_prediction = representative_targets[0]
    target_name = pipeline_prediction["target_name"]

    source_df = pd.read_csv(config.prediction.source_blup_path).set_index("Genotype")
    target_df = pd.read_csv(config.prediction.target_blup_path).set_index("Genotype")
    common_genotypes = sorted(set(source_df.index) & set(target_df.index))
    X = source_df.loc[common_genotypes]
    y = target_df.loc[common_genotypes, target_name].to_numpy()

    direct_result = logo_cv_predict(
        X=X, y=y, genotypes=common_genotypes, reduction_method="pls_latent"
    )

    np.testing.assert_allclose(
        pipeline_prediction["r2"], direct_result.r2, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        pipeline_prediction["y_pred"], direct_result.y_pred, rtol=1e-6, atol=1e-9
    )
