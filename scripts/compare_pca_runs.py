"""Compare two sleap-roots-analyze runs that differ only in pca.n_components.

Usage:
    uv run python scripts/compare_pca_runs.py
"""
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy"]
# ///
import json
from pathlib import Path

import pandas as pd
import numpy as np

NEW_ROOT = Path(
    r"Z:\users\eberrigan\20260425_Mauricio_Chiurazzi_Alfalfa_GWAS_groups_1-6_canola_models"
    r"\combined_analysis\sleap-roots-analyze-output\2026-04-26_203038"
)
OLD_ROOT = Path(
    r"Z:\users\eberrigan\20260425_Mauricio_Chiurazzi_Alfalfa_GWAS_groups_1-6_canola_models"
    r"\combined_analysis\sleap-roots-analyze-output\2026-04-26_190512"
)


def find_qc_dir(root: Path, age: str) -> Path:
    """Find the qc subdir for a given plant_age_days."""
    qc = root / "qc"
    matches = list(qc.glob(f"plant_age_days_{age}_*"))
    assert len(matches) == 1, f"expected 1 match got {matches}"
    return matches[0]


def find_viz_dir(root: Path, age: str) -> Path:
    viz = root / "viz" / f"plant_age_days_{age}"
    matches = list(viz.glob("*_viz_*"))
    assert len(matches) == 1, f"expected 1 match got {matches}"
    return matches[0]


def summarize_run(root: Path, label: str) -> dict:
    print(f"\n{'='*80}\n{label}: {root.name}\n{'='*80}")
    out = {}
    for age in ["9", "12"]:
        qc = find_qc_dir(root, age)
        viz = find_viz_dir(root, age)

        # 10_final_data
        final = pd.read_csv(qc / "10_final_data.csv")
        # 08_heritability_results
        herit = pd.read_csv(qc / "data" / "08_heritability_results.csv")

        # 10_pipeline_summary.json
        with open(qc / "10_pipeline_summary.json") as f:
            psum = json.load(f)

        # config
        with open(qc / "config.yaml") as f:
            cfg_text = f.read()

        # PCA explained variance
        pca_dir = viz / "data" / "pca"
        ev_path = pca_dir / "explained_variance.csv"
        if ev_path.exists():
            ev = pd.read_csv(ev_path)
        else:
            print(f"  [WARN] no explained_variance.csv in {pca_dir}")
            ev = None

        # PCA loadings
        load_path = pca_dir / "loadings.csv"
        if load_path.exists():
            loadings = pd.read_csv(load_path)
        else:
            loadings = None

        print(f"\n--- Day {age} ---")
        print(f"  10_final_data: {len(final)} rows x {len(final.columns)} cols")
        # try to find trait columns
        non_meta = [c for c in final.columns if c not in {
            "scan_id", "plant_qr_code", "plant_age_days", "wave",
            "experiment", "genotype", "wave_number", "treatment",
            "barcode", "replicate", "batch", "Replicate", "Genotype",
            "scan_path", "plant_id", "Wave", "Wave_number"
        }]
        print(f"  approx trait cols (excluding common metadata): {len(non_meta)}")
        n_genotypes = final["genotype"].nunique() if "genotype" in final.columns else (
            final["Genotype"].nunique() if "Genotype" in final.columns else None
        )
        print(f"  unique genotypes: {n_genotypes}")

        print(f"  08_heritability: {len(herit)} rows; mean H2 = {herit['heritability'].mean():.4f}; "
              f"median = {herit['heritability'].median():.4f}; "
              f"min = {herit['heritability'].min():.4f}; max = {herit['heritability'].max():.4f}")
        print(f"  heritability columns: {list(herit.columns)}")

        if ev is not None:
            print(f"  PCA n_components retained: {len(ev)}")
            print(f"  PCA columns: {list(ev.columns)}")
            print(ev.head(10).to_string(index=False))

        if loadings is not None:
            print(f"  loadings shape: {loadings.shape}; columns sample: {list(loadings.columns)[:8]}")

        out[age] = {
            "n_samples": len(final),
            "n_genotypes": n_genotypes,
            "n_traits_heritability": len(herit),
            "mean_h2": float(herit["heritability"].mean()),
            "median_h2": float(herit["heritability"].median()),
            "ev": ev,
            "loadings": loadings,
            "pipeline_summary": psum,
            "qc_dir": str(qc),
            "viz_dir": str(viz),
        }
    return out


print("Loading new run...")
new = summarize_run(NEW_ROOT, "NEW (pca.n_components=0.75)")
print("\n\nLoading old run...")
old = summarize_run(OLD_ROOT, "OLD (pca.n_components=0.95)")

print(f"\n\n{'='*80}\nCOMPARISON\n{'='*80}")
for age in ["9", "12"]:
    print(f"\n--- Day {age} ---")
    n = new[age]
    o = old[age]
    print(f"  samples:           new={n['n_samples']}  old={o['n_samples']}  match={n['n_samples']==o['n_samples']}")
    print(f"  genotypes:         new={n['n_genotypes']}  old={o['n_genotypes']}  match={n['n_genotypes']==o['n_genotypes']}")
    print(f"  trait count (H2):  new={n['n_traits_heritability']}  old={o['n_traits_heritability']}  match={n['n_traits_heritability']==o['n_traits_heritability']}")
    print(f"  mean H2:           new={n['mean_h2']:.4f}  old={o['mean_h2']:.4f}  diff={n['mean_h2']-o['mean_h2']:+.4f}")
    print(f"  median H2:         new={n['median_h2']:.4f}  old={o['median_h2']:.4f}  diff={n['median_h2']-o['median_h2']:+.4f}")
    if n["ev"] is not None and o["ev"] is not None:
        print(f"  PCs retained:      new={len(n['ev'])}  old={len(o['ev'])}")

print("\nDone.")
