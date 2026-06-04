"""Quick sanity check for new run."""
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas"]
# ///
import json
from pathlib import Path
import pandas as pd

NEW_ROOT = Path(
    r"Z:\users\eberrigan\20260425_Mauricio_Chiurazzi_Alfalfa_GWAS_groups_1-6_canola_models"
    r"\combined_analysis\sleap-roots-analyze-output\2026-04-26_203038"
)

for age in ["9", "12"]:
    qc = next((NEW_ROOT / "qc").glob(f"plant_age_days_{age}_*"))
    print(f"\n=== Day {age} ===  {qc.name}")
    final = pd.read_csv(qc / "10_final_data.csv")
    nan_total = final.isna().sum().sum()
    nan_cols = (final.isna().sum() > 0).sum()
    print(f"  10_final_data: {len(final)} rows x {len(final.columns)} cols")
    print(f"  total NaN cells: {nan_total}")
    print(f"  cols with any NaN: {nan_cols}")

    with open(qc / "10_pipeline_summary.json") as f:
        psum = json.load(f)
    print(f"  pipeline_summary keys: {list(psum.keys())[:15]}")
    if "n_traits_input" in psum:
        print(f"  traits in: {psum['n_traits_input']}")
    if "n_traits_output" in psum:
        print(f"  traits out: {psum['n_traits_output']}")
    if "n_samples_input" in psum:
        print(f"  samples in: {psum['n_samples_input']}")
    if "n_samples_output" in psum:
        print(f"  samples out: {psum['n_samples_output']}")
    # sometimes nested
    for k, v in psum.items():
        if isinstance(v, (int, float, str, bool)) and any(s in k for s in ["sample", "trait", "outlier", "removed"]):
            print(f"  {k}: {v}")

    # cleanup log
    with open(qc / "01_trait_cleanup_log.json") as f:
        tcl = json.load(f)
    print(f"  trait_cleanup_log keys: {list(tcl.keys())}")
    for k, v in tcl.items():
        if isinstance(v, (int, float, str)):
            print(f"    {k}: {v}")

    with open(qc / "02_cleanup_log.json") as f:
        scl = json.load(f)
    print(f"  sample_cleanup_log keys: {list(scl.keys())}")
    for k, v in scl.items():
        if isinstance(v, (int, float, str)):
            print(f"    {k}: {v}")

    # outlier log
    with open(qc / "07_outlier_removal_log.json") as f:
        ol = json.load(f)
    print(f"  outlier_removal_log keys: {list(ol.keys())}")
    for k, v in ol.items():
        if isinstance(v, (int, float, str)):
            print(f"    {k}: {v}")

    # stats summary
    stats_path = qc / "data" / "08_statistical_analysis_summary.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"  stats_summary keys: {list(stats.keys())}")
        for k, v in stats.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"    {k}: {v}")
