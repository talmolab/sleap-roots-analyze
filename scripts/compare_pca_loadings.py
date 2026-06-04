"""Top-loading traits per PC, both runs, both ages. Strategy=extreme means
top 5 positive AND top 5 negative loadings."""
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy"]
# ///
from pathlib import Path
import pandas as pd

NEW_ROOT = Path(
    r"Z:\users\eberrigan\20260425_Mauricio_Chiurazzi_Alfalfa_GWAS_groups_1-6_canola_models"
    r"\combined_analysis\sleap-roots-analyze-output\2026-04-26_203038"
)
OLD_ROOT = Path(
    r"Z:\users\eberrigan\20260425_Mauricio_Chiurazzi_Alfalfa_GWAS_groups_1-6_canola_models"
    r"\combined_analysis\sleap-roots-analyze-output\2026-04-26_190512"
)


def find_pca_dir(root: Path, age: str) -> Path:
    return next((root / "viz" / f"plant_age_days_{age}").glob("*_viz_*")) / "data" / "pca"


def top_loadings(loadings_df: pd.DataFrame, pc: str, n: int = 5):
    """Return top n positive and top n negative loadings for given PC."""
    s = loadings_df.set_index("Unnamed: 0")[pc]
    top_pos = s.nlargest(n)
    top_neg = s.nsmallest(n)
    return top_pos, top_neg


for label, root in [("NEW (n_components=0.75)", NEW_ROOT), ("OLD (n_components=0.95)", OLD_ROOT)]:
    print(f"\n{'#'*80}\n# {label}\n{'#'*80}")
    for age in ["9", "12"]:
        pca_dir = find_pca_dir(root, age)
        loadings = pd.read_csv(pca_dir / "loadings.csv")
        ev = pd.read_csv(pca_dir / "explained_variance.csv")
        print(f"\n=== Day {age} ===")
        for pc in ["PC1", "PC2", "PC3"]:
            evr = ev.loc[ev["Unnamed: 0"] == pc, "explained_variance_ratio"].values[0]
            print(f"\n  {pc} ({evr*100:.2f}% var)")
            pos, neg = top_loadings(loadings, pc, 5)
            print(f"    top + loadings:")
            for t, v in pos.items():
                print(f"      {v:+.4f}  {t}")
            print(f"    top - loadings:")
            for t, v in neg.items():
                print(f"      {v:+.4f}  {t}")
