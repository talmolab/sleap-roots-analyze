"""Investigate Scanline First Ind heritability inflation hypothesis.

Hypothesis: scanline_first_ind is an integer index (0..49), not a continuous
physical measurement. Discrete-valued traits with low variance can produce
artificially inflated heritability.
"""
import pandas as pd
import numpy as np

CSV = r"Z:\users\eberrigan\20260425_Mauricio_Chiurazzi_Alfalfa_GWAS_groups_1-6_canola_models\combined_analysis\combined_traits_summary.csv"

df = pd.read_csv(CSV, low_memory=False)
print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

# Find scanline first ind columns
sf_cols = [c for c in df.columns if "scanline_first_ind" in c.lower() or "scanline first ind" in c.lower()]
print(f"\nFound {len(sf_cols)} 'scanline first ind' columns:")
for c in sf_cols:
    print(f"  {c}")

# Distribution analysis: # unique values, dtype, range, uniqueness ratio
print("\n=== Distribution analysis ===")
print(f"{'col':<60} {'dtype':<10} {'n_unique':>8} {'min':>6} {'max':>6} {'unique_pct':>10}")
for c in sf_cols:
    s = df[c].dropna()
    n_unique = s.nunique()
    n_total = len(s)
    pct = 100 * n_unique / n_total if n_total else 0
    try:
        print(f"{c:<60} {str(s.dtype):<10} {n_unique:>8} {s.min():>6.2f} {s.max():>6.2f} {pct:>9.2f}%")
    except Exception as e:
        print(f"{c:<60} ERR {e}")

# Look at value counts for a few representative ones
print("\n=== Top value_counts for representative columns ===")
for c in sf_cols[:6]:
    s = df[c].dropna()
    vc = s.value_counts().head(10)
    print(f"\n--- {c} (n={len(s)}, n_unique={s.nunique()}) ---")
    print(vc)

# Compare to a known-continuous trait family for baseline
print("\n=== Compare to a continuous trait (primary length) ===")
cont_cols = [c for c in df.columns if "primary_length" in c.lower()][:3]
for c in cont_cols:
    s = df[c].dropna()
    print(f"{c:<60} unique={s.nunique()} total={len(s)} ({100*s.nunique()/len(s):.1f}%)")

# Within vs between genotype variance
print("\n=== Within vs between genotype variance ===")
geno_col = None
for cand in ["genotype", "Genotype", "accession_name", "plant_qr_code"]:
    if cand in df.columns:
        geno_col = cand
        break
print(f"Genotype column: {geno_col}")
print(f"N genotypes: {df[geno_col].nunique() if geno_col else 'N/A'}")

# Also check experiment / wave
exp_cols = [c for c in df.columns if c.lower() in {"experiment_name", "wave_number", "wave", "experiment"}]
print(f"Experiment-related columns: {exp_cols}")

if geno_col:
    # compute within-vs-between for each scanline first ind column
    print(f"\n{'col':<60} {'between_var':>12} {'within_var':>12} {'ratio':>8} {'ICC_naive':>10}")
    for c in sf_cols:
        s = df[[geno_col, c]].dropna()
        if len(s) == 0:
            continue
        grp = s.groupby(geno_col)[c]
        between = grp.mean().var()
        within = grp.var().mean()
        total = s[c].var()
        icc = between / total if total > 0 else np.nan
        ratio = between / within if within and within > 0 else np.nan
        print(f"{c:<60} {between:>12.4f} {within:>12.4f} {ratio:>8.2f} {icc:>10.3f}")

# Check experiment_name correlation
if "experiment_name" in df.columns:
    print("\n=== Per-experiment means for first 6 scanline first ind cols ===")
    for c in sf_cols[:6]:
        means = df.groupby("experiment_name")[c].agg(['mean', 'std', 'count'])
        print(f"\n--- {c} ---")
        print(means)
