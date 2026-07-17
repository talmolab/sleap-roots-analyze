"""One-off generator for the Tier 3.5 synthetic BLUP fixture pair.

Not part of the shipped package -- run once to (re)produce
tests/fixtures/synthetic/cross_platform_prediction/{source,target}_blup.csv.
Deterministic (fixed seed): a single realization is sufficient here since this
fixture backs a wiring-correctness equality check (pipeline output vs. a
direct logo_cv_predict() call on the same data), not a statistical
signal-recovery claim (see tasks.md Section 1.1).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(194)

n_genotypes = 19
genotypes = [f"G{i:02d}" for i in range(1, n_genotypes + 1)]

# Source platform BLUP table: 3 traits, no planted structure needed on this side.
source = pd.DataFrame(
    {
        "trait_a": rng.normal(10, 2, n_genotypes),
        "trait_b": rng.normal(50, 5, n_genotypes),
        "trait_c": rng.normal(0, 1, n_genotypes),
    },
    index=genotypes,
)
source.index.name = "Genotype"

# Target platform BLUP table: trait_x has a planted linear relationship to
# trait_a (a modest, deterministic signal); trait_y/trait_z are independent.
noise = rng.normal(0, 0.5, n_genotypes)
target = pd.DataFrame(
    {
        "trait_x": 2.0 * source["trait_a"].to_numpy() + noise,
        "trait_y": rng.normal(20, 3, n_genotypes),
        "trait_z": rng.normal(-5, 1, n_genotypes),
    },
    index=genotypes,
)
target.index.name = "Genotype"

out_dir = "tests/fixtures/synthetic/cross_platform_prediction"
source.reset_index().to_csv(f"{out_dir}/source_blup.csv", index=False)
target.reset_index().to_csv(f"{out_dir}/target_blup.csv", index=False)
print(source)
print(target)
