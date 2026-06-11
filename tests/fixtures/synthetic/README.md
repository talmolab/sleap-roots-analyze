# Synthetic analysis-input fixtures

The canonical synthetic analysis-input examples are **owned by
`sleap-roots-contracts`** (contracts#3) and ship inside that package. This repo does
**not** keep a divergent second copy — tests load them from the package accessor so
there is a single source of truth:

```python
from sleap_roots_contracts.examples import load_analysis_input_example

df = load_analysis_input_example("turface")             # replicate-present
df = load_analysis_input_example("cylinder_no_replicate")  # replicate-absent (#142)
```

`load_analysis_input_example` reads the role columns as strings, so the returned frame
passes `sleap_roots_contracts.validate_analysis_input()` directly. Available examples:
`cylinder`, `cylinder_no_replicate`, `field`, `turface`, `genotype_means`.

See `tests/test_pipeline_reproduction.py::test_canonical_examples_pass_contract`. These
run only when `sleap-roots-contracts[pandas]` is installed (it is not yet published;
once released, add `sleap-roots-contracts[pandas]>=0.1.0a1` to the dev dependency group
so the contract tests run in CI instead of being skipped).

The **full real reproduction data** (analyze-native `Genotype`/`Barcode`/`Replicate`
names) lives under `../real/wheat_edpie/`; the post-QC `10_final_data.csv` there is
canonicalized to the contract's role names before validation — see
`test_post_qc_input_passes_contract`.
