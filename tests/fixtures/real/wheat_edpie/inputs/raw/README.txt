RAW PIPELINE INPUTS — wheat EDPIE (pre-QC)
==========================================

These are the CSVs fed INTO the sleap-roots-analyze QC pipeline (the source data),
distinct from ../analysis_inputs_post_qc/ which holds the POST-QC, analysis-ready
tables (10_final_data.csv) used for the #120 reproduction (boundary A).

Use these to test/reproduce the QC steps themselves
(raw -> traits cleanup -> samples cleanup -> outlier removal -> heritability filter
 -> 10_final_data.csv), i.e. the full pipeline (boundary B).

Layout
------
  turface/    single traits CSV
                Turface_all_traits_2024_RSR_diameter_angle_traits_removed.csv
  cylinder/   single traits CSV (scanner-independent, 11 DAG)
                traits_11DAG_cleaned_qc_scanner_independent.csv
  field/      root-core ingest = 3 files merged by the root_core pipeline:
                rearranged_root_biomass_dw.csv   (root biomass)
                root_counting_cimmyt_edited.csv  (root counting)
                Field_2024_aboveground.csv       (above-ground, merged in)

Pipeline run anchor: pipeline_runs/2026-02-12_191823
Source configs: pipeline_runs/2026-02-12_191823/qc/<platform>/config.yaml (csv_path)
