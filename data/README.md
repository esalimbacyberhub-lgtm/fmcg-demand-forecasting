# Data

The datasets are not stored in this repository because of their size (the sales file is about 212 MB).

## Input

Place the raw sales file here before running the notebooks:

- `fmcg_sales_3years_1M_rows.csv`: FMCG multi-country daily sales, 1,100,000 rows, January 2021 to December 2023

## Files created by the pipeline

| File | Created by |
|---|---|
| `fmcg_clean.csv` | `notebooks/01_data_prep.ipynb` |
| `abc_xyz_segments.csv` | `notebooks/02_eda_segmentation.ipynb` |
| `promo_lift_by_series.csv` | `notebooks/03_promo_lift.ipynb` |
| `fmcg_features.csv` / `.parquet` | `notebooks/04_feature_engineering.ipynb`, or `python scripts/features.py` |
| `test_predictions.csv` | `notebooks/05_modeling.ipynb`, or `python scripts/train_evaluate.py` |

`climate_features.csv`, supplied with the original files, is not used: it has no date or location key to join to the sales data. Weather features come from the `temperature` and `rain_mm` columns of the sales file.
