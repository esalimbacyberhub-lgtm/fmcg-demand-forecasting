# FMCG Global Demand Planning & Forecasting

A 6-phase time-series forecasting pipeline built on 3 years of daily sales data across 15 countries and 350+ SKUs. The goal is to optimize global inventory allocation and move from reactive reporting to predictive demand management.

## Project phases

1. **Data prep & integrity**: schema validation, missing values, median lead time imputation, deduplication, channel standardization
2. **EDA & portfolio segmentation**: ABC/XYZ SKU segmentation, lead time bottleneck analysis
3. **Statistical rigor**: t-tests/ANOVA on baseline demand vs. promotional lift by channel
4. **Feature engineering**: lag features (t-1, t-7, t-30) and rolling statistics, with no chronological leakage
5. **Machine learning ensembles**: XGBoost, LightGBM, Random Forest with time-based cross-validation
6. **Explainability & ethics**: SHAP interpretability and an ethical channel allocation framework for traditional trade

## Repository layout

- `data/`: raw datasets (excluded from Git because of file size)
- `notebooks/`: Jupyter notebooks, one per phase
- `scripts/`: reusable Python scripts

## Data

The sales dataset (`fmcg_sales_3years_1M_rows.csv`, about 212 MB) is not stored in this repo. Place it in `data/` locally to run the notebooks.
