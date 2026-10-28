# FMCG Global Demand Planning & Forecasting

A 6-phase time-series forecasting pipeline built on 3 years of daily sales data (January 2021 to December 2023) across 7 countries, 13 stores and 102 SKUs (1,005 store-SKU series, about 1.1 million rows). The goal is to optimise global inventory allocation and move from reactive reporting to predictive demand management.

## Executive summary

- **Forecasts beat the naive baseline by about 15%.** On the held-out test period (July to December 2023), LightGBM reaches a WAPE of 0.267 (MAE 16.2 units), against 0.316 for a 30-day rolling-mean baseline and 0.291 for a linear regression. On days without stock-outs the figures are 0.244, 0.293 and 0.268.
- **Recent demand drives the forecast.** SHAP attributes 68% of model importance to recent demand history, 10.6% to calendar effects and 9.6% to the active promotion.
- **Promotions lift demand strongly, and demand is price-elastic.** For A-class products the net price elasticity is -2.54 (95% CI -2.56 to -2.52), and a 10% discount raises units by about 68%. The model implies a multiplier of about 1.6x at 10% off and about 2.0x at 30% off.
- **Stock-outs cannot be predicted from the available data.** They affect 3.0% of store-SKU days and cut sales by about 80% (average 11.7 vs 60.7 units), but the best classifier is no better than chance (ROC-AUC 0.502, PR-AUC equal to the 3.0% base rate). A useful stock-out model would need supply-side data such as delivery failures and replenishment events.
- **The forecast does not disadvantage Traditional Trade.** Convenience stores (the Traditional Trade proxy) are under-forecast by 2.05%, against 2.47% for Modern Trade; the 0.40 percentage point difference is not statistically detectable (95% CI -0.31 to 1.22). A simulated forecast-based allocation gives a fill-rate gap of 0.27 percentage points between channels. The result is sensitive: an extra under-forecast of roughly 1% for Traditional Trade would break a 1 percentage point fairness target, so we recommend monitoring this bias every planning cycle.

## Project phases

1. **Data prep & integrity** (`01_data_prep`): schema and integrity checks on about 1.1 million rows (data types, missing values, exact and key duplicates, category spellings, impossible values such as zero or negative sales). No missing values, duplicates or inconsistent channel names were found, so no imputation or deduplication was needed. The 3,092 zero-sales rows were kept because 82% of them are flagged stock-outs. The checked dataset is saved for later phases.
2. **EDA & portfolio segmentation** (`02_eda_segmentation`): ABC/XYZ SKU segmentation, lead time bottleneck analysis
3. **Promotion lift** (`03_promo_lift`): baseline demand vs promotional lift by channel, with statistical tests
4. **Feature engineering** (`04_feature_engineering`): 31 features including lags (1, 7, 14, 30 days) and rolling statistics computed on the previous day's demand, with no chronological leakage
5. **Modelling** (`05_modeling`): LightGBM, XGBoost and Random Forest compared with time-based cross-validation against a 30-day rolling-mean baseline. On the July to December 2023 test set LightGBM and XGBoost are effectively tied (WAPE 0.2667 and 0.2669), Random Forest scores 0.2733 and the baseline 0.3158. LightGBM was selected and improves on the baseline by 15.5%.
6. **Extended analysis and ethics**:
   - `06_stats_and_geo`: regional volatility, correlations, price elasticity, Mann-Whitney tests, stationarity, regional ABC, logistics
   - `07_model_requirements`: month-start/end flags, linear baseline, ten representative series, MAPE caveat, stock-out classification
   - `08_explainability_and_ethics`: SHAP explainability and a Traditional Trade bias check with an allocation stress test

## Repository layout

- `data/`: datasets (excluded from Git because of file size)
- `models/`: saved models (excluded from Git; created by notebook 05)
- `notebooks/`: Jupyter notebooks, one per phase, each ending with a summary of findings
- `reports/`: result tables (CSV) and figures (`reports/figures/`)
- `scripts/`: reusable Python scripts

## How to reproduce

1. Place the raw sales file (`fmcg_sales_3years_1M_rows.csv`, about 212 MB) in `data/`.
2. Install the requirements: `pip install -r requirements.txt`.
3. Run the notebooks in order, from `01` to `08`. Notebook 08 loads the model saved by notebook 05.

## Limitations

- The data appear to be simulated (7 countries, 13 stores and 102 SKUs, smaller than the brief suggests), so results may not transfer to real retail data.
- Results come from a single test period (July to December 2023).
- Convenience stores are only a proxy for Traditional Trade, and there are just 44 such series, so the intervals are wide.
- Price elasticity and SHAP values describe associations in observational data, not causal effects.