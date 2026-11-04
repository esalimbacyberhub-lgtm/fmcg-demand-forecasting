#!/usr/bin/env python
"""Phase 5 as a script: train, tune, evaluate and save the demand forecasting models.

Reproduces notebook 05_modeling.ipynb.

Pipeline
--------
1. Load the Phase 4 feature table (data/fmcg_features.parquet, or the CSV fallback).
2. Time-based splits (never shuffled):
       train 2021-01-31 .. 2022-12-31 | validation 2023-01-01 .. 2023-06-30 | test 2023-07-01 ..
3. Expanding-window cross-validation over 2022 Q1-Q4 for the three model families.
4. Tune LightGBM and XGBoost on the validation period with early stopping.
5. Pick the best family on VALIDATION, refit on train + validation, predict the test set ONCE.
6. Save the best model, the feature list, test predictions and report tables.

Usage (from the project root)
-----------------------------
    python scripts/train_evaluate.py            # full run (CV + tuning), about 1 hour
    python scripts/train_evaluate.py --skip-cv  # tuning + final test, about 40 minutes
    python scripts/train_evaluate.py --fast     # fixed settings from the notebook, about 15 minutes

Outputs
-------
    models/best_model.txt (or .json / .joblib)   models/feature_cols.json
    data/test_predictions.csv
    reports/model_comparison_test.csv            reports/tuning_results.csv   reports/cv_results.csv
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"

TARGET = "units_sold"
ID_COLS = ["series_id", "date"]
BASELINE = "roll_mean_30"        # best naive baseline from Phase 4
RF_SAMPLE = 250_000              # rows used to train the Random Forest (speed)

TRAIN_END = "2022-12-31"
VALID_START, VALID_END = "2023-01-01", "2023-06-30"
TEST_START = "2023-07-01"

# Expanding-window CV folds: (train end, validation start, validation end)
FOLDS = [("2021-12-31", "2022-01-01", "2022-03-31"),
         ("2022-03-31", "2022-04-01", "2022-06-30"),
         ("2022-06-30", "2022-07-01", "2022-09-30"),
         ("2022-09-30", "2022-10-01", "2022-12-31")]

# Settings found by the full tuning run in notebook 05 (used only with --fast)
FAST_CONFIGS = {
    "LightGBM": {"params": {"num_leaves": 127, "objective": "poisson"}, "trees": 699},
    "XGBoost": {"params": {"max_depth": 8, "objective": "count:poisson"}, "trees": 830},
}


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def metrics(y, p):
    """WAPE (total absolute error / total actual), MAE, RMSE and Bias_% (over/under-forecast).

    Predictions are clipped at 0 because units sold cannot be negative.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    return {"WAPE": np.abs(y - p).sum() / y.sum(),
            "MAE": np.abs(y - p).mean(),
            "RMSE": np.sqrt(((y - p) ** 2).mean()),
            "Bias_%": (p.sum() - y.sum()) / y.sum() * 100}


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_features():
    """Load the Phase 4 feature table and add the evaluation-only `so_today` column."""
    parquet, csv = DATA / "fmcg_features.parquet", DATA / "fmcg_features.csv"
    if parquet.exists():
        feats = pd.read_parquet(parquet)
    else:
        feats = pd.read_csv(csv, parse_dates=["date"])

    feature_cols = [c for c in feats.columns if c not in ID_COLS + [TARGET]]
    print(f"Rows: {len(feats):,} | Features: {len(feature_cols)}")

    # `so_today` (was there a stock-out today) is used ONLY to evaluate the models on normal days.
    # It is never given to a model as a feature, because it is only known after the day ends.
    clean = DATA / "fmcg_clean.csv"
    if clean.exists():
        flag = pd.read_csv(clean, usecols=["store_id", "sku_id", "date", "stock_out_flag"],
                           parse_dates=["date"])
        flag["series_id"] = flag["store_id"] + "_" + flag["sku_id"]
        feats = feats.merge(
            flag[["series_id", "date", "stock_out_flag"]].rename(columns={"stock_out_flag": "so_today"}),
            on=["series_id", "date"], how="left")
    else:
        print("data/fmcg_clean.csv not found: skipping the non-stock-out evaluation.")
    return feats, feature_cols


def split(feats):
    """Chronological train / validation / test split."""
    train = feats[feats["date"] <= TRAIN_END]
    valid = feats[(feats["date"] >= VALID_START) & (feats["date"] <= VALID_END)]
    test = feats[feats["date"] >= TEST_START]
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        print(f"{name:11s} {part['date'].min().date()} to {part['date'].max().date()}  rows: {len(part):,}")
    return train, valid, test


# ----------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------
def make_rf():
    return RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=50,
                                 max_features=0.5, n_jobs=-1, random_state=SEED)


def make_lgbm(**kw):
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                  importance_type="gain", random_state=SEED, n_jobs=-1, verbose=-1)
    params.update(kw)
    return lgb.LGBMRegressor(**params)


def make_xgb(**kw):
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=50, tree_method="hist",
                  random_state=SEED, n_jobs=-1)
    params.update(kw)
    return xgb.XGBRegressor(**params)


def fit_predict(name, train_df, pred_df, feature_cols):
    """Fit a model family with default settings on train_df and predict pred_df."""
    if name == "RandomForest":
        tr = train_df.sample(min(RF_SAMPLE, len(train_df)), random_state=SEED)
        model = make_rf()
    else:
        tr = train_df
        model = make_lgbm() if name == "LightGBM" else make_xgb()
    model.fit(tr[feature_cols], tr[TARGET])
    return model, model.predict(pred_df[feature_cols])


# ----------------------------------------------------------------------------------------
# Step 3: expanding-window cross-validation
# ----------------------------------------------------------------------------------------
def run_cv(feats, feature_cols):
    rows = []
    for i, (tr_end, va_start, va_end) in enumerate(FOLDS, 1):
        tr = feats[feats["date"] <= tr_end]
        va = feats[(feats["date"] >= va_start) & (feats["date"] <= va_end)]
        rows.append({"fold": i, "model": f"Baseline {BASELINE}",
                     "WAPE": metrics(va[TARGET], va[BASELINE])["WAPE"]})
        for name in ["RandomForest", "LightGBM", "XGBoost"]:
            t0 = time.time()
            _, pred = fit_predict(name, tr, va, feature_cols)
            wape = metrics(va[TARGET], pred)["WAPE"]
            rows.append({"fold": i, "model": name, "WAPE": wape})
            print(f"fold {i} {name:12s} WAPE {wape:.3f}  ({time.time() - t0:.0f}s)")

    cv = pd.DataFrame(rows)
    table = cv.pivot(index="model", columns="fold", values="WAPE")
    table["mean"] = table.mean(axis=1)
    table["std"] = table[[1, 2, 3, 4]].std(axis=1)
    table = table.sort_values("mean")
    print("\nWAPE by fold (lower is better):")
    print(table.round(3).to_string())
    table.to_csv(REPORTS / "cv_results.csv")


# ----------------------------------------------------------------------------------------
# Step 4: tuning on the validation period with early stopping
# ----------------------------------------------------------------------------------------
def run_tuning(train, valid, feature_cols):
    rows = []

    for obj in ["regression", "poisson"]:
        for leaves in [31, 63, 127]:
            t0 = time.time()
            m = make_lgbm(n_estimators=1000, num_leaves=leaves, objective=obj)
            m.fit(train[feature_cols], train[TARGET],
                  eval_set=[(valid[feature_cols], valid[TARGET])], eval_metric="l1",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            wape = metrics(valid[TARGET], m.predict(valid[feature_cols]))["WAPE"]
            key = f"LightGBM {obj} leaves={leaves}"
            rows.append({"config": key, "family": "LightGBM", "num_leaves": leaves, "objective": obj,
                         "trees": m.best_iteration_, "valid_WAPE": wape})
            print(f"{key:40s} trees {m.best_iteration_:4d}  WAPE {wape:.3f}  ({time.time() - t0:.0f}s)")

    for obj in ["reg:squarederror", "count:poisson"]:
        for depth in [6, 8, 10]:
            t0 = time.time()
            m = make_xgb(n_estimators=1000, early_stopping_rounds=50, eval_metric="mae",
                         max_depth=depth, objective=obj)
            m.fit(train[feature_cols], train[TARGET],
                  eval_set=[(valid[feature_cols], valid[TARGET])], verbose=False)
            wape = metrics(valid[TARGET], m.predict(valid[feature_cols]))["WAPE"]
            key = f"XGBoost {obj} depth={depth}"
            rows.append({"config": key, "family": "XGBoost", "max_depth": depth, "objective": obj,
                         "trees": m.best_iteration + 1, "valid_WAPE": wape})
            print(f"{key:40s} trees {m.best_iteration + 1:4d}  WAPE {wape:.3f}  ({time.time() - t0:.0f}s)")

    tune = pd.DataFrame(rows).sort_values("valid_WAPE")
    print("\nTuning results (lower WAPE is better):")
    print(tune[["config", "trees", "valid_WAPE"]].round(3).to_string(index=False))
    tune.to_csv(REPORTS / "tuning_results.csv", index=False)
    return tune


def best_configs_from_tuning(tune):
    """Best LightGBM and XGBoost configuration (lowest validation WAPE)."""
    lgb_row = tune[tune["family"] == "LightGBM"].iloc[0]
    xgb_row = tune[tune["family"] == "XGBoost"].iloc[0]
    return {
        "LightGBM": {"params": {"num_leaves": int(lgb_row["num_leaves"]), "objective": lgb_row["objective"]},
                     "trees": int(lgb_row["trees"]), "valid_WAPE": float(lgb_row["valid_WAPE"])},
        "XGBoost": {"params": {"max_depth": int(xgb_row["max_depth"]), "objective": xgb_row["objective"]},
                    "trees": int(xgb_row["trees"]), "valid_WAPE": float(xgb_row["valid_WAPE"])},
    }


def build_final(name, cfg):
    """Model with the chosen settings and number of trees."""
    if name == "LightGBM":
        return make_lgbm(n_estimators=cfg["trees"], **cfg["params"])
    return make_xgb(n_estimators=cfg["trees"], **cfg["params"])


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Phase 5 forecasting models.")
    parser.add_argument("--skip-cv", action="store_true", help="skip the cross-validation step")
    parser.add_argument("--fast", action="store_true",
                        help="skip CV and tuning; use the fixed settings found in notebook 05")
    args = parser.parse_args()

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    feats, feature_cols = load_features()
    train, valid, test = split(feats)

    print("\nBaseline", BASELINE)
    for name, part in [("validation", valid), ("test", test)]:
        print(f"  {name}:", {k: round(v, 3) for k, v in metrics(part[TARGET], part[BASELINE]).items()})

    # Step 3: cross-validation
    if not (args.skip_cv or args.fast):
        print("\n=== Expanding-window cross-validation ===")
        run_cv(feats, feature_cols)

    # Step 4: tuning (or fixed settings)
    if args.fast:
        print("\n=== Fast mode: fixed settings, validating on Jan-Jun 2023 ===")
        configs = {}
        for name, cfg in FAST_CONFIGS.items():
            m = build_final(name, cfg)
            m.fit(train[feature_cols], train[TARGET])
            configs[name] = {**cfg, "valid_WAPE": metrics(valid[TARGET], m.predict(valid[feature_cols]))["WAPE"]}
    else:
        print("\n=== Tuning on the validation period ===")
        configs = best_configs_from_tuning(run_tuning(train, valid, feature_cols))

    # Random Forest has no early stopping: one run with default settings
    t0 = time.time()
    _, rf_pred = fit_predict("RandomForest", train, valid, feature_cols)
    rf_wape = metrics(valid[TARGET], rf_pred)["WAPE"]
    print(f"\nRandomForest validation WAPE {rf_wape:.3f}  ({time.time() - t0:.0f}s)")

    validation_scores = {"RandomForest": rf_wape,
                         "LightGBM": configs["LightGBM"]["valid_WAPE"],
                         "XGBoost": configs["XGBoost"]["valid_WAPE"]}
    best_name = min(validation_scores, key=validation_scores.get)
    print("Validation WAPE by family:", {k: round(v, 4) for k, v in validation_scores.items()})
    print("Best family on validation:", best_name)

    # Step 5: refit on train + validation and predict the test set once
    print("\n=== Final test (Jul-Dec 2023), used once ===")
    train_valid = pd.concat([train, valid])
    final_models, test_preds = {}, {}
    for name in ["LightGBM", "XGBoost"]:
        m = build_final(name, configs[name])
        m.fit(train_valid[feature_cols], train_valid[TARGET])
        final_models[name], test_preds[name] = m, m.predict(test[feature_cols])
    final_models["RandomForest"], test_preds["RandomForest"] = fit_predict(
        "RandomForest", train_valid, test, feature_cols)

    base_label = f"Baseline {BASELINE}"
    rows = [{"model": base_label, **metrics(test[TARGET], test[BASELINE])}]
    rows += [{"model": n, **metrics(test[TARGET], p)} for n, p in test_preds.items()]
    results = pd.DataFrame(rows).set_index("model")
    results["improvement_vs_baseline_%"] = (1 - results["WAPE"] / results.loc[base_label, "WAPE"]) * 100
    print("TEST SET (Jul-Dec 2023):")
    print(results.round(3))
    results.to_csv(REPORTS / "model_comparison_test.csv")

    if "so_today" in test.columns:
        mask = (test["so_today"] == 0).values
        rows = [{"model": base_label, **metrics(test[TARGET].values[mask], test[BASELINE].values[mask])}]
        rows += [{"model": n, **metrics(test[TARGET].values[mask], p[mask])} for n, p in test_preds.items()]
        print("\nTest set, non-stock-out days only (stock-outs distort sales, not demand):")
        print(pd.DataFrame(rows).set_index("model").round(3))

    # Step 6: save for Phase 6 (SHAP)
    best_model = final_models[best_name]
    if best_name == "LightGBM":
        best_model.booster_.save_model(str(MODELS / "best_model.txt"))
        saved = "models/best_model.txt"
    elif best_name == "XGBoost":
        best_model.save_model(str(MODELS / "best_model.json"))
        saved = "models/best_model.json"
    else:
        import joblib
        joblib.dump(best_model, MODELS / "best_model.joblib", compress=3)
        saved = "models/best_model.joblib"

    with open(MODELS / "feature_cols.json", "w") as f:
        json.dump({"best_model": best_name, "features": feature_cols, "target": TARGET}, f, indent=2)

    out = test[["series_id", "date", TARGET]].rename(columns={TARGET: "y"}).copy()
    out["pred"] = np.clip(test_preds[best_name], 0, None)
    out.to_csv(DATA / "test_predictions.csv", index=False)

    print(f"\nSaved: {saved} | models/feature_cols.json | data/test_predictions.csv | reports/*.csv")


if __name__ == "__main__":
    main()
