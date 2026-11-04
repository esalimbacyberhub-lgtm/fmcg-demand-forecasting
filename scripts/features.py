"""Phase 4: feature engineering.

Reproduces notebook 04_feature_engineering.ipynb. Run from anywhere:
    python scripts/features.py
    python scripts/features.py --out /tmp/test_features.parquet
Reads data/fmcg_clean.csv (from data_prep.py) and data/abc_xyz_segments.csv (from notebook 02).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import time_split_masks  # shared date split (scripts/metrics.py)

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DEFAULT = ROOT / "data" / "fmcg_clean.csv"
SEGMENTS_DEFAULT = ROOT / "data" / "abc_xyz_segments.csv"
OUT_DEFAULT = ROOT / "data" / "fmcg_features.parquet"

TARGET = "units_sold"
LAG_FEATS = ["lag_1", "lag_7", "lag_14", "lag_30"]
ROLL_FEATS = ["roll_mean_7", "roll_std_7", "roll_mean_30", "roll_std_30"]
HIST_FEATS = ["promo_last7", "stockout_lag1", "stockout_last7"]
# Known in advance for the forecast day (calendar, planned promotions, price, weather).
KNOWN_FEATS = ["promo_flag", "discount_pct", "list_price", "is_holiday", "is_weekend",
               "weekday", "month", "day", "weekofyear", "temperature", "rain_mm"]
STATIC_FEATS = ["country_code", "channel_code", "category_code", "subcategory_code",
                "brand_code", "store_id_code", "sku_id_code", "abc_code", "xyz_rel_code"]
FEATURE_COLS = LAG_FEATS + ROLL_FEATS + HIST_FEATS + KNOWN_FEATS + STATIC_FEATS


def load_clean(path):
    """Read the checked dataset and put rows in time order within each series.

    Lags only make sense if rows are in time order WITHIN each store-SKU series.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["series_id"] = df["store_id"] + "_" + df["sku_id"]
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    print("Rows:", f"{len(df):,}", "| Series:", df["series_id"].nunique())
    return df


def check_panel_complete(df):
    """Stop if any series has missing days: row-based shifts would then be wrong."""
    span = df.groupby("series_id")["date"].agg(n_rows="count", first="min", last="max")
    span["expected_days"] = (span["last"] - span["first"]).dt.days + 1

    print("Duplicate (series, date) rows:", df.duplicated(["series_id", "date"]).sum())
    print("Series with missing days:", (span["n_rows"] != span["expected_days"]).sum())
    print("Distinct start dates:", span["first"].nunique(),
          "| distinct end dates:", span["last"].nunique())
    assert (span["n_rows"] == span["expected_days"]).all(), \
        "Gaps found: reindex to daily before lagging"
    print("Panel is complete: one row per day per series.")


def add_lags(df):
    """Sales k days ago, within each series (shift moves values down k rows)."""
    g = df.groupby("series_id")[TARGET]
    for k in [1, 7, 14, 30]:
        df[f"lag_{k}"] = g.shift(k)
    return df


def add_rolling(df):
    """Rolling mean/std computed on lag_1, so every window ends yesterday (no leakage)."""
    by_series = df.groupby("series_id")["lag_1"]
    for w in [7, 30]:
        # min_periods=w: only compute when a full window of history exists.
        df[f"roll_mean_{w}"] = by_series.rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = by_series.rolling(w, min_periods=w).std().reset_index(level=0, drop=True)
    return df


def add_history_flags(df):
    """Promotions and stock-outs in the previous 7 days (shifted one day first, then summed)."""
    df["promo_lag1"] = df.groupby("series_id")["promo_flag"].shift(1)
    df["promo_last7"] = (df.groupby("series_id")["promo_lag1"]
                           .rolling(7, min_periods=7).sum()
                           .reset_index(level=0, drop=True))
    df["stockout_lag1"] = df.groupby("series_id")["stock_out_flag"].shift(1)
    df["stockout_last7"] = (df.groupby("series_id")["stockout_lag1"]
                              .rolling(7, min_periods=7).sum()
                              .reset_index(level=0, drop=True))
    return df


def add_static_codes(df, segments_path):
    """Turn text attributes into integer codes and attach the Phase 2 ABC/XYZ segments."""
    for col in ["country", "channel", "category", "subcategory", "brand", "store_id", "sku_id"]:
        df[f"{col}_code"] = df[col].astype("category").cat.codes

    seg = pd.read_csv(segments_path)[["series_id", "abc_class", "xyz_rel"]]
    df = df.merge(seg, on="series_id", how="left")
    df["abc_code"] = df["abc_class"].map({"A": 0, "B": 1, "C": 2})
    df["xyz_rel_code"] = df["xyz_rel"].map({"X": 0, "Y": 1, "Z": 2})
    print("Series without a segment:", df["abc_code"].isna().sum())
    return df


def check_alignment(df, seed=42):
    """Spot-check one random series: every lag and rolling value must match the raw sales."""
    rng = np.random.default_rng(seed)
    sid = rng.choice(df["series_id"].unique())
    s = df[df["series_id"] == sid].reset_index(drop=True)
    i = 100

    assert s.loc[i, "lag_1"] == s.loc[i - 1, "units_sold"]
    assert s.loc[i, "lag_7"] == s.loc[i - 7, "units_sold"]
    assert s.loc[i, "lag_30"] == s.loc[i - 30, "units_sold"]
    assert np.isclose(s.loc[i, "roll_mean_7"], s.loc[i - 7:i - 1, "units_sold"].mean())
    assert np.isclose(s.loc[i, "roll_mean_30"], s.loc[i - 30:i - 1, "units_sold"].mean())
    assert df.groupby("series_id").head(1)["lag_1"].isna().all()
    assert df.groupby("series_id").head(30)["lag_30"].isna().groupby(df["series_id"]).sum().eq(30).all()
    print("All alignment tests passed for series", sid)


def build_features(df, segments_path):
    """Run every feature step in order and return the full feature table."""
    df = add_lags(df)
    df = add_rolling(df)
    df = add_history_flags(df)
    df = add_static_codes(df, segments_path)
    return df


def make_model_table(df):
    """Drop warm-up days (no full history yet), keep only the needed columns, shrink float64 to float32."""
    model_df = df.dropna(subset=LAG_FEATS + ROLL_FEATS + HIST_FEATS).copy()
    print("Rows before dropping warm-up days:", f"{len(df):,}", "| after:", f"{len(model_df):,}")

    out = model_df[["series_id", "date", TARGET] + FEATURE_COLS].copy()
    for c in out.select_dtypes("float64").columns:
        out[c] = out[c].astype("float32")  # trees do not need float64
    return out


def save_features(out, path):
    """Save as parquet; if no parquet engine is installed, fall back to CSV."""
    try:
        out.to_parquet(path, index=False)
        print("Saved parquet:", out.shape, "->", path)
    except Exception as e:
        csv_path = Path(path).with_suffix(".csv")
        print("Parquet not available (", type(e).__name__, "), saving CSV instead")
        out.to_csv(csv_path, index=False)
        print("Saved CSV:", out.shape, "->", csv_path)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 feature engineering.")
    parser.add_argument("--clean", type=Path, default=CLEAN_DEFAULT, help="checked sales CSV")
    parser.add_argument("--segments", type=Path, default=SEGMENTS_DEFAULT, help="ABC/XYZ segments CSV")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="output feature file")
    args = parser.parse_args()

    df = load_clean(args.clean)
    check_panel_complete(df)
    df = build_features(df, args.segments)
    check_alignment(df)

    out = make_model_table(df)
    print(len(FEATURE_COLS), "features")
    for name, m in zip(["train", "validation", "test"], time_split_masks(out["date"])):
        d = out.loc[m, "date"]
        print(f"{name:11s} {d.min().date()} to {d.max().date()}  rows: {m.sum():,}")
    save_features(out, args.out)


if __name__ == "__main__":
    main()
