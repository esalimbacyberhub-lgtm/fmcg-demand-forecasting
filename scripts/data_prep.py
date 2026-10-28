"""Phase 1: data integrity checks and the checked dataset.

Reproduces notebook 01_data_prep.ipynb. Run from anywhere:
    python scripts/data_prep.py
    python scripts/data_prep.py --raw path/to/raw.csv --out path/to/output.csv
"""
import argparse
from pathlib import Path

import pandas as pd

# Repo root = the folder above scripts/, so the script works from any working directory.
ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "fmcg_sales_3years_1M_rows.csv"
OUT_DEFAULT = ROOT / "data" / "fmcg_clean.csv"

# One record per day, store and SKU is expected.
KEY = ["date", "store_id", "sku_id"]


def load_raw(path):
    """Read the raw CSV and convert text dates to datetime (needed later for sorting, filters and lags)."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def print_report(df):
    """Print the integrity checks from notebook 01. Nothing is changed in the data."""
    print("Shape:", df.shape)
    print("Missing values (total):", int(df.isna().sum().sum()))
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("Unique days:", df["date"].nunique())

    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("Duplicate date-store-SKU records:", int(df.duplicated(subset=KEY).sum()))

    # Look for inconsistent spellings, e.g. 'Supermarket' vs 'supermarket'.
    for col in ["country", "channel", "category"]:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().to_string())

    print()
    for label, col in [("Countries", "country"), ("Cities", "city"), ("Stores", "store_id"),
                       ("SKUs", "sku_id"), ("Suppliers", "supplier_id")]:
        print(f"{label}: {df[col].nunique()}")
    print("Store-SKU series:", len(df.groupby(["store_id", "sku_id"]).size()))

    checks = {
        "units_sold <= 0": (df["units_sold"] <= 0).sum(),
        "lead_time_days <= 0": (df["lead_time_days"] <= 0).sum(),
        "discount_pct outside 0-1": ((df["discount_pct"] < 0) | (df["discount_pct"] > 1)).sum(),
        "net_sales > gross_sales": (df["net_sales"] > df["gross_sales"]).sum(),
        "stock_on_hand < 0": (df["stock_on_hand"] < 0).sum(),
    }
    print("\nValue checks (rows affected):")
    print(pd.Series(checks).to_string())

    # Zero or negative sales: are they stock-outs?
    bad = df[df["units_sold"] <= 0]
    print("\nZero sales:", int((bad["units_sold"] == 0).sum()))
    print("Negative sales:", int((bad["units_sold"] < 0).sum()))
    print("Stock-out flag among these rows:")
    print(bad["stock_out_flag"].value_counts().to_string())


def save_clean(df, out_path):
    """Save the checked dataset. It stays local because of .gitignore."""
    df.to_csv(out_path, index=False)
    print("\nSaved:", df.shape, "->", out_path)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 data checks and checked dataset.")
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT, help="raw sales CSV")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="where to write the checked CSV")
    args = parser.parse_args()

    df = load_raw(args.raw)
    print_report(df)
    save_clean(df, args.out)


if __name__ == "__main__":
    main()
