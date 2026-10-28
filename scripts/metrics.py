"""Shared evaluation helpers used across the project.

Definitions are copied from notebooks 04 and 07 so scripts and notebooks give identical numbers.
"""
import numpy as np
import pandas as pd

# Time-based split used throughout the project (no random shuffling: the test set is the future).
TRAIN_END = "2022-12-31"
VALID_START, VALID_END = "2023-01-01", "2023-06-30"
TEST_START = "2023-07-01"


def wape(y, yhat):
    """Weighted absolute percentage error: total absolute error / total actual sales."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.abs(y - yhat).sum() / y.sum()


def metrics(y, p, mask=None):
    """MAE/RMSE in units; MAPE on days with sales > 0 (MAPE is undefined at zero); WAPE = total abs error / total actual."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, None)  # a forecast cannot be negative
    if mask is not None:
        y, p = y[mask], p[mask]
    nz = y > 0
    return {"MAE": np.abs(y - p).mean(),
            "RMSE": np.sqrt(((y - p) ** 2).mean()),
            "MAPE_%": (np.abs(y[nz] - p[nz]) / y[nz]).mean() * 100,
            "WAPE": np.abs(y - p).sum() / y.sum()}


def time_split_masks(dates):
    """Boolean masks (train, validation, test) for a column of dates."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    train = (d <= TRAIN_END).values
    valid = ((d >= VALID_START) & (d <= VALID_END)).values
    test = (d >= TEST_START).values
    return train, valid, test


if __name__ == "__main__":
    # Quick self-test with numbers you can check by hand.
    y = [10, 0, 20]
    p = [8, 1, 25]
    print({k: round(float(v), 4) for k, v in metrics(y, p).items()})
    print("WAPE:", round(float(wape(y, p)), 4))
