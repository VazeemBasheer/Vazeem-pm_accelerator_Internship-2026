from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_clean.parquet"
SPLIT_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_split.parquet"

TRAIN_SHARE = 0.70
VALID_SHARE = 0.15
MAX_HORIZON = 7
N_SPLITS = 5


def get_split_boundaries(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Get cutoff dates for train (70%) and valid (15%)
    dates = np.sort(df["date"].unique())
    n_dates = len(dates)

    train_end_idx = int(n_dates * TRAIN_SHARE) - 1
    valid_end_idx = int(n_dates * (TRAIN_SHARE + VALID_SHARE)) - 1

    train_end = pd.Timestamp(dates[train_end_idx])
    valid_end = pd.Timestamp(dates[valid_end_idx])

    # Warn if train set has less than 1 full year
    train_days = (train_end - pd.Timestamp(dates[0])).days + 1
    if train_days < 365:
        print(f"Warning: train span is {train_days} days (< 1 full year).")

    return train_end, valid_end


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    # Assign train, valid, and test labels with a 7-day gap between splits
    df = df.copy()
    train_end, valid_end = get_split_boundaries(df)
    gap = pd.Timedelta(days=MAX_HORIZON)

    df["split"] = "drop"
    df.loc[df["date"] <= train_end - gap, "split"] = "train"
    df.loc[(df["date"] > train_end) & (df["date"] <= valid_end - gap), "split"] = "valid"
    df.loc[df["date"] > valid_end, "split"] = "test"

    # Remove the 7-day gap rows to stop target leakage
    return df[df["split"] != "drop"].reset_index(drop=True)


def iter_rolling_folds(train_df: pd.DataFrame, n_splits: int = N_SPLITS, gap: int = MAX_HORIZON):
    # Create expanding-window CV folds with a 7-day gap
    dates = np.sort(train_df["date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    for tr_idx, va_idx in tscv.split(dates):
        tr_dates = dates[tr_idx]
        va_dates = dates[va_idx]
        yield train_df[train_df["date"].isin(tr_dates)], train_df[train_df["date"].isin(va_dates)]


def iter_city_folds(df: pd.DataFrame, n_splits: int = N_SPLITS):
    # Create city-grouped folds to test on unseen cities
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in gkf.split(df, groups=df["location_name"]):
        yield df.iloc[tr_idx], df.iloc[va_idx]


def validate_splits(df: pd.DataFrame) -> None:
    # Check that dates are in order and separated by at least 7 days
    train_dates = df.loc[df["split"] == "train", "date"]
    valid_dates = df.loc[df["split"] == "valid", "date"]
    test_dates = df.loc[df["split"] == "test", "date"]

    min_gap = pd.Timedelta(days=MAX_HORIZON)
    assert valid_dates.min() - train_dates.max() > min_gap, "Insufficient gap between train and valid!"
    assert test_dates.min() - valid_dates.max() > min_gap, "Insufficient gap between valid and test!"


def main() -> None:
    # Load cleaned data, split by date, validate, and save
    df = pd.read_parquet(CLEAN_PARQUET)
    df = assign_splits(df)
    validate_splits(df)

    SPLIT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SPLIT_PARQUET, index=False)

    for name in ["train", "valid", "test"]:
        sub = df[df["split"] == name]
        print(f"{name:5s}: {len(sub):,} rows | {sub['date'].min().date()} to {sub['date'].max().date()}")


if __name__ == "__main__":
    main()