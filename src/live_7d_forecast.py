import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from clean import (
    IMPERIAL_COLS,
    clean_physical_values,
    engineer_structural_features,
    standardize_panel,
    validate,
)
from features_engineering import DROP_NOISE_COLS, add_engineered_features

DATASET_HANDLE = "nelgiriyewithana/global-weather-repository"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Global Weather Repository.csv"
TRAIN_REF_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_features.parquet"
LIVE_FEATURES_PARQUET = PROJECT_ROOT / "data" / "interim" / "live_recent_features.parquet"
LIVE_FORECASTS_PARQUET = PROJECT_ROOT / "data" / "processed" / "live_7d_forecasts.parquet"
DEFAULT_FORECASTS_PARQUET = PROJECT_ROOT / "data" / "processed" / "forecasts.parquet"

LOOKBACK_DAYS = 35
LATEST_WINDOW_DAYS = 7


def download_latest_csv(target_csv: Path, skip_download: bool = False) -> Path:
    if skip_download and target_csv.exists():
        return target_csv

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE, force_download=True))

    csv_files = list(cache_path.rglob("*.csv"))
    if csv_files:
        shutil.copy2(csv_files[0], target_csv)
        return target_csv

    for candidate in cache_path.rglob("*"):
        if candidate.is_file():
            try:
                if candidate.suffix.lower() == ".parquet":
                    pd.read_parquet(candidate).to_csv(target_csv, index=False)
                    return target_csv
                df_try = pd.read_csv(candidate)
                if "last_updated" in df_try.columns or "location_name" in df_try.columns:
                    df_try.to_csv(target_csv, index=False)
                    return target_csv
            except Exception:
                continue

    for file_name in ("GlobalWeatherRepository.csv", "Global Weather Repository.csv"):
        try:
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                DATASET_HANDLE,
                file_name,
            )
            df.to_csv(target_csv, index=False)
            return target_csv
        except Exception:
            continue

    raise FileNotFoundError(f"Could not locate dataset table in {cache_path}")


def build_live_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Clean using exact Stage 1 rules
    df = raw_df.drop(columns=[c for c in IMPERIAL_COLS if c in raw_df.columns], errors="ignore")
    df = standardize_panel(df)
    df = clean_physical_values(df)
    df = engineer_structural_features(df)
    df = df.sort_values(["coord_key", "date"]).reset_index(drop=True)
    validate(df)

    # 2. Keep only the recent lookback window per city for rolling and sequence history
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["date"] >= cutoff].copy().reset_index(drop=True)

    df = df.drop(columns=[c for c in DROP_NOISE_COLS if c in df.columns], errors="ignore")
    df["split"] = "live"
    df["month"] = df["date"].dt.month

    # 3. Forward-fill short gaps and fill remaining missing numerics from frozen train monthly medians
    exclude = {
        "date",
        "split",
        "coord_key",
        "location_name",
        "country",
        "latitude",
        "longitude",
        "lat_round",
        "lon_round",
        "month",
        "hour",
        "last_updated_epoch",
    }
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    df[num_cols] = df.groupby("coord_key", sort=False)[num_cols].ffill(limit=3)

    if TRAIN_REF_PARQUET.exists():
        ref_df = pd.read_parquet(TRAIN_REF_PARQUET)
        train_ref = ref_df[ref_df["split"] == "train"] if "split" in ref_df.columns else ref_df
        shared_cols = [c for c in num_cols if c in train_ref.columns]

        city_month_med = train_ref.groupby(["coord_key", "month"])[shared_cols].median()
        global_month_med = train_ref.groupby("month")[shared_cols].median()

        df = df.set_index(["coord_key", "month"])
        df[shared_cols] = df[shared_cols].fillna(city_month_med)
        df = df.reset_index().set_index("month")
        df[shared_cols] = df[shared_cols].fillna(global_month_med)
        df = df.reset_index()

        if "city_month_temp_clim" in train_ref.columns:
            clim = (
                train_ref[["coord_key", "month", "city_month_temp_clim"]]
                .drop_duplicates(subset=["coord_key", "month"])
            )
        else:
            clim = (
                train_ref.groupby(["coord_key", "month"])["temperature_celsius"]
                .mean()
                .rename("city_month_temp_clim")
                .reset_index()
            )
    else:
        clim = (
            df.groupby(["coord_key", "month"])["temperature_celsius"]
            .mean()
            .rename("city_month_temp_clim")
            .reset_index()
        )

    # 4. Build lag, rolling, interaction, and calendar features without dropping latest rows
    df = add_engineered_features(df)
    df = df.drop(columns=["city_month_temp_clim", "temp_vs_city_clim"], errors="ignore")
    df = df.merge(clim, on=["coord_key", "month"], how="left")
    df["city_month_temp_clim"] = df["city_month_temp_clim"].fillna(df["temperature_celsius"])
    df["temp_vs_city_clim"] = df["temperature_celsius"] - df["city_month_temp_clim"]
    df["is_outlier"] = 0

    return df.sort_values(["coord_key", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch latest Kaggle weather data and generate 7-day live forecasts."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing local RAW_CSV instead of downloading from Kaggle.",
    )
    parser.add_argument(
        "--update-dashboard",
        action="store_true",
        help="Also write the latest 7-day slice to data/processed/forecasts.parquet for app.py.",
    )
    args = parser.parse_args()

    csv_path = download_latest_csv(RAW_CSV, skip_download=args.skip_download)
    raw_df = pd.read_csv(csv_path)
    print(f"[live] Raw dataset max timestamp: {raw_df['last_updated'].max()}")

    feat_df = build_live_features(raw_df)
    LIVE_FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(LIVE_FEATURES_PARQUET, index=False)

    temp_full_preds = PROJECT_ROOT / "data" / "processed" / "_live_35d_preds.parquet"
    cmd = [
        sys.executable,
        str(_HERE / "predict.py"),
        "--input",
        str(LIVE_FEATURES_PARQUET),
        "--output",
        str(temp_full_preds),
    ]
    subprocess.run(cmd, check=True)

    preds_df = pd.read_parquet(temp_full_preds)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    max_dt = preds_df["date"].max()
    window_start = max_dt - pd.Timedelta(days=LATEST_WINDOW_DAYS - 1)
    latest_7d = (
        preds_df[preds_df["date"] >= window_start]
        .sort_values(["coord_key", "date"])
        .reset_index(drop=True)
    )

    LIVE_FORECASTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    latest_7d.to_parquet(LIVE_FORECASTS_PARQUET, index=False)
    temp_full_preds.unlink(missing_ok=True)

    if args.update_dashboard:
        latest_7d.to_parquet(DEFAULT_FORECASTS_PARQUET, index=False)
        print(f"[live] Updated main dashboard file: {DEFAULT_FORECASTS_PARQUET}")

    print(
        f"[live] Saved {len(latest_7d):,} rows ({window_start.date()} to {max_dt.date()}) -> {LIVE_FORECASTS_PARQUET}"
    )


if __name__ == "__main__":
    main()