from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

SPLIT_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_split.parquet"
FEATURES_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_features.parquet"
SCALER_PATH = PROJECT_ROOT / "models" / "feature_scalers.joblib"

# Collinear/noisy columns identified during EDA to drop
DROP_NOISE_COLS = [
    "air_quality_us-epa-index",
    "air_quality_gb-defra-index",
    "air_quality_PM10",
    "condition_text",
]

SKEWED_COLS = [
    "precip_mm",
    "gust_kph",
    "air_quality_Ozone",
    "air_quality_PM2.5",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Carbon_Monoxide",
    "air_quality_Sulphur_dioxide",
]

STANDARD_SCALE_COLS = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
]


def filter_sparse_cities(df: pd.DataFrame, max_missing: float = 0.20) -> pd.DataFrame:
    # Drop cities missing more than 20% of target history in the training split
    train_sub = df[df["split"] == "train"]
    miss_rate = train_sub.groupby("coord_key")[["temperature_celsius", "precip_mm"]].apply(
        lambda g: g.isna().mean().max()
    )
    valid_cities = miss_rate[miss_rate <= max_missing].index
    return df[df["coord_key"].isin(valid_cities)].reset_index(drop=True)


def impute_leakage_safe(df: pd.DataFrame) -> pd.DataFrame:
    # Forward-fill short gaps (<= 3 days) per city-split, then fill with train monthly medians
    df = df.sort_values(["coord_key", "date"]).reset_index(drop=True)
    df["month"] = df["date"].dt.month

    exclude = {
        "date", "split", "coord_key", "location_name", "country",
        "latitude", "longitude", "lat_round", "lon_round",
        "month", "hour", "last_updated_epoch",
    }
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    # Forward-fill up to 3 days inside each city and split block
    df[num_cols] = df.groupby(["coord_key", "split"], sort=False)[num_cols].ffill(limit=3)

    # Fit monthly medians strictly on training rows
    train_df = df[df["split"] == "train"]
    city_month_med = train_df.groupby(["coord_key", "month"])[num_cols].median()
    global_month_med = train_df.groupby("month")[num_cols].median()

    df = df.set_index(["coord_key", "month"])
    df[num_cols] = df[num_cols].fillna(city_month_med)
    df = df.reset_index().set_index("month")
    df[num_cols] = df[num_cols].fillna(global_month_med)
    df = df.reset_index()

    return df.sort_values(["coord_key", "date"]).reset_index(drop=True)

def flag_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    # Flag extreme temperature or rainfall events using train-fitted robust z-scores
    df = df.copy()
    train_df = df[df["split"] == "train"]

    stats = train_df.groupby(["coord_key", "month"])["temperature_celsius"].agg(
        temp_med="median",
        temp_mad=lambda s: np.median(np.abs(s - np.median(s))),
    )
    df = df.merge(stats, on=["coord_key", "month"], how="left")

    scale = 1.4826 * df["temp_mad"].replace(0, np.nan)
    robust_z = (df["temperature_celsius"] - df["temp_med"]) / scale
    df["is_outlier"] = (robust_z.abs() > z_thresh).fillna(False).astype(int)

    return df.drop(columns=["temp_med", "temp_mad"])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Build past-only lags, rolling stats, domain interactions, and cyclical calendar features
    df = df.sort_values(["coord_key", "date"]).reset_index(drop=True)
    g = df.groupby(["coord_key", "split"], sort=False)

    # Temperature lags & rolling volatility (shifted by 1 to prevent leakage)
    for lag in [1, 2, 3, 7]:
        df[f"temp_lag{lag}"] = g["temperature_celsius"].shift(lag)
    df["temp_diff_1d"] = df["temp_lag1"] - df["temp_lag2"]
    df["temp_roll_mean_3d"] = g["temperature_celsius"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["temp_roll_mean_7d"] = g["temperature_celsius"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["temp_roll_std_7d"] = g["temperature_celsius"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())

    # Precipitation lags & 3-day rain washout accumulation
    for lag in [1, 2, 3]:
        df[f"precip_lag{lag}"] = g["precip_mm"].shift(lag)
    df["rain_washout_3d"] = g["precip_mm"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())

    # Pressure, humidity, wind, and cloud momentum
    for col in ["humidity", "pressure_mb", "wind_kph", "cloud"]:
        if col in df.columns:
            df[f"{col}_lag1"] = g[col].shift(1)
            df[f"{col}_diff_1d"] = df[col] - df[f"{col}_lag1"]

    # Domain interactions identified from EDA correlations
    df["dewpoint_spread"] = (100.0 - df["humidity"]) / 5.0
    df["temp_x_humidity"] = df["temperature_celsius"] * df["humidity"]
    df["ventilation_proxy"] = df["wind_kph"] * (100.0 - df["cloud"])

    # Cyclical calendar & hemisphere-adjusted latitude seasonality
    doy = df["date"].dt.day_of_year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["lat_season"] = np.sign(df["latitude"]) * df["abs_lat"] * df["month_cos"]

    # City monthly climatology anomaly fitted strictly on training data
    city_clim = (
        df[df["split"] == "train"]
        .groupby(["coord_key", "month"])["temperature_celsius"]
        .mean()
        .rename("city_month_temp_clim")
    )
    df = df.merge(city_clim, on=["coord_key", "month"], how="left")
    df["temp_vs_city_clim"] = df["temperature_celsius"] - df["city_month_temp_clim"]

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Create 1-day and 7-day ahead regression and rain-occurrence targets per city-split
    df = df.sort_values(["coord_key", "date"]).reset_index(drop=True)
    g = df.groupby(["coord_key", "split"], sort=False)

    for h in [1, 7]:
        df[f"target_temp_h{h}"] = g["temperature_celsius"].shift(-h)
        df[f"target_precip_h{h}"] = g["precip_mm"].shift(-h)
        df[f"target_rain_h{h}"] = (df[f"target_precip_h{h}"] > 0.1).astype(float)
        df.loc[df[f"target_precip_h{h}"].isna(), f"target_rain_h{h}"] = np.nan

    # Drop trailing rows per city-split where 1-day targets are unavailable
    return df.dropna(subset=["target_temp_h1", "target_precip_h1"]).reset_index(drop=True)


def fit_and_save_scalers(df: pd.DataFrame) -> None:
    # Fit RobustScaler (log1p skewed features) and StandardScaler on train split only
    train_df = df[df["split"] == "train"]
    skew_cols = [c for c in SKEWED_COLS if c in df.columns]
    std_cols = [c for c in STANDARD_SCALE_COLS if c in df.columns]

    robust = RobustScaler().fit(np.log1p(train_df[skew_cols].clip(lower=0)))
    standard = StandardScaler().fit(train_df[std_cols])

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"robust": robust, "skew_cols": skew_cols, "standard": standard, "std_cols": std_cols}, SCALER_PATH)


def main() -> None:
    # Run leakage-safe imputation, feature engineering, target creation, and export
    df = pd.read_parquet(SPLIT_PARQUET)
    df = df.drop(columns=[c for c in DROP_NOISE_COLS if c in df.columns])

    df = filter_sparse_cities(df)
    df = impute_leakage_safe(df)
    df = flag_outliers(df)
    df = add_engineered_features(df)
    df = add_targets(df)

    fit_and_save_scalers(df)

    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, index=False)

    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    print(f"Saved {len(df):,} rows and {len(feature_cols)} features to {FEATURES_PARQUET}")


if __name__ == "__main__":
    main()