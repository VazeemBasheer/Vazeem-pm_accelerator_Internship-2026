from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Global Weather Repository.csv"
RAW_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_raw.parquet"
CLEAN_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_clean.parquet"

IMPERIAL_COLS = [
    "temperature_fahrenheit",
    "feels_like_fahrenheit",
    "wind_mph",
    "gust_mph",
    "pressure_in",
    "precip_in",
    "visibility_miles",
]

PHYSICAL_BOUNDS = {
    "temperature_celsius": (-90.0, 60.0),
    "feels_like_celsius": (-90.0, 60.0),
    "humidity": (0.0, 100.0),
    "cloud": (0.0, 100.0),
    "pressure_mb": (850.0, 1100.0),
    "precip_mm": (0.0, None),
    "wind_kph": (0.0, None),
    "gust_kph": (0.0, None),
    "visibility_km": (0.0, None),
    "uv_index": (0.0, None),
    "air_quality_Carbon_Monoxide": (0.0, None),
    "air_quality_Ozone": (0.0, None),
    "air_quality_Nitrogen_dioxide": (0.0, None),
    "air_quality_Sulphur_dioxide": (0.0, None),
    "air_quality_PM2.5": (0.0, None),
    "air_quality_PM10": (0.0, None),
}


def load_input() -> pd.DataFrame:
    # Read from interim parquet if available, else raw CSV
    if RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError(f"Neither {RAW_PARQUET} nor {RAW_CSV} found.")


def parse_clock_to_minutes(series: pd.Series) -> pd.Series:
    # Convert 12-hour clock text into minutes since midnight
    parsed = pd.to_datetime(series, format="%I:%M %p", errors="coerce")
    return parsed.dt.hour * 60.0 + parsed.dt.minute


def standardize_panel(df: pd.DataFrame) -> pd.DataFrame:
    # Clean names, parse dates, and deduplicate to one row per city per day
    df = df.copy()
    df["location_name"] = df["location_name"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["lat_round"] = df["latitude"].round(2)
    df["lon_round"] = df["longitude"].round(2)

    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df = df.dropna(subset=["last_updated", "location_name"])
    df["date"] = df["last_updated"].dt.normalize()
    df["hour"] = df["last_updated"].dt.hour

    df = df.drop_duplicates()
    df["_hour_dist"] = (df["hour"] - 12).abs()
    df = (
        df.sort_values(["location_name", "date", "_hour_dist", "last_updated"])
        .drop_duplicates(subset=["location_name", "date"], keep="first")
        .drop(columns=["_hour_dist"])
    )
    return df


def clean_physical_values(df: pd.DataFrame) -> pd.DataFrame:
    # Replace -9999 sentinels and out-of-range readings with NaN
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([-9999, -9999.0], np.nan)

    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        if low is not None:
            df.loc[df[col] < low, col] = np.nan
        if high is not None:
            df.loc[df[col] > high, col] = np.nan
    return df


def engineer_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    # Derive day length from sunrise/sunset and wind sine/cosine from wind_degree
    df = df.copy()

    if {"sunrise", "sunset"}.issubset(df.columns):
        sr_min = parse_clock_to_minutes(df["sunrise"])
        ss_min = parse_clock_to_minutes(df["sunset"])
        df["day_length_hours"] = ((ss_min - sr_min) % (24 * 60)) / 60.0
        df = df.drop(columns=["sunrise", "sunset", "moonrise", "moonset"], errors="ignore")

    if "wind_degree" in df.columns:
        rad = np.deg2rad(df["wind_degree"])
        df["wind_sin"] = np.sin(rad)
        df["wind_cos"] = np.cos(rad)
        df = df.drop(columns=["wind_degree", "wind_direction"], errors="ignore")

    if "condition_text" in df.columns:
        df["condition_text"] = df["condition_text"].astype("category")

    return df


def validate(df: pd.DataFrame) -> None:
    # Assert unique city-date keys and valid physical bounds
    dup_count = df.duplicated(subset=["location_name", "date"]).sum()
    assert dup_count == 0, f"Found {dup_count} duplicate (location_name, date) rows."

    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        valid_vals = df[col].dropna()
        if low is not None:
            assert (valid_vals >= low).all(), f"{col} has values below {low}"
        if high is not None:
            assert (valid_vals <= high).all(), f"{col} has values above {high}"


def main() -> None:
    # Run structural cleaning pipeline and export weather_clean.parquet
    df = load_input()
    df = df.drop(columns=[c for c in IMPERIAL_COLS if c in df.columns])
    df = standardize_panel(df)

    missing_before = (df.isna().mean() * 100).round(2)
    df = clean_physical_values(df)
    missing_after = (df.isna().mean() * 100).round(2)

    df = engineer_structural_features(df)
    df = df.sort_values(["location_name", "date"]).reset_index(drop=True)
    validate(df)

    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)

    audit = pd.DataFrame({"missing_before_%": missing_before, "missing_after_%": missing_after})
    print(audit[audit["missing_after_%"] > 0])
    print(f"\nSaved {len(df):,} clean rows across {df['location_name'].nunique()} cities to {CLEAN_PARQUET}")


if __name__ == "__main__":
    main()