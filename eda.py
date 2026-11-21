import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry_convert as pc
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_split.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_CORE_DIR = PROJECT_ROOT / "reports" / "figures" / "core"
FIG_ADV_DIR = PROJECT_ROOT / "reports" / "figures" / "advanced"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "eda_manifest.json"


def country_to_continent(country_name: str) -> str:
    # Map country name to continent 
    try:
        alpha2 = pc.country_name_to_country_alpha2(country_name)
        cont_code = pc.country_alpha2_to_continent_code(alpha2)
        return pc.convert_continent_code_to_continent_name(cont_code)
    except Exception:
        return "Other"


def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add month, rain flag, and continent columns for grouping
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["is_rain"] = (df["precip_mm"] > 0.1).astype(int)
    country_map = {c: country_to_continent(c) for c in df["country"].unique()}
    df["continent"] = df["country"].map(country_map)
    return df


def run_core_eda(train: pd.DataFrame) -> list[dict]:
    # Generate temperature, precipitation, and time-series plots
    FIG_CORE_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    # 1. Temperature distribution overall and by continent
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train["temperature_celsius"].dropna(), bins=40, kde=True, ax=axes[0])
    axes[0].set_title("Temperature Distribution (°C)")
    sns.boxplot(data=train, x="continent", y="temperature_celsius", ax=axes[1])
    axes[1].set_title("Temperature by Continent")
    axes[1].tick_params(axis="x", rotation=25)
    p1 = FIG_CORE_DIR / "temp_distribution.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    records.append({"section": "core", "id": "temp_dist", "title": "Temperature Distribution", "path": str(p1.relative_to(PROJECT_ROOT))})

    # 2. Temperature over time for 4 contrasting cities
    sample_cities = train.groupby("location_name")["date"].count().nlargest(4).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 4))
    for city in sample_cities:
        sub = train[train["location_name"] == city].sort_values("date")
        ax.plot(sub["date"], sub["temperature_celsius"], label=city, alpha=0.85)
    ax.set_title("Daily Temperature Over Time (Sample Cities)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    p2 = FIG_CORE_DIR / "temp_over_time.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    records.append({"section": "core", "id": "temp_time", "title": "Temperature Over Time", "path": str(p2.relative_to(PROJECT_ROOT))})

    # 3. STL seasonal decomposition on the city with longest clean series
    top_city = sample_cities[0]
    city_ts = (
        train[train["location_name"] == top_city]
        .set_index("date")["temperature_celsius"]
        .asfreq("D")
        .interpolate(limit=3)
        .dropna()
    )
    if len(city_ts) >= 14:
        stl = STL(city_ts, period=7, robust=True).fit()
        fig = stl.plot()
        fig.set_size_inches(11, 6)
        p3 = FIG_CORE_DIR / "stl_decomposition.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        records.append({"section": "core", "id": "stl_decomp", "title": f"STL Decomposition ({top_city})", "path": str(p3.relative_to(PROJECT_ROOT))})

    # Plot precipitation distribution and monthly rainfall
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rain_pos = train.loc[train["precip_mm"] > 0, "precip_mm"]
    sns.histplot(np.log1p(rain_pos), bins=35, ax=axes[0])
    zero_pct = (train["precip_mm"] == 0).mean() * 100
    axes[0].set_title(f"Log1p Precipitation (Zero-Rain Days: {zero_pct:.1f}%)")
    monthly_rain = train.groupby(["month", "continent"])["precip_mm"].mean().reset_index()
    sns.barplot(data=monthly_rain, x="month", y="precip_mm", hue="continent", ax=axes[1])
    axes[1].set_title("Mean Precipitation by Month & Continent")
    p4 = FIG_CORE_DIR / "precip_patterns.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    records.append({"section": "core", "id": "precip_patterns", "title": "Precipitation Distribution & Seasonality", "path": str(p4.relative_to(PROJECT_ROOT))})

    # 5. ACF / PACF and humidity-cloud rain signal
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if len(city_ts) > 20:
        plot_acf(city_ts, lags=min(21, len(city_ts) // 2 - 1), ax=axes[0])
        plot_pacf(city_ts, lags=min(21, len(city_ts) // 2 - 1), ax=axes[1])
    sns.boxplot(data=train, x="is_rain", y="humidity", ax=axes[2])
    axes[2].set_title("Humidity: Dry (0) vs Rain (1)")
    p5 = FIG_CORE_DIR / "autocorr_and_rain_signal.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    records.append({"section": "core", "id": "acf_pacf", "title": "Autocorrelation & Rain Drivers", "path": str(p5.relative_to(PROJECT_ROOT))})

    # Export core EDA summary CSV 
    core_summary = train.groupby("continent").agg(
        cities=("location_name", "nunique"),
        temp_mean=("temperature_celsius", "mean"),
        temp_std=("temperature_celsius", "std"),
        rain_day_pct=("is_rain", lambda s: s.mean() * 100),
        precip_mean=("precip_mm", "mean"),
    ).round(2)
    core_summary.to_csv(PROCESSED_DIR / "eda_core_summary.csv")

    return records


def run_advanced_eda(train: pd.DataFrame) -> list[dict]:
    # Generate geographical, air-quality, and extreme-value plots
    FIG_ADV_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    # 1. Latitude gradient vs mean city temperature
    city_geo = train.groupby(["location_name", "continent"], as_index=False).agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        temp_mean=("temperature_celsius", "mean"),
        temp_range=("temperature_celsius", lambda s: s.max() - s.min()),
        precip_mean=("precip_mm", "mean"),
        pm25_mean=("air_quality_PM2.5", "mean"),
    )
    city_geo["abs_lat"] = city_geo["latitude"].abs()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.regplot(data=city_geo, x="abs_lat", y="temp_mean", scatter_kws={"alpha": 0.6}, ax=ax)
    ax.set_title("Mean City Temperature vs. Absolute Latitude")
    ax.set_xlabel("Absolute Latitude (°)")
    ax.set_ylabel("Mean Temperature (°C)")
    p1 = FIG_ADV_DIR / "latitude_gradient.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    records.append({"section": "advanced", "id": "lat_gradient", "title": "Global Latitude Temperature Gradient", "path": str(p1.relative_to(PROJECT_ROOT))})

    # 2. Air quality vs weather Spearman correlation heatmap
    aq_cols = [c for c in train.columns if c.startswith("air_quality_") and train[c].dtype != "object"]
    wx_cols = [c for c in ["temperature_celsius", "humidity", "pressure_mb", "wind_kph", "cloud", "precip_mm"] if c in train.columns]
    if aq_cols and wx_cols:
        corr = train[aq_cols + wx_cols].corr(method="spearman").loc[aq_cols, wx_cols]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Air Quality vs. Weather (Spearman Correlation)")
        p2 = FIG_ADV_DIR / "air_quality_weather_corr.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        records.append({"section": "advanced", "id": "aq_corr", "title": "Air Quality & Weather Correlations", "path": str(p2.relative_to(PROJECT_ROOT))})

    # 3. Count extreme temperatures by continent
    med = train.groupby(["location_name", "month"])["temperature_celsius"].transform("median")
    mad = train.groupby(["location_name", "month"])["temperature_celsius"].transform(lambda s: np.median(np.abs(s - np.median(s))))
    robust_z = (train["temperature_celsius"] - med) / (1.4826 * mad.replace(0, np.nan))
    train_anom = train[robust_z.abs() > 3.5]

    fig, ax = plt.subplots(figsize=(8, 4))
    anom_counts = train_anom["continent"].value_counts()
    sns.barplot(x=anom_counts.index, y=anom_counts.values, ax=ax)
    ax.set_title("Extreme Temperature Events (|Robust Z| > 3.5) by Continent")
    ax.set_ylabel("Flagged City-Days")
    ax.tick_params(axis="x", rotation=25)
    p3 = FIG_ADV_DIR / "extreme_anomalies_by_continent.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    records.append({"section": "advanced", "id": "anom_counts", "title": "Extreme Weather Frequency", "path": str(p3.relative_to(PROJECT_ROOT))})

    # Export city-level geo summary CSV for dashboard maps and tables
    city_geo.round(3).to_csv(PROCESSED_DIR / "eda_advanced_city_geo.csv", index=False)

    return records


def main() -> None:
    # Run Core EDA and Advanced EDA on training data only and save a dashboard manifest
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SPLIT_PARQUET)
    train = add_helper_columns(df[df["split"] == "train"])

    manifest = {
        "core": run_core_eda(train),
        "advanced": run_advanced_eda(train),
        "tables": {
            "core_summary": "data/processed/eda_core_summary.csv",
            "advanced_city_geo": "data/processed/eda_advanced_city_geo.csv",
        },
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest['core'])} core EDA and {len(manifest['advanced'])} advanced EDA figures.")


if __name__ == "__main__":
    main()