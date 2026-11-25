import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

HOLDOUT_PARQUET = DATA_PROCESSED / "forecasts.parquet"
LIVE_7D_PARQUET = DATA_PROCESSED / "live_7d_forecasts.parquet"
WEIGHTS_JSON = MODELS_DIR / "ensemble_weights.json"

PM_ACCELERATOR_MISSION = (
    "The Product Manager Accelerator Program is designed to support PM professionals "
    "through every stage of their careers. From students looking for entry-level jobs "
    "to Directors looking to take on a leadership role, our program has helped over "
    "hundreds of students fulfill their career aspirations. Our Product Manager "
    "Accelerator community are ambitious and committed. Through our program they have "
    "learnt, honed and developed new PM and leadership skills, giving them a strong "
    "foundation for their future endeavors."
)

st.set_page_config(
    page_title="Weather Forecaster | Multi-Horizon ML Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize theme state
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

is_dark = st.session_state.theme_mode == "Dark"

# Theme palette
PALETTE = {
    "app_bg": "#0b0e14" if is_dark else "#f8fafc",
    "text": "#f8fafc" if is_dark else "#0f172a",
    "muted": "#94a3b8" if is_dark else "#475569",
    "control_bg": "#151922" if is_dark else "#ffffff",
    "control_border": "#262c3a" if is_dark else "#cbd5e1",
    "card_bg": "#131720" if is_dark else "#ffffff",
    "card_border": "#222734" if is_dark else "#e2e8f0",
    "card_hover": "#333b4f" if is_dark else "#cbd5e1",
    "grid": "#1b202c" if is_dark else "#e2e8f0",
    "axis_label": "#8b95a5" if is_dark else "#475569",
    "axis_title": "#cbd5e1" if is_dark else "#1e293b",
    "legend_label": "#e2e8f0" if is_dark else "#1e293b",
}

# Keep Streamlit's top-right ⋮ menu visible (for Record a screencast) while preventing clipping
st.markdown(
    f"""
    <style>
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    footer {{
        visibility: hidden;
    }}
    .stApp {{
        background-color: {PALETTE['app_bg']};
        color: {PALETTE['text']};
    }}
    .block-container {{
        padding-top: 3.1rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1220px;
    }}

    /* Unified Top Controls */
    div[data-baseweb="select"] > div {{
        background-color: {PALETTE['control_bg']} !important;
        border: 1px solid {PALETTE['control_border']} !important;
        border-radius: 12px !important;
        min-height: 42px !important;
        color: {PALETTE['text']} !important;
    }}
    div[data-testid="stRadio"] > div {{
        background-color: {PALETTE['control_bg']};
        border: 1px solid {PALETTE['control_border']};
        border-radius: 12px;
        padding: 5px 12px;
        min-height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    div[data-testid="stPopover"] > button,
    div[data-testid="stButton"] > button {{
        background-color: {PALETTE['control_bg']} !important;
        border: 1px solid {PALETTE['control_border']} !important;
        border-radius: 12px !important;
        min-height: 42px !important;
        color: {PALETTE['text']} !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stPopover"] > button:hover,
    div[data-testid="stButton"] > button:hover {{
        border-color: #f59e0b !important;
    }}

    /* Clean Tab Styling */
    button[data-baseweb="tab"] {{
        font-size: 0.94rem !important;
        font-weight: 500 !important;
        padding-top: 8px !important;
        padding-bottom: 10px !important;
    }}

    /* Hero Banner */
    .hero-banner {{
        background: linear-gradient(130deg, #f59e0b 0%, #ea7c00 55%, #d46500 100%);
        border-radius: 22px;
        padding: 28px 34px;
        color: #ffffff;
        margin-top: 6px;
        margin-bottom: 18px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
    }}
    .hero-top {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 12px;
    }}
    .hero-city {{
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        line-height: 1.2;
    }}
    .hero-sub {{
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 4px;
    }}
    .hero-right-title {{
        font-size: 1.02rem;
        font-weight: 700;
        text-align: right;
    }}
    .hero-right-sub {{
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: right;
        margin-top: 3px;
    }}
    .hero-bottom {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        margin-top: 20px;
        gap: 16px;
    }}
    .hero-temp-wrap {{
        display: flex;
        align-items: center;
        gap: 18px;
    }}
    .hero-sun-icon {{
        font-size: 3.9rem;
        line-height: 1;
        filter: drop-shadow(0 6px 14px rgba(255, 235, 140, 0.4));
    }}
    .hero-temp {{
        font-size: 4.2rem;
        font-weight: 300;
        line-height: 1;
        letter-spacing: -0.03em;
    }}
    .hero-unit {{
        font-size: 1.45rem;
        font-weight: 400;
        vertical-align: top;
        margin-left: 3px;
    }}
    .hero-stats {{
        text-align: right;
        font-size: 0.92rem;
        line-height: 1.7;
        color: rgba(255, 255, 255, 0.96);
    }}

    /* Forecast Horizon Cards */
    .f-card {{
        background-color: {PALETTE['card_bg']};
        border: 1px solid {PALETTE['card_border']};
        border-radius: 18px;
        padding: 22px 26px;
        color: {PALETTE['text']};
        margin-bottom: 16px;
        transition: border-color 0.2s ease;
    }}
    .f-card:hover {{
        border-color: {PALETTE['card_hover']};
    }}
    .f-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }}
    .f-card-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {PALETTE['text']};
    }}
    .f-card-date {{
        font-size: 0.85rem;
        color: {PALETTE['muted']};
    }}
    .f-card-body {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }}
    .f-card-temp {{
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
        color: {PALETTE['text']};
    }}
    .f-card-cond {{
        font-size: 0.9rem;
        color: {PALETTE['muted']};
        margin-top: 4px;
    }}
    .f-card-icon {{
        font-size: 2.75rem;
        line-height: 1;
    }}
    .pill-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .pill-gold {{
        background-color: rgba(234, 179, 8, 0.14);
        color: {'#facc15' if is_dark else '#a16207'};
        border: 1px solid rgba(234, 179, 8, 0.28);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .pill-blue {{
        background-color: rgba(56, 189, 248, 0.14);
        color: {'#7dd3fc' if is_dark else '#0369a1'};
        border: 1px solid rgba(56, 189, 248, 0.28);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .pill-indigo {{
        background-color: rgba(129, 140, 248, 0.14);
        color: {'#c7d2fe' if is_dark else '#4338ca'};
        border: 1px solid rgba(129, 140, 248, 0.28);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_forecasts(path_str: str) -> pd.DataFrame:
    df = pd.read_parquet(path_str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "country" in df.columns:
        df["city_label"] = df["location_name"].astype(str) + ", " + df["country"].astype(str)
    else:
        df["city_label"] = df["location_name"].astype(str)
    return df.sort_values(["city_label", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ensemble_weights(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def to_unit(val_c: float, unit: str) -> float:
    return val_c * 9.0 / 5.0 + 32.0 if unit == "°F" else val_c


def convert_series(s: pd.Series, unit: str) -> pd.Series:
    return s * 9.0 / 5.0 + 32.0 if unit == "°F" else s


def describe_weather(temp_c: float, rain_prob: float) -> tuple[str, str]:
    if not pd.isna(rain_prob):
        if rain_prob >= 0.65:
            return "🌧️", "Rain Likely"
        if rain_prob >= 0.35:
            return "🌦️", "Scattered Showers"
    if not pd.isna(temp_c):
        if temp_c >= 30.0:
            return "☀️", "Hot & Sunny"
        if temp_c >= 20.0:
            return "☀️", "Warm & Clear"
        if temp_c >= 10.0:
            return "⛅", "Mild & Partly Cloudy"
        return "❄️", "Cool & Crisp"
    return "☀️", "Clear"


# ---------------------------------------------------------------------------
# Top Control Bar
# ---------------------------------------------------------------------------
if not LIVE_7D_PARQUET.exists() and not HOLDOUT_PARQUET.exists():
    st.error(
        "No forecast files found. Run `uv run src/live_7d_forecast.py` or "
        "`uv run src/predict.py --split test` first."
    )
    st.stop()

window_options: list[str] = []
if LIVE_7D_PARQUET.exists():
    window_options.append("Latest 7 days (Live)")
if HOLDOUT_PARQUET.exists():
    window_options.extend(["Last 30 days", "Last 90 days", "All Test Days"])

default_df = load_forecasts(str(LIVE_7D_PARQUET if LIVE_7D_PARQUET.exists() else HOLDOUT_PARQUET))
city_labels = sorted(default_df["city_label"].dropna().unique().tolist())
default_idx = 0
for pref in ("New Delhi, India", "Kochi, India", "London, United Kingdom"):
    if pref in city_labels:
        default_idx = city_labels.index(pref)
        break

c_city, c_win, c_unit, c_pm, c_tools = st.columns([2.4, 1.6, 0.95, 1.35, 1.0], vertical_alignment="center")

with c_city:
    selected_city_label = st.selectbox(
        "City",
        options=city_labels,
        index=default_idx,
        label_visibility="collapsed",
    )

with c_win:
    selected_window = st.selectbox(
        "Window",
        options=window_options,
        index=0,
        label_visibility="collapsed",
    )

with c_unit:
    unit = st.radio(
        "Unit",
        options=["°C", "°F"],
        horizontal=True,
        label_visibility="collapsed",
    )

with c_pm:
    with st.popover("ℹ️ PM Accelerator", use_container_width=True):
        st.markdown("### PM Accelerator Mission")
        st.write(PM_ACCELERATOR_MISSION)
        st.caption("Built by Vazeem Basheer")

with c_tools:
    with st.popover("⚙️ Tools", use_container_width=True):
        st.markdown("#### Appearance")
        new_theme = st.radio(
            "Theme",
            options=["Dark", "Light"],
            index=0 if is_dark else 1,
            horizontal=True,
        )
        if new_theme != st.session_state.theme_mode:
            st.session_state.theme_mode = new_theme
            st.rerun()

        st.markdown("#### Data & Recording")
        if st.button("🧹 Clear Cache & Reload", use_container_width=True):
            st.cache_data.clear()
            st.toast("Cache cleared!", icon="🧹")
            st.rerun()

        st.caption("🎥 **Record Screencast:** Click the top-right **⋮** menu → **Record a screencast**.")

# Load active slice
if selected_window == "Latest 7 days (Live)" and LIVE_7D_PARQUET.exists():
    active_df = load_forecasts(str(LIVE_7D_PARQUET))
else:
    active_df = load_forecasts(str(HOLDOUT_PARQUET))

if selected_city_label not in active_df["city_label"].values:
    selected_city_label = sorted(active_df["city_label"].unique().tolist())[0]

city_df = active_df[active_df["city_label"] == selected_city_label].sort_values("date").reset_index(drop=True)

if selected_window == "Last 30 days":
    city_df = city_df.tail(30).reset_index(drop=True)
elif selected_window == "Last 90 days":
    city_df = city_df.tail(90).reset_index(drop=True)

latest_row = city_df.iloc[-1]

# ---------------------------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------------------------
tab_forecast, tab_benchmarks, tab_ensemble = st.tabs(
    ["☀️ Weather Forecast", "📊 Model Benchmarks", "⚖️ Ensemble Details"]
)

# --- TAB 1: Weather Forecast ---
with tab_forecast:
    obs_date = latest_row["date"]
    obs_date_str = obs_date.strftime("%A, %b %d, %Y")
    coord_str = latest_row.get("coord_key", "")

    obs_temp_c = float(latest_row.get("temperature_celsius", float("nan")))
    h1_temp_c = float(latest_row.get("pred_temp_h1_c", float("nan")))
    h7_temp_c = float(latest_row.get("pred_temp_h7_c", float("nan")))

    h1_rain = float(latest_row.get("pred_rain_h1_prob", 0.0))
    h7_rain = float(latest_row.get("pred_rain_h7_prob", 0.0))
    h1_precip = float(latest_row.get("pred_precip_h1_mm", 0.0))
    h7_precip = float(latest_row.get("pred_precip_h7_mm", 0.0))

    hero_icon, hero_cond = describe_weather(obs_temp_c, h1_rain)
    h1_icon, h1_cond = describe_weather(h1_temp_c, h1_rain)
    h7_icon, h7_cond = describe_weather(h7_temp_c, h7_rain)

    obs_disp = to_unit(obs_temp_c, unit)
    h1_disp = to_unit(h1_temp_c, unit)
    h7_disp = to_unit(h7_temp_c, unit)

    delta_h1 = h1_disp - obs_disp
    delta_h7 = h7_disp - obs_disp

    t1_date_str = (obs_date + pd.Timedelta(days=1)).strftime("%a, %b %d")
    t7_date_str = (obs_date + pd.Timedelta(days=7)).strftime("%a, %b %d")

    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-top">
                <div>
                    <div class="hero-city">{selected_city_label}</div>
                    <div class="hero-sub">Observed {obs_date_str}{f' • Coordinates {coord_str}' if coord_str else ''}</div>
                </div>
                <div>
                    <div class="hero-right-title">{hero_cond}</div>
                    <div class="hero-right-sub">7-Model ML Ensemble</div>
                </div>
            </div>
            <div class="hero-bottom">
                <div class="hero-temp-wrap">
                    <div class="hero-sun-icon">{hero_icon}</div>
                    <div class="hero-temp">{obs_disp:.1f}<span class="hero-unit">{unit}</span></div>
                </div>
                <div class="hero-stats">
                    <div>Tomorrow's Rain Chance: <b>{h1_rain * 100:.1f}%</b></div>
                    <div>Tomorrow's Precipitation: <b>{h1_precip:.2f} mm</b></div>
                    <div>7-Day Outlook: <b>{h7_disp:.1f}{unit} ({h7_rain * 100:.1f}% rain)</b></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_h1, col_h7 = st.columns(2, gap="medium")

    with col_h1:
        st.markdown(
            f"""
            <div class="f-card">
                <div class="f-card-header">
                    <div class="f-card-title">Tomorrow (+1 Day Forecast)</div>
                    <div class="f-card-date">{t1_date_str}</div>
                </div>
                <div class="f-card-body">
                    <div>
                        <div class="f-card-temp">{h1_disp:.1f}{unit}</div>
                        <div class="f-card-cond">{h1_cond}</div>
                    </div>
                    <div class="f-card-icon">{h1_icon}</div>
                </div>
                <div class="pill-row">
                    <span class="pill-gold">{delta_h1:+.1f}{unit} vs today</span>
                    <span class="pill-blue">💧 {h1_rain * 100:.1f}% rain</span>
                    <span class="pill-indigo">☔ {h1_precip:.2f} mm</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_h7:
        st.markdown(
            f"""
            <div class="f-card">
                <div class="f-card-header">
                    <div class="f-card-title">Next Week (+7 Day Forecast)</div>
                    <div class="f-card-date">{t7_date_str}</div>
                </div>
                <div class="f-card-body">
                    <div>
                        <div class="f-card-temp">{h7_disp:.1f}{unit}</div>
                        <div class="f-card-cond">{h7_cond}</div>
                    </div>
                    <div class="f-card-icon">{h7_icon}</div>
                </div>
                <div class="pill-row">
                    <span class="pill-gold">{delta_h7:+.1f}{unit} vs today</span>
                    <span class="pill-blue">💧 {h7_rain * 100:.1f}% rain</span>
                    <span class="pill-indigo">☔ {h7_precip:.2f} mm</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trend Chart Tabs
    chart_df = city_df.copy()
    chart_df["Observed"] = convert_series(chart_df["temperature_celsius"], unit)
    chart_df["Pred +1d"] = convert_series(chart_df["pred_temp_h1_c"], unit)
    chart_df["Pred +7d"] = convert_series(chart_df["pred_temp_h7_c"], unit)
    chart_df["Rain +1d (%)"] = chart_df["pred_rain_h1_prob"] * 100.0
    chart_df["Rain +7d (%)"] = chart_df["pred_rain_h7_prob"] * 100.0
    chart_df["date_str"] = chart_df["date"].dt.strftime("%b %d")
    date_order = chart_df["date_str"].drop_duplicates().tolist()

    sub_temp, sub_rain, sub_precip = st.tabs(
        [f"🌡️ Temperature ({unit})", "💧 Rain Probability (%)", "☔ Precipitation (mm)"]
    )

    with sub_temp:
        temp_long = chart_df.melt(
            id_vars=["date"],
            value_vars=["Observed", "Pred +1d", "Pred +7d"],
            var_name="Series",
            value_name="Value",
        )
        temp_chart = (
            alt.Chart(temp_long)
            .mark_line(strokeWidth=2.5, point=len(chart_df) <= 14)
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(grid=False, labelColor=PALETTE["axis_label"])),
                y=alt.Y(
                    "Value:Q",
                    title=f"Temperature ({unit})",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        gridColor=PALETTE["grid"],
                        labelColor=PALETTE["axis_label"],
                        titleColor=PALETTE["axis_title"],
                        tickCount=5,
                    ),
                ),
                color=alt.Color(
                    "Series:N",
                    legend=alt.Legend(orient="top-left", title=None, labelColor=PALETTE["legend_label"], symbolType="stroke"),
                    scale=alt.Scale(
                        domain=["Observed", "Pred +1d", "Pred +7d"],
                        range=["#0284c7" if not is_dark else "#38bdf8", "#f59e0b", "#9333ea" if not is_dark else "#a855f7"],
                    ),
                ),
                strokeDash=alt.StrokeDash(
                    "Series:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=["Observed", "Pred +1d", "Pred +7d"],
                        range=[[1, 0], [1, 0], [5, 4]],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("Value:Q", title=f"Temp ({unit})", format=".1f"),
                ],
            )
            .properties(height=265, background=PALETTE["app_bg"])
            .configure_view(strokeWidth=0)
            .interactive()
        )
        st.altair_chart(temp_chart, use_container_width=True)

    with sub_rain:
        rain_long = chart_df.melt(
            id_vars=["date", "date_str"],
            value_vars=["Rain +1d (%)", "Rain +7d (%)"],
            var_name="Horizon",
            value_name="Probability",
        )
        rain_chart = (
            alt.Chart(rain_long)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, opacity=0.9)
            .encode(
                x=alt.X(
                    "date_str:O",
                    sort=date_order,
                    title=None,
                    axis=alt.Axis(grid=False, labelColor=PALETTE["axis_label"], labelAngle=-35),
                ),
                y=alt.Y(
                    "Probability:Q",
                    title="Rain Probability (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(
                        gridColor=PALETTE["grid"],
                        labelColor=PALETTE["axis_label"],
                        titleColor=PALETTE["axis_title"],
                    ),
                ),
                color=alt.Color(
                    "Horizon:N",
                    legend=alt.Legend(orient="top-left", title=None, labelColor=PALETTE["legend_label"]),
                    scale=alt.Scale(
                        domain=["Rain +1d (%)", "Rain +7d (%)"],
                        range=["#38bdf8", "#818cf8"],
                    ),
                ),
                xOffset="Horizon:N",
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Horizon:N", title="Horizon"),
                    alt.Tooltip("Probability:Q", title="Chance (%)", format=".1f"),
                ],
            )
            .properties(height=265, background=PALETTE["app_bg"])
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(rain_chart, use_container_width=True)

    with sub_precip:
        precip_long = chart_df.melt(
            id_vars=["date", "date_str"],
            value_vars=["pred_precip_h1_mm", "pred_precip_h7_mm"],
            var_name="Horizon",
            value_name="Precip_mm",
        )
        precip_long["Horizon"] = precip_long["Horizon"].map(
            {"pred_precip_h1_mm": "Precip +1d (mm)", "pred_precip_h7_mm": "Precip +7d (mm)"}
        )
        precip_chart = (
            alt.Chart(precip_long)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, opacity=0.9)
            .encode(
                x=alt.X(
                    "date_str:O",
                    sort=date_order,
                    title=None,
                    axis=alt.Axis(grid=False, labelColor=PALETTE["axis_label"], labelAngle=-35),
                ),
                y=alt.Y(
                    "Precip_mm:Q",
                    title="Precipitation (mm)",
                    axis=alt.Axis(
                        gridColor=PALETTE["grid"],
                        labelColor=PALETTE["axis_label"],
                        titleColor=PALETTE["axis_title"],
                    ),
                ),
                color=alt.Color(
                    "Horizon:N",
                    legend=alt.Legend(orient="top-left", title=None, labelColor=PALETTE["legend_label"]),
                    scale=alt.Scale(
                        domain=["Precip +1d (mm)", "Precip +7d (mm)"],
                        range=["#22d3ee", "#6366f1"],
                    ),
                ),
                xOffset="Horizon:N",
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Horizon:N", title="Horizon"),
                    alt.Tooltip("Precip_mm:Q", title="Precip (mm)", format=".2f"),
                ],
            )
            .properties(height=265, background=PALETTE["app_bg"])
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(precip_chart, use_container_width=True)

# --- TAB 2: Model Benchmarks ---
with tab_benchmarks:
    png_files = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if png_files:
        for i in range(0, len(png_files), 2):
            cols = st.columns(2, gap="medium")
            for col, img_path in zip(cols, png_files[i : i + 2]):
                with col:
                    st.image(str(img_path), caption=img_path.stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("No benchmark PNGs found in `reports/figures/`. Run `uv run src/plot_leaderboards.py` to generate them.")

    csv_metrics = sorted(DATA_PROCESSED.glob("*metrics*.csv")) + sorted(DATA_PROCESSED.glob("*benchmark*.csv"))
    if csv_metrics:
        for csv_path in csv_metrics:
            with st.expander(csv_path.name, expanded=False):
                st.dataframe(pd.read_csv(csv_path), use_container_width=True)

# --- TAB 3: Ensemble Details ---
with tab_ensemble:
    if WEIGHTS_JSON.exists():
        weights_cfg = load_ensemble_weights(str(WEIGHTS_JSON))
        rows = []
        for target, cfg in weights_cfg.items():
            row = {
                "Target": target,
                "Task": cfg.get("task", ""),
                "Hurdle Threshold": cfg.get("hurdle_threshold", 0.0),
            }
            for model_name, w in cfg.get("weights", {}).items():
                row[model_name] = round(float(w), 4)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).fillna(0.0), use_container_width=True, hide_index=True)

        with st.expander("View Raw JSON Configuration", expanded=False):
            st.json(weights_cfg)
    else:
        st.warning(f"Ensemble weights file not found at `{WEIGHTS_JSON}`.")