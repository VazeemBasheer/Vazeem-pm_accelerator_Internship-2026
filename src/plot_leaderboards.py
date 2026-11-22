import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless rendering (no GUI popup required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

DEFAULT_VALID_CSV = PROJECT_ROOT / "data" / "processed" / "benchmark_leaderboard.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_leaderboard.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_ORDER = ["baseline", "linear", "histgb", "lgbm", "catboost", "resnet", "seq", "ensemble"]
MODEL_COLORS = {
    "baseline": "#95a5a6",
    "linear": "#3498db",
    "histgb": "#2ecc71",
    "lgbm": "#e67e22",
    "catboost": "#9b59b6",
    "resnet": "#e84393",
    "seq": "#00cec9",
    "ensemble": "#1e8449",
}

PLOT_SPECS = [
    ("target_temp_h1", "MAE", "Temp H1 — MAE (°C, lower is better)"),
    ("target_temp_h7", "MAE", "Temp H7 — MAE (°C, lower is better)"),
    ("target_precip_h1", "MAE", "Precip H1 — MAE (mm, lower is better)"),
    ("target_precip_h7", "MAE", "Precip H7 — MAE (mm, lower is better)"),
    ("target_rain_h1", "LogLoss", "Rain H1 — LogLoss (lower is better)"),
    ("target_rain_h7", "LogLoss", "Rain H7 — LogLoss (lower is better)"),
]


def plot_single_leaderboard(df: pd.DataFrame, title: str, out_path: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.8))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    for (target, metric, subtitle), ax in zip(PLOT_SPECS, axes.flat):
        sub = df[df["target"] == target].copy()
        if sub.empty or metric not in sub.columns:
            ax.set_visible(False)
            continue

        present_models = [m for m in MODEL_ORDER if m in set(sub["model"])]
        sub = sub.set_index("model").loc[present_models]
        vals = sub[metric].to_numpy(dtype=float)
        colors = [MODEL_COLORS.get(m, "#34495e") for m in present_models]

        bars = ax.bar(present_models, vals, color=colors, edgecolor="black", linewidth=0.65, alpha=0.92)

        base_val = None
        if "baseline" in present_models:
            base_val = float(sub.loc["baseline", metric])
            ax.axhline(base_val, color="#d63031", linestyle="--", linewidth=1.2, label=f"Baseline ({base_val:.4f})")

        ax.set_title(subtitle, fontsize=10.5, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", linestyle=":", alpha=0.55)

        ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
        span = (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.2, 0.1)
        ax.set_ylim(max(0.0, ymin - span * 0.28), ymax + span * 0.45)

        best_idx = int(np.nanargmin(vals))
        for i, (bar, v) in enumerate(zip(bars, vals)):
            is_best = i == best_idx
            is_ens_worse = (present_models[i] == "ensemble") and (base_val is not None) and (v > base_val)
            fw = "bold" if (is_best or present_models[i] == "ensemble") else "normal"
            col = "#1e8449" if is_best else ("#c0392b" if is_ens_worse else "#222222")
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + span * 0.03,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight=fw,
                color=col,
            )

        if base_val is not None:
            ax.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def plot_valid_vs_test(val_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    fig.suptitle("Validation vs. Holdout Test Generalization Across All Models", fontsize=14, fontweight="bold", y=0.98)

    for (target, metric, subtitle), ax in zip(PLOT_SPECS, axes.flat):
        v_sub = val_df[val_df["target"] == target].set_index("model")
        t_sub = test_df[test_df["target"] == target].set_index("model")
        present = [m for m in MODEL_ORDER if m in v_sub.index and m in t_sub.index]
        if not present:
            ax.set_visible(False)
            continue

        v_vals = v_sub.loc[present, metric].to_numpy(dtype=float)
        t_vals = t_sub.loc[present, metric].to_numpy(dtype=float)

        x = np.arange(len(present))
        width = 0.38

        b1 = ax.bar(x - width / 2, v_vals, width, label="Valid", color="#5dade2", edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x + width / 2, t_vals, width, label="Test (Holdout)", color="#1f618d", edgecolor="black", linewidth=0.6)

        ax.set_title(subtitle, fontsize=10.5, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=28)
        ax.grid(axis="y", linestyle=":", alpha=0.55)

        all_vals = np.concatenate([v_vals, t_vals])
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        span = (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.2, 0.1)
        ax.set_ylim(max(0.0, ymin - span * 0.25), ymax + span * 0.38)

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + span * 0.02,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    rotation=90,
                )

        ax.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert weather_forcaster leaderboard CSVs into bar graphs.")
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID_CSV, help="Path to validation leaderboard CSV.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_CSV, help="Path to test leaderboard CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory to save PNG charts.")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    args = parser.parse_args()

    val_df = pd.read_csv(args.valid) if args.valid.exists() else None
    test_df = pd.read_csv(args.test) if args.test.exists() else None

    if val_df is None and test_df is None:
        raise FileNotFoundError(f"Neither {args.valid} nor {args.test} was found.")

    if val_df is not None:
        plot_single_leaderboard(
            val_df,
            "Validation Benchmark Leaderboard (All 8 Models × 6 Targets)",
            args.out_dir / "valid_leaderboard.png",
            dpi=args.dpi,
        )

    if test_df is not None:
        plot_single_leaderboard(
            test_df,
            "Holdout Test Leaderboard (All 8 Models × 6 Targets)",
            args.out_dir / "test_leaderboard.png",
            dpi=args.dpi,
        )

    if val_df is not None and test_df is not None:
        plot_valid_vs_test(
            val_df,
            test_df,
            args.out_dir / "valid_vs_test_comparison.png",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()