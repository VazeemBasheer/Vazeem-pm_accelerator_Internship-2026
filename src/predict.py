import argparse
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "weather_features.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "forecasts.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
WEIGHTS_JSON = MODEL_DIR / "ensemble_weights.json"

# Order matters: rain probabilities must be computed before precipitation hurdle gating
EVAL_ORDER = [
    "target_rain_h1",
    "target_rain_h7",
    "target_temp_h1",
    "target_temp_h7",
    "target_precip_h1",
    "target_precip_h7",
]

# Fixed head ordering used when training multitask_resnet.pt and multitask_seq.pt
NN_TARGETS = [
    "target_temp_h1",
    "target_temp_h7",
    "target_precip_h1",
    "target_precip_h7",
    "target_rain_h1",
    "target_rain_h7",
]

RESIDUAL_ANCHORS = {
    "target_temp_h1": "temperature_celsius",
    "target_temp_h7": "city_month_temp_clim",
}

OUTPUT_COL_MAP = {
    "target_temp_h1": "pred_temp_h1_c",
    "target_temp_h7": "pred_temp_h7_c",
    "target_precip_h1": "pred_precip_h1_mm",
    "target_precip_h7": "pred_precip_h7_mm",
    "target_rain_h1": "pred_rain_h1_prob",
    "target_rain_h7": "pred_rain_h7_prob",
}


# ---------------------------------------------------------------------------
# Neural Architectures (must match saved state_dict layers)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 192, num_blocks: int = 3, dropout: float = 0.15):
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, len(NN_TARGETS))
        self.linear_skip = nn.Linear(in_features, len(NN_TARGETS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.head_norm(h)
        return self.head(h) + self.linear_skip(x)


class HybridConvGRU(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim + in_features)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + in_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(NN_TARGETS)),
        )
        self.linear_skip = nn.Linear(in_features, len(NN_TARGETS))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        last_step = x_seq[:, -1, :]
        c = self.act(self.conv1(x_seq.transpose(1, 2))).transpose(1, 2)
        gru_out, _ = self.gru(c)
        h = self.norm(torch.cat([gru_out[:, -1, :], last_step], dim=-1))
        return self.head(h) + self.linear_skip(last_step)


# ---------------------------------------------------------------------------
# Sequence Windowing & Model Inference Helpers
# ---------------------------------------------------------------------------
def build_causal_windows(
    eval_df: pd.DataFrame,
    X_scaled: np.ndarray,
    seq_len: int = 7,
    history_df: pd.DataFrame | None = None,
    X_history_scaled: np.ndarray | None = None,
) -> np.ndarray:
    n_rows, n_feats = X_scaled.shape
    windows = np.zeros((n_rows, seq_len, n_feats), dtype=np.float32)

    if "coord_key" in eval_df.columns:
        group_col = "coord_key"
    elif "location_name" in eval_df.columns:
        group_col = "location_name"
    else:
        group_col = None

    cities = eval_df[group_col].to_numpy() if group_col else np.zeros(n_rows, dtype=int)
    dates = pd.to_datetime(eval_df["date"]).to_numpy() if "date" in eval_df.columns else np.arange(n_rows)

    hist_lookup: dict = {}
    if history_df is not None and X_history_scaled is not None and len(history_df) > 0:
        h_cities = history_df[group_col].to_numpy() if group_col else np.zeros(len(history_df), dtype=int)
        h_dates = pd.to_datetime(history_df["date"]).to_numpy() if "date" in history_df.columns else np.arange(len(history_df))
        h_order = pd.DataFrame({
            "pos": np.arange(len(history_df)),
            "city": h_cities,
            "date": h_dates,
        }).sort_values(["city", "date", "pos"], kind="mergesort")
        for city, grp in h_order.groupby("city", sort=False):
            hist_lookup[city] = (grp["date"].to_numpy(), X_history_scaled[grp["pos"].to_numpy()])

    order_df = pd.DataFrame({
        "pos": np.arange(n_rows),
        "city": cities,
        "date": dates,
    }).sort_values(["city", "date", "pos"], kind="mergesort")

    pad_len = seq_len - 1
    for city, grp in order_df.groupby("city", sort=False):
        positions = grp["pos"].to_numpy()
        city_dates = grp["date"].to_numpy()
        city_x = X_scaled[positions]
        first_date = city_dates[0]

        prefix = None
        if city in hist_lookup:
            h_dates, h_x = hist_lookup[city]
            past_mask = h_dates < first_date
            if past_mask.any():
                prefix = h_x[past_mask][-pad_len:]

        if prefix is None or len(prefix) == 0:
            prefix = np.repeat(city_x[0:1], pad_len, axis=0)
        elif len(prefix) < pad_len:
            shortfall = pad_len - len(prefix)
            prefix = np.vstack([np.repeat(prefix[0:1], shortfall, axis=0), prefix])

        padded = np.vstack([prefix, city_x])
        for i, orig_pos in enumerate(positions):
            windows[orig_pos] = padded[i : i + seq_len]

    return windows


def decode_nn_outputs(raw_preds: np.ndarray, eval_df: pd.DataFrame, train_means: dict) -> dict[str, np.ndarray]:
    decoded: dict[str, np.ndarray] = {}
    for idx, target in enumerate(NN_TARGETS):
        col = raw_preds[:, idx].astype(np.float64)
        if target in RESIDUAL_ANCHORS and RESIDUAL_ANCHORS[target] in eval_df.columns:
            anchor = eval_df[RESIDUAL_ANCHORS[target]].fillna(train_means[target]).to_numpy(dtype=np.float64)
            decoded[target] = col + anchor
        elif "precip" in target:
            decoded[target] = np.clip(col, 0.0, None)
        else:
            decoded[target] = 1.0 / (1.0 + np.exp(-np.clip(col, -30.0, 30.0)))
    return decoded


def predict_resnet(eval_df: pd.DataFrame, device: torch.device) -> dict[str, np.ndarray]:
    ckpt_path = MODEL_DIR / "multitask_resnet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing ResNet checkpoint: {ckpt_path}")

    # weights_only=False required because sklearn imputer/scaler are bundled in the checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    features = ckpt["features"]
    X = np.clip(ckpt["scaler"].transform(ckpt["imputer"].transform(eval_df[features])), -15.0, 15.0).astype(np.float32)

    model = TabularResNet(in_features=len(features), hidden_dim=192, num_blocks=3, dropout=0.15).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), 4096):
            bx = torch.from_numpy(X[start : start + 4096]).to(device)
            chunks.append(model(bx).cpu().numpy())

    return decode_nn_outputs(np.vstack(chunks), eval_df, ckpt["train_means"])


def predict_seq(eval_df: pd.DataFrame, history_df: pd.DataFrame | None, device: torch.device) -> dict[str, np.ndarray]:
    ckpt_path = MODEL_DIR / "multitask_seq.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Conv1D-GRU checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    features = ckpt["features"]
    seq_len = int(ckpt.get("seq_len", 7))

    X_eval_2d = np.clip(ckpt["scaler"].transform(ckpt["imputer"].transform(eval_df[features])), -15.0, 15.0).astype(np.float32)
    X_hist_2d = None
    if history_df is not None and len(history_df) > 0:
        X_hist_2d = np.clip(ckpt["scaler"].transform(ckpt["imputer"].transform(history_df[features])), -15.0, 15.0).astype(np.float32)

    X_seq = build_causal_windows(eval_df, X_eval_2d, seq_len=seq_len, history_df=history_df, X_history_scaled=X_hist_2d)

    model = HybridConvGRU(in_features=len(features), hidden_dim=128, dropout=0.15).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_seq), 2048):
            bx = torch.from_numpy(X_seq[start : start + 2048]).to(device)
            chunks.append(model(bx).cpu().numpy())

    return decode_nn_outputs(np.vstack(chunks), eval_df, ckpt["train_means"])


def predict_tabular_member(target: str, model_name: str, eval_df: pd.DataFrame) -> np.ndarray:
    artifact_path = MODEL_DIR / f"{target}_{model_name}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing tabular artifact: {artifact_path}")

    bundle = joblib.load(artifact_path)
    est = bundle["model"]
    features = bundle["features"]
    task = bundle["task"]
    ridge_stage = bundle.get("ridge_stage")
    anchor_col = bundle.get("residual_anchor")
    fallback_mean = float(bundle.get("fallback_mean", 0.0))

    X = eval_df[features]
    raw_preds = est.predict_proba(X)[:, 1] if task == "classification" else est.predict(X)
    raw_preds = np.asarray(raw_preds, dtype=np.float64)

    ridge_offset = ridge_stage.predict(X) if ridge_stage is not None else 0.0

    if anchor_col and anchor_col in eval_df.columns:
        anchor = eval_df[anchor_col].fillna(fallback_mean).to_numpy(dtype=np.float64)
        return raw_preds + ridge_offset + anchor
    if "precip" in target.lower():
        return np.clip(raw_preds, 0.0, None)
    return raw_preds


def evaluate_predictions(y_true: np.ndarray, preds: np.ndarray, task: str) -> dict[str, float]:
    if task == "regression":
        return {
            "MAE": round(float(mean_absolute_error(y_true, preds)), 4),
            "RMSE": round(float(root_mean_squared_error(y_true, preds)), 4),
        }
    probs = np.clip(preds, 1e-6, 1.0 - 1e-6)
    binary_preds = (probs >= 0.5).astype(int)
    return {
        "LogLoss": round(float(log_loss(y_true, probs, labels=[0, 1])), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, probs)), 4),
        "F1": round(float(f1_score(y_true, binary_preds, zero_division=0)), 4),
        "Accuracy": round(float(accuracy_score(y_true, binary_preds)), 4),
    }


def load_input_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone inference runner for the weather_forcaster ensemble.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA, help="Path to feature parquet/csv file.")
    parser.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"], help="Optional split slice.")
    parser.add_argument("--city", type=str, default=None, help="Optional case-insensitive city filter (location_name).")
    parser.add_argument("--latest", action="store_true", help="Output only the most recent date per city.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .parquet or .csv path.")
    parser.add_argument("--evaluate", action="store_true", help="Print metrics if ground-truth target_* columns exist.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if not WEIGHTS_JSON.exists():
        raise FileNotFoundError(f"Ensemble weights not found at {WEIGHTS_JSON}. Run src/ensemble.py --eval-split valid first.")

    with open(WEIGHTS_JSON, "r", encoding="utf-8") as f:
        ens_config: dict[str, dict] = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_input_frame(args.input)

    if args.city:
        if "location_name" not in df.columns:
            raise ValueError("--city filter requires a 'location_name' column in the input file.")
        mask = df["location_name"].astype(str).str.contains(args.city, case=False, na=False)
        df = df[mask].copy()
        if df.empty:
            raise ValueError(f"No rows matched --city '{args.city}'.")

    # Separate eval_df and strictly prior history_df for causal sequence windowing
    history_df: pd.DataFrame | None = None
    if args.split and "split" in df.columns:
        eval_df = df[df["split"] == args.split].copy()
        if args.split == "valid":
            history_df = df[df["split"] == "train"].copy()
        elif args.split == "test":
            history_df = df[df["split"].isin(["train", "valid"])].copy()
    else:
        eval_df = df.copy()

    if eval_df.empty:
        raise ValueError("No rows available for inference after filtering.")

    eval_df = eval_df.reset_index(drop=True)
    print(f"[predict] Rows={len(eval_df):,} | Split={args.split or 'all'} | Device={device}")

    # Determine which neural models are actually referenced by ensemble_weights.json
    active_models = {m for cfg in ens_config.values() for m, w in cfg["weights"].items() if w > 0.0}
    nn_cache: dict[str, dict[str, np.ndarray]] = {}

    if "resnet" in active_models:
        print("[predict] Running Tabular ResNet inference...")
        nn_cache["resnet"] = predict_resnet(eval_df, device)

    if "seq" in active_models:
        print("[predict] Running 7-Day Sequence Conv1D-GRU inference...")
        nn_cache["seq"] = predict_seq(eval_df, history_df, device)

    # Compute ensemble predictions in hurdle-safe order
    ensemble_preds: dict[str, np.ndarray] = {}
    for target in EVAL_ORDER:
        if target not in ens_config:
            continue
        cfg = ens_config[target]
        weight_map: dict[str, float] = cfg["weights"]

        blended = np.zeros(len(eval_df), dtype=np.float64)
        total_w = 0.0
        for model_name, w in weight_map.items():
            if w <= 0.0:
                continue
            if model_name in nn_cache:
                member_pred = nn_cache[model_name][target]
            else:
                member_pred = predict_tabular_member(target, model_name, eval_df)
            blended += w * member_pred
            total_w += w

        blended /= max(total_w, 1e-12)

        hurdle_target = cfg.get("hurdle_rain_target")
        hurdle_thresh = float(cfg.get("hurdle_threshold", 0.0))
        if hurdle_target and hurdle_thresh > 0.0 and hurdle_target in ensemble_preds:
            blended = np.where(ensemble_preds[hurdle_target] < hurdle_thresh, 0.0, blended)

        if "precip" in target:
            blended = np.clip(blended, 0.0, None)

        ensemble_preds[target] = blended

    # Optional metric evaluation against ground truth
    if args.evaluate:
        print("\n=== Standalone Inference Evaluation ===")
        for target in EVAL_ORDER:
            if target in eval_df.columns and target in ensemble_preds:
                valid_mask = eval_df[target].notna().to_numpy()
                if valid_mask.any():
                    y_true = eval_df.loc[valid_mask, target].to_numpy(dtype=np.float64)
                    metrics = evaluate_predictions(y_true, ensemble_preds[target][valid_mask], ens_config[target]["task"])
                    print(f"  {target:18s} | rows={int(valid_mask.sum()):6d} | {metrics}")

    # Assemble clean forecast dataframe
    meta_cols = [c for c in ["coord_key", "location_name", "country", "date", "temperature_celsius"] if c in eval_df.columns]
    out_df = eval_df[meta_cols].copy()

    for target, col_name in OUTPUT_COL_MAP.items():
        if target in ensemble_preds:
            out_df[col_name] = np.round(ensemble_preds[target], 4)
            if "rain" in target:
                label_col = col_name.replace("_prob", "_label")
                out_df[label_col] = (ensemble_preds[target] >= 0.5).astype(int)

    if args.latest and "date" in out_df.columns:
        group_col = "coord_key" if "coord_key" in out_df.columns else ("location_name" if "location_name" in out_df.columns else None)
        out_df["_dt"] = pd.to_datetime(out_df["date"])
        if group_col:
            out_df = out_df.sort_values("_dt").groupby(group_col, as_index=False).tail(1)
        else:
            out_df = out_df.sort_values("_dt").tail(1)
        out_df = out_df.drop(columns=["_dt"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        out_df.to_csv(args.output, index=False)
    else:
        out_df.to_parquet(args.output, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\n[predict] Saved {len(out_df):,} forecast rows to {args.output} in {elapsed:.2f}s")

    if args.city or args.latest:
        preview_cols = [c for c in ["location_name", "date", "pred_temp_h1_c", "pred_temp_h7_c", "pred_precip_h1_mm", "pred_rain_h1_prob", "pred_rain_h7_prob"] if c in out_df.columns]
        print("\nForecast Preview:")
        print(out_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()