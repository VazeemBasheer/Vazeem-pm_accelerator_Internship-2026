import argparse
from datetime import datetime
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "weather_features.parquet"
DEFAULT_SELECTED = PROJECT_ROOT / "data" / "processed" / "selected_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_FILE = PROJECT_ROOT / "logs" / "overnight_train.log"
LEADERBOARD_CSV = OUTPUT_DIR / "benchmark_leaderboard.csv"

TARGETS = [
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

FORBIDDEN_COLS = {
    "date",
    "split",
    "coord_key",
    "location_name",
    "country",
    "last_updated",
}


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_union_features(selected_path: Path, df: pd.DataFrame) -> list[str]:
    feats: list[str] = []
    if selected_path.exists():
        sel_df = pd.read_csv(selected_path)
        if {"target", "selected", "feature"}.issubset(sel_df.columns):
            sub = sel_df[sel_df["target"].isin(TARGETS) & sel_df["selected"].astype(bool)]
            feats.extend(sub["feature"].tolist())

    for anchor in RESIDUAL_ANCHORS.values():
        if anchor in df.columns:
            feats.append(anchor)

    valid = [
        c for c in dict.fromkeys(feats)
        if c in df.columns and c not in FORBIDDEN_COLS and not c.startswith("target_")
    ]
    if valid:
        return valid

    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in FORBIDDEN_COLS and not c.startswith("target_")
    ]


def build_causal_windows(split_df: pd.DataFrame, X_scaled: np.ndarray, seq_len: int = 7) -> np.ndarray:
    """
    Constructs strictly causal 7-day rolling windows per city within a single split.
    Left-pads the first (seq_len - 1) days of each city with its earliest split row
    so every row aligns 1-to-1 without crossing split boundaries.
    """
    n_rows, n_feats = X_scaled.shape
    windows = np.zeros((n_rows, seq_len, n_feats), dtype=np.float32)

    group_col = "coord_key" if "coord_key" in split_df.columns else "location_name"
    order_df = pd.DataFrame({
        "pos": np.arange(n_rows),
        "city": split_df[group_col].to_numpy(),
        "date": pd.to_datetime(split_df["date"]).to_numpy(),
    }).sort_values(["city", "date", "pos"], kind="mergesort")

    for _, grp in order_df.groupby("city", sort=False):
        positions = grp["pos"].to_numpy()
        city_x = X_scaled[positions]
        # Pad top with (seq_len - 1) copies of day 0
        padded = np.vstack([np.repeat(city_x[0:1], seq_len - 1, axis=0), city_x])
        for i, orig_pos in enumerate(positions):
            windows[orig_pos] = padded[i : i + seq_len]

    return windows


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
            nn.Linear(hidden_dim, len(TARGETS)),
        )
        # Direct linear skip from current day (t = -1) preserves linear trends
        self.linear_skip = nn.Linear(in_features, len(TARGETS))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch, seq_len, features)
        last_step = x_seq[:, -1, :]
        c = self.act(self.conv1(x_seq.transpose(1, 2))).transpose(1, 2)
        gru_out, _ = self.gru(c)
        h = self.norm(torch.cat([gru_out[:, -1, :], last_step], dim=-1))
        return self.head(h) + self.linear_skip(last_step)


def compute_multitask_loss(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=preds.device)
    weights = [1.0, 0.8, 2.5, 2.5, 1.5, 1.5]
    for idx, w in enumerate(weights):
        valid = masks[:, idx] > 0.5
        if not valid.any():
            continue
        p = preds[valid, idx]
        y = targets[valid, idx]
        if idx < 4:
            loss_i = nn.functional.l1_loss(p, y)
        else:
            loss_i = nn.functional.binary_cross_entropy_with_logits(p, y)
        total_loss = total_loss + w * loss_i
    return total_loss


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


def upsert_leaderboard(new_rows: list[dict], model_tag: str) -> None:
    new_df = pd.DataFrame(new_rows)
    if LEADERBOARD_CSV.exists():
        old_df = pd.read_csv(LEADERBOARD_CSV)
        old_df = old_df[old_df["model"] != model_tag]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values(["target", "model"]).reset_index(drop=True)
    combined.to_csv(LEADERBOARD_CSV, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Head 7-Day Sequence Conv1D-GRU trainer.")
    parser.add_argument("--eval-split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--seq-len", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    log(f"=== Starting 7-Day Sequence Conv1D-GRU Training (split={args.eval_split}, device={device}, dry_run={args.dry_run}) ===")

    df = pd.read_parquet(DEFAULT_DATA)
    if args.dry_run:
        df = df.groupby("split", group_keys=False).head(2000).copy()
        args.epochs = 3

    features = get_union_features(DEFAULT_SELECTED, df)
    train_df = df[df["split"] == "train"].copy()
    eval_df = df[df["split"] == args.eval_split].copy()
    assert len(train_df.index.intersection(eval_df.index)) == 0, "Split overlap detected!"

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train_2d = np.clip(scaler.fit_transform(imputer.fit_transform(train_df[features])), -15.0, 15.0).astype(np.float32)
    X_eval_2d = np.clip(scaler.transform(imputer.transform(eval_df[features])), -15.0, 15.0).astype(np.float32)

    log("Building causal 7-day city windows (strictly within each split)...")
    X_train_seq = build_causal_windows(train_df, X_train_2d, seq_len=args.seq_len)
    X_eval_seq = build_causal_windows(eval_df, X_eval_2d, seq_len=args.seq_len)

    train_means = {t: float(train_df[t].mean()) for t in TARGETS}
    Y_train = np.zeros((len(train_df), len(TARGETS)), dtype=np.float32)
    M_train = np.zeros((len(train_df), len(TARGETS)), dtype=np.float32)

    for idx, t in enumerate(TARGETS):
        valid_mask = train_df[t].notna().to_numpy()
        M_train[:, idx] = valid_mask.astype(np.float32)
        vals = train_df[t].fillna(0.0).to_numpy(dtype=np.float32)
        if t in RESIDUAL_ANCHORS:
            anchor = train_df[RESIDUAL_ANCHORS[t]].fillna(train_means[t]).to_numpy(dtype=np.float32)
            vals = vals - anchor
        Y_train[:, idx] = np.where(valid_mask, vals, 0.0)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(Y_train), torch.from_numpy(M_train)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    model = HybridConvGRU(in_features=len(features), hidden_dim=128, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by, bm in train_loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = compute_multitask_loss(out, by, bm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item() * len(bx)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            log(f"  Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {running_loss / len(train_df):.4f}")

    # Batched inference to keep VRAM usage tiny
    model.eval()
    eval_preds_list = []
    with torch.no_grad():
        for start in range(0, len(X_eval_seq), 2048):
            batch_x = torch.from_numpy(X_eval_seq[start : start + 2048]).to(device)
            eval_preds_list.append(model(batch_x).cpu().numpy())
    raw_eval_preds = np.vstack(eval_preds_list)

    elapsed_total = round(time.perf_counter() - t0, 2)
    meta_cols = [c for c in ["coord_key", "location_name", "country", "date"] if c in eval_df.columns]
    results = []

    for idx, target in enumerate(TARGETS):
        task = "classification" if "rain" in target else "regression"
        valid_mask = eval_df[target].notna().to_numpy()
        y_true = eval_df.loc[valid_mask, target].to_numpy(dtype=np.float64)
        pred_col = raw_eval_preds[valid_mask, idx].astype(np.float64)

        if target in RESIDUAL_ANCHORS:
            anchor = eval_df.loc[valid_mask, RESIDUAL_ANCHORS[target]].fillna(train_means[target]).to_numpy(dtype=np.float64)
            preds = pred_col + anchor
        elif "precip" in target:
            preds = np.clip(pred_col, 0.0, None)
        else:
            preds = 1.0 / (1.0 + np.exp(-np.clip(pred_col, -30.0, 30.0)))

        metrics = evaluate_predictions(y_true, preds, task)

        if not args.dry_run:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_df = eval_df.loc[valid_mask, meta_cols].copy()
            out_df["actual"] = y_true
            out_df["predicted"] = preds
            out_df.to_parquet(OUTPUT_DIR / f"preds_{target}_seq_{args.eval_split}.parquet", index=False)

        row = {
            "model": "seq",
            "target": target,
            "task": task,
            "residual_mode": target in RESIDUAL_ANCHORS,
            "features": len(features),
            "train_rows": int(M_train[:, idx].sum()),
            "eval_rows": int(valid_mask.sum()),
            "time_s": elapsed_total,
            **metrics,
        }
        results.append(row)
        log(f"  seq      | {target:18s} | {metrics}")

    if not args.dry_run:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "features": features,
                "seq_len": args.seq_len,
                "imputer": imputer,
                "scaler": scaler,
                "train_means": train_means,
            },
            MODEL_DIR / "multitask_seq.pt",
        )
        upsert_leaderboard(results, "seq")
        log(f"Saved multitask_seq.pt and updated {LEADERBOARD_CSV}")


if __name__ == "__main__":
    main()