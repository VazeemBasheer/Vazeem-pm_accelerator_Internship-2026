import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gc
from pathlib import Path
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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

FORBIDDEN_COLS = {
    "date",
    "split",
    "coord_key",
    "location_name",
    "country",
    "last_updated",
}


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "split" not in df.columns:
        raise ValueError("Dataset must contain a 'split' column ('train', 'valid', 'test').")
    return df


def verify_no_leakage(train_df: pd.DataFrame, eval_df: pd.DataFrame, features: list[str], target: str) -> None:
    leaked_cols = [c for c in features if c in FORBIDDEN_COLS or c.startswith("target_")]
    if leaked_cols:
        raise RuntimeError(f"Leakage detected in feature list for {target}: {leaked_cols}")

    if len(train_df.index.intersection(eval_df.index)) > 0:
        raise RuntimeError("Index overlap detected between train and eval splits.")


def get_feature_list(selected_path: Path, target: str, df: pd.DataFrame) -> list[str]:
    if selected_path.exists():
        sel_df = pd.read_csv(selected_path)
        if {"target", "selected", "feature"}.issubset(sel_df.columns):
            mask = (sel_df["target"] == target) & (sel_df["selected"].astype(bool))
            cols = sel_df.loc[mask, "feature"].tolist()
            valid_cols = [
                c for c in cols
                if c in df.columns and c not in FORBIDDEN_COLS and not c.startswith("target_")
            ]
            if valid_cols:
                return valid_cols

    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in FORBIDDEN_COLS and not c.startswith("target_")
    ]


def infer_task(y: pd.Series, target: str) -> str:
    unique_vals = set(y.dropna().unique())
    if "rain" in target.lower() or unique_vals.issubset({0, 1, 0.0, 1.0}):
        return "classification"
    return "regression"


def predict_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, target: str) -> np.ndarray:
    if "temp" in target:
        if "h7" in target and "city_month_temp_clim" in eval_df.columns:
            fallback = float(train_df[target].mean())
            return eval_df["city_month_temp_clim"].fillna(fallback).to_numpy(dtype=np.float64)
        if "temperature_celsius" in eval_df.columns:
            fallback = float(train_df[target].mean())
            return eval_df["temperature_celsius"].fillna(fallback).to_numpy(dtype=np.float64)
        return np.full(len(eval_df), float(train_df[target].mean()), dtype=np.float64)

    if "precip" in target:
        if "precip_mm" in eval_df.columns:
            return eval_df["precip_mm"].fillna(0.0).to_numpy(dtype=np.float64)
        return np.full(len(eval_df), float(train_df[target].mean()), dtype=np.float64)

    rain_rate = float(train_df[target].mean())
    return np.full(len(eval_df), rain_rate, dtype=np.float64)


def build_estimator(model_name: str, task: str, parallel_mode: bool = False, use_gpu: bool = True):
    name = model_name.lower()

    if name == "linear":
        if task == "regression":
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
                ("model", Ridge(alpha=1.0)),
            ])
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ])

    if name == "histgb":
        if task == "regression":
            return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)

    if name == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        n_jobs = 4 if parallel_mode else -1
        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            random_state=42,
            n_jobs=n_jobs,
            verbose=-1,
        )
        return LGBMRegressor(**params) if task == "regression" else LGBMClassifier(**params)

    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        params = dict(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        if use_gpu:
            params["task_type"] = "GPU"
            params["gpu_ram_part"] = 0.30 if parallel_mode else 0.70
        else:
            params["task_type"] = "CPU"
            params["thread_count"] = 4 if parallel_mode else -1
        return CatBoostRegressor(**params) if task == "regression" else CatBoostClassifier(**params)

    raise ValueError(f"Unsupported model: {model_name}")


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


def fit_and_eval_one(
    target: str,
    model_name: str,
    eval_split: str,
    features: list[str],
    tr_df: pd.DataFrame,
    ev_df: pd.DataFrame,
    parallel_mode: bool = False,
    dry_run: bool = False,
) -> dict:
    t0 = time.perf_counter()
    verify_no_leakage(tr_df, ev_df, features, target)

    task = infer_task(tr_df[target], target)
    y_train = tr_df[target].to_numpy()
    y_eval = ev_df[target].to_numpy()

    if model_name == "baseline":
        preds = predict_baseline(tr_df, ev_df, target)
    else:
        X_train = tr_df[features]
        X_eval = ev_df[features]

        if model_name == "catboost":
            try:
                est = build_estimator(model_name, task, parallel_mode=parallel_mode, use_gpu=True)
                est.fit(X_train, y_train)
            except Exception:
                est = build_estimator(model_name, task, parallel_mode=parallel_mode, use_gpu=False)
                est.fit(X_train, y_train)
        else:
            est = build_estimator(model_name, task, parallel_mode=parallel_mode)
            est.fit(X_train, y_train)

        preds = (
            est.predict_proba(X_eval)[:, 1]
            if task == "classification"
            else est.predict(X_eval)
        )

        if not dry_run:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": est, "features": features, "target": target, "task": task},
                MODEL_DIR / f"{target}_{model_name}.joblib",
            )

    metrics = evaluate_predictions(y_eval, preds, task)

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        meta_cols = [c for c in ["coord_key", "location_name", "country", "date"] if c in ev_df.columns]
        out_df = ev_df[meta_cols].copy()
        out_df["actual"] = y_eval
        out_df["predicted"] = preds
        out_df.to_parquet(OUTPUT_DIR / f"preds_{target}_{model_name}_{eval_split}.parquet", index=False)

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "model": model_name,
        "target": target,
        "task": task,
        "features": len(features),
        "train_rows": len(tr_df),
        "eval_rows": len(ev_df),
        "time_s": elapsed,
        **metrics,
    }


def prepare_slice(
    df: pd.DataFrame,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    target: str,
    features: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = [c for c in ["coord_key", "location_name", "country", "date"] if c in df.columns]
    baseline_cols = [c for c in ["temperature_celsius", "precip_mm", "city_month_temp_clim"] if c in df.columns]
    extra = baseline_cols if model_name == "baseline" else []
    keep_cols = list(dict.fromkeys(meta_cols + features + extra + [target]))

    tr_slice = df.loc[train_mask & df[target].notna(), keep_cols].copy()
    ev_slice = df.loc[eval_mask & df[target].notna(), keep_cols].copy()
    return tr_slice, ev_slice


def main() -> None:
    parser = argparse.ArgumentParser(description="Leak-proof overnight benchmark runner.")
    parser.add_argument("--eval-split", type=str, default="valid", choices=["valid", "test"], help="Evaluation split")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to feature parquet")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED, help="Path to selected_features.csv")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on a 2,000-row subset per split without writing models, predictions, or leaderboard files",
    )
    args = parser.parse_args()

    t_total = time.perf_counter()
    log(f"=== Starting Overnight Benchmark Run (split={args.eval_split}, dry_run={args.dry_run}) ===")

    df = load_dataset(args.data)
    if args.dry_run:
        df = df.groupby("split", group_keys=False).head(2000).copy()
        log(f"Dry-run active: sampled {len(df):,} rows total (read-only mode).")

    train_mask = df["split"] == "train"
    eval_mask = df["split"] == args.eval_split

    if not train_mask.any() or not eval_mask.any():
        raise ValueError(f"Empty split detected: train={train_mask.sum()}, {args.eval_split}={eval_mask.sum()}")

    available_targets = [t for t in TARGETS if t in df.columns]
    if not available_targets:
        raise KeyError(f"None of expected targets {TARGETS} found in dataset.")

    target_features = {t: get_feature_list(args.selected, t, df) for t in available_targets}
    for t, feats in target_features.items():
        log(f"Loaded feature schema | {t:18s}: {len(feats)} features -> {feats}")

    model_schedule = [
        ("baseline", "sequential"),
        ("linear", "sequential"),
        ("histgb", "sequential"),
        ("lgbm", "parallel"),
        ("catboost", "parallel"),
    ]

    all_rows = []
    for model_name, mode in model_schedule:
        log(f"--- Running model family: {model_name} ({mode}) ---")
        try:
            if mode == "sequential":
                for target in available_targets:
                    feats = target_features[target]
                    tr_slice, ev_slice = prepare_slice(df, train_mask, eval_mask, target, feats, model_name)
                    row = fit_and_eval_one(
                        target,
                        model_name,
                        args.eval_split,
                        feats,
                        tr_slice,
                        ev_slice,
                        parallel_mode=False,
                        dry_run=args.dry_run,
                    )
                    all_rows.append(row)
                    log(f"  {model_name:8s} | {target:18s} | {row}")
            else:
                tasks = []
                for target in available_targets:
                    feats = target_features[target]
                    tr_slice, ev_slice = prepare_slice(df, train_mask, eval_mask, target, feats, model_name)
                    tasks.append((target, model_name, args.eval_split, feats, tr_slice, ev_slice, True, args.dry_run))

                workers = 2 if model_name == "catboost" else 3
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(fit_and_eval_one, *t) for t in tasks]
                    for fut in as_completed(futures):
                        row = fut.result()
                        all_rows.append(row)
                        log(f"  {model_name:8s} | {row['target']:18s} | {row}")

            if not args.dry_run:
                board_df = pd.DataFrame(all_rows).sort_values(["target", "model"]).reset_index(drop=True)
                board_df.to_csv(LEADERBOARD_CSV, index=False)

        except Exception as exc:
            log(f"ERROR in model family '{model_name}': {exc}")
            log(traceback.format_exc())

        gc.collect()

    elapsed_min = (time.perf_counter() - t_total) / 60.0
    log(f"=== Overnight Run Complete in {elapsed_min:.2f} mins ({len(all_rows)} runs completed) ===")
    if args.dry_run:
        log("Dry run finished (no models, predictions, or leaderboard files were written).")
    else:
        log(f"Leaderboard saved to: {LEADERBOARD_CSV}")


if __name__ == "__main__":
    main()