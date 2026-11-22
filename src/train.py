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
from sklearn.model_selection import GroupKFold, KFold
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

# Tree models train on 5-fold date-grouped OOF Ridge residuals for these targets
HYBRID_RIDGE_TARGETS = {"target_temp_h7"}
TREE_MODELS = {"histgb", "lgbm", "catboost"}

FORBIDDEN_COLS = {
    "date",
    "split",
    "coord_key",
    "location_name",
    "country",
    "last_updated",
}


def detect_gpu() -> bool:
    try:
        from catboost.utils import get_gpu_device_count
        return get_gpu_device_count() > 0
    except Exception:
        return False


HAS_GPU = detect_gpu()


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


def get_anchor_series(df: pd.DataFrame, target: str, fallback_mean: float) -> np.ndarray | None:
    anchor_col = RESIDUAL_ANCHORS.get(target)
    if anchor_col and anchor_col in df.columns:
        return df[anchor_col].fillna(fallback_mean).to_numpy(dtype=np.float64)
    return None


def predict_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, target: str) -> np.ndarray:
    fallback = float(train_df[target].mean())
    if "temp" in target:
        anchor = get_anchor_series(eval_df, target, fallback)
        if anchor is not None:
            return anchor
        return np.full(len(eval_df), fallback, dtype=np.float64)

    if "precip" in target:
        if "precip_mm" in eval_df.columns:
            return eval_df["precip_mm"].fillna(0.0).to_numpy(dtype=np.float64)
        return np.full(len(eval_df), fallback, dtype=np.float64)

    return np.full(len(eval_df), fallback, dtype=np.float64)


def make_ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def fit_oof_ridge_stage(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    dates: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """Fits leak-free 5-fold date-grouped OOF Ridge on train + full Ridge for eval."""
    oof_preds = np.zeros(len(X_train), dtype=np.float64)
    if dates is not None and dates.nunique() >= 5:
        splitter = GroupKFold(n_splits=5).split(X_train, y_train, groups=dates)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42).split(X_train)

    for tr_idx, va_idx in splitter:
        fold_ridge = make_ridge_pipeline()
        fold_ridge.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_preds[va_idx] = fold_ridge.predict(X_train.iloc[va_idx])

    full_ridge = make_ridge_pipeline()
    full_ridge.fit(X_train, y_train)
    eval_preds = full_ridge.predict(X_eval)
    return oof_preds, eval_preds, full_ridge


def build_estimator(
    model_name: str,
    task: str,
    target: str,
    parallel_mode: bool = False,
    use_gpu: bool = True,
):
    name = model_name.lower()
    is_precip = "precip" in target.lower()

    if name == "linear":
        if task == "regression":
            return make_ridge_pipeline()
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ])

    if name == "histgb":
        if task == "regression":
            return HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=350,
                learning_rate=0.05,
                random_state=42,
            )
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)

    if name == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        n_jobs = 4 if parallel_mode else -1
        params = dict(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            random_state=42,
            n_jobs=n_jobs,
            verbose=-1,
        )
        if task == "classification":
            return LGBMClassifier(**params)
        if is_precip:
            params["objective"] = "tweedie"
            params["tweedie_variance_power"] = 1.5
        else:
            params["objective"] = "regression_l1"
        return LGBMRegressor(**params)

    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        params = dict(
            iterations=450,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        if task == "regression":
            params["loss_function"] = "Tweedie:variance_power=1.5" if is_precip else "MAE"

        if use_gpu:
            params["task_type"] = "GPU"
            params["gpu_ram_part"] = 0.30 if parallel_mode else 0.70
            params["metric_period"] = 5
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
    y_train_raw = tr_df[target].to_numpy(dtype=np.float64)
    y_eval_raw = ev_df[target].to_numpy(dtype=np.float64)

    anchor_col = RESIDUAL_ANCHORS.get(target)
    train_mean = float(np.mean(y_train_raw))
    train_anchor = get_anchor_series(tr_df, target, train_mean)
    eval_anchor = get_anchor_series(ev_df, target, train_mean)
    use_residual = (anchor_col is not None) and (train_anchor is not None) and (eval_anchor is not None)
    use_hybrid_ridge = (target in HYBRID_RIDGE_TARGETS) and (model_name in TREE_MODELS) and use_residual

    if model_name == "baseline":
        preds = predict_baseline(tr_df, ev_df, target)
    else:
        X_train = tr_df[features]
        X_eval = ev_df[features]
        y_fit = (y_train_raw - train_anchor) if use_residual else y_train_raw

        ridge_stage = None
        ridge_eval_offset = 0.0
        if use_hybrid_ridge:
            dates = tr_df["date"] if "date" in tr_df.columns else None
            ridge_oof, ridge_eval_offset, ridge_stage = fit_oof_ridge_stage(X_train, y_fit, X_eval, dates)
            y_fit = y_fit - ridge_oof

        if model_name == "catboost":
            try:
                est = build_estimator(model_name, task, target, parallel_mode=parallel_mode, use_gpu=HAS_GPU)
                est.fit(X_train, y_fit)
            except Exception:
                est = build_estimator(model_name, task, target, parallel_mode=parallel_mode, use_gpu=False)
                est.fit(X_train, y_fit)
        else:
            est = build_estimator(model_name, task, target, parallel_mode=parallel_mode)
            est.fit(X_train, y_fit)

        raw_preds = (
            est.predict_proba(X_eval)[:, 1]
            if task == "classification"
            else est.predict(X_eval)
        )

        if use_residual:
            preds = raw_preds + ridge_eval_offset + eval_anchor
        elif "precip" in target.lower():
            preds = np.clip(raw_preds, 0.0, None)
        else:
            preds = raw_preds

        if not dry_run:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": est,
                    "ridge_stage": ridge_stage,
                    "features": features,
                    "target": target,
                    "task": task,
                    "residual_anchor": anchor_col if use_residual else None,
                    "fallback_mean": train_mean,
                },
                MODEL_DIR / f"{target}_{model_name}.joblib",
            )

    metrics = evaluate_predictions(y_eval_raw, preds, task)

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        meta_cols = [c for c in ["coord_key", "location_name", "country", "date"] if c in ev_df.columns]
        out_df = ev_df[meta_cols].copy()
        out_df["actual"] = y_eval_raw
        out_df["predicted"] = preds
        out_df.to_parquet(OUTPUT_DIR / f"preds_{target}_{model_name}_{eval_split}.parquet", index=False)

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "model": model_name,
        "target": target,
        "task": task,
        "residual_mode": use_residual,
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
    extra_cols = []
    if target in RESIDUAL_ANCHORS and RESIDUAL_ANCHORS[target] in df.columns:
        extra_cols.append(RESIDUAL_ANCHORS[target])
    if model_name == "baseline":
        extra_cols.extend([c for c in ["temperature_celsius", "precip_mm", "city_month_temp_clim"] if c in df.columns])

    keep_cols = list(dict.fromkeys(meta_cols + features + extra_cols + [target]))
    tr_slice = df.loc[train_mask & df[target].notna(), keep_cols].copy()
    ev_slice = df.loc[eval_mask & df[target].notna(), keep_cols].copy()
    return tr_slice, ev_slice


def main() -> None:
    parser = argparse.ArgumentParser(description="Leak-proof benchmark runner with hybrid Ridge+Tree temp_h7.")
    parser.add_argument("--eval-split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    leaderboard_csv = OUTPUT_DIR / (
        "benchmark_leaderboard.csv" if args.eval_split == "valid" else f"{args.eval_split}_leaderboard.csv"
    )

    t_total = time.perf_counter()
    log(f"=== Starting Benchmark Run (split={args.eval_split}, dry_run={args.dry_run}, has_gpu={HAS_GPU}) ===")

    df = load_dataset(args.data)
    if args.dry_run:
        df = df.groupby("split", group_keys=False).head(2000).copy()
        log(f"Dry-run active: sampled {len(df):,} rows total (read-only mode).")

    train_mask = df["split"] == "train"
    eval_mask = df["split"] == args.eval_split

    if not train_mask.any() or not eval_mask.any():
        raise ValueError(f"Empty split detected: train={train_mask.sum()}, {args.eval_split}={eval_mask.sum()}")

    available_targets = [t for t in TARGETS if t in df.columns]
    target_features = {t: get_feature_list(args.selected, t, df) for t in available_targets}
    for t, feats in target_features.items():
        hybrid_note = " | hybrid_ridge=True" if t in HYBRID_RIDGE_TARGETS else ""
        log(f"Loaded feature schema | {t:18s}: {len(feats)} features{hybrid_note}")

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
                        target, model_name, args.eval_split, feats, tr_slice, ev_slice,
                        parallel_mode=False, dry_run=args.dry_run,
                    )
                    all_rows.append(row)
                    log(f"  {model_name:8s} | {target:18s} | {row}")
            else:
                tasks = []
                for target in available_targets:
                    feats = target_features[target]
                    tr_slice, ev_slice = prepare_slice(df, train_mask, eval_mask, target, feats, model_name)
                    tasks.append((target, model_name, args.eval_split, feats, tr_slice, ev_slice, True, args.dry_run))

                workers = 2 if (model_name == "catboost" and HAS_GPU) else 3
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(fit_and_eval_one, *t) for t in tasks]
                    for fut in as_completed(futures):
                        row = fut.result()
                        all_rows.append(row)
                        log(f"  {model_name:8s} | {row['target']:18s} | {row}")

            if not args.dry_run:
                new_df = pd.DataFrame(all_rows)
                if leaderboard_csv.exists():
                    old_df = pd.read_csv(leaderboard_csv)
                    old_df = old_df[~old_df["model"].isin(new_df["model"].unique())]
                    board_df = pd.concat([old_df, new_df], ignore_index=True)
                else:
                    board_df = new_df
                board_df = board_df.sort_values(["target", "model"]).reset_index(drop=True)
                board_df.to_csv(leaderboard_csv, index=False)

        except Exception as exc:
            log(f"ERROR in model family '{model_name}': {exc}")
            log(traceback.format_exc())

        gc.collect()

    elapsed_min = (time.perf_counter() - t_total) / 60.0
    log(f"=== Run Complete in {elapsed_min:.2f} mins ({len(all_rows)} runs completed) ===")
    if not args.dry_run:
        log(f"Leaderboard saved to: {leaderboard_csv}")


if __name__ == "__main__":
    main()