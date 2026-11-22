import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_FILE = PROJECT_ROOT / "logs" / "overnight_train.log"
WEIGHTS_JSON = MODEL_DIR / "ensemble_weights.json"
VALID_LEADERBOARD_CSV = OUTPUT_DIR / "benchmark_leaderboard.csv"
TEST_LEADERBOARD_CSV = OUTPUT_DIR / "test_leaderboard.csv"

TARGETS = [
    "target_rain_h1",
    "target_rain_h7",
    "target_temp_h1",
    "target_temp_h7",
    "target_precip_h1",
    "target_precip_h7",
]

CANDIDATE_MODELS = ["linear", "histgb", "lgbm", "catboost", "resnet", "seq"]

PRECIP_TO_RAIN = {
    "target_precip_h1": "target_rain_h1",
    "target_precip_h7": "target_rain_h7",
}


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def load_model_predictions(
    target: str, eval_split: str, models: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    loaded_models: list[str] = []
    pred_cols: list[np.ndarray] = []
    ref_meta: pd.DataFrame | None = None
    ref_actual: np.ndarray | None = None

    for m in models:
        path = OUTPUT_DIR / f"preds_{target}_{m}_{eval_split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        actual = df["actual"].to_numpy(dtype=np.float64)
        pred = df["predicted"].to_numpy(dtype=np.float64)

        if ref_actual is None:
            ref_actual = actual
            meta_cols = [c for c in ["coord_key", "location_name", "country", "date"] if c in df.columns]
            ref_meta = df[meta_cols].copy()
        else:
            if len(actual) != len(ref_actual) or not np.allclose(actual, ref_actual, atol=1e-5, equal_nan=True):
                raise RuntimeError(f"Row alignment mismatch for {target} ({m} vs {loaded_models[0]}) on split={eval_split}")

        loaded_models.append(m)
        pred_cols.append(pred)

    if not loaded_models or ref_actual is None or ref_meta is None:
        raise FileNotFoundError(f"No prediction files found for {target} (split={eval_split}) in {OUTPUT_DIR}")

    return ref_meta, ref_actual, np.column_stack(pred_cols), loaded_models


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def optimize_simplex_weights(P: np.ndarray, y: np.ndarray, task: str, seed: int = 42) -> np.ndarray:
    n_models = P.shape[1]

    def loss_from_weights(w: np.ndarray) -> float:
        blend = P @ w
        if task == "regression":
            return float(mean_absolute_error(y, blend))
        probs = np.clip(blend, 1e-6, 1.0 - 1e-6)
        return float(log_loss(y, probs, labels=[0, 1]))

    indiv_losses = [loss_from_weights(np.eye(n_models)[i]) for i in range(n_models)]
    best_single = int(np.argmin(indiv_losses))

    rng = np.random.default_rng(seed)
    best_w = np.eye(n_models)[best_single]
    best_loss = indiv_losses[best_single]

    eq_w = np.full(n_models, 1.0 / n_models)
    eq_loss = loss_from_weights(eq_w)
    if eq_loss < best_loss:
        best_w, best_loss = eq_w, eq_loss

    candidates = rng.dirichlet(np.ones(n_models), size=400)
    for w_cand in candidates:
        cand_loss = loss_from_weights(w_cand)
        if cand_loss < best_loss:
            best_w, best_loss = w_cand, cand_loss

    if task == "classification":
        res = minimize(
            loss_from_weights,
            best_w,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"ftol": 1e-7, "maxiter": 200},
        )
        if res.success and res.fun < best_loss:
            w_opt = np.clip(res.x, 0.0, 1.0)
            best_w = w_opt / np.sum(w_opt)
    else:
        init_logits = np.log(np.clip(best_w, 1e-4, 1.0))
        res = minimize(
            lambda logits: loss_from_weights(softmax(logits)),
            init_logits,
            method="Powell",
            options={"ftol": 1e-6, "maxiter": 500},
        )
        w_opt = softmax(res.x)
        if loss_from_weights(w_opt) < best_loss:
            best_w = w_opt

    best_w = np.where(best_w < 0.005, 0.0, best_w)
    best_w = best_w / np.sum(best_w)
    return best_w


def optimize_precip_hurdle(precip_preds: np.ndarray, rain_probs: np.ndarray, y_true: np.ndarray) -> float:
    best_thresh = 0.0
    best_mae = float(mean_absolute_error(y_true, precip_preds))
    for thresh in np.linspace(0.02, 0.45, 44):
        gated = np.where(rain_probs < thresh, 0.0, precip_preds)
        mae = float(mean_absolute_error(y_true, gated))
        if mae < best_mae - 1e-6:
            best_mae = mae
            best_thresh = float(round(thresh, 3))
    return best_thresh


def upsert_leaderboard(path: Path, new_rows: list[dict], model_tag: str = "ensemble") -> None:
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old_df = pd.read_csv(path)
        old_df = old_df[old_df["model"] != model_tag]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values(["target", "model"]).reset_index(drop=True)
    combined.to_csv(path, index=False)


def run_valid_fit(n_splits: int = 5) -> None:
    t0 = time.perf_counter()
    log(f"=== Fitting Leak-Proof {n_splits}-Fold Date-Grouped OOF Ensemble on valid ===")

    learned_config: dict[str, dict] = {}
    ensemble_rain_oof: dict[str, np.ndarray] = {}
    leaderboard_rows: list[dict] = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for target in TARGETS:
        task = "classification" if "rain" in target else "regression"
        meta_df, y_true, P, models = load_model_predictions(target, "valid", CANDIDATE_MODELS)

        oof_preds = np.zeros(len(y_true), dtype=np.float64)
        rain_target = PRECIP_TO_RAIN.get(target)
        rain_probs = ensemble_rain_oof.get(rain_target) if rain_target else None

        if "date" in meta_df.columns and meta_df["date"].nunique() >= n_splits:
            splits = list(GroupKFold(n_splits=n_splits).split(P, y_true, groups=meta_df["date"]))
        else:
            splits = list(kf.split(P))

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            w_fold = optimize_simplex_weights(P[tr_idx], y_true[tr_idx], task, seed=42 + fold_idx)
            va_pred = P[va_idx] @ w_fold

            if rain_probs is not None and len(rain_probs) == len(y_true):
                tr_pred = P[tr_idx] @ w_fold
                thresh = optimize_precip_hurdle(tr_pred, rain_probs[tr_idx], y_true[tr_idx])
                va_pred = np.where(rain_probs[va_idx] < thresh, 0.0, va_pred)

            if "precip" in target:
                va_pred = np.clip(va_pred, 0.0, None)

            oof_preds[va_idx] = va_pred

        if task == "classification":
            ensemble_rain_oof[target] = oof_preds.copy()

        final_w = optimize_simplex_weights(P, y_true, task, seed=42)
        final_hurdle = 0.0
        if rain_probs is not None and len(rain_probs) == len(y_true):
            final_hurdle = optimize_precip_hurdle(P @ final_w, rain_probs, y_true)

        oof_metrics = evaluate_predictions(y_true, oof_preds, task)

        out_df = meta_df.copy()
        out_df["actual"] = y_true
        out_df["predicted"] = oof_preds
        out_df.to_parquet(OUTPUT_DIR / f"preds_{target}_ensemble_valid.parquet", index=False)

        weight_map = {m: round(float(w), 4) for m, w in zip(models, final_w) if w > 0.0}
        learned_config[target] = {
            "task": task,
            "weights": weight_map,
            "hurdle_rain_target": rain_target if final_hurdle > 0.0 else None,
            "hurdle_threshold": final_hurdle,
            "oof_metrics": oof_metrics,
        }

        row = {
            "model": "ensemble",
            "target": target,
            "task": task,
            "residual_mode": "temp" in target,
            "features": len(weight_map),
            "train_rows": len(y_true),
            "eval_rows": len(y_true),
            "time_s": round(time.perf_counter() - t0, 2),
            **oof_metrics,
        }
        leaderboard_rows.append(row)

        hurdle_str = f" | hurdle={final_hurdle:.3f}" if final_hurdle > 0.0 else ""
        log(f"  ensemble | {target:18s} | {oof_metrics} | weights={weight_map}{hurdle_str}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(learned_config, f, indent=2)

    upsert_leaderboard(VALID_LEADERBOARD_CSV, leaderboard_rows, model_tag="ensemble")
    log(f"Saved frozen ensemble weights to {WEIGHTS_JSON}")
    log(f"Updated {VALID_LEADERBOARD_CSV} with ensemble OOF rows.")


def run_test_eval() -> None:
    if not WEIGHTS_JSON.exists():
        raise FileNotFoundError(f"Frozen weights not found at {WEIGHTS_JSON}. Run --eval-split valid first.")

    with open(WEIGHTS_JSON, "r", encoding="utf-8") as f:
        config = json.load(f)

    log("=== Running Frozen-Weight Holdout Evaluation on test ===")
    all_test_rows: list[dict] = []
    ensemble_rain_test: dict[str, np.ndarray] = {}

    for target in TARGETS:
        cfg = config[target]
        task = cfg["task"]
        weight_map: dict[str, float] = cfg["weights"]

        avail_models = ["baseline"] + CANDIDATE_MODELS
        meta_df, y_true, P_all, loaded_models = load_model_predictions(target, "test", avail_models)

        model_preds_dict = {m: P_all[:, i] for i, m in enumerate(loaded_models)}
        for m in loaded_models:
            metrics = evaluate_predictions(y_true, model_preds_dict[m], task)
            all_test_rows.append({
                "model": m,
                "target": target,
                "task": task,
                "eval_rows": len(y_true),
                **metrics,
            })

        ens_pred = np.zeros(len(y_true), dtype=np.float64)
        total_w = 0.0
        for m, w in weight_map.items():
            if m not in model_preds_dict:
                raise FileNotFoundError(f"Missing test prediction file for required ensemble member: {target}_{m}_test")
            ens_pred += w * model_preds_dict[m]
            total_w += w
        ens_pred /= max(total_w, 1e-12)

        hurdle_target = cfg.get("hurdle_rain_target")
        hurdle_thresh = float(cfg.get("hurdle_threshold", 0.0))
        if hurdle_target and hurdle_thresh > 0.0 and hurdle_target in ensemble_rain_test:
            ens_pred = np.where(ensemble_rain_test[hurdle_target] < hurdle_thresh, 0.0, ens_pred)

        if "precip" in target:
            ens_pred = np.clip(ens_pred, 0.0, None)

        if task == "classification":
            ensemble_rain_test[target] = ens_pred.copy()

        ens_metrics = evaluate_predictions(y_true, ens_pred, task)
        all_test_rows.append({
            "model": "ensemble",
            "target": target,
            "task": task,
            "eval_rows": len(y_true),
            **ens_metrics,
        })

        out_df = meta_df.copy()
        out_df["actual"] = y_true
        out_df["predicted"] = ens_pred
        out_df.to_parquet(OUTPUT_DIR / f"preds_{target}_ensemble_test.parquet", index=False)
        log(f"  TEST ensemble | {target:18s} | {ens_metrics}")

    test_df = pd.DataFrame(all_test_rows).sort_values(["target", "model"]).reset_index(drop=True)
    test_df.to_csv(TEST_LEADERBOARD_CSV, index=False)
    log(f"=== Holdout Test Leaderboard saved to {TEST_LEADERBOARD_CSV} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leak-proof OOF ensemble optimizer & frozen test evaluator.")
    parser.add_argument("--eval-split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--folds", type=int, default=5, help="Number of OOF folds for validation weight fitting")
    args = parser.parse_args()

    if args.eval_split == "valid":
        run_valid_fit(n_splits=args.folds)
    else:
        run_test_eval()


if __name__ == "__main__":
    main()