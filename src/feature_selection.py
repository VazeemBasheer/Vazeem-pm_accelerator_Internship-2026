from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE

FEATURES_PARQUET = PROJECT_ROOT / "data" / "interim" / "weather_features.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SELECTED_CSV = PROCESSED_DIR / "selected_features.csv"
CURVE_CSV = PROCESSED_DIR / "feature_selection_summary.csv"

MAX_HORIZON = 7
N_SPLITS = 5
CORR_THRESHOLD = 0.92
MIN_STABILITY = 0.70

META_COLS = {
    "date", "split", "coord_key", "location_name", "country",
    "latitude", "longitude", "lat_round", "lon_round",
    "last_updated", "last_updated_epoch", "hour", "month",
}

TARGETS = [
    ("target_temp_h1", "regression"),
    ("target_temp_h7", "regression"),
    ("target_precip_h1", "regression"),
    ("target_precip_h7", "regression"),
    ("target_rain_h1", "classification"),
    ("target_rain_h7", "classification"),
]


def get_model(task: str):
    # Return a fast gradient-boosted estimator for regression or classification
    if HAS_LGBM:
        if task == "regression":
            return LGBMRegressor(
                n_estimators=150, learning_rate=0.05, num_leaves=31,
                subsample_for_bin=50000, random_state=42, n_jobs=-1, verbose=-1
            )
        return LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            subsample_for_bin=50000, random_state=42, n_jobs=-1, verbose=-1
        )
    if task == "regression":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=42)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=42)


def get_candidate_features(df: pd.DataFrame) -> list[str]:
    # Select numeric predictor columns excluding metadata and targets
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in META_COLS and not c.startswith("target_")
    ]


def prune_collinear_features(
    X: pd.DataFrame, y: pd.Series, task: str, threshold: float = CORR_THRESHOLD
) -> tuple[list[str], pd.Series]:
    # Drop zero-variance features and prune Spearman pairs > threshold using Mutual Information
    non_const = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X_sub = X[non_const].fillna(X[non_const].median())

    # Subsample up to 40k rows for fast MI and Spearman estimation
    if len(X_sub) > 40000:
        sample_idx = X_sub.sample(40000, random_state=42).index
        X_sample, y_sample = X_sub.loc[sample_idx], y.loc[sample_idx]
    else:
        X_sample, y_sample = X_sub, y

    if task == "regression":
        mi_vals = mutual_info_regression(X_sample, y_sample, random_state=42)
    else:
        mi_vals = mutual_info_classif(X_sample, y_sample.astype(int), random_state=42)
    mi_series = pd.Series(mi_vals, index=non_const)

    corr_matrix = X_sample.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for partner in high_corr:
            if col in to_drop or partner in to_drop:
                continue
            # Keep whichever feature has higher Mutual Information with the target
            if mi_series[col] >= mi_series[partner]:
                to_drop.add(partner)
            else:
                to_drop.add(col)

    kept = [c for c in non_const if c not in to_drop]
    return kept, mi_series


def build_spatiotemporal_folds(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    # Combine 5 expanding time folds (gap=7) and 5 city-grouped folds
    folds = []

    # 1. Temporal folds
    dates = np.sort(df["date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=MAX_HORIZON)
    for tr_d, va_d in tscv.split(dates):
        tr_idx = df.index[df["date"].isin(dates[tr_d])].to_numpy()
        va_idx = df.index[df["date"].isin(dates[va_d])].to_numpy()
        folds.append(("temporal", tr_idx, va_idx))

    # 2. Spatial folds (unseen cities)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr_pos, va_pos in gkf.split(df, groups=df["coord_key"]):
        folds.append(("spatial", df.index[tr_pos].to_numpy(), df.index[va_pos].to_numpy()))

    return folds


def score_predictions(y_true: pd.Series, model, X_val: pd.DataFrame, task: str) -> float:
    # Return lower-is-better error metric (MAE for regression, log-loss for rain)
    if task == "regression":
        preds = model.predict(X_val)
        return float(mean_absolute_error(y_true, preds))
    probs = model.predict_proba(X_val)[:, 1]
    return float(log_loss(y_true.astype(int), probs, labels=[0, 1]))


def rank_by_permutation_stability(
    df: pd.DataFrame, features: list[str], target: str, task: str, folds: list
) -> pd.DataFrame:
    # Measure out-of-fold permutation importance across all temporal and spatial folds
    scoring = "neg_mean_absolute_error" if task == "regression" else "neg_log_loss"
    imp_records = []

    for fold_type, tr_idx, va_idx in folds:
        X_tr, y_tr = df.loc[tr_idx, features], df.loc[tr_idx, target]
        X_va, y_va = df.loc[va_idx, features], df.loc[va_idx, target]

        model = get_model(task)
        model.fit(X_tr, y_tr)

        perm = permutation_importance(
            model, X_va, y_va, scoring=scoring, n_repeats=3, random_state=42, n_jobs=-1
        )
        imp_records.append(pd.Series(perm.importances_mean, index=features))

    imp_df = pd.DataFrame(imp_records)
    summary = pd.DataFrame({
        "feature": features,
        "perm_imp_mean": imp_df.mean(axis=0).values,
        "perm_imp_std": imp_df.std(axis=0).values,
        "stability_rate": (imp_df > 0).mean(axis=0).values,
    })
    return summary.sort_values(["perm_imp_mean", "stability_rate"], ascending=False).reset_index(drop=True)


def select_one_se_subset(
    df: pd.DataFrame, ranked_features: list[str], target: str, task: str, folds: list
) -> tuple[list[str], pd.DataFrame]:
    # Pick the smallest top-k feature subset within 1 standard error of the best CV score
    max_k = len(ranked_features)
    k_grid = sorted(set([k for k in [5, 8, 12, 16, 20, 25, 30, 40, max_k] if k <= max_k]))

    curve_rows = []
    for k in k_grid:
        subset = ranked_features[:k]
        fold_scores = []
        for _, tr_idx, va_idx in folds:
            model = get_model(task)
            model.fit(df.loc[tr_idx, subset], df.loc[tr_idx, target])
            fold_scores.append(score_predictions(df.loc[va_idx, target], model, df.loc[va_idx, subset], task))

        mean_err = float(np.mean(fold_scores))
        se_err = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
        curve_rows.append({"target": target, "k": k, "cv_error_mean": mean_err, "cv_error_se": se_err})

    curve_df = pd.DataFrame(curve_rows)
    best_row = curve_df.loc[curve_df["cv_error_mean"].idxmin()]
    threshold = best_row["cv_error_mean"] + best_row["cv_error_se"]

    # Smallest k whose CV error falls inside the 1-SE tolerance band
    optimal_k = int(curve_df.loc[curve_df["cv_error_mean"] <= threshold, "k"].min())
    curve_df["is_chosen_k"] = curve_df["k"] == optimal_k

    return ranked_features[:optimal_k], curve_df


def main() -> None:
    # Run 3-stage leakage-safe feature selection on the training split only
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PARQUET)
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    candidate_cols = get_candidate_features(train_df)
    all_selected_rows = []
    all_curve_rows = []

    for target, task in TARGETS:
        valid_train = train_df.dropna(subset=[target]).reset_index(drop=True)
        folds = build_spatiotemporal_folds(valid_train)

        # Stage A: Prune collinear features
        kept_cols, mi_series = prune_collinear_features(valid_train[candidate_cols], valid_train[target], task)

        # Stage B: Rank by out-of-fold permutation stability across time & cities
        rank_df = rank_by_permutation_stability(valid_train, kept_cols, target, task, folds)
        rank_df["mutual_info"] = rank_df["feature"].map(mi_series)
        stable_features = rank_df.loc[
            (rank_df["stability_rate"] >= MIN_STABILITY) & (rank_df["perm_imp_mean"] > 0), "feature"
        ].tolist()

        # Fallback to top 10 features if strict stability threshold is too narrow
        if len(stable_features) < 5:
            stable_features = rank_df["feature"].head(10).tolist()

        # Stage C: Apply 1-standard-error cutoff
        chosen_features, curve_df = select_one_se_subset(valid_train, stable_features, target, task, folds)
        all_curve_rows.append(curve_df)

        rank_df["target"] = target
        rank_df["rank"] = np.arange(1, len(rank_df) + 1)
        rank_df["selected"] = rank_df["feature"].isin(chosen_features)
        all_selected_rows.append(rank_df)

        print(f"{target:18s} | pruned: {len(kept_cols):2d} -> stable: {len(stable_features):2d} -> selected (1-SE): {len(chosen_features):2d}")

    selected_out = pd.concat(all_selected_rows, ignore_index=True)
    curves_out = pd.concat(all_curve_rows, ignore_index=True)

    selected_out.to_csv(SELECTED_CSV, index=False)
    curves_out.to_csv(CURVE_CSV, index=False)
    print(f"\nSaved feature rankings to {SELECTED_CSV}")
    print(f"Saved 1-SE selection curves to {CURVE_CSV}")


if __name__ == "__main__":
    main()