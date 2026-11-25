Weather Forecaster

Setup and Installation

After cloning the repository, install the dependencies declared in pyproject.toml (Python 3.12) before running any stage of the pipeline.

Using uv (recommended):

git clone <https://github.com/VazeemBasheer/Vazeem-pm_accelerator_Internship-2026.git>
cd weather_forcaster
uv sync


Using standard pip:

git clone <https://github.com/VazeemBasheer/Vazeem-pm_accelerator_Internship-2026.git>
cd weather_forcaster
python -m venv .venv
.venv\Scripts\activate
pip install -e .


End-to-End Pipeline Execution Order

Run Stages 0 through 9 sequentially to reproduce the dataset preparation, exploratory charts, feature tables, model benchmarks, frozen ensemble weights, holdout evaluations, standalone forecasts, latest 7-day inference, and Streamlit dashboard.

Stage 0: Data Ingestion

Loads the raw weather observations into the local workspace and validates required schema fields.

uv run src/ingest.py


Stage 1: Data Cleaning

Standardizes timestamps, resolves missing values and unit inconsistencies, deduplicates city-date records, and constructs composite coordinate keys (coord_key) to distinguish cities sharing identical names across regions.

uv run src/clean.py


Stage 2: Chronological Train, Validation, and Test Splitting

Partitions the cleaned dataset strictly by time into contiguous train, valid, and holdout test windows prior to any feature transformation. Enforcing chronological boundaries at Stage 2 prevents future observations from leaking into rolling statistics, target encodings, or neural sequence windows.

uv run src/split.py


Stage 3: Core and Advanced Exploratory Data Analysis

Generates correlation heatmaps, target distribution plots, seasonal drift diagnostics, and regional anomaly charts used to guide downstream modeling decisions.

uv run src/eda.py


Key EDA Findings:

Zero-Inflated Precipitation: Daily rainfall is heavily skewed toward zero (0.0 mm) with occasional heavy spikes, motivating a two-part hurdle architecture that pairs calibrated rain probabilities with continuous precipitation regression.

Seasonal Non-Stationarity: Temperature distributions shift significantly across seasons, meaning tree-based models struggle to extrapolate linear trends without hybrid residual modeling or smooth neural representations.

Spatial and Temporal Autocorrelation: Short-term persistence (h1) is dominated by recent lag trajectories and diurnal pressure/humidity shifts, whereas 7-day (h7) forecasts rely heavily on city-month climatology baselines.

Stage 4: Feature Engineering

Constructs 31 domain-specific, lag, rolling, and climatological features saved to data/interim/weather_features.parquet.

uv run src/features_engineering.py


Engineered Feature Groups:

Autoregressive Lags and Deltas: Multi-day lags (1d, 2d, 3d, 7d) and first-order differences for temperature, humidity, surface pressure, wind speed, and cloud cover.

Rolling Window Statistics: Causal rolling means, standard deviations, and min/max ranges over 3-day, 7-day, and 14-day lookback windows computed strictly from past rows.

Moisture and Thermodynamic Indicators: Dew-point spread approximations, vapor pressure deficit proxies, and rolling wet-day frequencies.

Climatology Baselines: Out-of-fold / training-only city-month historical temperature and precipitation averages to anchor 7-day (h7) seasonal expectations.

Stage 5: Feature Selection

Evaluates predictor stability across time folds and prunes redundant collinear variables, narrowing the feature set down to the 5 to 12 most stable predictors per target and horizon.

uv run src/feature_selection.py


Feature Selection Methodology:

Collinearity Filtering: Removes highly correlated duplicates (|r| > 0.92) while retaining the feature with stronger out-of-sample Mutual Information with the target.

Cross-Fold Stability Selection: Ranks features using permutation importance across temporal and spatial sub-folds, keeping only predictors that consistently reduce validation error across seasons and geographic regions.

Target-Specific Subsets: Applies a 1-standard-error cutoff to assign compact, specialized feature subsets (5 to 12 features) per target (temp, precip, rain) and horizon (h1, h7) to prevent tree overfitting on noisy atmospheric measurements.

Stage 6: Model Training and Validation Benchmarks

Trains seven distinct model families on the train split and evaluates them on valid, recording all metrics in data/processed/benchmark_leaderboard.csv.

uv run src/train.py --eval-split valid
uv run src/train_resnet.py --eval-split valid
uv run src/train_seq.py --eval-split valid


(Note: src/train.py --dry-run can be used to verify the pipeline end-to-end without writing model artifacts.)

Model Architectures:

Persistence Baseline (baseline): Predicts the most recent observed value (or historical wet-day rate) as a benchmark floor.

Regularized Linear Models (linear): Ridge regression for continuous targets and L2-regularized Logistic Regression for rain classification, providing smooth trend extrapolation.

Gradient Boosted Decision Trees (histgb, lgbm, catboost): Non-linear tabular ensembles trained with early stopping. For target_temp_h7, tree models use a hybrid Ridge-residual formulation—fitting trees on the residuals of a 5-fold date-grouped out-of-fold Ridge model—to overcome tree extrapolation limits during seasonal temperature shifts.

Multi-Head Tabular ResNet (resnet): A PyTorch residual neural network with LayerNorm, SiLU activations, and dropout that jointly learns shared atmospheric representations across all six targets.

Causal 7-Day Conv1D-GRU (seq): A PyTorch sequence architecture combining 1D causal convolutions with a Gated Recurrent Unit over strictly past 7-day city histories ((B, 7, F)), capturing multi-day weather front momentum without cross-split boundary leakage.

Stage 7: Out-of-Fold Ensemble Optimization

Combines individual model predictions using 5-fold date-grouped cross-validation (GroupKFold grouped by calendar date) and saves the frozen configuration to models/ensemble_weights.json.

uv run src/ensemble.py --eval-split valid --folds 5


Ensembling and Hurdle Calibration:

Date-Grouped Out-of-Fold Blending: Grouping folds by date prevents simultaneous global weather events from leaking across folds during weight optimization.

Simplex Weight Optimization: Solves for non-negative weights summing to 1.0 per target (SLSQP minimization) directly optimizing MAE for continuous targets and LogLoss for rain probabilities.

Precipitation Hurdle Gating: Multiplies continuous precipitation predictions by a threshold gate derived from the ensemble's calibrated rain probability (pred_rain_prob >= threshold), eliminating false micro-drizzle noise on dry days.

Stage 8: Holdout Test Evaluation

Evaluates all trained models and the frozen ensemble on the unseen test split without refitting weights or thresholds, writing final generalization results to data/processed/test_leaderboard.csv.

uv run src/train.py --eval-split test
uv run src/train_resnet.py --eval-split test
uv run src/train_seq.py --eval-split test
uv run src/ensemble.py --eval-split test


Evaluation Methodology:

Temperature (target_temp_h1, target_temp_h7): Evaluated via Mean Absolute Error (MAE in °C) and Root Mean Squared Error (RMSE) to measure direct degree accuracy and sensitivity to extreme heat/cold waves.

Precipitation Volume (target_precip_h1, target_precip_h7): Evaluated via MAE (in mm) and RMSE to capture both dry-day precision and storm magnitude accuracy.

Rain Occurrence (target_rain_h1, target_rain_h7): Evaluated via Binary Cross-Entropy (LogLoss) for probability calibration and ROC-AUC for ranking discrimination across dry and wet regimes.

Stage 9: Live 7-Day Refresh, Visualization, and Dashboard

Generates batch and city-specific forecasts from frozen artifacts, pulls the latest 7-day observation window for fresh multi-horizon predictions without retraining, renders comparison bar charts into reports/figures/, and launches the interactive Streamlit dashboard (app.py).

Holdout Batch and City Inference:

uv run src/predict.py --split test --evaluate
uv run src/predict.py --split test --city "New Delhi" --latest --output data/processed/delhi_forecast.csv
uv run src/plot_leaderboards.py
uv run streamlit run app.py


Latest 7-Day Live Refresh Inference:

To generate fresh +1d and +7d forecasts from the newest available snapshot in the Global Weather Repository without retraining models or shifting the frozen train/valid/test boundaries in weather_split.parquet, run src/live_7d_forecast.py:

uv run src/live_7d_forecast.py


To write the latest 7-day forecast window directly into data/processed/forecasts.parquet so app.py displays it immediately:

uv run src/live_7d_forecast.py --update-dashboard
uv run streamlit run app.py


(You can also pass --skip-download to reuse the existing local raw CSV, or run uv run src/predict.py --split test at any time to restore the full holdout test set in forecasts.parquet.)

How Latest 7-Day Inference Works:

Lookback Preservation: Extracts a 35-day lookback window per city from the newest snapshot so all lag features, rolling windows, and the 7-day causal Conv1D-GRU sequence inputs have complete history.

Leakage-Safe Feature Reuse: Applies the exact Stage 1 cleaning rules and Stage 4 feature transformations using frozen training-set monthly medians and city-month climatology from weather_features.parquet, while skipping future target dropping so the most recent observation day is retained for prediction.

Frozen Ensemble Execution: Calls src/predict.py against the frozen tabular, neural, and ensemble weight artifacts and slices the output to the latest 7 days in data/processed/live_7d_forecasts.parquet.


Summary of Results

Validation Split (valid)

The 5-fold date-grouped ensemble matched or outperformed every individual model across all six targets:

target_precip_h7: 0.1025 mm MAE (-39.81% vs. persistence baseline)

target_precip_h1: 0.1019 mm MAE (-34.13% vs. persistence baseline)

target_rain_h1: 0.3253 LogLoss (0.8159 ROC-AUC)

target_rain_h7: 0.3477 LogLoss (0.7688 ROC-AUC)

target_temp_h7: 3.1288 °C MAE (-14.04% vs. persistence baseline)

target_temp_h1: 1.6708 °C MAE

Holdout Test Split (test)

On the unseen holdout split, the frozen ensemble demonstrated strong cross-regime generalization:

target_rain_h1: Ranked #1 with 0.3020 LogLoss and 0.8366 ROC-AUC

target_rain_h7: Ranked #1 with 0.3187 LogLoss and 0.8081 ROC-AUC

target_precip_h7: Tied #1 with 0.1265 mm MAE (-39.44% vs. persistence baseline)

target_precip_h1: Ranked #2 with 0.1244 mm MAE (-36.47% vs. persistence baseline)

Temperature Generalization: Under a seasonal temperature regime shift in the test window, the neural architectures achieved the lowest holdout temperature errors, with seq (Conv1D-GRU) leading target_temp_h1 (1.4913 °C MAE) and resnet leading target_temp_h7 (2.1724 °C MAE).