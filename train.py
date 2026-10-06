"""
Explainable Machine Learning and Deep Neural Networks for Concrete Compressive Strength Prediction
===================================================================================================
Full Training Pipeline
Reproduces all model artifacts and evaluation CSV files from raw UCI data.

Usage:
    python train.py

Outputs (saved to project root):
    - xgboost.joblib, random_forest.joblib, gradient_boosting.joblib,
      svr.joblib, linear_regression.joblib          (trained ML models)
    - ann_model.keras, ann_scaler.joblib,
      ann_weights.joblib                             (ANN model + scaler)
    - model_comparison.csv                           (test-set metrics)
    - 10_fold_cross_validation.csv                   (10-fold CV metrics)
    - feature_importance.csv                         (XGBoost gain-based)
    - test_predictions.csv                           (per-sample predictions)

Environment:
    pip install -r requirements.txt
    Requires: scikit-learn, xgboost, tensorflow/keras, pandas, numpy, joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 10
ANN_EPOCHS = 100
ANN_BATCH_SIZE = 32
ANN_LR = 0.001

# UCI Concrete dataset URL (Yeh, 1998)
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "concrete/compressive/Concrete_Data.xls"
)

FEATURE_NAMES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]
TARGET_NAME = "Actual Strength"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Helpers ──────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    """Return dict of MAE, RMSE, R2."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def save_path(filename):
    """Absolute path in project root."""
    return os.path.join(OUTPUT_DIR, filename)


# ─── 1. Load & Prepare Data ──────────────────────────────────────────────────
def load_data():
    """Load UCI Concrete dataset and return features (X) and target (y)."""
    print("[1/6] Loading UCI Concrete dataset ...")

    # Try local cache first, then download
    local_file = save_path("Concrete_Data.xls")
    if os.path.exists(local_file):
        df = pd.read_excel(local_file)
    else:
        try:
            # First try standard download
            df = pd.read_excel(DATA_URL)
            df.to_excel(local_file, index=False)
            print(f"      Cached dataset to {local_file}")
        except Exception as e:
            try:
                # SSL fallback for legacy UCI server cert issues
                import ssl
                import urllib.request

                context = ssl._create_unverified_context()
                with urllib.request.urlopen(DATA_URL, context=context) as response:
                    file_content = response.read()
                with open(local_file, "wb") as f:
                    f.write(file_content)
                df = pd.read_excel(local_file)
                print(f"      Downloaded & cached dataset (SSL unverified fallback) to {local_file}")
            except Exception as e2:
                raise FileNotFoundError(
                    f"Could not download UCI Concrete dataset: {e2}. "
                    "Please place 'Concrete_Data.xls' in the project root."
                )

    # Standardize column names
    df.columns = FEATURE_NAMES + [TARGET_NAME]
    print(f"      Dataset shape: {df.shape}")
    print(f"      Features: {FEATURE_NAMES}")

    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values
    return X, y, df


# ─── 2. Train-Test Split ─────────────────────────────────────────────────────
def split_data(X, y):
    """80/20 train-test split with fixed random state."""
    print("[2/6] Splitting data (80% train / 20% test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"      Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ─── 3. Build & Train Models ─────────────────────────────────────────────────
def build_ann(input_dim):
    """Build the 6-layer Deep ANN (MLP) using Keras."""
    import keras
    from keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ANN_LR),
        loss="mse",
    )
    return model


def train_all_models(X_train, y_train):
    """
    Train all 6 standalone models + ANN.
    Returns dict of {name: model} and the fitted ANN scaler.

    Hyperparameter selection protocol:
    All hyperparameters are fixed a priori (see Section 3.1.2 of the
    manuscript). No automated search (grid/random/Bayesian) is performed.
    """
    print("[3/6] Training models ...")

    models = {}

    # --- Linear Regression (baseline) ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr
    joblib.dump(lr, save_path("linear_regression.joblib"))
    print("      [OK] Linear Regression")

    # --- SVR (RBF kernel) ---
    # SVR requires scaled inputs for convergence
    svr_scaler = StandardScaler()
    X_train_scaled = svr_scaler.fit_transform(X_train)
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.10, gamma="scale")
    svr.fit(X_train_scaled, y_train)
    models["SVR"] = (svr, svr_scaler)  # store scaler alongside
    joblib.dump(svr, save_path("svr.joblib"))
    print("      [OK] SVR")

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=100, max_features="sqrt",
        min_samples_split=2, random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    joblib.dump(rf, save_path("random_forest.joblib"))
    print("      [OK] Random Forest")

    # --- Gradient Boosting ---
    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.10,
        max_depth=3, loss="squared_error", random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = gb
    joblib.dump(gb, save_path("gradient_boosting.joblib"))
    print("      [OK] Gradient Boosting")

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.10,
        max_depth=6, subsample=0.80, colsample_bytree=0.80,
        random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model
    joblib.dump(xgb_model, save_path("xgboost.joblib"))
    print("      [OK] XGBoost")

    # --- Deep ANN ---
    ann_scaler = StandardScaler()
    X_train_ann = ann_scaler.fit_transform(X_train)

    ann = build_ann(input_dim=X_train.shape[1])
    ann.fit(
        X_train_ann, y_train,
        epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE,
        verbose=0,
    )
    models["Artificial Neural Network"] = (ann, ann_scaler)
    ann.save(save_path("ann_model.keras"))
    joblib.dump(ann_scaler, save_path("ann_scaler.joblib"))
    # Save weights separately for portability
    joblib.dump(
        {f"layer_{i}": w for i, w in enumerate(ann.get_weights())},
        save_path("ann_weights.joblib"),
    )
    print("      [OK] Artificial Neural Network (6-layer MLP)")

    return models, ann_scaler


# ─── 4. Evaluate on Test Set ─────────────────────────────────────────────────
def predict_with_model(name, model_or_tuple, X, ann_scaler=None):
    """Generate predictions, handling SVR and ANN scaling."""
    if name == "SVR":
        svr_model, svr_scaler = model_or_tuple
        return svr_model.predict(svr_scaler.transform(X))
    elif name == "Artificial Neural Network":
        ann_model, scaler = model_or_tuple
        return ann_model.predict(scaler.transform(X), verbose=0).flatten()
    else:
        return model_or_tuple.predict(X)


def evaluate_test_set(models, X_test, y_test, ann_scaler):
    """Evaluate all models + hybrid on the held-out test set."""
    print("[4/6] Evaluating on held-out test set ...")

    results = []
    predictions = {}

    for name, model in models.items():
        y_pred = predict_with_model(name, model, X_test, ann_scaler)
        metrics = evaluate(y_test, y_pred)
        results.append({"Model": name, **metrics})
        predictions[name] = y_pred
        print(f"      {name:35s}  R²={metrics['R2']:.4f}  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}")

    # --- Hybrid XGBoost + ANN (equal-weight blend) ---
    y_xgb = predictions["XGBoost"]
    y_ann = predictions["Artificial Neural Network"]
    y_hybrid = 0.5 * y_xgb + 0.5 * y_ann

    metrics_hybrid = evaluate(y_test, y_hybrid)
    results.append({"Model": "Hybrid XGBoost + ANN", **metrics_hybrid})
    predictions["Hybrid XGBoost + ANN"] = y_hybrid
    print(f"      {'Hybrid XGBoost + ANN':35s}  R²={metrics_hybrid['R2']:.4f}  MAE={metrics_hybrid['MAE']:.2f}  RMSE={metrics_hybrid['RMSE']:.2f}")

    # Sort by R2 descending
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    results_df.to_csv(save_path("model_comparison.csv"), index=False)
    print(f"      -> Saved model_comparison.csv")

    return results_df, predictions


# --- 5. 10-Fold Cross-Validation & Nested CV Audit --------------------------
def run_nested_cv_audit(X_train, y_train):
    """
    5x10 Nested Cross-Validation audit to empirically confirm that model selection
    is strictly independent of performance estimation.
    """
    print("      Running 5x10 Nested CV audit for tuning validation ...")
    from sklearn.model_selection import GridSearchCV

    outer_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    param_grids = {
        "XGBoost": (
            xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.05, 0.10, 0.20],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [50, 100, 200],
                "max_features": ["sqrt", "log2"],
                "min_samples_split": [2, 5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.10, 0.20],
            },
        ),
        "SVR": (
            SVR(kernel="rbf"),
            {
                "C": [1.0, 10.0, 100.0],
                "epsilon": [0.01, 0.10, 0.20],
            },
        ),
    }

    nested_results = []
    for name, (base_estimator, grid) in param_grids.items():
        outer_scores = []
        for train_idx, val_idx in outer_kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            clf = GridSearchCV(base_estimator, grid, cv=inner_cv, scoring="r2", n_jobs=-1)

            if name == "SVR":
                clf.fit(X_tr_s, y_tr)
                y_pred = clf.predict(X_val_s)
            else:
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_val)

            outer_scores.append(r2_score(y_val, y_pred))

        nested_r2_mean = np.mean(outer_scores)
        nested_r2_std = np.std(outer_scores)
        print(f"      [Nested CV] {name:20s} Nested R² = {nested_r2_mean:.4f} +/- {nested_r2_std:.4f}")
        nested_results.append({
            "Model": name,
            "Nested CV R2 Mean": nested_r2_mean,
            "Nested CV R2 Std": nested_r2_std,
        })

def run_repeated_seed_sensitivity_audit(X, y, seeds=[42, 100, 2024, 7, 13, 99, 123, 456, 789, 2026]):
    """
    10-Seed Repeated 80/20 Hold-Out Sensitivity Audit.
    Evaluates whether performance metrics and model rankings are sensitive to random partitioning seeds.
    """
    print("      Running 10-seed repeated 80/20 hold-out sensitivity audit ...")

    models_to_test = {
        "XGBoost": lambda s: xgb.XGBRegressor(n_estimators=100, learning_rate=0.10, max_depth=6, subsample=0.80, colsample_bytree=0.80, random_state=s, verbosity=0),
        "Random Forest": lambda s: RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_split=2, random_state=s),
        "Gradient Boosting": lambda s: GradientBoostingRegressor(n_estimators=100, learning_rate=0.10, max_depth=3, loss="squared_error", random_state=s),
        "SVR": lambda s: SVR(kernel="rbf", C=10.0, epsilon=0.10, gamma="scale"),
        "Linear Regression": lambda s: LinearRegression(),
    }

    seed_results = {name: [] for name in models_to_test}

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, model_fn in models_to_test.items():
            model = model_fn(seed)
            if name == "SVR":
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)
            else:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

            score = r2_score(y_te, y_pred)
            seed_results[name].append(score)

    print("      --- Multi-Seed Partitioning Sensitivity (10 Seeds) ---")
    for name, scores in seed_results.items():
        print(f"      [Multi-Seed] {name:20s} Mean R² = {np.mean(scores):.4f} +/- {np.std(scores):.4f} (Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f})")

    return seed_results


def cross_validate_models(models, X_train, y_train, X=None, y=None):
    """
    10-fold CV on the training partition only under prespecified selection protocol.
    Scaling is re-fitted per fold to prevent leakage.
    Also executes nested CV audit and multi-seed sensitivity audit.
    """
    print("[5/6] Running 10-fold cross-validation, Nested CV Audit & Multi-Seed Audit ...")

    # Run nested CV audit first
    nested_df = run_nested_cv_audit(X_train, y_train)

    # Run multi-seed sensitivity audit if full dataset is provided
    if X is not None and y is not None:
        run_repeated_seed_sensitivity_audit(X, y)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {
        "XGBoost":                   {"r2": [], "mae": [], "rmse": []},
        "Hybrid XGBoost + ANN":      {"r2": [], "mae": [], "rmse": []},
        "Random Forest":             {"r2": [], "mae": [], "rmse": []},
        "Gradient Boosting":         {"r2": [], "mae": [], "rmse": []},
        "Artificial Neural Network": {"r2": [], "mae": [], "rmse": []},
        "SVR":                       {"r2": [], "mae": [], "rmse": []},
        "Linear Regression":         {"r2": [], "mae": [], "rmse": []},
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        fold_scaler = StandardScaler()
        X_tr_s = fold_scaler.fit_transform(X_tr)
        X_val_s = fold_scaler.transform(X_val)

        # 1. XGBoost
        m_xgb = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.10, max_depth=6,
            subsample=0.80, colsample_bytree=0.80,
            random_state=RANDOM_STATE, verbosity=0,
        )
        m_xgb.fit(X_tr, y_tr)
        pred_xgb = m_xgb.predict(X_val)

        # 2. Random Forest
        m_rf = RandomForestRegressor(
            n_estimators=100, max_features="sqrt",
            min_samples_split=2, random_state=RANDOM_STATE,
        )
        m_rf.fit(X_tr, y_tr)
        pred_rf = m_rf.predict(X_val)

        # 3. Gradient Boosting
        m_gb = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.10,
            max_depth=3, loss="squared_error", random_state=RANDOM_STATE,
        )
        m_gb.fit(X_tr, y_tr)
        pred_gb = m_gb.predict(X_val)

        # 4. SVR
        m_svr = SVR(kernel="rbf", C=10.0, epsilon=0.10, gamma="scale")
        m_svr.fit(X_tr_s, y_tr)
        pred_svr = m_svr.predict(X_val_s)

        # 5. Linear Regression
        m_lr = LinearRegression()
        m_lr.fit(X_tr, y_tr)
        pred_lr = m_lr.predict(X_val)

        # 6. Deep ANN
        m_ann = build_ann(input_dim=X_tr.shape[1])
        m_ann.fit(X_tr_s, y_tr, epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE, verbose=0)
        pred_ann = m_ann.predict(X_val_s, verbose=0).flatten()

        # 7. Hybrid XGBoost + ANN
        pred_hybrid = 0.5 * pred_xgb + 0.5 * pred_ann

        fold_preds = {
            "XGBoost":                   pred_xgb,
            "Hybrid XGBoost + ANN":      pred_hybrid,
            "Random Forest":             pred_rf,
            "Gradient Boosting":         pred_gb,
            "Artificial Neural Network": pred_ann,
            "SVR":                       pred_svr,
            "Linear Regression":         pred_lr,
        }

        for mname, pval in fold_preds.items():
            fold_metrics[mname]["r2"].append(r2_score(y_val, pval))
            fold_metrics[mname]["mae"].append(mean_absolute_error(y_val, pval))
            fold_metrics[mname]["rmse"].append(rmse(y_val, pval))

    cv_results = []
    for mname, mdata in fold_metrics.items():
        r2_m = np.mean(mdata["r2"])
        r2_s = np.std(mdata["r2"])
        mae_m = np.mean(mdata["mae"])
        rmse_m = np.mean(mdata["rmse"])
        cv_results.append({
            "Model": mname,
            "CV MAE Mean": mae_m,
            "CV RMSE Mean": rmse_m,
            "CV R2 Mean": r2_m,
            "CV R2 Std": r2_s,
        })
        print(f"      {mname:28s}  CV R2={r2_m:.4f} +/- {r2_s:.4f}")

    cv_df = pd.DataFrame(cv_results).sort_values("CV R2 Mean", ascending=False)
    cv_df.to_csv(save_path("10_fold_cross_validation.csv"), index=False)
    print(f"      -> Saved 10_fold_cross_validation.csv")

    return cv_df


# --- 6. Feature Importance & Predictions --------------------------------------
def save_feature_importance(models):
    """Extract and save XGBoost gain-based feature importance."""
    print("[6/6] Saving feature importance & test predictions ...")

    xgb_model = models["XGBoost"]
    importances = xgb_model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    fi_df.to_csv(save_path("feature_importance.csv"), index=False)
    print(f"      -> Saved feature_importance.csv")
    for _, row in fi_df.iterrows():
        print(f"         {row['Feature']:25s} {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

    return fi_df


def save_test_predictions(X_test, y_test, predictions):
    """Save per-sample predictions with residuals for the hybrid model."""
    y_hybrid = predictions["Hybrid XGBoost + ANN"]

    pred_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    pred_df["Actual Strength"] = y_test
    pred_df["Hybrid Predicted Strength"] = y_hybrid
    pred_df["Absolute Error"] = np.abs(y_test - y_hybrid)

    pred_df.to_csv(save_path("test_predictions.csv"), index=False)
    print(f"      -> Saved test_predictions.csv ({len(pred_df)} samples)")

    within_5 = (pred_df["Absolute Error"] <= 5.0).mean() * 100
    print(f"      -> {within_5:.1f}% of hybrid predictions within +/-5 MPa")


# --- Main ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Concrete Compressive Strength - Full Training Pipeline")
    print("=" * 70)

    print()

    # Step 1: Load data
    X, y, df = load_data()

    # Step 2: Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Train all models
    models, ann_scaler = train_all_models(X_train, y_train)

    # Step 4: Evaluate on test set
    results_df, predictions = evaluate_test_set(models, X_test, y_test, ann_scaler)

    # Step 5: 10-fold CV & Multi-Seed Audit
    cv_df = cross_validate_models(models, X_train, y_train, X, y)

    # Step 6: Feature importance & predictions
    fi_df = save_feature_importance(models)
    save_test_predictions(X_test, y_test, predictions)

    print()
    print("=" * 70)
    print("  Pipeline complete. All artifacts saved to project root.")
    print("=" * 70)
    print()
    print("Generated files:")
    for f in [
        "model_comparison.csv", "10_fold_cross_validation.csv",
        "feature_importance.csv", "test_predictions.csv",
        "xgboost.joblib", "random_forest.joblib", "gradient_boosting.joblib",
        "svr.joblib", "linear_regression.joblib",
        "ann_model.keras", "ann_scaler.joblib", "ann_weights.joblib",
    ]:
        path = save_path(f)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"  {status} {f}")


if __name__ == "__main__":
    main()
