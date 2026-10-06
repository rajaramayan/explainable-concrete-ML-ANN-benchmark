"""
Explainable Machine Learning and Deep Neural Networks for Concrete Compressive Strength Prediction
===================================================================================================
SHAP Feature Attribution Analysis Pipeline
Computes game-theoretic SHAP (SHapley Additive exPlanations) values for the
XGBoost, Random Forest, and Gradient Boosting models, plus a kernel-based
approximation for the Deep ANN.

Explainer strategy:
  - Tree models  → shap.Explainer(model.predict, background)
    Uses PermutationExplainer or an exact sampler with a small background set.
  - Deep ANN     → shap.KernelExplainer(model.predict, background)
    Uses Kernel SHAP (model-agnostic, slower but universally applicable).

Usage:
    python shap_analysis.py

Outputs (saved to project root):
    shap_summary.csv               Mean |SHAP| per feature, across all models
    shap_summary_xgboost.png       SHAP beeswarm dot plot  — XGBoost
    shap_bar_xgboost.png           SHAP mean |SHAP| bar chart — XGBoost
    shap_bar_rf.png                SHAP mean |SHAP| bar chart — Random Forest
    shap_bar_gb.png                SHAP mean |SHAP| bar chart — Gradient Boosting
    shap_bar_ann.png               SHAP mean |SHAP| bar chart — Deep ANN (Kernel)
    shap_waterfall_sample.png      Waterfall for specimen index 0 — XGBoost
    shap_dependence_age.png        SHAP dependence plot: Age vs. prediction — XGBoost

Environment:
    pip install -r requirements.txt
    Requires: shap>=0.42.0, xgboost, scikit-learn, tensorflow, matplotlib, joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
BACKGROUND_N   = 100      # Background samples for Kernel / Permutation explainer
# ANN SHAP now evaluates the full 1,030-specimen dataset.
# Runtime is ~35–40 min for ExactExplainer over 1030 samples.
# To limit it, set ANN_SAMPLE_N to a smaller integer (e.g. 200).
ANN_SAMPLE_N   = None     # None → use entire dataset

FEATURE_NAMES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]
TARGET_NAME = "Actual Strength"

PALETTE = {
    "XGBoost":           "#1a73e8",
    "Random Forest":     "#34a853",
    "Gradient Boosting": "#ff6d00",
    "Deep ANN":          "#9c27b0",
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_data():
    """Load UCI Concrete dataset, return fixed 80/20 split AND the full dataset."""
    local = save_path("Concrete_Data.xls")
    if os.path.exists(local):
        df = pd.read_excel(local)
    else:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "concrete/compressive/Concrete_Data.xls"
        )
        df = pd.read_excel(url)
        df.to_excel(local, index=False)

    df.columns = FEATURE_NAMES + [TARGET_NAME]
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Return the full dataset as well (all 1,030 specimens)
    return X_train, X_test, y_train, y_test, X, y


# ─── Model Loading ────────────────────────────────────────────────────────────
def load_ml_models(X_train, y_train):
    """
    Load pre-trained ML model binaries.
    If a binary is missing, re-train with the canonical hyperparameter configuration.
    """
    models = {}

    # XGBoost
    p = save_path("xgboost.joblib")
    if os.path.exists(p):
        m = joblib.load(p)
        print(f"      Loaded XGBoost from {p}")
    else:
        m = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.10, max_depth=6,
            subsample=0.80, colsample_bytree=0.80,
            random_state=RANDOM_STATE, verbosity=0,
        )
        m.fit(X_train.values, y_train)
        print("      Trained XGBoost (no binary found)")
    models["XGBoost"] = m

    # Random Forest
    p = save_path("random_forest.joblib")
    if os.path.exists(p):
        m = joblib.load(p)
        print(f"      Loaded Random Forest from {p}")
    else:
        m = RandomForestRegressor(
            n_estimators=100, max_features="sqrt",
            min_samples_split=2, random_state=RANDOM_STATE,
        )
        m.fit(X_train.values, y_train)
        print("      Trained Random Forest (no binary found)")
    models["Random Forest"] = m

    # Gradient Boosting
    p = save_path("gradient_boosting.joblib")
    if os.path.exists(p):
        m = joblib.load(p)
        print(f"      Loaded Gradient Boosting from {p}")
    else:
        m = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.10, max_depth=3,
            loss="squared_error", random_state=RANDOM_STATE,
        )
        m.fit(X_train.values, y_train)
        print("      Trained Gradient Boosting (no binary found)")
    models["Gradient Boosting"] = m

    return models


def load_ann(X_train, y_train):
    """Load the Deep ANN model + its scaler, re-training if necessary."""
    scaler_path = save_path("ann_scaler.joblib")
    model_path  = save_path("ann_model.keras")

    scaler = StandardScaler()

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        scaler = joblib.load(scaler_path)
        from keras.models import load_model
        ann = load_model(model_path)
        print(f"      Loaded Deep ANN from {model_path}")
    else:
        import keras
        from keras import layers

        scaler.fit(X_train.values)
        X_s = scaler.transform(X_train.values)

        ann = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1,  activation="linear"),
        ])
        ann.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        ann.fit(X_s, y_train, epochs=100, batch_size=32, verbose=0)
        ann.save(model_path)
        joblib.dump(scaler, scaler_path)
        print("      Trained Deep ANN (no binary found)")

    return ann, scaler


# ─── SHAP Utilities ───────────────────────────────────────────────────────────
def build_background(X_ref: pd.DataFrame, n: int) -> np.ndarray:
    """
    Sub-sample a balanced background dataset for Kernel / Permutation explainers.
    shap.sample() or random choice are both valid; we use shap.sample for consistency.
    """
    bg = shap.sample(X_ref.values, nsamples=n, random_state=RANDOM_STATE)
    return bg


def compute_shap_values(predict_fn, background: np.ndarray, X_eval, label: str):
    """
    Dispatch to an appropriate explainer.
    X_eval can be a DataFrame (preferred — preserves feature names) or ndarray.
    Returns a (vals_array, shap_explanation) tuple.
    """
    print(f"      Computing SHAP values for {label} ...")
    explainer = shap.Explainer(predict_fn, background)
    sv = explainer(X_eval)
    vals = sv.values if hasattr(sv, "values") else np.array(sv)
    if vals.ndim == 3:
        vals = vals[:, :, 0]
    return vals, sv


# ─── Plotting Helpers ─────────────────────────────────────────────────────────
def plot_bar_chart(mean_abs_shap: np.ndarray, model_name: str, color: str, out_path: str):
    """Horizontal bar chart of mean |SHAP| values, sorted descending."""
    order = np.argsort(mean_abs_shap)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [FEATURE_NAMES[i] for i in order],
        mean_abs_shap[order],
        color=color, edgecolor="white", linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (MPa)", fontsize=11)
    ax.set_title(f"{model_name} — Mean |SHAP| Feature Attribution", fontsize=13, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved -> {out_path}")


def plot_beeswarm(sv_obj, X_df: pd.DataFrame, model_name: str, out_path: str):
    """Beeswarm / dot summary plot coloured by feature value."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv_obj, X_df, show=False)
    plt.title(f"{model_name} — SHAP Feature Contribution Summary (Test Set)", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved -> {out_path}")


def plot_waterfall(sv_obj, sample_idx: int, model_name: str, out_path: str):
    """Waterfall explanation for a single test specimen."""
    try:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sv_obj[sample_idx], show=False)
        plt.title(f"{model_name} — SHAP Waterfall: Test Specimen #{sample_idx}", fontsize=12, weight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"      Saved -> {out_path}")
    except Exception as e:
        print(f"      Waterfall plot skipped ({e})")


def plot_dependence(sv_vals: np.ndarray, X_df: pd.DataFrame, feature: str, out_path: str):
    """SHAP dependence scatter plot for a single feature."""
    idx = FEATURE_NAMES.index(feature)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        X_df[feature], sv_vals[:, idx],
        c=sv_vals[:, idx], cmap="coolwarm",
        s=25, alpha=0.75, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="SHAP value (MPa)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"{feature} (kg/m³ or days)", fontsize=11)
    ax.set_ylabel("SHAP Value (Impact on predicted strength, MPa)", fontsize=11)
    ax.set_title(f"XGBoost — SHAP Dependence Plot: {feature}", fontsize=13, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved -> {out_path}")


def plot_multi_model_comparison(summary_dict: dict, out_path: str):
    """
    Grouped horizontal bar chart comparing mean |SHAP| values
    across all four models for each feature.
    """
    models = list(summary_dict.keys())
    n_models  = len(models)
    n_features = len(FEATURE_NAMES)

    # Sort features by XGBoost mean |SHAP| descending
    xgb_vals = summary_dict["XGBoost"]
    order = np.argsort(xgb_vals)[::-1]
    sorted_features = [FEATURE_NAMES[i] for i in order]

    x = np.arange(n_features)
    bar_h = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_h

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (mname, vals) in enumerate(summary_dict.items()):
        sorted_vals = np.array([vals[j] for j in order])
        ax.barh(
            x + offsets[i], sorted_vals,
            height=bar_h, label=mname,
            color=PALETTE[mname], edgecolor="white", linewidth=0.5,
        )

    ax.set_yticks(x)
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (MPa)", fontsize=11)
    ax.set_title(
        "Cross-Model SHAP Feature Attribution Comparison\n(XGBoost · Random Forest · Gradient Boosting · Deep ANN)",
        fontsize=13, weight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved cross-model comparison -> {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Concrete Compressive Strength — SHAP Feature Attribution Analysis")
    print("  SHAP version:", shap.__version__)
    print("=" * 70)

    # 1. Data
    print("\n[1/5] Loading dataset and preparing partitions ...")
    X_train, X_test, y_train, y_test, X_full, y_full = load_data()
    X_train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    X_test_df  = pd.DataFrame(X_test,  columns=FEATURE_NAMES)
    X_full_df  = pd.DataFrame(X_full,  columns=FEATURE_NAMES)  # all 1,030 specimens
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,} | Full: {len(X_full):,} specimens")
    print(f"      SHAP will be evaluated across all {len(X_full):,} specimens.")

    # Background dataset for model-agnostic explainers (NumPy for the explainer baseline)
    bg = build_background(X_train_df, BACKGROUND_N)
    print(f"      Background (for SHAP): {BACKGROUND_N} training samples (shap.sample)")

    # 2. ML Models
    print("\n[2/5] Loading / training ML models ...")
    ml_models = load_ml_models(X_train_df, y_train)

    # 3. Deep ANN
    print("\n[3/5] Loading Deep ANN ...")
    ann, ann_scaler = load_ann(X_train_df, y_train)
    ann_bg = ann_scaler.transform(bg)

    def ann_predict(X_arr):
        Xs = ann_scaler.transform(X_arr)
        return ann.predict(Xs, verbose=0).flatten()

    # 4. SHAP Computation
    print("\n[4/5] Computing SHAP values for all models ...")

    # Tree-based models: evaluate SHAP on the full 1,030-specimen dataset
    # so that global importance plots reflect the complete data distribution.
    X_shap_labeled = X_full_df          # DataFrame — preserves column names in plots
    X_shap_arr     = X_full_df.values   # NumPy — for ANN preprocessing

    # ANN: use full dataset or a cap defined by ANN_SAMPLE_N
    ann_n = len(X_full_df) if ANN_SAMPLE_N is None else min(ANN_SAMPLE_N, len(X_full_df))
    X_ann_shap = X_shap_arr[:ann_n]
    print(f"      Tree models: {len(X_shap_labeled):,} specimens | ANN: {ann_n:,} specimens")

    summary_vals = {}

    for mname, model in ml_models.items():
        def _predict(X_arr, _m=model):
            return _m.predict(X_arr)

        # Evaluate over the full 1,030-specimen dataset with column names
        sv_arr, sv_obj = compute_shap_values(_predict, bg, X_shap_labeled, mname)
        mean_abs = np.mean(np.abs(sv_arr), axis=0)
        summary_vals[mname] = mean_abs

        # Bar chart per model
        color = PALETTE[mname]
        bar_fname = f"shap_bar_{mname.lower().replace(' ', '_')}.png"
        plot_bar_chart(mean_abs, mname, color, save_path(bar_fname))

        # Beeswarm and extra plots for XGBoost (best model)
        if mname == "XGBoost":
            xgb_sv_obj = sv_obj
            xgb_sv_arr = sv_arr
            plot_beeswarm(sv_obj, X_shap_labeled, mname, save_path("shap_summary_xgboost.png"))
            plot_waterfall(sv_obj, 0, mname, save_path("shap_waterfall_sample.png"))
            plot_dependence(sv_arr, X_full_df, "Age", save_path("shap_dependence_age.png"))
            plot_dependence(sv_arr, X_full_df, "Cement", save_path("shap_dependence_cement.png"))

    # ANN SHAP — full dataset (pass NumPy; scaler transforms before ANN inference)
    ann_sv_arr, _ = compute_shap_values(ann_predict, ann_bg, X_ann_shap, "Deep ANN")
    ann_mean_abs = np.mean(np.abs(ann_sv_arr), axis=0)
    summary_vals["Deep ANN"] = ann_mean_abs
    plot_bar_chart(ann_mean_abs, "Deep ANN", PALETTE["Deep ANN"], save_path("shap_bar_ann.png"))

    # Cross-model grouped comparison
    plot_multi_model_comparison(summary_vals, save_path("shap_comparison_all_models.png"))

    # 5. Export Summary CSV
    print("\n[5/5] Exporting SHAP summary table ...")
    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        **{f"{m} Mean |SHAP| (MPa)": vals for m, vals in summary_vals.items()},
    }).sort_values("XGBoost Mean |SHAP| (MPa)", ascending=False)

    csv_path = save_path("shap_summary.csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"      Saved shap_summary.csv -> {csv_path}")

    print("\n--- SHAP Feature Attribution Summary (Mean |SHAP|, MPa) ---")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(shap_df.to_string(index=False))

    # Output artefacts list
    artefacts = [
        "shap_summary.csv",
        "shap_summary_xgboost.png",
        "shap_bar_xgboost.png",
        "shap_bar_random_forest.png",
        "shap_bar_gradient_boosting.png",
        "shap_bar_ann.png",
        "shap_comparison_all_models.png",
        "shap_waterfall_sample.png",
        "shap_dependence_age.png",
        "shap_dependence_cement.png",
    ]
    print("\n" + "=" * 70)
    print("  SHAP Analysis Complete!")
    print("=" * 70)
    print("\nGenerated artefacts:")
    for f in artefacts:
        p = save_path(f)
        status = "[OK]" if os.path.exists(p) else "[MISSING]"
        print(f"  {status} {f}")


if __name__ == "__main__":
    main()
