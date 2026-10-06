# 🏗️ Explainable Machine Learning and Deep Neural Networks for Concrete Compressive Strength Prediction: A Benchmark Study with SHAP Feature Attribution

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-111111?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-7B2FBE?style=for-the-badge)](https://shap.readthedocs.io/)

An interactive machine learning and deep learning research platform for predicting **Concrete Compressive Strength (MPa)** based on 8 concrete mix parameters and curing age. The platform integrates game-theoretic **SHAP feature attribution analysis** for transparent, explainable model interpretation.

---

## 📌 Research Overview

Predicting the compressive strength of concrete is crucial in civil and structural engineering to ensure structural performance, material optimization, and safety. This research platform evaluates six standalone Machine Learning (ML) regressors and a 6-layer Deep Artificial Neural Network (ANN), alongside an operationally defined **Hybrid Equal-Weight Blending Ensemble** ($\hat{y}_{hybrid} = 0.5 \hat{y}_{XGBoost} + 0.5 \hat{y}_{ANN}$) across 1,030 empirical concrete formulations from the UCI Repository (Yeh, 1998).

The evaluation pipeline enforces strict data leakage prevention and model selection independence: feature standard scaling ($z$-score) parameters are fit strictly on the 80% training set, and all hyperparameter configurations are fixed *a priori* via an explicit prespecified selection protocol (validated via a 5×10 nested cross-validation audit). Models are evaluated across held-out test data (20%) and 10-fold cross-validation ($10\text{-CV}$). SHAP-based feature attribution analysis, dynamic sensitivity simulations (1–180 days curing age, $w/c$ ratio), and an interactive Streamlit web application prototype are integrated for decision support.

### 📊 Model Performance Comparison

| Model | MAE (MPa) | RMSE (MPa) | R² Score |
| :--- | :---: | :---: | :---: |
| **XGBoost** 🏆 | **2.61** | **4.20** | **0.941** |
| **Hybrid XGBoost + ANN** | 3.09 | 4.79 | 0.923 |
| **Gradient Boosting** | 3.65 | 5.02 | 0.915 |
| **Random Forest** | 3.51 | 5.19 | 0.910 |
| **Support Vector Regressor (SVR)** | 4.02 | 5.97 | 0.880 |
| **Artificial Neural Network (ANN)** | 4.30 | 6.10 | 0.875 |
| **Linear Regression** | 8.90 | 11.19 | 0.580 |

> [!NOTE]
> Reported metrics correspond to the `random_state=42` fixed split. A 10-seed sensitivity audit produced XGBoost mean $R^2 = 0.938 \pm 0.006$ (range 0.928–0.947), confirming ranking stability across partitions.

---

> [!WARNING]
> **Field Deployment & Recalibration Notice**:
> The models function as interpolation tools bounded strictly by the empirical training dataset limits (Cement 102–540 kg/m³, Water 127–247 kg/m³, Age 1–365 days, lab moist curing ~20°C). Field engineers must recalibrate model parameters with local batch plant trial mixes before applying predictions in commercial structural compliance.

---

## 🔍 SHAP Feature Attribution Analysis

This platform includes a dedicated **SHAP (SHapley Additive exPlanations)** analysis module (`shap_analysis.py`) that provides game-theoretic, model-agnostic feature attribution for all ML models.

### Running the SHAP Analysis

```bash
python shap_analysis.py
```

### SHAP Outputs

| Artefact | Description |
| :--- | :--- |
| `shap_summary.csv` | Mean \|SHAP\| per feature across all 4 models |
| `shap_summary_xgboost.png` | Beeswarm dot plot — SHAP impact coloured by feature value |
| `shap_bar_xgboost.png` | XGBoost mean \|SHAP\| horizontal bar chart |
| `shap_bar_random_forest.png` | Random Forest mean \|SHAP\| bar chart |
| `shap_bar_gradient_boosting.png` | Gradient Boosting mean \|SHAP\| bar chart |
| `shap_bar_ann.png` | Deep ANN Kernel SHAP bar chart |
| `shap_comparison_all_models.png` | Cross-model grouped SHAP comparison chart |
| `shap_waterfall_sample.png` | Waterfall explanation for a single test specimen |
| `shap_dependence_age.png` | SHAP dependence plot: Curing Age → strength contribution |
| `shap_dependence_cement.png` | SHAP dependence plot: Cement Content → strength contribution |

### Explainer Strategy

| Model Family | SHAP Explainer Used |
| :--- | :--- |
| XGBoost, Random Forest, Gradient Boosting | `shap.TreeExplainer` / `shap.Explainer` — closed-form tree SHAP |
| Deep ANN | `shap.Explainer(ann_predict_fn, background)` — model-agnostic Permutation/ExactExplainer |

> [!NOTE]
> **ANN SHAP Caveat**: Model-agnostic explainers applied to neural networks with correlated inputs under marginal background distributions produce inflated attribution magnitudes not directly comparable to tree-model SHAP values. ANN SHAP values should be read as global input sensitivity indicators, not ranked feature importance.

> [!NOTE]
> SHAP analysis uses 100 background samples (drawn via `shap.sample`) and evaluates all 1,030 specimens across all models. **Top XGBoost SHAP rankings**: Curing Age (8.29 MPa), Cement Content (5.88 MPa), Water Content (4.62 MPa) — concordant across all three tree-based models and with gain-based importance.

---

## ⚙️ Features of the Web Application

- 🧪 **Interactive Mix Strength Predictor**: Adjust ingredient proportions and curing age with real-time strength prediction, concrete category classification, and recommended applications.
- ⚡ **Mix Formulation Presets**: One-click presets (*Standard 28-Day*, *High-Strength*, *Eco Fly-Ash*, *Early 7-Day*).
- 🧮 **Derived Engineering Metrics**: Water-to-Binder ratio ($w/b$), Total Binder content ($kg/m^3$), and estimated concrete density.
- 📊 **Multi-Model Comparison**: Compare predictions across all 7 models simultaneously.
- 📁 **Batch CSV Predictor & Exporter**: Upload batch concrete mix datasets and export predictions.
- 🔍 **Explainability & Sensitivity Simulator**: Gain-based feature importance, SHAP attribution, 1–180 day age curing growth curve simulator, and water-to-cement ratio sensitivity curves.
- 📁 **Research Dataset Explorer**: Search, filter, and download test predictions and cross-validation metrics.

---

## 🧠 Deep Learning & Ensemble Architecture

### Artificial Neural Network (ANN)
- **Input Layer**: 8 Mix Features (`Cement`, `Blast Furnace Slag`, `Fly Ash`, `Water`, `Superplasticizer`, `Coarse Aggregate`, `Fine Aggregate`, `Age`)
- **Hidden Layers**: Dense (128, ReLU) → Dense (64, ReLU) → Dropout (0.2) → Dense (32, ReLU) → Dense (16, ReLU)
- **Output Layer**: Dense (1, Linear)
- **Optimizer**: Adam (lr=0.001), Loss: MSE, Scaler: StandardScaler

### Hybrid XGBoost + ANN Equal-Weight Blending Ensemble
$$\hat{y}_{hybrid} = 0.5 \cdot \hat{y}_{XGBoost} + 0.5 \cdot \hat{y}_{ANN}$$

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/rajaramayan/explainable-concrete-ML-ANN-benchmark.git
cd explainable-concrete-ML-ANN-benchmark
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Reproduce All Models & Artifacts (Optional)
```bash
python train.py
```

### 5. Run SHAP Feature Attribution Analysis (Optional)
```bash
python shap_analysis.py
```

> Outputs SHAP plots and `shap_summary.csv` to the project root. Runtime: ~3–10 minutes depending on hardware.

### 6. Run Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
├── app.py                          # Main Streamlit web application
├── train.py                        # Full training pipeline (reproduces all artifacts)
├── shap_analysis.py                # SHAP feature attribution analysis script
├── requirements.txt                # Python dependencies (incl. shap>=0.42.0)
├── README.md                       # Project documentation
├── Thesis.md                       # Full empirical benchmark manuscript
├── generate_all_plots.py           # Publication-quality static figure generator
├── Concrete_Data.xls               # UCI Concrete dataset (Yeh, 1998)
│
├── ── Trained Model Binaries ──
├── xgboost.joblib                  # Saved XGBoost model
├── ann_model.keras                 # Keras ANN model
├── ann_scaler.joblib               # StandardScaler for ANN inputs
├── ann_weights.joblib              # ANN layer weights (portability)
├── random_forest.joblib            # Saved Random Forest model
├── gradient_boosting.joblib        # Saved Gradient Boosting model
├── svr.joblib                      # Saved SVR model
├── linear_regression.joblib        # Saved Linear Regression model
│
├── ── Evaluation Reports ──
├── model_comparison.csv            # Out-of-sample test-set metrics (all models)
├── 10_fold_cross_validation.csv    # 10-fold CV mean R², MAE, RMSE ± std
├── feature_importance.csv          # XGBoost gain-based feature importance
├── test_predictions.csv            # Per-specimen predictions & residuals
│
└── ── SHAP Artefacts (generated by shap_analysis.py) ──
    ├── shap_summary.csv            # Mean |SHAP| table across all models
    ├── shap_summary_xgboost.png    # Beeswarm summary plot
    ├── shap_bar_xgboost.png        # XGBoost SHAP bar chart
    ├── shap_bar_random_forest.png  # Random Forest SHAP bar chart
    ├── shap_bar_gradient_boosting.png
    ├── shap_bar_ann.png            # ANN Kernel SHAP bar chart
    ├── shap_comparison_all_models.png  # Cross-model grouped comparison
    ├── shap_waterfall_sample.png   # Single specimen waterfall
    ├── shap_dependence_age.png     # Age SHAP dependence plot
    └── shap_dependence_cement.png  # Cement SHAP dependence plot
```

---

## 💻 Computational Infrastructure & Environment

| Component | Package / Specification | Exact Version |
| :--- | :--- | :--- |
| **Language Runtime** | CPython (x86_64) | `v3.10.2` |
| **Deep Learning** | TensorFlow / Keras | `v2.20.0` |
| **Machine Learning** | Scikit-Learn | `v1.6.1` |
| **Gradient Boosting** | XGBoost | `v3.2.0` |
| **Explainability** | SHAP | `v0.49.1` |
| **Data Processing** | Pandas / NumPy | `pandas 2.2.3` / `numpy 2.2.6` |
| **Visualization** | Matplotlib / Plotly | `matplotlib 3.7+` / `plotly 5.18.0` |
| **Model Persistence** | Joblib / OpenPyXL | `joblib 1.5.0` / `openpyxl 3.0.0` |
| **Web UI Framework** | Streamlit | `streamlit 1.45.1` |
| **Release Identifier** | GitHub Release Tag / Zenodo DOI | `v1.2.0-reproducible` / `10.5281/zenodo.14892021` |

---

## 📜 License
This project is open-source under the MIT License.
