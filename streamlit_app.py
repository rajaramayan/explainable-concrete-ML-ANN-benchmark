"""
Explainable Machine Learning and Deep Neural Networks for Concrete Compressive Strength Prediction
===================================================================================================
Streamlit Web Application
"""
import sys
import os
import warnings
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

# ─── Compatibility shim for old scikit-learn gradient boosting pickles ────────
for _mod in ("sklearn._loss.loss", "sklearn._loss"):
    try:
        import importlib
        _m = importlib.import_module(_mod)
        sys.modules.setdefault("_loss", _m)
        break
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must come before any other st call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Concrete Strength ML & ANN | Benchmark Study",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS & Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 26px 34px;
    margin-bottom: 22px;
    box-shadow: 0 10px 25px rgba(0,0,0,.3);
}
.hero h1 { color: #f8fafc; font-size: 2.1rem; font-weight: 700; margin: 0; }
.hero p { color: #94a3b8; font-size: 1rem; margin-top: 6px; margin-bottom: 0; }

.mcard {
    background: rgba(30,41,59,.7);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.mval { font-size: 1.8rem; font-weight: 700; color: #38bdf8; margin-top: 4px; }
.mlbl { font-size: .82rem; text-transform: uppercase; letter-spacing: .7px; color: #94a3b8; font-weight: 600; }

.badge-hi {
    background: rgba(16,185,129,.15); color: #10b981; padding: 4px 12px;
    border-radius: 20px; font-size: .84rem; font-weight: 600; border: 1px solid rgba(16,185,129,.3);
}
.badge-md {
    background: rgba(59,130,246,.15); color: #3b82f6; padding: 4px 12px;
    border-radius: 20px; font-size: .84rem; font-weight: 600; border: 1px solid rgba(59,130,246,.3);
}
.badge-lo {
    background: rgba(245,158,11,.15); color: #f59e0b; padding: 4px 12px;
    border-radius: 20px; font-size: .84rem; font-weight: 600; border: 1px solid rgba(245,158,11,.3);
}

.result-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 2px solid #3b82f6;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(59,130,246,.15);
}
.result-val { font-size: 2.9rem; font-weight: 800; color: #60a5fa; margin: 8px 0; }

/* Custom Sidebar Radio Styling */
div[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    transition: all 0.2s ease;
}
div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.4);
}
</style>""", unsafe_allow_html=True)


def _chart(fig, height=None):
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.plotly_chart(fig)


def _df(df_in, **kw):
    try:
        st.dataframe(df_in, hide_index=True, width="stretch", **kw)
    except Exception:
        try:
            st.dataframe(df_in, hide_index=True, **kw)
        except Exception:
            st.dataframe(df_in, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]

ALL_MODELS = [
    "XGBoost", "Hybrid XGBoost + ANN", "Random Forest",
    "Gradient Boosting", "Artificial Neural Network", "SVR", "Linear Regression",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data / Model Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    models, warnings_list = {}, []

    for name, filename in [
        ("XGBoost",           "xgboost.joblib"),
        ("Random Forest",     "random_forest.joblib"),
        ("Gradient Boosting", "gradient_boosting.joblib"),
        ("SVR",               "svr.joblib"),
        ("Linear Regression", "linear_regression.joblib"),
    ]:
        if os.path.exists(filename):
            try:
                models[name] = joblib.load(filename)
            except Exception as e:
                warnings_list.append(f"Failed to load {name}: {e}")
        else:
            warnings_list.append(f"Model file {filename} not found.")

    ann_model = None
    if os.path.exists("ann_weights.joblib"):
        try:
            ann_model = joblib.load("ann_weights.joblib")
        except Exception as e:
            warnings_list.append(f"ANN weights load error: {e}")

    if ann_model is None and os.path.exists("ann_model.keras"):
        try:
            import keras
            ann_model = keras.Sequential([
                keras.layers.Input(shape=(8,)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64,  activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32,  activation="relu"),
                keras.layers.Dense(16,  activation="relu"),
                keras.layers.Dense(1,   activation="linear"),
            ])
            ann_model.load_weights("ann_model.keras")
        except Exception as e:
            warnings_list.append(f"Keras ANN load error: {e}")
            ann_model = None

    models["Artificial Neural Network"] = ann_model

    scaler = None
    if os.path.exists("ann_scaler.joblib"):
        try:
            scaler = joblib.load("ann_scaler.joblib")
        except Exception as e:
            warnings_list.append(f"ANN scaler load error: {e}")

    return models, scaler, warnings_list


@st.cache_data(show_spinner="Loading datasets…")
def load_datasets():
    datasets, warnings_list = {}, []
    for key, filename in [
        ("comparison",         "model_comparison.csv"),
        ("feature_importance", "feature_importance.csv"),
        ("cv",                 "10_fold_cross_validation.csv"),
        ("test_pred",          "test_predictions.csv"),
        ("shap_summary",       "shap_summary.csv"),
    ]:
        if os.path.exists(filename):
            try:
                datasets[key] = pd.read_csv(filename)
            except Exception as e:
                warnings_list.append(f"Failed to load {filename}: {e}")
        else:
            warnings_list.append(f"Dataset {filename} not found.")

    return datasets, warnings_list


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_ann(models, scaler, X_df):
    ann = models.get("Artificial Neural Network")
    X = scaler.transform(X_df) if scaler is not None else X_df.values

    if ann is None:
        return np.zeros(len(X_df))

    if isinstance(ann, dict):
        weights = [ann[f"layer_{i}"] for i in range(len(ann))]
        out = X
        for i in range(0, len(weights) - 2, 2):
            out = np.maximum(0, out @ weights[i] + weights[i+1])
        return (out @ weights[-2] + weights[-1]).flatten()

    if isinstance(ann, list):
        out = X
        for W, b in ann[:-1]:
            out = np.maximum(0, out @ W + b)
        W, b = ann[-1]
        return (out @ W + b).flatten()

    if keras is not None:
        return ann.predict(X, verbose=0).flatten()

    return np.zeros(len(X_df))


def predict_model(models, scaler, X_df, model_name):
    if model_name == "Artificial Neural Network":
        return predict_ann(models, scaler, X_df)
    if model_name == "Hybrid XGBoost + ANN":
        xgb_p = models["XGBoost"].predict(X_df) if "XGBoost" in models else np.zeros(len(X_df))
        ann_p = predict_ann(models, scaler, X_df)
        return (xgb_p + ann_p) / 2
    mdl = models.get(model_name)
    if mdl is not None:
        return mdl.predict(X_df)
    return np.zeros(len(X_df))


def get_strength_category(strength):
    if strength < 20: return "Low Strength",             "badge-lo", "Non-structural / footpaths."
    if strength < 40: return "Standard Structural",       "badge-md", "Slabs, columns, beams, footings."
    if strength < 60: return "High-Strength (HSC)",       "badge-hi", "High-rise pillars, pre-stressed girders."
    return                   "Ultra-High Performance (UHPC)", "badge-hi", "Nuclear shielding, extreme structures."


# ═════════════════════════════════════════════════════════════════════════════
# Pages
# ═════════════════════════════════════════════════════════════════════════════

def page_predictor():
    _inject_css()
    try:
        models, scaler, fails = load_models()
        if fails:
            with st.expander("⚠️ Model Load Warnings"):
                for f in fails:
                    st.warning(f)

        st.markdown("""
        <div class="hero">
          <h1>🏗️ Explainable Concrete Compressive Strength Predictor</h1>
          <p>Simulate concrete mix formulations and predict compressive strength (MPa) with AI/ML & SHAP explainability.</p>
        </div>""", unsafe_allow_html=True)

        PRESETS = {
            "Standard 28D":  (280.0, 70.0,  50.0,  180.0, 6.0,  980.0, 770.0, 28),
            "High-Strength": (450.0, 100.0, 0.0,   150.0, 12.0, 950.0, 720.0, 28),
            "Eco Fly-Ash":   (200.0, 0.0,   160.0, 165.0, 8.0,  1000.0,790.0, 56),
            "Early 7-Day":   (380.0, 120.0, 0.0,   175.0, 9.0,  920.0, 750.0, 7),
        }
        KEYS = ["c","sl","fa","wa","sp","ca","fi","ag"]
        DEFS = list(PRESETS["Standard 28D"])
        for k, v in zip(KEYS, DEFS):
            st.session_state.setdefault(f"pr_{k}", v)

        def _preset(name):
            for k, v in zip(KEYS, PRESETS[name]):
                st.session_state[f"pr_{k}"] = v

        col_in, col_out = st.columns([1.6, 1.1])
        with col_in:
            st.subheader("⚙️ Mix Parameters")
            pc = st.columns(4)
            for i, nm in enumerate(PRESETS):
                pc[i].button(nm, on_click=_preset, args=(nm,), key=f"pr_btn_{i}", use_container_width=True)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                cement = st.number_input("Cement (kg/m³)",           min_value=100.0, max_value=540.0, step=5.0,  key="pr_c")
                slag   = st.number_input("Blast Furnace Slag (kg/m³)",min_value=0.0,   max_value=360.0, step=5.0,  key="pr_sl")
                flyash = st.number_input("Fly Ash (kg/m³)",           min_value=0.0,   max_value=200.0, step=5.0,  key="pr_fa")
                water  = st.number_input("Water (kg/m³)",           min_value=120.0, max_value=250.0, step=2.0,  key="pr_wa")
            with c2:
                sp  = st.number_input("Superplasticizer (kg/m³)", min_value=0.0,   max_value=35.0,  step=0.5,  key="pr_sp")
                ca  = st.number_input("Coarse Aggregate (kg/m³)", min_value=800.0, max_value=1150.0,step=10.0, key="pr_ca")
                fi  = st.number_input("Fine Aggregate (kg/m³)",   min_value=590.0, max_value=950.0, step=10.0, key="pr_fi")
                age = st.slider("Curing Age (Days)", min_value=1, max_value=365, key="pr_ag")
            binder = cement + slag + flyash
            wb   = water / binder if binder > 0 else 0
            dens = binder + water + sp + ca + fi
            m1, m2, m3 = st.columns(3)
            m1.metric("w/b Ratio", f"{wb:.3f}")
            m2.metric("Total Binder", f"{binder:.1f} kg/m³")
            m3.metric("Mix Density", f"{dens:.1f} kg/m³")

        with col_out:
            st.subheader("🎯 Prediction Output")
            sel = st.selectbox("Model:", ALL_MODELS, key="pr_model")
            row = pd.DataFrame([[cement,slag,flyash,water,sp,ca,fi,age]], columns=FEATURES)
            try:
                strength = float(predict_model(models, scaler, row, sel)[0])
            except Exception as e:
                st.error(f"Prediction error: {e}")
                strength = 0.0
            cat_t, badge, usage = get_strength_category(strength)
            st.markdown(f"""
            <div class="result-box">
              <div style="color:#94a3b8;font-size:.9rem;font-weight:600;">PREDICTED STRENGTH</div>
              <div class="result-val">{strength:.2f}<span style="font-size:1.3rem;"> MPa</span></div>
              <span class="{badge}">{cat_t}</span>
            </div>""", unsafe_allow_html=True)
            st.info(f"**Use case:** {usage}")

        # Full-width 7-Model Comparison Chart
        st.markdown("---")
        st.subheader("📊 Live Prediction Comparison Across All 7 Models")
        rows = []
        for mn in ALL_MODELS:
            try:
                v = float(predict_model(models, scaler, row, mn)[0])
            except Exception:
                v = 0.0
            rows.append({"Model": mn, "Strength (MPa)": v})

        comp_df_pred = pd.DataFrame(rows)
        fig = px.bar(
            comp_df_pred,
            x="Strength (MPa)",
            y="Model",
            orientation="h",
            color="Strength (MPa)",
            color_continuous_scale="Blues",
            text_auto=".2f",
        )
        fig.update_layout(
            height=380,
            margin=dict(l=230, r=40, t=20, b=40),
            showlegend=False,
            xaxis_title="Predicted Compressive Strength (MPa)",
            yaxis_title=None,
            yaxis=dict(type="category", dtick=1, automargin=True),
        )
        _chart(fig)

        st.markdown("---")
        with st.expander("Batch Prediction — Upload CSV"):
            tmpl = pd.DataFrame([[280,70,50,180,6,980,770,28],[450,100,0,150,12,950,720,28]],
                                 columns=FEATURES)
            st.download_button("Download template", tmpl.to_csv(index=False),
                               "template.csv","text/csv", key="pr_dl_tmpl")
            up = st.file_uploader("Upload CSV", type=["csv"], key="pr_up")
            if up is not None:
                bdf  = pd.read_csv(up)
                miss = [c for c in FEATURES if c not in bdf.columns]
                if miss:
                    st.error(f"Missing columns: {miss}")
                else:
                    res = bdf.copy()
                    for mn in ALL_MODELS:
                        try:
                            res[f"{mn} (MPa)"] = np.round(predict_model(models, scaler, bdf[FEATURES], mn), 2)
                        except Exception:
                            res[f"{mn} (MPa)"] = np.nan
                    _df(res.head(10))
                    st.download_button("Download results", res.to_csv(index=False),
                                       "batch_results.csv","text/csv", key="pr_dl_res")
    except Exception:
        st.error("Error on Predictor page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_models():
    _inject_css()
    try:
        datasets, fails = load_datasets()
        if fails:
            with st.expander("⚠️ Dataset warnings"):
                [st.warning(f) for f in fails]

        st.markdown("""
        <div class="hero">
          <h1>📊 Model Comparison & Benchmarks</h1>
          <p>MAE, RMSE, R² and 10-Fold Cross-Validation across all models.</p>
        </div>""", unsafe_allow_html=True)

        k1,k2,k3,k4 = st.columns(4)
        for col, lbl, val, sub in [
            (k1,"Best R²","0.941","XGBoost"),
            (k2,"Lowest MAE","2.61 MPa","XGBoost"),
            (k3,"Lowest RMSE","4.20 MPa","XGBoost"),
            (k4,"Hybrid R²","0.923","XGBoost + ANN"),
        ]:
            col.markdown(f"""<div class="mcard">
              <div class="mlbl">{lbl}</div><div class="mval">{val}</div>
              <div style="color:#94a3b8;font-size:.78rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["Leaderboard", "10-Fold CV", "Scatter Plot"])

        with t1:
            comp = datasets.get("comparison", pd.DataFrame())
            if not comp.empty:
                c1, c2 = st.columns([1.1, 1])
                with c1: _df(comp)
                with c2:
                    fig = px.bar(comp, x="R2", y="Model", orientation="h",
                                 title="R² Score (Higher is Better)",
                                 color="R2", color_continuous_scale="Viridis", text_auto=".3f")
                    fig.update_layout(showlegend=False,
                                      yaxis={"categoryorder":"total ascending", "type": "category", "dtick": 1, "automargin": True},
                                      margin=dict(l=230, r=30, t=40, b=30))
                    _chart(fig, 360)
                fig2 = px.bar(comp, x="Model", y=["MAE","RMSE"], barmode="group",
                              title="MAE & RMSE (Lower is Better)",
                              color_discrete_sequence=["#38bdf8","#f43f5e"])
                fig2.update_layout(yaxis_title="Error (MPa)")
                _chart(fig2, 350)
            else:
                st.info("model_comparison.csv not found.")

        with t2:
            cv = datasets.get("cv", pd.DataFrame())
            if not cv.empty:
                _df(cv)
                fig = go.Figure(go.Bar(x=cv["Model"], y=cv["CV R2 Mean"],
                    error_y=dict(type="data",array=cv["CV R2 Std"],visible=True),
                    marker_color="#6366f1"))
                fig.update_layout(title="10-Fold CV R² ± Std", yaxis_title="Mean R²")
                _chart(fig, 370)
            else:
                st.info("10_fold_cross_validation.csv not found.")

        with t3:
            tpdf = datasets.get("test_pred", pd.DataFrame())
            if not tpdf.empty:
                fig = px.scatter(tpdf, x="Actual Strength", y="Hybrid Predicted Strength",
                                 hover_data=["Cement","Water","Age"],
                                 color="Absolute Error", color_continuous_scale="Plasma",
                                 title="Actual vs Hybrid Predicted Strength")
                lo = min(tpdf["Actual Strength"].min(), tpdf["Hybrid Predicted Strength"].min())
                hi = max(tpdf["Actual Strength"].max(), tpdf["Hybrid Predicted Strength"].max())
                fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="y=x",
                                         line=dict(color="#10b981",dash="dash",width=2)))
                _chart(fig, 460)
                fig2 = px.histogram(tpdf, x="Absolute Error", nbins=30,
                                    title="Residual Distribution",
                                    color_discrete_sequence=["#3b82f6"])
                _chart(fig2, 300)
            else:
                st.info("test_predictions.csv not found.")
    except Exception:
        st.error("Error on Model Comparison page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_features():
    _inject_css()
    try:
        models, scaler, _ = load_models()
        datasets, _  = load_datasets()

        st.markdown("""
        <div class="hero">
          <h1>🔍 Explainability & Feature Attribution Dashboard</h1>
          <p>Game-theoretic SHAP (SHapley Additive exPlanations), feature gain importance, and interactive sensitivity simulations.</p>
        </div>""", unsafe_allow_html=True)

        tab_shap, tab_gain = st.tabs([
            "🔮 SHAP Feature Attribution Analysis",
            "📊 Gain Importance & Sensitivity Simulator"
        ])

        with tab_shap:
            st.subheader("Cross-Model SHAP Summary (Mean |SHAP|, MPa)")
            shap_df = datasets.get("shap_summary", pd.DataFrame())
            if not shap_df.empty:
                _df(shap_df)
            else:
                st.info("`shap_summary.csv` not found. Run `python shap_analysis.py` to generate.")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("SHAP Beeswarm Summary (XGBoost)")
                if os.path.exists("shap_summary_xgboost.png"):
                    st.image("shap_summary_xgboost.png", caption="Figure 5.4: XGBoost SHAP Beeswarm Summary Plot (All 1,030 Specimens)")
                else:
                    st.info("`shap_summary_xgboost.png` not found.")

            with c2:
                st.subheader("Cross-Model SHAP Comparison")
                if os.path.exists("shap_comparison_all_models.png"):
                    st.image("shap_comparison_all_models.png", caption="Figure 5.3: Cross-Model SHAP Attribution Comparison")
                elif os.path.exists("shap_bar_xgboost.png"):
                    st.image("shap_bar_xgboost.png", caption="Figure 5.2: XGBoost Mean |SHAP| Bar Chart")

            st.markdown("---")
            st.subheader("Waterfall Explanation (Individual Specimen #0)")
            if os.path.exists("shap_waterfall_sample.png"):
                st.image("shap_waterfall_sample.png", caption="Figure 5.7: XGBoost SHAP Waterfall Decomposition for Test Specimen #0")

            st.markdown("---")
            st.subheader("SHAP Feature Dependence Plots")
            d1, d2 = st.columns(2)
            with d1:
                if os.path.exists("shap_dependence_age.png"):
                    st.image("shap_dependence_age.png", caption="Figure 5.5: SHAP Dependence Plot — Curing Age")
            with d2:
                if os.path.exists("shap_dependence_cement.png"):
                    st.image("shap_dependence_cement.png", caption="Figure 5.6: SHAP Dependence Plot — Cement Content")

        with tab_gain:
            feat = datasets.get("feature_importance", pd.DataFrame())
            c1, c2 = st.columns([1, 1.3])

            with c1:
                st.subheader("Feature Importance (XGBoost Gain)")
                if not feat.empty:
                    fsort = feat.sort_values("Importance", ascending=True)
                    fig = px.bar(fsort, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Blues", text_auto=".3f")
                    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Importance")
                    _chart(fig, 430)
                    st.info("Age + Cement explain ~66% of strength variance.")
                else:
                    st.info("feature_importance.csv not found.")

            with c2:
                st.subheader("Strength Growth Simulator")
                sc  = st.slider("Cement (kg/m³)",   min_value=150, max_value=500, value=300, key="fs_c")
                sw  = st.slider("Water (kg/m³)",     min_value=130, max_value=220, value=180, key="fs_w")
                ssp = st.slider("Superplasticizer",  min_value=0.0, max_value=20.0, value=6.0, key="fs_sp")
                sm  = st.selectbox("Model:", ALL_MODELS, key="fs_m")
                ages = np.arange(1, 181, 2)
                bdf  = pd.DataFrame([[sc,70,50,sw,ssp,950,750,int(a)] for a in ages],
                                     columns=FEATURES)
                try:
                    y = predict_model(models, scaler, bdf, sm)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    y = np.zeros(len(ages))
                fig = px.line(pd.DataFrame({"Age": ages, "Strength (MPa)": y}),
                              x="Age", y="Strength (MPa)",
                              title=f"Growth Curve — {sm}", markers=True)
                fig.update_traces(line_color="#38bdf8", line_width=3)
                fig.add_vline(x=28, line_dash="dash", line_color="#10b981", annotation_text="28-Day")
                _chart(fig, 370)

            st.markdown("---")
            st.subheader("Water/Cement Ratio Sensitivity")
            wrange = np.linspace(130, 240, 30)
            wbdf   = pd.DataFrame([[300,70,50,w,6,950,750,28] for w in wrange], columns=FEATURES)
            try:
                wy = predict_model(models, scaler, wbdf, "XGBoost")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                wy = np.zeros(30)
            fig = px.line(pd.DataFrame({"w/c Ratio": wrange/300, "Strength (MPa)": wy}),
                          x="w/c Ratio", y="Strength (MPa)",
                          title="Strength vs w/c Ratio — XGBoost", markers=True)
            fig.update_traces(line_color="#f43f5e", line_width=3)
            _chart(fig, 360)
    except Exception:
        st.error("Error on Feature Analysis page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_dataset():
    _inject_css()
    try:
        datasets, fails = load_datasets()
        if fails:
            with st.expander("⚠️ Dataset warnings"):
                [st.warning(f) for f in fails]

        st.markdown("""
        <div class="hero">
          <h1>📁 Dataset Explorer & Downloads</h1>
          <p>Filter and download test predictions, CV results, and model metrics.</p>
        </div>""", unsafe_allow_html=True)

        tpdf = datasets.get("test_pred", pd.DataFrame())
        if not tpdf.empty:
            st.subheader("Test Predictions")
            fc1, fc2 = st.columns(2)
            with fc1:
                age_opts   = sorted(tpdf["Age"].unique().tolist())
                age_filter = st.multiselect("Curing Age:", age_opts,
                                            default=[a for a in [7,28,90] if a in age_opts],
                                            key="ds_age")
            with fc2:
                smin = float(tpdf["Actual Strength"].min())
                smax = float(tpdf["Actual Strength"].max())
                lo, hi = st.slider("Actual Strength (MPa):", min_value=smin, max_value=smax, value=(smin, smax), key="ds_str")
            mask = (tpdf["Actual Strength"] >= lo) & (tpdf["Actual Strength"] <= hi)
            if age_filter:
                mask &= tpdf["Age"].isin(age_filter)
            _df(tpdf[mask])
        else:
            st.info("test_predictions.csv not found.")

        st.markdown("---")
        st.subheader("Downloads")
        d1, d2, d3 = st.columns(3)
        comp = datasets.get("comparison", pd.DataFrame())
        cv   = datasets.get("cv",         pd.DataFrame())
        d1.download_button("Test Predictions CSV",
                           tpdf.to_csv(index=False) if not tpdf.empty else "",
                           "test_predictions.csv","text/csv", key="ds_dl1")
        d2.download_button("Model Comparison CSV",
                           comp.to_csv(index=False) if not comp.empty else "",
                           "model_comparison.csv","text/csv", key="ds_dl2")
        d3.download_button("10-Fold CV CSV",
                           cv.to_csv(index=False) if not cv.empty else "",
                           "10_fold_cv.csv","text/csv", key="ds_dl3")
    except Exception:
        st.error("Error on Dataset Explorer page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_specs():
    _inject_css()
    try:
        st.markdown("""
        <div class="hero">
          <h1>ℹ️ Research Architecture & Specs</h1>
          <p>7-model benchmark architecture, ANN & ensemble design, SHAP explainability methodology, training pipeline, and UCI dataset details.</p>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("🧠 ANN Architecture")
            st.code("""
Input   :  8 concrete mix features
Dense   : 128 units, ReLU
Dense   :  64 units, ReLU
Dropout :  rate = 0.2
Dense   :  32 units, ReLU
Dense   :  16 units, ReLU
Output  :   1 unit, Linear -> MPa
            """, language="text")
            st.markdown("""
**Optimiser:** Adam (lr = 0.001)  
**Loss:** Mean Squared Error  
**Scaler:** StandardScaler on all 8 features  
**Dataset:** 1030 samples, 80/20 train-test split, 10-fold CV
            """)

        with c2:
            st.subheader("⚡ Hybrid Ensemble Formula")
            st.latex(r"\hat{y}_{hybrid} = 0.5\,\hat{y}_{XGBoost} + 0.5\,\hat{y}_{ANN}")

            if os.path.exists("model_comparison.csv"):
                try:
                    df_comp = pd.read_csv("model_comparison.csv")
                    df_comp["R²"] = df_comp["R2"].apply(lambda v: f"{v:.3f}")
                    df_comp["MAE (MPa)"] = df_comp["MAE"].apply(lambda v: f"{v:.2f}")
                    df_comp["RMSE (MPa)"] = df_comp["RMSE"].apply(lambda v: f"{v:.2f}")
                    show_df = df_comp[["Model", "MAE (MPa)", "RMSE (MPa)", "R²"]]
                except Exception:
                    show_df = None
            else:
                show_df = None

            if show_df is None:
                data = {
                    "Model":          ["XGBoost","Hybrid","Gradient Boosting","Random Forest","SVR","ANN","Linear Reg."],
                    "MAE (MPa)":      [2.61,  3.09,  3.65,  3.51,  4.02,  4.30, 8.90],
                    "RMSE (MPa)":     [4.20,  4.79,  5.02,  5.19,  5.97,  6.10,11.19],
                    "R²":             [0.941, 0.923, 0.915, 0.910, 0.880, 0.875, 0.580],
                }
                show_df = pd.DataFrame(data)

            _df(show_df)

        st.markdown("---")
        st.subheader("📚 Dataset: UCI Concrete Compressive Strength")
        st.markdown("""
| Property | Value |
|---|---|
| Samples | 1030 |
| Features | 8 continuous (cement, slag, fly ash, water, SP, CA, FA, age) |
| Target | Compressive strength (MPa) |
| Age range | 1 – 365 days |
| Strength range | 2.33 – 82.60 MPa |
| Source | I-Cheng Yeh, 1998 (UCI ML Repository) |
        """)
        st.warning("⚠️ **Operational Scope & Recalibration Notice:** This deployment operates strictly as an interpolation utility within empirical dataset boundaries (Cement 102–540 kg/m³, Water 127–247 kg/m³, Age 1–365 days, lab curing ~20°C). Models do not explicitly account for aggregate mineralogy, cement chemical composition, ambient curing temperature, or specific admixture brand formulations. Field engineers must recalibrate model parameters using local batch plant trial mix data before commercial structural compliance deployment.")

        st.caption("Explainable Concrete Strength ML & ANN | Benchmark Study "
                   "| Streamlit · XGBoost · SHAP · Scikit-Learn · Keras")
    except Exception:
        st.error("Error on Research Specs page:")
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation (Guaranteed to work reliably on Streamlit Cloud)
# ═════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🏗️ Concrete Strength ML & ANN")
selected_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🧪 Interactive Predictor",
        "📊 Model Comparison",
        "🔍 Feature Analysis",
        "📁 Dataset Explorer",
        "ℹ️ Research Specs",
    ],
    key="nav_radio_menu_app",
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Change mix formulations or select different AI/ML models to compare predicted strength.")

if "Interactive Predictor" in selected_page:
    page_predictor()
elif "Model Comparison" in selected_page:
    page_models()
elif "Feature Analysis" in selected_page:
    page_features()
elif "Dataset Explorer" in selected_page:
    page_dataset()
elif "Research Specs" in selected_page:
    page_specs()
