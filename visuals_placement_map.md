# Thesis Visual Placement Map
## *Explainable Machine Learning and Deep Neural Networks for Concrete Compressive Strength Prediction: A Benchmark Study with SHAP Feature Attribution*
### All Tables, Figures, Plots & Curves — Exact Location in `Thesis.md`

---

## Legend

| Symbol | Meaning |
|:---:|:---|
| ✅ | Already present as inline markdown table — no file embed needed |
| ⬜ | PNG file exists in project root — needs `![caption](filename)` embed added to Thesis.md |
| ⚠️ | Must first be exported as a static PNG from the Streamlit app, then embedded |

---

## SECTION 4 — Experimental Results & Performance Analysis

---

### §4.1 · Test Dataset Performance Benchmark (≈ lines 505–570)

| Visual | File Name | Insert After | Status |
|:---|:---|:---|:---:|
| **Table 2** — Model Performance Metrics Summary | `model_comparison.csv` (inline) | Line 540 heading | ✅ |
| **Figure 4.1** — Model R² Leaderboard Bar Chart | `model_r2_comparison.png` | Line 553 heading | ⬜ |
| **Figure 4.2** — MAE vs. RMSE Bar Chart | `model_error_comparison.png` | Line 562 heading | ⬜ |

**Embed syntax for Figure 4.1:**
```markdown
**Figure 4.1: Model Out-of-Sample Leaderboard Bar Chart**

![Figure 4.1 – R² leaderboard bar chart across all 7 models](model_r2_comparison.png)
```

**Embed syntax for Figure 4.2:**
```markdown
**Figure 4.2: Model Error Benchmark Bar Chart — MAE vs. RMSE**

![Figure 4.2 – MAE and RMSE comparison across all 7 models](model_error_comparison.png)
```

---

### §4.2 · 10-Fold Cross-Validation Metrics (≈ lines 571–602)

| Visual | File Name | Insert After | Status |
|:---|:---|:---|:---:|
| **Table 3** — 10-Fold CV Results | `10_fold_cross_validation.csv` (inline) | Line 585 heading | ✅ |
| **Figure 4.3** — CV R² with Std-Dev Error Bars | `cross_validation_r2.png` | Line 596 heading | ⬜ |

**Embed syntax for Figure 4.3:**
```markdown
**Figure 4.3: 10-Fold Cross-Validation Mean R² with Standard Deviation Error Bars**

![Figure 4.3 – 10-fold CV mean R² with ± std error bars across models](cross_validation_r2.png)
```

---

### §4.3 · Out-of-Sample Predictions & Residual Distribution (≈ lines 604–618)

| Visual | File Name | Insert After | Status |
|:---|:---|:---|:---:|
| **Figure 4.4** — Actual vs. Predicted Scatter Plot | `actual_vs_predicted.png` | Line 608 (blank lines before Figure heading) | ⬜ |

**Embed syntax for Figure 4.4:**
```markdown
**Figure 4.4: Actual vs. Hybrid Predicted Compressive Strength Scatter Plot**

![Figure 4.4 – Actual vs. predicted scatter plot for the Hybrid XGBoost+ANN model](actual_vs_predicted.png)
```

---

### §4.4 · Hybrid Ensemble Ablation & Component Isolation Study (≈ lines 620–650)

| Visual | File Name | Insert After | Status |
|:---|:---|:---|:---:|
| **Table 4** — Ablation Study Results | Inline markdown | Line 624 heading | ✅ |
| **Figure 4.5** — Residual Error Distribution Histogram | `residual_histogram.png` | Line 641 (blank lines before Figure heading) | ⬜ |

**Embed syntax for Figure 4.5:**
```markdown
**Figure 4.5: Model Residual Error Distribution Histogram**

![Figure 4.5 – Distribution of absolute prediction errors across the test set](residual_histogram.png)
```

---

## SECTION 5 — Feature Importance & Sensitivity Analysis

---

### §5.1 · Gain-Based Feature Importance (≈ lines 652–691)

| Visual | File Name | Insert After | Status |
|:---|:---|:---|:---:|
| **Figure 5.1** — XGBoost Feature Importance Bar Chart | `feature_importance_bar.png` | Line 682 heading | ⬜ |

**Embed syntax for Figure 5.1:**
```markdown
**Figure 5.1: XGBoost Relative Feature Importance Bar Chart**

![Figure 5.1 – XGBoost gain-based feature importance for all 8 concrete mix parameters](feature_importance_bar.png)
```

---

### §5.1.2 · SHAP Feature Attribution Analysis (≈ lines 693–741)

| Visual | File Name | Insert After | Status |
|:---|:---|:---|:---:|
| **Table 5** — Cross-Model SHAP Summary | `shap_summary.csv` (inline) | Line 709 heading | ✅ |
| **Figure 5.2** — SHAP Bar Chart: XGBoost | `shap_bar_xgboost.png` | After Table 5 (≈ line 726), before "Key SHAP Findings" | ⬜ |
| **Figure 5.3** — Cross-Model SHAP Comparison | `shap_comparison_all_models.png` | After Figure 5.2 | ⬜ |
| **Figure 5.4** — SHAP Beeswarm Plot (XGBoost) | `shap_summary_xgboost.png` | Within SHAP Finding #1 (≈ line 729) | ⬜ |
| **Figure 5.5** — SHAP Dependence: Curing Age | `shap_dependence_age.png` | After Figure 5.4 / within Finding #1 | ⬜ |
| **Figure 5.6** — SHAP Dependence: Cement Content | `shap_dependence_cement.png` | Within SHAP Finding #2 (≈ line 731) | ⬜ |
| **Figure 5.7** — SHAP Waterfall: Specimen #0 | `shap_waterfall_sample.png` | End of "Waterfall Explanation" sub-section (≈ line 741) | ⬜ |

**Embed syntax for Figure 5.2:**
```markdown
**Figure 5.2: XGBoost — Mean |SHAP Value| Feature Attribution Bar Chart**

![Figure 5.2 – XGBoost mean absolute SHAP value per feature, sorted by importance](shap_bar_xgboost.png)
```

**Embed syntax for Figure 5.3:**
```markdown
**Figure 5.3: Cross-Model SHAP Attribution Comparison — XGBoost · Random Forest · Gradient Boosting · ANN**

![Figure 5.3 – Grouped bar chart comparing mean |SHAP| per feature across all four models](shap_comparison_all_models.png)
```

**Embed syntax for Figure 5.4:**
```markdown
**Figure 5.4: XGBoost SHAP Beeswarm Summary Plot — All 1,030 Specimens**

![Figure 5.4 – SHAP beeswarm plot: feature value (colour) vs. SHAP contribution (x-axis) for XGBoost](shap_summary_xgboost.png)
```

**Embed syntax for Figure 5.5:**
```markdown
**Figure 5.5: SHAP Dependence Plot — Curing Age vs. Strength Contribution (XGBoost)**

![Figure 5.5 – SHAP values for Curing Age plotted against actual age values (days)](shap_dependence_age.png)
```

**Embed syntax for Figure 5.6:**
```markdown
**Figure 5.6: SHAP Dependence Plot — Cement Content vs. Strength Contribution (XGBoost)**

![Figure 5.6 – SHAP values for Cement Content plotted against actual cement content (kg/m³)](shap_dependence_cement.png)
```

**Embed syntax for Figure 5.7:**
```markdown
**Figure 5.7: XGBoost SHAP Waterfall — Test Specimen #0 (Predicted: 52.01 MPa)**

![Figure 5.7 – Waterfall plot decomposing the XGBoost prediction for specimen #0 into additive SHAP contributions](shap_waterfall_sample.png)
```

---

### §5.2 · Dynamic Sensitivity & Growth Curve Simulations (≈ lines 743–771)

| Visual | File Name | Insert After | Status |
|:---|:---|:---|:---:|
| **Figure 5.8** — Curing Age Growth Curve (1–180 days) | `age_growth_curve.png` | Line 747 (blank lines before Figure heading) | ⚠️ Export from Streamlit |
| **Figure 5.9** — w/c Ratio Sensitivity Curve | `wc_sensitivity_curve.png` | Line 762 (before Figure heading) | ⚠️ Export from Streamlit |

> **How to export Figures 5.8 & 5.9:**
> 1. Run `streamlit run app.py`
> 2. Navigate to the Sensitivity Simulator panel
> 3. Use your browser's right-click → "Save image as" on each plot
> 4. Save as `age_growth_curve.png` and `wc_sensitivity_curve.png` in the project root

**Embed syntax for Figure 5.8:**
```markdown
**Figure 5.8: Dynamic Compressive Strength Development Growth Simulator Curve (1–180 Days)**

![Figure 5.8 – XGBoost-predicted strength growth curve from 1 to 180 days curing age](age_growth_curve.png)
```

**Embed syntax for Figure 5.9:**
```markdown
**Figure 5.9: Compressive Strength Sensitivity vs. Water-to-Cement Ratio (w/c) Curve**

![Figure 5.9 – XGBoost-predicted strength vs. w/c ratio showing inverse strength relationship](wc_sensitivity_curve.png)
```

---

## Master Summary Table

| # | Figure / Table Label | File Name | Section | Status |
|:---:|:---|:---|:---|:---:|
| T2 | Table 2 — Model Performance Metrics | inline (`model_comparison.csv`) | §4.1 | ✅ Embedded |
| F4.1 | R² Leaderboard Bar Chart | `model_r2_comparison.png` | §4.1 | ✅ Embedded |
| F4.2 | MAE vs RMSE Bar Chart | `model_error_comparison.png` | §4.1 | ✅ Embedded |
| T3 | Table 3 — 10-Fold CV Results | inline (`10_fold_cross_validation.csv`) | §4.2 | ✅ Embedded |
| F4.3 | 10-Fold CV R² Error Bar Chart | `cross_validation_r2.png` | §4.2 | ✅ Embedded |
| F4.4 | Actual vs. Predicted Scatter Plot | `actual_vs_predicted.png` | §4.3 | ✅ Embedded |
| T4 | Table 4 — Ablation Study | inline markdown | §4.4 | ✅ Embedded |
| F4.5 | Residual Error Histogram | `residual_histogram.png` | §4.4 | ✅ Embedded |
| F5.1 | XGBoost Feature Importance Bar | `feature_importance_bar.png` | §5.1 | ✅ Embedded |
| T5 | Table 5 — SHAP Attribution Summary | inline (`shap_summary.csv`) | §5.1.2 | ✅ Embedded |
| F5.2 | SHAP Bar Chart — XGBoost | `shap_bar_xgboost.png` | §5.1.2 | ✅ Embedded |
| F5.3 | Cross-Model SHAP Comparison | `shap_comparison_all_models.png` | §5.1.2 | ✅ Embedded |
| F5.4 | SHAP Beeswarm Plot (XGBoost) | `shap_summary_xgboost.png` | §5.1.2 | ✅ Embedded |
| F5.5 | SHAP Dependence — Curing Age | `shap_dependence_age.png` | §5.1.2 | ✅ Embedded |
| F5.6 | SHAP Dependence — Cement Content | `shap_dependence_cement.png` | §5.1.2 | ✅ Embedded |
| F5.7 | SHAP Waterfall — Specimen #0 | `shap_waterfall_sample.png` | §5.1.2 | ✅ Embedded |
| F5.8 | Curing Age Growth Curve | `age_growth_curve.png` | §5.2 | ✅ Embedded |
| F5.9 | w/c Ratio Sensitivity Curve | `wc_sensitivity_curve.png` | §5.2 | ✅ Embedded |

**Total: 18 visuals — All 4 tables inline ✅ · All 14 PNG plot figures generated & embedded into Thesis.md ✅**
