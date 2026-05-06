"""
DS-3002 Data Mining — Assignment #4
Part E: Local Front-End Dashboard (10 Marks)
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Disease Screening",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
.risk-positive { background:#ffe0e0; border-left:6px solid #cc0000;
                 padding:14px; border-radius:8px; font-size:1.2rem; }
.risk-negative { background:#e0ffe0; border-left:6px solid #007700;
                 padding:14px; border-radius:8px; font-size:1.2rem; }
.metric-box    { background:#f5f5f5; border-radius:8px; padding:12px;
                 text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    data      = joblib.load('models/preprocessed_data.pkl')
    rf_model  = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    return data, rf_model, xgb_model

data, rf_model, xgb_model = load_artifacts()
scaler      = data['scaler']
feat_names  = data['feature_names']   # post-encoding columns

CONTINUOUS  = ['age','trestbps','chol','thalach','oldpeak','ca']
CATEGORICAL = ['cp','restecg','slope','thal']

# Pre-populate with a real test patient (first row of X_test_raw)
X_test_raw = data['X_test_raw']
sample = X_test_raw.iloc[0]

# ── Title ────────────────────────────────────────────────────────
st.title("🫀 CardioAI — Heart Disease Risk Screening")
st.markdown("""
*Decision-support tool for community cardiologists.*  
Enter patient measurements below and click **Predict** to get an instant risk assessment.
""")

# ════════════════════════════════════════════════════════════════
# E1  Input Form
# ════════════════════════════════════════════════════════════════
st.header("📋 Patient Input Form")
st.info("Form pre-populated with a real test patient. Edit values as needed.")

col1, col2, col3 = st.columns(3)

with col1:
    age      = st.number_input("Age (years) — range 20–80",
                               min_value=20, max_value=80, value=int(sample['age']))
    sex      = st.selectbox("Sex",   options=[0, 1],
                            format_func=lambda x: "Female (0)" if x==0 else "Male (1)",
                            index=int(sample['sex']))
    cp       = st.selectbox("Chest Pain Type (cp)",
                            options=[0,1,2,3],
                            format_func=lambda x: {0:'Typical Angina',1:'Atypical',
                                                    2:'Non-anginal',3:'Asymptomatic'}[x],
                            index=int(sample['cp']))
    trestbps = st.number_input("Resting BP mmHg — range 90–200",
                               min_value=90, max_value=200, value=int(sample['trestbps']))
    chol     = st.number_input("Serum Cholesterol mg/dl — range 100–600",
                               min_value=100, max_value=600, value=int(sample['chol']))

with col2:
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                            options=[0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)",
                            index=int(sample['fbs']))
    restecg  = st.selectbox("Resting ECG Results (restecg)",
                            options=[0,1,2],
                            format_func=lambda x: {0:'Normal',1:'ST-T wave abnormality',
                                                    2:'LV hypertrophy'}[x],
                            index=int(sample['restecg']))
    thalach  = st.number_input("Max Heart Rate Achieved — range 70–210",
                               min_value=70, max_value=210, value=int(sample['thalach']))
    exang    = st.selectbox("Exercise-Induced Angina",
                            options=[0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)",
                            index=int(sample['exang']))

with col3:
    oldpeak  = st.number_input("ST Depression (oldpeak) — range 0.0–6.2",
                               min_value=0.0, max_value=6.2,
                               value=float(sample['oldpeak']), step=0.1, format="%.1f")
    slope    = st.selectbox("Slope of ST Segment",
                            options=[0,1,2],
                            format_func=lambda x: {0:'Upsloping',1:'Flat',2:'Downsloping'}[x],
                            index=int(sample['slope']))
    ca       = st.number_input("Major Vessels by Fluoroscopy (ca) — range 0–3",
                               min_value=0, max_value=3, value=int(sample['ca']))
    thal     = st.selectbox("Thalassemia (thal)",
                            options=[1,2,3],
                            format_func=lambda x: {1:'Normal',2:'Fixed Defect',
                                                    3:'Reversible Defect'}[x],
                            index=[1,2,3].index(int(sample['thal'])))

model_choice = st.radio("Select Model", ['XGBoost (Recommended)', 'Random Forest'],
                         horizontal=True)

predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════════════
# E2  Results Panel
# ════════════════════════════════════════════════════════════════
if predict_btn:
    st.divider()
    st.header("📊 Prediction Result")

    # Build raw feature row
    raw_row = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    # Encode categoricals
    enc_row = pd.get_dummies(raw_row, columns=CATEGORICAL)
    enc_row = enc_row.reindex(columns=feat_names, fill_value=0)
    enc_row[CONTINUOUS] = scaler.transform(enc_row[CONTINUOUS])
    X_input = enc_row.values

    model = xgb_model if 'XGBoost' in model_choice else rf_model
    pred  = int(model.predict(X_input)[0])
    prob  = float(model.predict_proba(X_input)[0][1])

    # Display result
    risk_label = "🔴 Disease Present" if pred == 1 else "🟢 No Disease Detected"
    css_class  = "risk-positive" if pred == 1 else "risk-negative"
    conf_pct   = prob * 100 if pred == 1 else (1 - prob) * 100

    st.markdown(f'<div class="{css_class}"><b>{risk_label}</b></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Confidence Score", f"{conf_pct:.1f}%")
        st.metric("Disease Probability", f"{prob*100:.1f}%")
    with c2:
        st.metric("Model Used", model_choice.split(" ")[0])

    # Top 3 feature contributions
    st.subheader("🔍 Top 3 Feature Contributions")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.feature_importances_)

    top3_idx  = np.argsort(importances)[::-1][:3]
    top3_names = [feat_names[i] for i in top3_idx]
    top3_vals  = importances[top3_idx]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    colors = ['#cc3333' if pred==1 else '#1a7a1a'] * 3
    ax.barh(top3_names[::-1], top3_vals[::-1], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 3 Feature Importances')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close()

    # Plain-English explanation
    st.subheader("💬 Clinical Summary")
    if pred == 1:
        st.warning(f"""
**Elevated cardiac risk detected.** This patient's profile — including 
{"a low maximum heart rate" if thalach < 140 else "exercise-induced angina" if exang else "ST depression during exercise"} 
and {"ST depression level (oldpeak = " + str(oldpeak) + ")" if oldpeak > 1.5 else "vessel involvement (ca = " + str(ca) + ")"} — 
are among the strongest indicators of coronary artery disease. The cardiologist 
should consider further diagnostic workup (e.g., stress echo or angiography). 
Do not discharge without specialist review.
""")
    else:
        st.success(f"""
**Low cardiac risk detected.** Current measurements — including max heart rate of {thalach} bpm 
and minimal ST depression (oldpeak = {oldpeak}) — are within reassuring ranges for cardiac health. 
Routine follow-up is recommended; advise the patient on lifestyle factors (diet, exercise) 
and repeat screening in 12 months or sooner if symptoms develop.
""")

    st.caption(f"Model: {model_choice} | Prediction threshold: 0.50")

st.divider()
st.caption("CardioAI Labs — Assignment #4 Demo | DS-3002 Data Mining | FAST-NUCES 2026")
