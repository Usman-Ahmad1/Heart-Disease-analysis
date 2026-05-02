import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── Title ─────────────────────────────────────────────────────
st.title("🩺 Heart Disease Risk Predictor")
st.markdown("Enter patient details below to predict heart disease risk using an XGBoost model trained on 920 patients.")
st.markdown("---")

# ── Load and train model on the fly ──────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("heart_disease_uci.csv")

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

    df['target'] = (df['num'] > 0).astype(int)
    df_model = df.drop(columns=['id', 'num'], errors='ignore')

    le = LabelEncoder()
    cat_cols = df_model.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = df_model.drop(columns=['target'])
    y = df_model['target']

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss'
    )
    model.fit(X, y)
    return model, encoders, list(X.columns)

model, encoders, feature_cols = load_model()

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [
        "typical angina", "atypical angina", "non-anginal", "asymptomatic"
    ])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])

with col2:
    thalch = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["True", "False"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
    dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    input_dict = {
        'age': age,
        'sex': sex,
        'dataset': dataset,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical columns
    for col, mapping in encoders.items():
        if col in input_df.columns:
            val = str(input_df[col].iloc[0])
            input_df[col] = mapping.get(val, 0)

    # Align columns
    input_df = input_df[feature_cols]

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("🔬 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** — Confidence: {prob*100:.1f}%")
        st.markdown("This patient shows indicators associated with heart disease. Please consult a medical professional.")
    else:
        st.success(f"✅ **Low Risk of Heart Disease** — Confidence: {(1-prob)*100:.1f}%")
        st.markdown("This patient shows no strong indicators of heart disease based on the provided data.")

    # Risk meter
    st.markdown("#### Risk Probability")
    st.progress(float(prob))
    st.caption(f"Model confidence: {prob*100:.1f}% probability of heart disease")

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with ❤️ | Lagos Techies 3-Day Credibility Challenge 🔥")
