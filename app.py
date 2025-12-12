
import streamlit as st
import pandas as pd
import joblib
import os
from config import UI_ORDER, INPUT_CONFIG
from ui import (
    load_video,
    display_video,
    create_sidebar,
    display_predictions,
    display_insights,
    display_project_overview,
    create_afr_graph,
)
from processing import prepare_features

# =====================
# Load Models & Scalers
# =====================
lr_em = joblib.load("logreg_emission.pkl")
scaler_em = joblib.load("scaler_emission.pkl")
features_em = joblib.load("features_emission.pkl")

lr_q = joblib.load("logreg_combustion.pkl")
scaler_q = joblib.load("scaler_combustion.pkl")
features_q = joblib.load("features_combustion.pkl")

# =====================
# Streamlit Layout
# =====================
st.set_page_config(page_title="Engine Emission & Combustion Predictor", layout="wide")
st.title("🚗 Engine Emission & Combustion Quality Predictor")
st.markdown("Predict **Emission Category** 🌫️ and **Combustion Quality** 🔥 based on input engine parameters.")

# =====================
# Video Display
# =====================
video_path = os.path.join(os.path.dirname(__file__), "videos", "DEMO_ENGINE_APP.mp4")
encoded_video = load_video(video_path)
if encoded_video:
    display_video(encoded_video)
else:
    st.error("DEMO_ENGINE_APP.mp4 not found — make sure it is in the videos/ folder.")

# =====================
# Sidebar and Inputs
# =====================
input_data = create_sidebar()
input_df = pd.DataFrame([input_data])

st.subheader("🔎 Input Preview")
st.dataframe(input_df)

# =====================
# Prediction
# =====================
if st.button("Predict 🚀"):
    X_em, X_q = prepare_features(input_df, features_em, features_q)
    
    # Emission Prediction
    X_em_scaled = scaler_em.transform(X_em)
    em_pred = lr_em.predict(X_em_scaled)[0]
    
    # Combustion Quality Prediction
    X_q_scaled = scaler_q.transform(X_q)
    q_pred = lr_q.predict(X_q_scaled)[0]

    st.success("✅ Predictions Completed!")
    
    display_predictions(em_pred, q_pred)
    display_insights()
    display_project_overview()
    
    st.subheader("📊 Thermodynamic AFR Graph")
    afr_fig = create_afr_graph()
    st.plotly_chart(afr_fig, use_container_width=True)
