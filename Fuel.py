import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

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
# Input Config
# =====================
INPUT_CONFIG = {
    "Water fraction [0-1]": {"min": 0.0, "max": 0.5, "step": 0.01, "default": 0.25},
    "Rotational speed (rpm)": {"min": 2000.0, "max": 4000.0, "step": 50.0, "default": 3000.0},
    "Compression ratio": {"min": 7.44, "max": 9.44, "step": 0.01, "default": 8.44},
    "Fuel temperature (℃)": {"min": 29.5, "max": 37.5, "step": 0.1, "default": 32.3},
    "Exhaust temperature (℃)": {"min": 435.8, "max": 767.5, "step": 1.0, "default": 620.0},
    "Oil temperature (℃)": {"min": 63.9, "max": 166.1, "step": 0.1, "default": 115.0},
    "Torque (N m)": {"min": 5.7, "max": 12.5, "step": 0.1, "default": 10.0},
    "Fuel consumption (g/s)": {"min": 0.21, "max": 1.26, "step": 0.01, "default": 0.54797},
    "Power (kW)": {"min": 1.759292, "max": 5.235988, "step": 0.01, "default": 3.38},
    "Specific fuel consumption (g/kWh)": {"min": 265.00, "max": 1846.00, "step": 10.0, "default": 614.00},
    "Efficiency (%)": {"min": 20.00, "max": 60.50, "step": 10.0, "default": 40.0},
    "CO (%)": {"min": 0.16, "max": 0.96, "step": 0.01, "default": 0.598},
    "CO2 (%)": {"min": 7.00, "max": 12.6, "step": 0.01, "default": 10.00},
    "O2 (%)": {"min": 12.00, "max": 18.50, "step": 0.10, "default": 15.00},
    "HC (ppm)": {"min": 8.00, "max": 1917.00, "step": 10.00, "default": 288.00},
    "NOx (ppm)": {"min": 0.00, "max": 2804.000, "step": 10.000, "default": 580.00},
    "Ignition Advance Angle (°BTDC)": {"min": 20.00, "max": 75.00, "step": 1.00, "default": 40.0}
}

# Exact feature order for UI
UI_ORDER = [
    "Water fraction [0-1]", "Rotational speed (rpm)", "Compression ratio",
    "Fuel temperature (℃)", "Exhaust temperature (℃)", "Oil temperature (℃)",
    "Torque (N m)", "Fuel consumption (g/s)", "Power (kW)",
    "Specific fuel consumption (g/kWh)", "Efficiency (%)", "CO (%)",
    "CO2 (%)", "O2 (%)", "HC (ppm)", "NOx (ppm)", "Ignition Advance Angle (°BTDC)"
]


# =====================
# Preprocessing
# =====================
def apply_transformations(df):
    df_transformed = df.copy()
    if "NOx (ppm)" in df_transformed.columns:
        df_transformed["NOx (ppm)_sqrt"] = np.sqrt(df_transformed["NOx (ppm)"])
    right_skew = ["HC (ppm)", "Fuel temperature (℃)", "Ignition Advance Angle (°BTDC)",
                  "Fuel consumption (g/s)", "Specific fuel consumption (g/kWh)"]
    for col in right_skew:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    left_skew = ["Torque (N m)"]
    for col in left_skew:
        if col in df_transformed.columns:
            max_val = df_transformed[col].max() + 1
            df_transformed[col] = df_transformed[col].apply(lambda x: np.log1p(max_val - x) if x < max_val else 0)
    return df_transformed


def prepare_features(input_df):
    df_transformed = apply_transformations(input_df)
    for feat in features_em + features_q:
        if feat not in df_transformed.columns:
            df_transformed[feat] = 0
    X_em = df_transformed[features_em]
    X_q = df_transformed[features_q]
    return X_em, X_q


# =====================
# Streamlit Layout
# =====================
st.set_page_config(page_title="Engine Emission & Combustion Predictor", layout="wide")
st.title("🚗 Engine Emission & Combustion Quality Predictor")
st.markdown("Predict **Emission Category** 🌫️ and **Combustion Quality** 🔥 based on input engine parameters.")

# =====================
# Video Display (Deployment Safe)
# =====================
# Path to your video
import streamlit as st
import os
import base64

# Page title
st.title("Engine Crankshaft Demo")  # Clear and precise

# Path to video
video_path = os.path.join(os.path.dirname(__file__), "videos", "DEMO_ENGINE_APP.mp4")

if os.path.exists(video_path):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        encoded_video = base64.b64encode(video_bytes).decode()

    # Embed video centered
    st.markdown(
        f"""
        <div style="text-align:center;">
            <video autoplay loop muted playsinline 
                   style="width:600px; height:auto; border-radius:10px;">
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <p style="margin-top:10px; font-size:14px; color:gray;">
                Simulation of crankshaft motion in a 4-cylinder engine (mechanical demo).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error("DEMO_ENGINE_APP.mp4 not found — make sure it is in the videos/ folder in the repo.")


st.title("Engine Simulation Demo")

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Input Parameters")
    input_data = {}
    for feature in UI_ORDER:
        cfg = INPUT_CONFIG[feature]
        input_data[feature] = st.number_input(
            f"{feature}", min_value=cfg["min"], max_value=cfg["max"],
            value=cfg["default"], step=cfg["step"]
        )

# Convert to DataFrame
input_df = pd.DataFrame([input_data])
st.subheader("🔎 Input Preview")
st.dataframe(input_df)

# =====================
# Prediction
# =====================
if st.button("Predict 🚀"):
    X_em, X_q = prepare_features(input_df)
    X_em_scaled = scaler_em.transform(X_em)
    em_pred = lr_em.predict(X_em_scaled)[0]
    X_q_scaled = scaler_q.transform(X_q)
    q_pred = lr_q.predict(X_q_scaled)[0]

    st.success("✅ Predictions Completed!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌫️ Emission Category", em_pred)
    with col2:
        st.metric("🔥 Combustion Quality", q_pred)

    # =====================
    # Insights
    # =====================
    st.subheader("💡 Insights (Thermodynamics Focused)")

    st.markdown(
        """
    **🚨 Emission Notes:**  
    🌬 **NOx (Nitrogen Oxides):** Formed at high combustion temperatures from N₂ + O₂.  
    High NOx → indicates hot combustion or advanced ignition timing, which can increase thermal efficiency but also environmental pollutants.

    ⛽ **HC (Hydrocarbons):** Unburned fuel molecules in exhaust.  
    High HC → incomplete combustion caused by rich mixtures, misfires, or poor fuel atomization, leading to energy loss.

    **⚖ Stoichiometric Air-Fuel Ratio (AFR):**  
    - Ideal ratio where all fuel burns completely: gasoline ≈ 14.7:1.  
    - Lean (too much air) → higher flame temperatures → more NOx.  
    - Rich (too much fuel) → incomplete combustion → more HC and CO.  
    - Maintaining AFR near stoichiometric maximizes efficiency and minimizes pollutants.

    **🔥 Thermodynamics Notes:**  
    - High torque & power with low emissions → indicates efficient energy conversion.  
    - High exhaust temperature → more heat released, possibly more NOx formation.  
    - Oil & coolant temperatures → need to be optimal to maintain thermal efficiency and prevent overheating.  

    **💡 Practical Insight:**  
    Monitoring emissions and engine parameters together allows for tuning the air-fuel ratio, ignition timing, and cooling to achieve **maximum thermodynamic efficiency** while keeping NOx and HC emissions in check.
    """
    )

    if em_pred == "Low":
        st.success("✅ Emissions are low — combustion temp & air-fuel ratio near optimal.")
    elif em_pred == "Medium":
        st.warning("⚠ Moderate emissions — consider tuning ignition timing, air/fuel ratio, or cooling.")
    else:
        st.error("❌ High NOx/HC — check ignition timing, EGR, water injection, or cooling.")

    if q_pred == "High":
        st.success("🔥 Combustion quality high — efficient energy conversion, minimal thermodynamic losses.")
    elif q_pred == "Medium":
        st.warning("⚡ Combustion quality average — slight tuning may improve efficiency.")
    else:
        st.error("💨 Combustion quality low — misfires, rich/lean mixtures, or timing issues reducing efficiency.")

    st.info("""
**📊 Project Overview:**  
Monitors key engine parameters (Fuel, Exhaust, Oil temp, Torque, Power, Fuel consumption, Emissions)  
Predicts **combustion quality** & **emission levels** using thermodynamic principles (energy conversion efficiency, heat release, mixture optimization).  
💡 Goal: Improve efficiency, reduce pollutants, provide actionable tuning insights.
""")

    # =====================
    # Thermodynamic AFR Graph
    # =====================

    # AFR range: lean to rich
    afr = np.linspace(12, 17, 100)

    # Simplified emission trends
    nox = np.exp(-(afr - 14.7) ** 2 / 0.5) * 100  # NOx peaks on lean side
    hc = np.exp(-(afr - 14.7) ** 2 / 0.8) * 80  # HC peaks on rich side

    # Simplified efficiency curve: max at stoichiometric
    efficiency = np.exp(-(afr - 14.7) ** 2 / 1.0) * 100

    # Safe AFR range (for low emissions and high efficiency)
    safe_afr_min = 14.3
    safe_afr_max = 15.1

    # Thermodynamics summary text
    thermo_text = (
        "📌 Thermodynamics Summary:\n"
        "- Optimal combustion at stoichiometric AFR (~14.7)\n"
        "- Efficiency peaks in safe AFR zone\n"
        "- Lean → high temp → more NOx\n"
        "- Rich → incomplete → more HC\n"
        "- Power & torque → energy conversion from fuel"
    )

    # Create figure
    fig = go.Figure()

    # NOx curve
    fig.add_trace(go.Scatter(
        x=afr, y=nox,
        mode='lines',
        name='🌬 NOx (ppm)',
        line=dict(color='red', width=3)
    ))

    # HC curve
    fig.add_trace(go.Scatter(
        x=afr, y=hc,
        mode='lines',
        name='⛽ HC (ppm)',
        line=dict(color='blue', width=3)
    ))

    # Efficiency curve
    fig.add_trace(go.Scatter(
        x=afr, y=efficiency,
        mode='lines',
        name='⚡ Efficiency (%)',
        line=dict(color='orange', width=3, dash='dot')
    ))

    # Stoichiometric AFR line
    fig.add_trace(go.Scatter(
        x=[14.7, 14.7], y=[0, max(max(nox), max(hc), max(efficiency))],
        mode='lines',
        name='⚖ Stoichiometric AFR',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Safe operating zone (green shaded area)
    fig.add_shape(type="rect", x0=safe_afr_min, x1=safe_afr_max,
                  y0=0, y1=max(max(nox), max(hc), max(efficiency)),
                  fillcolor="green", opacity=0.1, layer="below", line_width=0)

    # Shaded areas for NOx and HC zones
    fig.add_shape(type="rect", x0=12, x1=14.7, y0=0, y1=max(nox),
                  fillcolor="red", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=14.7, x1=17, y0=0, y1=max(hc),
                  fillcolor="blue", opacity=0.1, layer="below", line_width=0)

    # Arrows to highlight zones
    fig.add_annotation(x=13.0, y=max(nox) * 0.8,
                       ax=13.5, ay=max(nox) * 0.9,
                       xref='x', yref='y',
                       text="🔥 High Temp → NOx",
                       showarrow=True, arrowhead=2, arrowcolor="red")
    fig.add_annotation(x=16.0, y=max(hc) * 0.8,
                       ax=15.5, ay=max(hc) * 0.9,
                       xref='x', yref='y',
                       text="⛽ Rich Mixture → HC",
                       showarrow=True, arrowhead=2, arrowcolor="blue")
    fig.add_annotation(x=(safe_afr_min + safe_afr_max) / 2, y=max(max(nox), max(hc), max(efficiency)) * 0.95,
                       text="✅ Safe Operating Zone",
                       showarrow=False,
                       font=dict(color="green", size=14),
                       bgcolor="white", bordercolor="green", borderwidth=1, borderpad=4)

    # Thermodynamics summary annotation below the graph

    # Layout
    fig.update_layout(
        title='Engine Emission & Efficiency vs Air-Fuel Ratio',
        xaxis_title='Air-Fuel Ratio (AFR)',
        yaxis_title='Relative Emission / Efficiency Level',
        template='plotly_white',
        legend=dict(x=0.7, y=0.95),
        margin=dict(t=50, b=150)  # extra space at bottom for annotation
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)














