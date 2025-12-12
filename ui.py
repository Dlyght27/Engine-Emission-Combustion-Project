
import streamlit as st
import base64
import os
import plotly.graph_objects as go
import numpy as np
from config import UI_ORDER, INPUT_CONFIG

def load_video(video_path):
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            encoded_video = base64.b64encode(video_bytes).decode()
        return encoded_video
    return None

def display_video(encoded_video):
    st.markdown(
        f'''
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
        ''',
        unsafe_allow_html=True,
    )

def create_sidebar():
    with st.sidebar:
        st.header("⚙️ Input Parameters")
        input_data = {}
        for feature in UI_ORDER:
            cfg = INPUT_CONFIG[feature]
            input_data[feature] = st.number_input(
                f"{feature}", min_value=cfg["min"], max_value=cfg["max"],
                value=cfg["default"], step=cfg["step"]
            )
    return input_data

def display_predictions(em_pred, q_pred):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌫️ Emission Category", em_pred)
    with col2:
        st.metric("🔥 Combustion Quality", q_pred)

    if em_pred == "Low":
        st.success("✅ Emissions are low — combustion temp & air-fuel ratio near optimal.")
    elif em_pred == "Medium":
        st.warning("⚠️ Moderate emissions — consider tuning ignition timing, air/fuel ratio, or cooling.")
    else:
        st.error("❌ High NOx/HC — check ignition timing, EGR, water injection, or cooling.")

    if q_pred == "High":
        st.success("🔥 Combustion quality high — efficient energy conversion, minimal thermodynamic losses.")
    elif q_pred == "Medium":
        st.warning("⚡ Combustion quality average — slight tuning may improve efficiency.")
    else:
        st.error("💨 Combustion quality low — misfires, rich/lean mixtures, or timing issues reducing efficiency.")

def display_insights():
    with st.expander("💡 View Thermodynamic Insights"):
        st.markdown(
            '''
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
        '''
        )

def display_project_overview():
    st.info('''
**📊 Project Overview:**  
Monitors key engine parameters (Fuel, Exhaust, Oil temp, Torque, Power, Fuel consumption, Emissions)  
Predicts **combustion quality** & **emission levels** using thermodynamic principles (energy conversion efficiency, heat release, mixture optimization).  
💡 Goal: Improve efficiency, reduce pollutants, provide actionable tuning insights.
''')

def create_afr_graph():
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

    # Layout
    fig.update_layout(
        title='Engine Emission & Efficiency vs Air-Fuel Ratio',
        xaxis_title='Air-Fuel Ratio (AFR)',
        yaxis_title='Relative Emission / Efficiency Level',
        template='plotly_white',
        legend=dict(x=0.7, y=0.95),
        margin=dict(t=50, b=150)  # extra space at bottom for annotation
    )
    
    return fig
