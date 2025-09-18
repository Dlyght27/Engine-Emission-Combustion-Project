# 🚗 Predictive Maintenance & Combustion Analysis System

A machine learning-powered system for analyzing and predicting internal combustion engine performance and emissions.

## 📌 Features
- Interactive **Streamlit app** for real-time predictions.
- Collects user inputs for **engine operating parameters**:
  - Water fraction [0–1]
  - Rotational speed (rpm)
  - Compression ratio
  - Fuel & exhaust temperatures
  - Oil temperature
  - Torque (N·m)
  - Fuel consumption (g/s)
  - Power (kW)
  - Specific fuel consumption (g/kWh)
  - Efficiency
  - Emission gases (CO, CO₂, O₂, HC, NOx)
  - Ignition advance angle
- **Automatic transformations** (log, sqrt) for skewed variables.
- Predicts **engine health, performance, and emissions**.
- Configurable input ranges (min, max, step) based on dataset statistics.

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit** – Web interface
- **scikit-learn** – Machine learning
- **pandas, numpy** – Data processing
- **joblib** – Model persistence

## 📂 Project Structure
├── app.py # Streamlit app
├── model.pkl # Saved trained ML model
├── data/ # Dataset (optional, not needed if model is saved)
└── README.md # Project documentation
