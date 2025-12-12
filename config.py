
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

UI_ORDER = [
    "Water fraction [0-1]", "Rotational speed (rpm)", "Compression ratio",
    "Fuel temperature (℃)", "Exhaust temperature (℃)", "Oil temperature (℃)",
    "Torque (N m)", "Fuel consumption (g/s)", "Power (kW)",
    "Specific fuel consumption (g/kWh)", "Efficiency (%)", "CO (%)",
    "CO2 (%)", "O2 (%)", "HC (ppm)", "NOx (ppm)", "Ignition Advance Angle (°BTDC)"
]
