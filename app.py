import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/battery_voltage_model.joblib"
METADATA_PATH = "models/metadata.json"

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

st.set_page_config(page_title="Battery Voltage Prediction", page_icon="🔋")
st.title("🔋 Battery Voltage Prediction")
st.write("Predict `Voltage_load` from battery measurements using the trained machine-learning pipeline.")

st.sidebar.header("Battery measurements")

voltage_measured = st.sidebar.number_input("Voltage measured", value=4.0, format="%.6f")
current_measured = st.sidebar.number_input("Current measured", value=-1.0, format="%.6f")
temperature_measured = st.sidebar.number_input("Temperature measured", value=24.0, format="%.6f")
current_load = st.sidebar.number_input("Current load", value=1.0, format="%.6f")
time_value = st.sidebar.number_input("Time", value=100.0, min_value=0.0, format="%.3f")

if st.button("Predict Voltage Load"):
    X = pd.DataFrame([{
        "Voltage_measured": voltage_measured,
        "Current_measured": current_measured,
        "Temperature_measured": temperature_measured,
        "Current_load": current_load,
        "Time": time_value,
    }])
    prediction = float(model.predict(X)[0])
    st.metric("Predicted Voltage_load", f"{prediction:.4f} V")
    st.caption(f"Model: {metadata['model']}")

st.divider()
st.subheader("Model information")
st.write(f"**Features:** {', '.join(metadata['features'])}")
st.write(f"**Target:** {metadata['target']}")
st.write(f"**Validation split:** {metadata['split']}")
