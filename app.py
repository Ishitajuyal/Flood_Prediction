import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load("best_flood_prediction_model.pkl")

st.title("Flood Prediction App")
st.write("Enter parameters to check flood risk:")

# Inputs from user
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 150.0)
river_discharge = st.number_input("River Discharge (m³/s)", 0.0, 5000.0, 2000.0)
water_level = st.number_input("Water Level (m)", 0.0, 10.0, 5.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)

# Default values for the remaining 9 features (replace with meaningful defaults if you have them)
default_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 9 zeros to make total 13 features

# Combine all features into a single row
all_features = [rainfall, river_discharge, water_level, humidity] + default_values

# Column names (must match exactly with training data)
columns = ['Rainfall (mm)', 'River Discharge (m³/s)', 'Water Level (m)', 'Humidity (%)',
           'Feature5','Feature6','Feature7','Feature8','Feature9','Feature10','Feature11','Feature12','Feature13']

input_data = pd.DataFrame([all_features], columns=columns)

# Prediction
if st.button("Predict Flood Risk"):
    pred = model.predict(input_data)[0]
    st.write("🔍 Prediction Result:")
    if pred == 1:
        st.error("Flood Likely to Occur!")
    else:
        st.success("No Flood Risk Detected.")
