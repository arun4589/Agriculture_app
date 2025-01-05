import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="centered"
)

# Title and description
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <h1 style="color: #4CAF50;">🌾 Crop Yield Predictor</h1>
        <p style="font-size: 18px; color: #555;">Predict crop yield using advanced machine learning models</p>
    </div>
    """, unsafe_allow_html=True
)

# Load pre-trained models and preprocessors
with open('processor.joblib', 'rb') as f:
    preprocessor = load(f)
with open('pipeline_cat.joblib', 'rb') as f:
    cat = load(f)    
with open('pipeline_gb.joblib', 'rb') as f:
    gb = load(f) 
with open('pipeline_lgb.joblib', 'rb') as f:
    lgb = load(f)  

# Input fields with enhanced layout
st.markdown("### Enter the following details:")
col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("Region", ['North', 'South', 'West', 'East'])
    soil = st.selectbox("Type of Soil", ['Clay', 'Sandy', 'Loam', 'Silt', 'Peaty', 'Chalky'])
    crop = st.selectbox("Crop Type", ['Wheat', 'Rice', 'Maize', 'Barley', 'Soybean', 'Cotton'])
    rain = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
    ferti = st.selectbox("Fertilizer Used", ['False', 'True'])

with col2:
    temp = st.number_input("Temperature (°C)", min_value=-10.0, step=0.5)
    irri = st.selectbox("Irrigation Used", ['False', 'True'])
    weather = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy'])
    days = st.number_input("Days to Harvest", min_value=0, step=1)

# Prediction function
def predict_combined(input_data):
    x_transformed = preprocessor.transform(input_data)
    pred_gb = gb.predict(x_transformed)
    pred_lgb = lgb.predict(x_transformed)
    pred_cat = cat.predict(x_transformed)

    final_pred = (0.35 * pred_gb + 0.34 * pred_lgb + 0.31 * pred_cat)
    return final_pred

# Prediction button
if st.button('Predict Yield'):
    # Prepare input for prediction
    query = np.array([region, soil, crop, rain, temp, ferti, irri, weather, days])
    query = query.reshape(1, 9)
    query = pd.DataFrame(query, columns=['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius', 
                                         'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest'])
    
    # Generate prediction
    prediction = predict_combined(query)[0]

    # Display result
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px; background-color: #F9F9F9; padding: 20px; border-radius: 10px;">
            <h2 style="color: #4CAF50;">🌟 Predicted Yield</h2>
            <p style="font-size: 24px; font-weight: bold;">{prediction:.2f} tons per hectare</p>
        </div>
        """, unsafe_allow_html=True
    )
