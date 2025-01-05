import streamlit as st
import numpy as np
from joblib import dump, load
import pandas as pd








st.title('Yield Predictor')
# st.write('')
with open('processor.joblib','rb') as f:
    preprocessor=load(f)
with open('pipeline_cat.joblib','rb') as f:
    cat=load(f)    
with open('pipeline_gb.joblib','rb') as f:
    gb=load(f) 
with open('pipeline_lgb.joblib','rb') as f:
    lgb=load(f)  


region=st.selectbox("Region",['North','South','West','East'])
soil=st.selectbox("Type of Soil",['Clay', 'Sandy', 'Loam', 'Silt', 'Peaty', 'Chalky'])
crop=st.selectbox("Crop Type",['Wheat', 'Rice', 'Maize', 'Barley', 'Soybean', 'Cotton'])
rain=st.number_input("Rain in mm")
temp=st.number_input("Temperature(C)")
ferti=st.selectbox("Fertilizer",['False','True'])
irri=st.selectbox("Irrigation",['False','True'])
weather=st.selectbox("Predominant Weather Condition",['Sunny', 'Rainy', 'Cloudy'])
days=st.number_input("Days to Harvest")
def predict_combined(input_data):
    x_transformed = preprocessor.transform(input_data)
    pred_gb = gb.predict(x_transformed)
    pred_lgb = lgb.predict(x_transformed)
    pred_cat = cat.predict(x_transformed)

    final_pred = ( 0.35*pred_gb + 0.34*pred_lgb + 0.31*pred_cat)
    return final_pred

if st.button('Predict Yield'):
    query=np.array([region,soil,crop,rain,temp,ferti,irri,weather,days])
    query=query.reshape(1,9)
    query=pd.DataFrame(query,columns=['Region','Soil_Type','Crop','Rainfall_mm','Temperature_Celsius','Fertilizer_Used','Irrigation_Used','Weather_Condition','Days_to_Harvest'])

    
    st.title("For above data Crop Yield will be " + str(predict_combined(query)[0]) + " tons per hectare")