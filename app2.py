import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown

# --- STEP 1: DOWNLOAD & LOAD MODEL FROM GOOGLE DRIVE ---
file_id = "1Kd7cGy-kJrICWkGPW9Egndy5UjI5xF7"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "car_price_model.pkl"

try:
    gdown.download(url, model_path, quiet=False)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# --- STEP 2: APP TITLE ---
st.title("🚗 Car Price Prediction App")

# --- STEP 3: CITY DROPDOWN ---
city_list = [
    "noida", "gurgaon", "bengaluru", "new delhi", "mumbai", 
    "pune", "hyderabad", "chennai", "kolkata", 
    "ahmedabad", "faridabad", "ghaziabad", "lucknow"
]

# --- STEP 4: USER INPUT FORM ---
yr_mfr = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
kms_run = st.number_input("Kilometers Driven", min_value=0, step=500)
city = st.selectbox("City", city_list)
body_type = st.selectbox("Body Type", ["SUV", "Sedan", "Hatchback", "MUV", "Coupe"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
model_name = st.text_input("Car Model Name (e.g., Swift, Creta, Nexon)")
total_owners = st.number_input("Number of Previous Owners", min_value=0, max_value=10, step=1)
original_price = st.number_input("Original Price (in lakhs)", min_value=1.0, step=0.5)
car_rating = st.slider("Car Rating (1 to 5)", 1.0, 5.0, 3.0)

# --- STEP 5: PREDICTION ---
if st.button("Predict Selling Price"):
    try:
        input_data = pd.DataFrame({
            "yr_mfr": [yr_mfr],
            "fuel_type": [fuel_type],
            "kms_run": [kms_run],
            "city": [city],
            "body_type": [body_type],
            "transmission": [transmission],
            "model": [model_name],
            "total_owners": [total_owners],
            "original_price": [original_price],
            "car_rating": [car_rating]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Selling Price: ₹ {round(prediction, 2)} lakhs")

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

