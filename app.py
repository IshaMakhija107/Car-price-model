import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.pkl")

st.title("🚗 Car Sale Price Prediction")
st.markdown("Enter car details to predict the sale price:")

# User inputs
yr_mfr = st.number_input("Year of Manufacture", min_value=1990, max_value=2035, value=2015)
kms_run = st.number_input("Kms Run", min_value=0, value=10000)
original_price = st.number_input("Original Price (INR)", min_value=0, value=500000)
total_owners = st.number_input("Total Owners", min_value=0, value=1)
car_rating = st.selectbox("Car Rating", ["excellent", "great", "good", "average", "poor"])
fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "electric", "petrol & CNG"])
city = st.selectbox("City", ["delhi", "mumbai", "bangalore", "hyderabad", "chennai", "pune", "others"])
model_name = st.text_input("Car Model", value="maruti swift")

if st.button("Predict Sale Price"):
    # Preprocess input (numeric + log transformations)
    X_input = pd.DataFrame([{
        'yr_mfr': yr_mfr,
        'kms_run': kms_run,
        'total_owners': total_owners,
        'original_price': original_price,
        'log_kms_run': np.log1p(kms_run),
        'log_original_price': np.log1p(original_price),
        'model_encoded': 0,  # replace with proper encoding if needed
        # add other one-hot / encoded columns if your model uses them
    }])
    
    pred = model.predict(X_input)[0]
    st.success(f"Predicted Sale Price: ₹{round(pred, 2)}")
