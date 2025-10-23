import streamlit as st
import numpy as np
import pickle

# Load the trained model
try:
    with open("car_price_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

st.title("🚗 Car Price Prediction App")
st.write("Enter the details below to estimate the car's selling price.")

# Input fields (customize these based on your dataset)
yr_mfr = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2018)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
kms_run = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000)
city = st.text_input("City", "Delhi")
body_type = st.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
model_name = st.text_input("Model Name", "Swift")
total_owners = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=1)
original_price = st.number_input("Original Price (in Lakhs)", min_value=1.0, max_value=200.0, value=10.0)
car_rating = st.slider("Car Rating (out of 5)", 1.0, 5.0, 3.5)

# Convert categorical values to numerics (You MUST adjust this mapping the way your model was trained)
# Example mappings (change these if different in your encoding):
fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "Hybrid": 4}
transmission_mapping = {"Manual": 0, "Automatic": 1}
body_mapping = {"Hatchback": 0, "Sedan": 1, "SUV": 2, "MUV": 3, "Coupe": 4, "Convertible": 5}

# Replace categorical values
fuel_type_encoded = fuel_mapping.get(fuel_type, 0)
transmission_encoded = transmission_mapping.get(transmission, 0)
body_type_encoded = body_mapping.get(body_type, 0)

# NOTE: If city and model_name were encoded using One-Hot Encoding, you must replicate that here.
# For now, let's assume they were label encoded numerically (adjust later if needed):
city_encoded = hash(city) % 1000  # Temporary numeric conversion
model_name_encoded = hash(model_name) % 1000  # Temporary numeric conversion

# Prepare input for prediction
input_data = np.array([[yr_mfr, fuel_type_encoded, kms_run, city_encoded, body_type_encoded,
                        transmission_encoded, model_name_encoded, total_owners, original_price, car_rating]])

if st.button("🔍 Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹{round(prediction, 2)} Lakhs")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
