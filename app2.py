import streamlit as st
import joblib
import os
import gdown

# Google Drive link (shared by you)
file_url = "https://drive.google.com/uc?id=1Kd7cGy-kJrICWkGPWWgEedny5Ujl5xF7"

# Download model if not already present
if not os.path.exists("rf_model.pkl"):
    st.write("Downloading model from Google Drive...")
    gdown.download(file_url, "rf_model.pkl", quiet=False)

# Try loading the model
try:
    model = joblib.load("rf_model.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
