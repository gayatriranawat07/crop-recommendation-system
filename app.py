import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Crop Recommendation System")

# Load trained model
model = joblib.load("crop_recommendation_model.pkl")

st.title("🌱 Crop Recommendation System")
st.write("Enter soil and climate details to get crop recommendations")

# User inputs
N = st.number_input("Nitrogen (N)", 0.0, 200.0)
P = st.number_input("Phosphorus (P)", 0.0, 200.0)
K = st.number_input("Potassium (K)", 0.0, 200.0)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
ph = st.number_input("pH Value", 0.0, 14.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)

# Prediction
if st.button("🌾 Recommend Crop"):
    input_data = pd.DataFrame({
        'Nitrogen': [N],
        'Phosphorus': [P],
        'Potassium': [K],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'pH_Value': [ph],
        'Rainfall': [rainfall]
    })

    probabilities = model.predict_proba(input_data)[0]
    top3_idx = probabilities.argsort()[-3:][::-1]
    top3_crops = model.classes_[top3_idx]

    st.success("Top 3 Recommended Crops:")
    for crop, prob in zip(top3_crops, probabilities[top3_idx]):
        st.write(f"🌱 **{crop}** → {prob*100:.2f}% confidence")
