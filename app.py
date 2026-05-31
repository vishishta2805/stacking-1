import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL & SCALER
# =========================

model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# STREAMLIT UI
# =========================

st.title("Stacking Classification App")

st.subheader("Breast Cancer Prediction")

st.write("Enter feature values below:")

# =========================
# USER INPUTS
# =========================

radius_mean = st.number_input("Radius Mean")
texture_mean = st.number_input("Texture Mean")
perimeter_mean = st.number_input("Perimeter Mean")
area_mean = st.number_input("Area Mean")
smoothness_mean = st.number_input("Smoothness Mean")
compactness_mean = st.number_input("Compactness Mean")
concavity_mean = st.number_input("Concavity Mean")
concave_points_mean = st.number_input("Concave Points Mean")
symmetry_mean = st.number_input("Symmetry Mean")
fractal_dimension_mean = st.number_input("Fractal Dimension Mean")

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    input_data = pd.DataFrame([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean
    ]], columns=[
        'radius_mean',
        'texture_mean',
        'perimeter_mean',
        'area_mean',
        'smoothness_mean',
        'compactness_mean',
        'concavity_mean',
        'concave points_mean',
        'symmetry_mean',
        'fractal_dimension_mean'
    ])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)

    # Probability
    probability = model.predict_proba(scaled_data)

    # =========================
    # OUTPUT
    # =========================

    if prediction[0] == 1:
        st.error("Malignant Cancer Detected")
    else:
        st.success("Benign Cancer Detected")

    st.write("Prediction Probability:")
    st.write(probability)