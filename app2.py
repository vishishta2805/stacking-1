import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL & SCALER
# =========================

model = joblib.load(
    "stacking_regression_model.pkl"
)

scaler = joblib.load(
    "regression_scaler.pkl"
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠"
)

# =========================
# TITLE
# =========================

st.title("🏠 Housing Price Prediction")

st.subheader(
    "Stacking Regression Model"
)

st.write(
    "Enter housing details below:"
)

# =========================
# USER INPUTS
# =========================

longitude = st.number_input(
    "Longitude",
    value=-122.23
)

latitude = st.number_input(
    "Latitude",
    value=37.88
)

housing_median_age = st.number_input(
    "Housing Median Age",
    min_value=1.0,
    value=41.0
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1.0,
    value=880.0
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1.0,
    value=129.0
)

population = st.number_input(
    "Population",
    min_value=1.0,
    value=322.0
)

households = st.number_input(
    "Households",
    min_value=1.0,
    value=126.0
)

median_income = st.number_input(
    "Median Income",
    min_value=0.0,
    value=8.3252
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Price"):

    # Base input dictionary
    input_dict = {

        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,

        # Dummy columns
        'ocean_proximity_INLAND': 0,
        'ocean_proximity_ISLAND': 0,
        'ocean_proximity_NEAR BAY': 0,
        'ocean_proximity_NEAR OCEAN': 0
    }

    # Set selected category to 1
    input_dict[f'ocean_proximity_{ocean_proximity}'] = 1

    # Convert to DataFrame
    input_data = pd.DataFrame([input_dict])

    # =========================
    # SCALE INPUT
    # =========================

    scaled_data = scaler.transform(
        input_data
    )

    # =========================
    # PREDICTION
    # =========================

    prediction = model.predict(
        scaled_data
    )

    predicted_price = prediction[0]

    # =========================
    # OUTPUT
    # =========================

    st.success(
        f"Predicted House Price: ${predicted_price:,.2f}"
    )

    st.write(
        "Prediction generated using Stacking Regression."
    )