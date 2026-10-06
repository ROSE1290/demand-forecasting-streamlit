import streamlit as st
import pandas as pd
import numpy as np
import pickle


# --------------------------------------------------
# Load trained model and label encoders
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, label_encoders = load_artifacts()


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.title("Demand Forecasting App")

st.divider()

st.header("Input Features")


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

Category = st.selectbox(
    "Category",
    label_encoders["Category"].classes_.tolist()
)

Inventory_Level = st.number_input(
    "Inventory Level",
    min_value=0,
    value=100
)

Price = st.number_input(
    "Price",
    min_value=0.0,
    value=50.0
)

Discount = st.number_input(
    "Discount (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

Promotion = st.selectbox(
    "Promotion",
    [0, 1]
)

Competitor_Pricing = st.number_input(
    "Competitor Pricing",
    min_value=0.0,
    value=50.0
)


# --------------------------------------------------
# Create input DataFrame
# IMPORTANT: Same columns and order as Xtrain
# --------------------------------------------------

input_data = pd.DataFrame({
    "Category": [Category],
    "Inventory Level": [Inventory_Level],
    "Price": [Price],
    "Discount": [Discount],
    "Promotion": [Promotion],
    "Competitor Pricing": [Competitor_Pricing]
})


# --------------------------------------------------
# Convert categorical values to numbers
# --------------------------------------------------

for col, encoder in label_encoders.items():

    if col in input_data.columns:

        input_data[col] = encoder.transform(
            input_data[col]
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("Predict Demand"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Demand: {int(prediction)} Units"
    )