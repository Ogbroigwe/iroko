import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained model
with open("logreg_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load image
image = Image.open("churn_logo.png")

# Set app title and image
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.image(image, use_column_width=True)
st.title("🔮 Churn Prediction App")
st.markdown("Predict whether a customer is likely to **churn** based on their features.")

# User input form
with st.form("churn_form"):
    st.subheader("📋 Enter Customer Details")

    charge_ratio = st.number_input("Charge Ratio", min_value=0.0, max_value=1.0, step=0.00)
    internet_fiber = st.selectbox("Internet Service - Fiber Optic?", ["No", "Yes"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    tenure = st.slider("Tenure (months)", min_value=0.0, max_value=1.0, step=0.01)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_echeck = st.selectbox("Payment Method - Electronic Check?", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    online_security = st.selectbox("Online Security?", ["No", "Yes"])

    submit = st.form_submit_button("Predict Churn")

# Feature transformation
if submit:
    features = [
        charge_ratio,
        1 if internet_fiber == "Yes" else 0,
        monthly_charges,
        tenure,
        1 if contract == "One year" else 0,
        1 if contract == "Two year" else 0,
        1 if payment_echeck == "Yes" else 0,
        1 if multiple_lines == "Yes" else 0,
        1 if gender == "Male" else 0,
        1 if online_security == "Yes" else 0
    ]

    prediction = model.predict([features])[0]
    prob = model.predict_proba([features])[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to **Churn** (Probability: {prob:.2%})")
    else:
        st.success(f"✅ The customer is likely to **Stay** (Probability: {prob:.2%})")