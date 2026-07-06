# Import required libraries

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("customer_churn_model.pkl")

# Load feature names
feature_columns = joblib.load("feature_columns.pkl")

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction",layout="wide")

# Title
st.title("Customer Churn Prediction System")
st.caption("Machine Learning based Customer Churn Prediction using Logistic Regression")
st.caption("Predicting customer churn using machine learning model trained on IBM Telco dataset")

st.write("Enter customer details to predict whether the customer is likely to churn.")

st.markdown("---")

# Sidebar
st.sidebar.header("Project Information")

st.sidebar.caption("""**Model:** Logistic Regression

**Dataset:** IBM Telco Customer Churn

**Accuracy:** 82%

**Type:** Classification""")
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Gauranshi Bhatt")

st.header("Customer Information")

st.write("Please enter the customer details below.")

col1, col2 = st.columns(2, gap="large")

with col1:

    gender = st.selectbox("Gender",["Male", "Female"])

    senior = st.selectbox("Senior Citizen",[0, 1])

    partner = st.selectbox("Partner",["Yes", "No"])

    dependents = st.selectbox("Dependents",["Yes", "No"])

    tenure = st.number_input("Tenure (Months)",min_value=0,max_value=72,value=12)

    phone = st.selectbox("Phone Service",["Yes", "No"])

    multiline = st.selectbox("Multiple Lines",["Yes", "No", "No phone service"])

    internet = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])

    security = st.selectbox("Online Security",["Yes", "No", "No internet service"])

with col2:

    backup = st.selectbox("Online Backup",["Yes", "No", "No internet service"])

    protection = st.selectbox("Device Protection",["Yes", "No", "No internet service"])

    support = st.selectbox("Tech Support",["Yes", "No", "No internet service"])

    tv = st.selectbox("Streaming TV",["Yes", "No", "No internet service"])

    movies = st.selectbox("Streaming Movies",["Yes", "No", "No internet service"])

    contract = st.selectbox("Contract",["Month-to-month", "One year", "Two year"])

    paperless = st.selectbox("Paperless Billing",["Yes", "No"])

    payment = st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

    monthly = st.number_input("Monthly Charges", min_value=0.0,value=50.0)   

    total = st.number_input("Total Charges",min_value=0.0,value=500.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    predict = st.button("Predict Churn", use_container_width=True)

if predict:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiline,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    raw_df = pd.DataFrame([customer_data])

    st.subheader("Input Data (Raw)")
    st.dataframe(raw_df)

    processed_df = pd.get_dummies(raw_df, drop_first=True)

    st.subheader("Processed Data (After Encoding)")
    st.dataframe(processed_df)

    for col in feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    # Prediction
    prediction = model.predict(processed_df)[0]
    probability = model.predict_proba(processed_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(" Customer is likely to Churn")
    else:
        st.success(" Customer is likely to Stay")

    st.info(f"Churn Probability: {probability:.2f}")
    st.progress(float(probability))
    
    