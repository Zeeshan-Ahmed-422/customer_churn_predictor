import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load your trained model from the pkl file
with open('churn_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.markdown("Enter the customer details below to predict the likelihood of churn.")

# Create input fields for your model's features
with st.form("prediction_form"):
    st.subheader("Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Personal & Demographic Information
        st.markdown("**Personal Information**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Partner = st.selectbox("Partner", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        # Account Information
        st.markdown("**Account Information**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        
        # Service Charges
        st.markdown("**Service Charges**")
        MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
        TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
        
    with col2:
        # Phone Services
        st.markdown("**Phone Services**")
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
        # Internet Services
        st.markdown("**Internet Services**")
        InternetService = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
        
        # Additional Services
        st.markdown("**Additional Services**")
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        # Contract & Payment
        st.markdown("**Contract & Payment**")
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", 
            "Credit card (automatic)", 
            "Electronic check", 
            "Mailed check"
        ])
    
    submitted = st.form_submit_button("Predict Churn")

# Process the form when submitted
if submitted:
    # Convert categorical inputs to numerical values
    gender_encoded = 1 if gender == "Male" else 0
    SeniorCitizen_encoded = 1 if SeniorCitizen == "Yes" else 0
    Partner_encoded = 1 if Partner == "Yes" else 0
    Dependents_encoded = 1 if Dependents == "Yes" else 0
    PhoneService_encoded = 1 if PhoneService == "Yes" else 0
    PaperlessBilling_encoded = 1 if PaperlessBilling == "Yes" else 0
    
    # Encode MultipleLines
    if MultipleLines == "No":
        MultipleLines_encoded = 0
    elif MultipleLines == "Yes":
        MultipleLines_encoded = 1
    else:  # "No phone service"
        MultipleLines_encoded = 2
    
    # Encode service features with "No internet service" option
    def encode_service(feature_value):
        if feature_value == "No":
            return 0
        elif feature_value == "Yes":
            return 1
        else:  # "No internet service"
            return 2
    
    OnlineSecurity_encoded = encode_service(OnlineSecurity)
    OnlineBackup_encoded = encode_service(OnlineBackup)
    DeviceProtection_encoded = encode_service(DeviceProtection)
    TechSupport_encoded = encode_service(TechSupport)
    StreamingTV_encoded = encode_service(StreamingTV)
    StreamingMovies_encoded = encode_service(StreamingMovies)
    
    # Encode Internet Service (one-hot encoded in your model)
    InternetService_DSL = 1 if InternetService == "DSL" else 0
    InternetService_Fiber_optic = 1 if InternetService == "Fiber optic" else 0
    InternetService_No = 1 if InternetService == "No" else 0
    
    # Encode Contract (one-hot encoded in your model)
    Contract_Month_to_month = 1 if Contract == "Month-to-month" else 0
    Contract_One_year = 1 if Contract == "One year" else 0
    Contract_Two_year = 1 if Contract == "Two year" else 0
    
    # Encode Payment Method (one-hot encoded in your model)
    PaymentMethod_Bank_transfer = 1 if PaymentMethod == "Bank transfer (automatic)" else 0
    PaymentMethod_Credit_card = 1 if PaymentMethod == "Credit card (automatic)" else 0
    PaymentMethod_Electronic_check = 1 if PaymentMethod == "Electronic check" else 0
    PaymentMethod_Mailed_check = 1 if PaymentMethod == "Mailed check" else 0
    
    # Create input array in the EXACT order of your model's features
    input_data = pd.DataFrame([[
        # Original features (0-16)
        gender_encoded, SeniorCitizen_encoded, Partner_encoded, Dependents_encoded,
        tenure, PhoneService_encoded, MultipleLines_encoded, OnlineSecurity_encoded,
        OnlineBackup_encoded, DeviceProtection_encoded, TechSupport_encoded,
        StreamingTV_encoded, StreamingMovies_encoded, PaperlessBilling_encoded,
        MonthlyCharges, TotalCharges,
        
        # One-hot encoded Internet Service features (17-19)
        InternetService_DSL, InternetService_Fiber_optic, InternetService_No,
        
        # One-hot encoded Payment Method features (20-23)
        PaymentMethod_Bank_transfer, PaymentMethod_Credit_card, 
        PaymentMethod_Electronic_check, PaymentMethod_Mailed_check,
        
        # One-hot encoded Contract features (24-26)
        Contract_Month_to_month, Contract_One_year, Contract_Two_year
    ]])
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 High Churn Risk: {prediction_proba[1]:.2%} probability")
            st.warning("This customer is likely to leave! Take immediate action.")
        else:
            st.success(f"✅ Low Churn Risk: {prediction_proba[0]:.2%} probability")
            st.info("This customer is likely to stay with us.")
            
        # Show probability gauge
        col1, col2, col3 = st.columns(3)
        with col2:
            st.metric("Churn Probability", f"{prediction_proba[1]:.2%}")
        
        # Show feature importance insights
        st.subheader("📈 Key Factors Influencing This Prediction")
        st.write("Based on churn analysis, these factors significantly impact customer retention:")
        
        factor_col1, factor_col2 = st.columns(2)
        with factor_col1:
            if Contract_Month_to_month == 1:
                st.write("• **Contract Type**: Month-to-month contracts have higher churn risk")
            if InternetService_Fiber_optic == 1:
                st.write("• **Internet Service**: Fiber optic users tend to churn more")
            if TechSupport_encoded == 0:
                st.write("• **Tech Support**: Lack of tech support increases churn risk")
                
        with factor_col2:
            if tenure < 12:
                st.write("• **Tenure**: New customers (under 12 months) have higher churn risk")
            if OnlineSecurity_encoded == 0:
                st.write("• **Online Security**: No online security increases churn risk")
            if MonthlyCharges > 70:
                st.write("• **Monthly Charges**: Higher monthly charges may increase churn risk")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check that all features are correctly configured.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ and courage")