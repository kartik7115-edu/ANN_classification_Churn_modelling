import os
from groq import Groq

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Initialize Groq clientfrom groq import Groq

client = Groq(api_key="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")

# ------------------------------
# AI Strategy Generator
# ------------------------------
def generate_strategy(customer_data, risk, prob):

    prompt = f"""
You are a banking customer retention expert.

Customer details:
Age: {customer_data['Age']}
Tenure: {customer_data['Tenure']}
Balance: {customer_data['Balance']}
Products: {customer_data['NumOfProducts']}
Active Member: {customer_data['IsActiveMember']}

Churn probability: {prob:.2f}
Risk Level: {risk}

Suggest 2 specific retention strategies a bank could apply.
Explain briefly why each strategy may reduce churn.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ------------------------------
# Load Model
# ------------------------------
model = tf.keras.models.load_model('model.h5')

# ------------------------------
# Load Encoders
# ------------------------------
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    Scaler = pickle.load(file)


# ------------------------------
# Risk Level Function
# ------------------------------
def risk_level(prob):
    if prob > 0.8:
        return "HIGH"
    elif prob > 0.6:
        return "MEDIUM"
    else:
        return "LOW"


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Customer Churn Prediction AI Agent")

st.subheader("Customer Information")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)

age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)

balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')

num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Churn"):

    # Prepare input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale data
    input_data_scaled = Scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled, verbose=0)
    prediction_proba = prediction[0][0]

    risk = risk_level(prediction_proba)

    # ------------------------------
    # Prediction Output
    # ------------------------------
    st.subheader("Prediction Result")

    st.write(f"Churn Probability: {prediction_proba:.2f}")

    st.progress(float(prediction_proba))

    # Risk Indicator
    st.subheader("Customer Risk Indicator")

    if risk == "HIGH":
        st.error("🔴 HIGH CHURN RISK")

    elif risk == "MEDIUM":
        st.warning("🟡 MEDIUM CHURN RISK")

    else:
        st.success("🟢 LOW CHURN RISK")

    # Basic classification message
    if prediction_proba > 0.5:
        st.error("The customer is likely to churn.")

    else:
        st.success("The customer is not likely to churn.")

    # ------------------------------
    # AI Retention Strategy
    # ------------------------------
    st.subheader("AI Retention Playbook")

    customer_dict = {
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "IsActiveMember": is_active_member
    }

    with st.spinner("AI Agent analyzing customer behavior..."):
        strategy = generate_strategy(customer_dict, risk, prediction_proba)

    st.write(strategy)

    # ------------------------------
    # Business Insight
    # ------------------------------
    st.subheader("Business Insight")

    if prediction_proba > 0.8:
        st.write("This customer has a very high probability of churn. Immediate retention action is recommended.")

    elif prediction_proba > 0.6:
        st.write("This customer shows moderate churn risk. Monitoring and targeted engagement may help retention.")

    else:
        st.write("This customer appears stable with low churn risk.")


st.markdown("**Kartik | Built with ❤️, TensorFlow & Streamlit**")
