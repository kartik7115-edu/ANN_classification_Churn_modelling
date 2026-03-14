import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import pandas as pd
import pickle

## Load the model
model = tf.keras.models.load_model('model.h5')

## Load Encoders and Scalers
with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open ('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open ('Scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction AI Agent')

## User Inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

## Prepare input data
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

## One hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

## Combine
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale
input_data_scaled = Scaler.transform(input_data)

## Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.subheader("Prediction Result")
st.write(f'Churn/Exit Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')

# -------------------------------
# AI AGENT LOGIC
# -------------------------------

def risk_level(prob):
    if prob > 0.8:
        return "HIGH"
    elif prob > 0.6:
        return "MEDIUM"
    else:
        return "LOW"

def retention_strategy(prob, tenure, balance):
    if prob > 0.8:
        return "Offer 25% retention discount and assign customer success manager."
    elif prob > 0.6:
        return "Provide loyalty rewards or service upgrade."
    elif prob > 0.4:
        return "Send engagement email and promotional offers."
    else:
        return "No action required."

def explain_churn(age, tenure, balance):
    reasons = []

    if tenure < 3:
        reasons.append("Customer tenure is very low.")

    if balance > 100000:
        reasons.append("Customer has high account balance.")

    if age > 60:
        reasons.append("Customer belongs to higher age group.")

    return reasons

# Agent outputs
risk = risk_level(prediction_proba)
action = retention_strategy(prediction_proba, tenure, balance)
reasons = explain_churn(age, tenure, balance)

st.subheader("AI Agent Decision")

st.write(f"Risk Level: **{risk}**")
st.write(f"Recommended Action: {action}")

if reasons:
    st.write("Possible Reasons for Churn:")
    for r in reasons:
        st.write(f"- {r}")

st.markdown("**Kartik | Built with ❤️, TensorFlow & Streamlit**")