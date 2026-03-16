from groq import Groq

client = Groq(api_key= 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)


import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import pandas as pd
import pickle

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

Suggest a retention strategy for this customer.
Explain briefly why this strategy may reduce churn.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content



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

def risk_level(prob):
    if prediction_proba  > 0.8:
        return "HIGH"
    elif prediction_proba  > 0.6:
        return "MEDIUM"
    else:
        return "LOW"

st.subheader("AI Agent Recommendation")

customer_dict = input_data.iloc[0].to_dict()

strategy = generate_strategy(customer_dict, risk, prediction_proba)

st.write(strategy)

st.markdown("**Kartik | Built with ❤️, TensorFlow & Streamlit**")