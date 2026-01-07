# Customer Churn Prediction using Artificial Neural Networks (ANN)

This project builds an end-to-end machine learning pipeline to predict whether a bank customer is likely to **churn (exit)**.  
It covers data cleaning, feature engineering, model training, evaluation, and deployment through an interactive Streamlit application.

---

## 1 Project Overview

Customer churn is a critical problem for financial institutions. Accurately identifying customers at risk enables better retention strategies and improved business outcomes.

This project demonstrates:

- End-to-end ML workflow
- Practical feature preprocessing
- ANN-based binary classification
- Deployment for real-time predictions

---

## 2 Dataset Summary

Each record represents a customer with features such as:

- Geography, Gender  
- Age, Tenure  
- Balance, Estimated Salary  
- Credit Score  
- Number of Products  
- Has Credit Card  
- Is Active Member  

**Target variable:**  
`Exited` → `0` (stays) or `1` (churns)

---

## 3 Model Architecture

The predictive model is built using **TensorFlow/Keras**:

- Input layer (processed numeric + encoded categorical variables)
- Two hidden layers with ReLU activation
- Output layer with Sigmoid activation

**Training details:**

- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Regular evaluation on validation data  
- Early stopping to prevent overfitting  

---

## 4 Evaluation Metrics

Performance is assessed using:

- Accuracy  
- Precision and Recall  
- Confusion Matrix  
- Churn probability outputs

---

## 5 Streamlit Application

The web app allows users to input customer details and receive:

- Predicted churn probability
- Clear interpretation of the result

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
