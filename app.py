import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st


# Loading training model
model = load_model("./model.keras")

# Loading encodrs and scalers
with open("./gender_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)
    
with open("./ohe_geo.pkl", "rb") as file:
    ohe_geo = pickle.load(file)

with open("./scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
    
# Streamlit App
st.title("Customer Churn Prediction App")
geography = st.selectbox("Geography", ohe_geo.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 6)
has_crdt_card = st.selectbox("Has Credit Card", ["Yes", "No"])
has_credit_card = 1 if has_crdt_card == "Yes" else 0
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
is_active = 1 if is_active_member == "Yes" else 0


input_data = ({
"CreditScore": [credit_score],
"Gender": [gender_encoder.transform([gender])[0]],
"Age":[age],
"Tenure" :[tenure],          
"Balance":[balance] ,
"NumOfProducts":[num_of_products],    
"HasCrCard":[has_credit_card],
"IsActiveMember":[is_active],
"EstimatedSalary":[estimated_salary] 
})

geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Convert input_data dictionary to a DataFrame
input_data_df = pd.DataFrame(input_data)

# Concatenate input_data_df with geo_encoded_df
input_data = pd.concat([input_data_df, geo_encoded_df], axis=1)

# Scaling the data
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

st.write(f"Exit Probability: {prediction[0][0]:.2f}")

if prediction[0][0] > 0.5:
    st.write("The customer is likely to exit")
else:
    st.write("The customer is not likely to exit")
