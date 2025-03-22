import streamlit as st
import pandas as pd
import joblib  # To load the trained model
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")  # Ensure you have a trained model saved as model.pkl

# Function to make predictions
def predict_delivery_time(product_category, customer_location, shipping_method):
    # Encode categorical inputs (assuming the model requires numerical inputs)
    category_mapping = {"Electronics": 0, "Clothing": 1, "Home & Kitchen": 2}
    location_mapping = {"Urban": 0, "Suburban": 1, "Rural": 2}
    shipping_mapping = {"Standard": 0, "Express": 1, "Same-day": 2}
    
    category = category_mapping.get(product_category, 0)
    location = location_mapping.get(customer_location, 0)
    shipping = shipping_mapping.get(shipping_method, 0)
    
    # Create input array
    input_data = np.array([[category, location, shipping]])
    
    # Predict delivery time
    predicted_time = model.predict(input_data)[0]
    return predicted_time

# Streamlit UI
st.title("Order to Delivery Time Prediction")
st.write("Enter order details to predict the expected delivery time.")

# Input fields
product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home & Kitchen"])
customer_location = st.selectbox("Customer Location", ["Urban", "Suburban", "Rural"])
shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Same-day"])

# Prediction button
if st.button("Predict Delivery Time"):
    result = predict_delivery_time(product_category, customer_location, shipping_method)
    st.success(f"Estimated Delivery Time: {result:.2f} days")
