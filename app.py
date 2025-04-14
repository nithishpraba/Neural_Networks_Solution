import streamlit as st
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.logger import get_logger

logger = get_logger(__name__)

st.title("Neural Network Prediction App")
st.write("Enter the input details to predict the admission chance.")

# Define a custom loss function alias
def mse_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Pass custom_objects to load_model so that "mse" is resolved correctly
custom_objects = {'mse': mse_loss}

model_path = os.path.join("models", "model.h5")
try:
    model = load_model(model_path, custom_objects=custom_objects)
    st.success("Neural network model loaded successfully!")
    logger.info("Model loaded from %s", model_path)
except Exception as e:
    st.error("Failed to load the neural network model. Please run the training script first.")
    logger.error("Error loading model: %s", e)
    st.stop()

# Load the saved scaler for consistent input scaling
scaler_path = os.path.join("models", "scaler.joblib")
try:
    scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded from %s", scaler_path)
except Exception as e:
    st.error("Failed to load the scaler. Please re-run the training script.")
    logger.error("Error loading scaler: %s", e)
    st.stop()

# Define input fields for your features.
# Adjust these fields according to the features used during training.
gre_score = st.number_input("GRE Score", min_value=0.0, value=300.0)
toefl_score = st.number_input("TOEFL Score", min_value=0.0, value=100.0)
university_rating = st.number_input("University Rating", min_value=0.0, value=4.0)
sop = st.number_input("SOP", min_value=0.0, value=4.0)
lor = st.number_input("LOR", min_value=0.0, value=4.0)
cgpa = st.number_input("CGPA", min_value=0.0, value=8.0)
research = st.number_input("Research (0 or 1)", min_value=0, max_value=1, value=1)

if st.button("Predict"):
    try:
        # Create a NumPy array for the input features
        input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
        
        # Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_data)
        
        # Make a prediction using the model
        prediction = model.predict(input_scaled)
        
        st.write("Predicted Admission Chance:", prediction[0][0])
        logger.info("Prediction made with input: %s", input_data.tolist())
    except Exception as e:
        st.error("Error during prediction.")
        logger.error("Prediction error: %s", e)
# Add a footer with the project name and GitHub link
st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 14px;">Project Name: <a href="https://github.com/yourusername/yourproject" target="_blank">Your Project</a></p>
    </div>
""", unsafe_allow_html=True)