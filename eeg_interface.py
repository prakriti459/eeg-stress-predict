import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("eeg_model.pkl")

st.set_page_config(page_title="EEG Score Predictor", layout="centered")

st.title("🧠 EEG Score Predictor")
st.markdown("Enter the following values to predict the **Trial 2 Maths** score:")

# Input fields
trial_1_maths = st.number_input("Trial 1 Maths", value=3.0)
trial_1_symmetry = st.number_input("Trial 1 Symmetry", value=2.5)
trial_1_stroop = st.number_input("Trial 1 Stroop", value=3.2)

# Derived features
trial_1_avg = (trial_1_maths + trial_1_symmetry + trial_1_stroop) / 3
math_sym_diff = trial_1_maths - trial_1_symmetry

# Predict button
if st.button("Predict"):
    input_data = np.array([[trial_1_maths, trial_1_symmetry, trial_1_stroop, trial_1_avg, math_sym_diff]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"🧮 Predicted Trial 2 Maths Score: **{prediction:.2f}**")
