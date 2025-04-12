import streamlit as st
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="EEG Score Predictor", layout="centered")

# --- LOAD MODEL ---
model = joblib.load("eeg_model.pkl")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
            color: #333;
        }
        h1 {
            color: #1e90ff;
            text-align: center;
        }
        .stNumberInput input {
            background-color: #fff;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
        }
        .stButton>button {
            background-color: #1e90ff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- APP TITLE ---
st.title("🧠 EEG Score Predictor")

# --- INSTRUCTIONS ---
st.markdown("""
This app predicts the **Trial 2 Maths** score based on inputs from **Trial 1** in three tests:

- **Maths**
- **Symmetry**
- **Stroop**

Enter the values for **Trial 1** and click **Predict** to get the predicted **Trial 2 Maths Score**.

**Score Interpretation:**  
- 🟢 **Low Stress**: 0.0 - 1.9  
- 🟡 **Moderate Stress**: 2.0 - 3.9  
- 🔴 **High Stress**: 4.0 and above
""")

# --- INPUTS ---
trial_1_maths = st.number_input("Trial 1 Maths", value=3.0)
trial_1_symmetry = st.number_input("Trial 1 Symmetry", value=2.5)
trial_1_stroop = st.number_input("Trial 1 Stroop", value=3.2)

# --- FEATURES ---
trial_1_avg = (trial_1_maths + trial_1_symmetry + trial_1_stroop) / 3
math_sym_diff = trial_1_maths - trial_1_symmetry

# --- PREDICT BUTTON ---
if st.button("Predict"):
    input_data = np.array([[trial_1_maths, trial_1_symmetry, trial_1_stroop, trial_1_avg, math_sym_diff]])
    prediction = model.predict(input_data)[0]

    st.success(f"🧮 Predicted Trial 2 Maths Score: **{prediction:.2f}**")

    # Score interpretation
    def interpret_score(score):
        if score < 2:
            return "🟢 Low Stress"
        elif 2 <= score < 4:
            return "🟡 Moderate Stress"
        else:
            return "🔴 High Stress"

    st.info(interpret_score(prediction))

           
