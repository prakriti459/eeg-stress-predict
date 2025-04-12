import streamlit as st
import joblib
import numpy as np

# Set up Streamlit page config
st.set_page_config(page_title="EEG Score Predictor", layout="centered")

# Apply custom CSS (only for messages and headings)
st.markdown("""
    <style>
        h1 {
            color: #1e90ff;
            text-align: center;
        }

        .stAlert-success {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
        }

        .stAlert-info {
            background-color: #fff3cd;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
model = joblib.load("eeg_model.pkl")

# Title and description
st.title("🧠 EEG Score Predictor")
st.markdown("""
This app predicts the **Trial 2 Maths** score based on inputs from **Trial 1** in three tests:
- **Maths**
- **Symmetry**
- **Stroop**

Enter the values for **Trial 1** and click "Predict" to get the predicted **Trial 2 Maths Score**.

**Score Interpretation**:  
- **Low Stress**: 0.0 - 1.9  
- **Moderate Stress**: 2.0 - 3.9  
- **High Stress**: 4.0 and above
""")

# Input fields (no CSS applied to these)
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

    # Display predicted score
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
