import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
model_path = os.path.join("..", "model", "student_model.pkl")
model = joblib.load(model_path)
print("Model loaded successfully:", model)

st.title("🎓 Student Performance Predictor")

# Input fields
study_hours = st.slider("Study Hours per Day", 0, 12, 6)
attendance = st.slider("Attendance (%)", 0, 100, 75)
assignment_score = st.slider("Assignment Score", 0, 100, 70)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([[study_hours, attendance, assignment_score]],
                            columns=["StudyHours", "attendance", "PreviousScore"])
    print("Input to model:", input_df)
    print(f"Study Hours: {study_hours}")
    print(f"Attendance: {attendance}%")
    print(f"Previous Score: {assignment_score}%")
    prediction = model.predict(input_df)[0]
    print("Prediction result:", prediction)
    
    # Display input values
    st.subheader("📊 Input Values:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Study Hours", f"{study_hours} hours/day")
    with col2:
        st.metric("Attendance", f"{attendance}%")
    with col3:
        st.metric("Previous Score", f"{assignment_score}%")
    
    # Display prediction result
    st.subheader("🎯 Prediction Result:")
    if prediction == 1:
        st.success("✅ The student is likely to PASS.")
    else:
        st.error("❌ The student is likely to FAIL.")