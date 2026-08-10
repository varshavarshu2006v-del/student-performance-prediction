import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict a student's final score using academic details.")

# Sample training data
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 90, 95, 98],
    "PreviousMarks": [40, 45, 50, 55, 60, 65, 70, 75, 82, 90],
    "AssignmentMarks": [35, 40, 45, 50, 55, 60, 65, 70, 80, 90],
    "FinalScore": [42, 47, 52, 57, 62, 67, 72, 78, 85, 93]
}

df = pd.DataFrame(data)

# Features and target
X = df[[
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "AssignmentMarks"
]]

y = df["FinalScore"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_marks = st.number_input(
    "Previous Exam Marks (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

assignment_marks = st.number_input(
    "Assignment Marks (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

if st.button("🔮 Predict Performance"):

    input_data = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousMarks": previous_marks,
        "AssignmentMarks": assignment_marks
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(
        f"🎯 Predicted Final Score: {prediction:.2f}%"
    )

    if prediction >= 75:
        st.info("🌟 Performance Level: Excellent")
    elif prediction >= 50:
        st.info("👍 Performance Level: Good")
    else:
        st.warning("📚 Performance Level: Needs Improvement")