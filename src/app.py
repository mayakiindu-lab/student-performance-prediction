import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Locate the saved model files
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "student_performance_model.pkl"
FEATURES_PATH = PROJECT_DIR / "models" / "model_features.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# Dashboard title
st.sidebar.title("🎓 Student Analytics")

st.sidebar.write("### About This Project")
st.sidebar.write(
    "This application uses machine learning to predict "
    "students' final mathematics grades."
)

st.sidebar.write("### Technologies Used")
st.sidebar.write("- Python")
st.sidebar.write("- Pandas")
st.sidebar.write("- Scikit-learn")
st.sidebar.write("- Streamlit")

st.sidebar.write("### Model")
st.sidebar.write("Random Forest Regressor")
st.title("🎓 Student Performance Prediction")
st.write(
    "Enter a student's information to predict their final mathematics grade."
)

st.info("The predicted grade is an estimate out of 20.")
st.header("📈 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Mean Absolute Error", "1.038")
col2.metric("Root Mean Squared Error", "1.732")
col3.metric("R² Score", "0.854")

# Student information inputs
st.header("Student Information")

age = st.number_input(
    "Student age",
    min_value=15,
    max_value=22,
    value=17
)

studytime = st.selectbox(
    "Weekly study time",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Less than 2 hours",
        2: "2–5 hours",
        3: "5–10 hours",
        4: "More than 10 hours"
    }[x]
)

failures = st.number_input(
    "Number of past class failures",
    min_value=0,
    max_value=4,
    value=0
)

absences = st.number_input(
    "Number of school absences",
    min_value=0,
    max_value=100,
    value=0
)

g1 = st.number_input(
    "First period grade (G1)",
    min_value=0,
    max_value=20,
    value=10
)

g2 = st.number_input(
    "Second period grade (G2)",
    min_value=0,
    max_value=20,
    value=10
)

# Predict final grade
if st.button("Predict Final Grade"):
    student_data = pd.DataFrame(
        [[age, studytime, failures, absences, g1, g2]],
        columns=features
    )

    prediction = model.predict(student_data)[0]

    # Keep the displayed grade within the expected range
    prediction = max(0, min(20, prediction))

    st.subheader("Prediction Result")
    st.metric("Predicted Final Grade", f"{prediction:.2f} / 20")
    st.subheader("📊 Predicted Grade Visualization")

    chart_data = pd.DataFrame({
        "Grade Type": ["Predicted Grade", "Maximum Grade"],
        "Score": [prediction, 20]
    })

    st.bar_chart(
        chart_data,
        x="Grade Type",
        y="Score",
        y_label="Grade out of 20"
    )

    if prediction >= 16:
        st.success("Performance category: Excellent")
    elif prediction >= 12:
        st.success("Performance category: Good")
    elif prediction >= 10:
        st.warning("Performance category: Average")
    else:
        st.error("Performance category: Needs Improvement")

    st.caption(
        "This prediction is an estimate based on a machine-learning model, "
        "not a guarantee of the student's actual grade."
    )