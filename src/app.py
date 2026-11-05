import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Analytics | AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F5F6FC;
}

[data-testid="stSidebar"] {
    background-color: #171B3A;
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #20254A;
}

.subtitle {
    color: #747994;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #E6E8F2;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

div.stButton > button {
    background-color: #6558D3;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #5144B8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "student_performance_model.pkl"
FEATURES_PATH = PROJECT_DIR / "models" / "model_features.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

model, features = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("# 🎓 STUDENT AI")

st.sidebar.caption("ACADEMIC INTELLIGENCE PLATFORM")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Grade Prediction",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### About the Project")

st.sidebar.write(
    "An intelligent academic analytics application "
    "that estimates students' final mathematics grades "
    "using machine learning."
)

st.sidebar.markdown("### Technologies")

st.sidebar.write(
    "🐍 Python\n\n"
    "📊 Pandas\n\n"
    "🤖 Scikit-learn\n\n"
    "🎨 Streamlit"
)

st.sidebar.divider()

st.sidebar.caption("Developed as a Machine Learning Project")

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">Student Performance Dashboard</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Academic insights powered by machine learning.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Model Performance Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mean Absolute Error",
            "1.038",
            help="Average absolute prediction error in grade points."
        )

    with col2:
        st.metric(
            "Root Mean Squared Error",
            "1.732",
            help="Measures prediction error, penalising larger errors."
        )

    with col3:
        st.metric(
            "R² Score",
            "0.854",
            help="Coefficient of determination on the test dataset."
        )

    st.divider()

    st.subheader("About the Prediction System")

    st.write(
        "This system uses a Random Forest Regression model "
        "to estimate students' final mathematics grades "
        "based on academic and attendance information."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Prediction capability**\n\n"
            "Estimate a student's final grade using previous "
            "academic performance and other selected factors."
        )

    with col2:
        st.success(
            "**Analytical objective**\n\n"
            "Explore how study time, previous failures, "
            "absences and earlier grades relate to performance."
        )

    st.warning(
        "Model performance is based on the original test dataset. "
        "Predictions are estimates, not guaranteed outcomes."
    )

# --------------------------------------------------
# GRADE PREDICTION PAGE
# --------------------------------------------------

elif page == "Grade Prediction":

    st.markdown(
        '<p class="main-title">Student Grade Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Enter academic information to generate a prediction.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    with st.form("prediction_form"):

        st.subheader("Student Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Student Age",
                min_value=15,
                max_value=22,
                value=17
            )

            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Less than 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "More than 10 hours"
                }[x]
            )

            failures = st.number_input(
                "Previous Class Failures",
                min_value=0,
                max_value=4,
                value=0
            )

        with col2:

            absences = st.number_input(
                "School Absences",
                min_value=0,
                max_value=100,
                value=0
            )

            g1 = st.number_input(
                "First Period Grade (G1)",
                min_value=0,
                max_value=20,
                value=10
            )

            g2 = st.number_input(
                "Second Period Grade (G2)",
                min_value=0,
                max_value=20,
                value=10
            )

        submitted = st.form_submit_button(
            "✨ Generate Prediction",
            use_container_width=True
        )

    if submitted:

        student_data = pd.DataFrame(
            [[age, studytime, failures, absences, g1, g2]],
            columns=features
        )

        prediction = float(model.predict(student_data)[0])

        prediction = max(0, min(20, prediction))

        st.divider()

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Final Grade",
                f"{prediction:.2f} / 20"
            )

        if prediction >= 16:
            category = "Excellent"
            st.success("🏆 Performance Category: Excellent")

        elif prediction >= 12:
            category = "Good"
            st.success("🌟 Performance Category: Good")

        elif prediction >= 10:
            category = "Average"
            st.warning("📘 Performance Category: Average")

        else:
            category = "Needs Improvement"
            st.error("📌 Performance Category: Needs Improvement")

        chart_data = pd.DataFrame({
            "Grade": ["Predicted Grade", "Remaining Points"],
            "Score": [prediction, 20 - prediction]
        })

        st.subheader("Grade Visualization")

        st.bar_chart(
            chart_data,
            x="Grade",
            y="Score"
        )

        st.progress(prediction / 20)

        st.caption(
            "This is a machine-learning estimate, not a guaranteed "
            "final grade or an assessment of the student's ability."
        )

# --------------------------------------------------
# MODEL INFORMATION PAGE
# --------------------------------------------------

elif page == "Model Information":

    st.markdown(
        '<p class="main-title">Machine Learning Information</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Understanding the intelligence behind the system.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Selected Algorithm")

    st.write("**Random Forest Regressor**")

    st.write(
        "Random Forest is an ensemble learning algorithm "
        "that combines predictions from multiple decision trees "
        "to produce a numerical prediction."
    )

    st.subheader("Input Features")

    st.write("""
    - Age
    - Weekly study time
    - Previous class failures
    - School absences
    - First period grade (G1)
    - Second period grade (G2)
    """)

    st.subheader("Model Evaluation")

    st.write("""
    The model was evaluated using a separate test dataset.

    - MAE: 1.038
    - RMSE: 1.732
    - R²: 0.854
    """)

    st.info(
        "The model was trained using historical student mathematics "
        "performance data. Its predictions may not generalise to "
        "every school or student population."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<center>🎓 Student Analytics | Machine Learning Project</center>",
    unsafe_allow_html=True
)