# Student Performance Analysis and Prediction

## 1. Project Overview

This project uses data analysis and machine learning to analyze
student academic performance and predict final mathematics grades.

The project uses the Student Performance dataset from the UCI
Machine Learning Repository.

## 2. Project Objectives

- Analyze student academic performance.
- Identify patterns associated with final grades.
- Visualize student performance using charts.
- Build a machine learning model to predict final grades.
- Develop an interactive dashboard for predictions.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Visual Studio Code

## 4. Machine Learning Models

Two regression models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor

The Random Forest model performed better on the test dataset.

### Model Evaluation Results

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 1.334 | 1.038 |
| RMSE | 2.112 | 1.732 |
| R² Score | 0.782 | 0.854 |

## 5. Dashboard Features

- Student information input.
- Final mathematics grade prediction.
- Predicted performance category.
- Grade visualization.
- Model evaluation metrics.
- Project information sidebar.

## 6. Project Structure

data-project/
├── data/
│   └── student-mat.csv
├── notebooks/
│   └── 01_data_loading.ipynb
├── src/
│   └── app.py
├── models/
│   ├── student_performance_model.pkl
│   └── model_features.pkl
├── reports/
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md

## 7. Installation and Setup

Clone or download the project files.

Install the required dependencies:

pip install -r requirements.txt

## 8. Run the Application

Activate the project's virtual environment.

Run the following command from the project root:

streamlit run src/app.py

Open the local address displayed in your terminal.

## 9. How the Prediction Works

The Random Forest Regressor uses the following features:

- Student age
- Weekly study time
- Number of past class failures
- Number of school absences
- First period grade (G1)
- Second period grade (G2)

The model predicts the student's final mathematics grade (G3)
on a scale of 0 to 20.

## 10. Limitations

Predictions are estimates and are not guaranteed to match
actual student grades.

The model was trained on a specific dataset, so its results
may not generalize to every school or student population.

## 11. Conclusion

This project demonstrates data cleaning, exploratory data
analysis, data visualization, machine learning, model evaluation,
and interactive application development using Python.