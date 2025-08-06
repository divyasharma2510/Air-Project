import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the data
df = pd.read_csv('Student_Performance.csv')

# Encode categorical column
lb = LabelEncoder()
df['Extracurricular Activities'] = lb.fit_transform(df['Extracurricular Activities'])

# Split features and target
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regression = LinearRegression()
regression.fit(X_train, y_train)

# Evaluation metrics
y_predict = regression.predict(X_test)
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

# Streamlit App Interface
st.title("🎓 Student Performance Prediction App")
st.write("Enter the student details to predict the performance index and eligibility for placement.")

# Input fields
hours_studied = st.slider("Hours Studied (per day)", 0, 12, 6)
attendance = st.slider("Attendance (%)", 0, 100, 75)
extracurricular = st.radio("Participates in Extracurricular Activities?", ['Yes', 'No'])
past_grade = st.slider("Past Exam Grade (out of 10)", 0, 10, 7)
internet_access = st.radio("Has Internet Access?", ['Yes', 'No'])

# Convert categorical inputs
extracurricular_val = 1 if extracurricular == 'Yes' else 0
internet_access_val = 1 if internet_access == 'Yes' else 0

# Make prediction
if st.button("Predict Performance"):
    input_data = np.array([[hours_studied, attendance, extracurricular_val, past_grade, internet_access_val]])
    prediction = regression.predict(input_data)[0]
    st.success(f"📊 Predicted Performance Index: {prediction:.2f}")

    if prediction >= 60:
        st.markdown("✅ *You are eligible for placement.*")
    else:
        st.markdown("❌ *Sorry, you are not eligible for placement.*")

# Show metrics
st.write("---")
st.write("### Model Evaluation Metrics")
st.write(f"📈 R² Score: {r2*100:.2f}%")
st.write(f"📉 Root Mean Squared Error: {rmse:.2f}")