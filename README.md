# Heart Disease Risk Prediction System

This project is a machine learning–based web application that predicts the risk of heart disease using clinical patient data.  
It demonstrates an end-to-end ML workflow — from data preprocessing and model training to deployment using Streamlit.

The goal of the project is to support early risk assessment by providing a simple, interpretable, and user-friendly prediction system.

---

## Problem Statement
Can we predict whether a person is at risk of heart disease based on common health metrics such as age, cholesterol levels, blood pressure, heart rate, and ECG-related indicators?

Early prediction can help in preventive care and better clinical decision-making.

---

## Dataset
- **Source:** UCI Heart Disease Dataset (via Kaggle)
- **Records:** 297
- **Target Variable:**  
  - `1` → Presence of heart disease  
  - `0` → No heart disease

---

## Approach

### 1. Data Preprocessing
- Verified data quality (no missing or duplicate values)
- One-hot encoded categorical features
- Scaled numerical features using `StandardScaler`
- Performed stratified train-test split (80/20)

### 2. Exploratory Data Analysis (EDA)
- Analyzed feature distributions and correlations
- Identified medically relevant indicators such as:
  - ST depression (`oldpeak`)
  - Maximum heart rate (`thalach`)
  - Chest pain type

### 3. Feature Engineering
- Created medically meaningful features:
  - Risk score
  - Heart rate reserve
  - Cholesterol-to-age ratio
- Selected top features using feature importance analysis

### 4. Model Training & Evaluation
Trained and evaluated multiple classification models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

Models were evaluated using:
- Recall
- Precision
- F1-score
- ROC–AUC

Logistic Regression was selected for its balance of performance and interpretability.

### 5. Threshold Optimization
- Tuned the decision threshold to reduce false negatives
- Prioritized recall due to the medical nature of the problem

---

## Deployment
- Built an interactive web application using **Streamlit**
- Users can input health parameters through a simple UI
- The app displays:
  - Prediction result
  - Probability score
  - Risk level
- Preprocessing artifacts and model are safely loaded to avoid feature mismatch errors

---

## Tech Stack
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
- **Web Framework:** Streamlit
- **Model Persistence:** Joblib

---

## Project Structure
├── app.py
├── best_model_tuned.joblib
├── scaler.joblib
├── feature_columns.joblib
├── requirements.txt
├── README.md


---

## Key Takeaways
- Focused on **interpretability and reliability**, not just accuracy
- Demonstrates a complete ML lifecycle
- Designed with real-world healthcare constraints in mind

---

## Disclaimer
This project is for educational purposes only and should not be used as a substitute for professional medical advice or diagnosis.
