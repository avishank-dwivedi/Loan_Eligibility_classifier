# 🏦 Loan Eligibility Predictor using Machine Learning

This project predicts whether a loan application should be **approved or rejected** based on applicant information such as income, employment status, credit history, and more. It uses machine learning (Logistic Regression) trained on real-world loan data.

---

## 🚀 Features

- Cleaned and preprocessed applicant data
- Label encoding for categorical features
- Trained with Logistic Regression model
- 80–85% accuracy achieved
- Real-time prediction using a Streamlit Web App

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- NumPy

---

## 📁 Dataset

We used the public [Loan Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

It includes fields like:

- Gender, Married, Dependents  
- Education, Self_Employed  
- ApplicantIncome, CoapplicantIncome  
- LoanAmount, Loan_Amount_Term  
- Credit_History, Property_Area  
- **Loan_Status (Target)**

---

## 🧹 Data Preprocessing

- Filling missing values (mode/median)
- Label encoding categorical variables
- Train-test spl
