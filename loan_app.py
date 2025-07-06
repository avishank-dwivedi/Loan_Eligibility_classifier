import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    'Credit_Score': [720, 680, 690, 710, 650, 600, 720, 690, 630, 750],
    'Annual_Income': [80000, 50000, 60000, 75000, 40000, 35000, 90000, 62000, 30000, 95000],
    'Years_in_Job': [5, 3, 4, 6, 2, 1, 7, 4, 1, 8],
    'Home_Ownership': ['Mortgage', 'Rent', 'Own', 'Mortgage', 'Rent', 'Rent', 'Own', 'Own', 'Rent', 'Mortgage'],
    'Loan_Purpose': ['Debt Consolidation', 'Home Improvement', 'Education', 'Major Purchase', 'Debt Consolidation',
                     'Small Business', 'Home Improvement', 'Education', 'Major Purchase', 'Debt Consolidation'],
    'Current_Loan_Amount': [15000, 8000, 12000, 20000, 7000, 5000, 25000, 14000, 6000, 30000],
    'Loan_Status': [1, 1, 1, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Encode categorical columns
le_home = LabelEncoder()
le_purpose = LabelEncoder()
df['Home_Ownership'] = le_home.fit_transform(df['Home_Ownership'])
df['Loan_Purpose'] = le_purpose.fit_transform(df['Loan_Purpose'])

# Train model
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("💰 Loan Eligibility Predictor")
st.write("Enter applicant details to predict loan approval.")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
annual_income = st.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
years_in_job = st.slider("Years in Current Job", 0, 40, 5)
home_ownership = st.selectbox("Home Ownership", le_home.classes_)
loan_purpose = st.selectbox("Loan Purpose", le_purpose.classes_)
current_loan_amount = st.number_input("Current Loan Amount", min_value=1000, max_value=1000000, value=10000)

if st.button("Predict"):
    # Encode inputs
    home_encoded = le_home.transform([home_ownership])[0]
    purpose_encoded = le_purpose.transform([loan_purpose])[0]

    input_data = pd.DataFrame([[
        credit_score,
        annual_income,
        years_in_job,
        home_encoded,
        purpose_encoded,
        current_loan_amount
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")
