import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "final_model.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    expected_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    return model, label_encoder, list(expected_columns)


model, label_encoder, expected_columns = load_artifacts()

# Categorical encodings (alphabetical order, same as sklearn LabelEncoder)
GENDER = {"Female": 0, "Male": 1}
MARRIED = {"No": 0, "Yes": 1}
DEPENDENTS = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED = {"No": 0, "Yes": 1}
PROPERTY_AREA = {"Rural": 0, "Semiurban": 1, "Urban": 2}
CREDIT_HISTORY = {"Good (1)": 1.0, "Bad (0)": 0.0}

st.title("🏦 Loan Approval Prediction by Taimoor")
st.markdown("Fill in the applicant details to check loan approval chances.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", list(GENDER))
    married = st.selectbox("Married", list(MARRIED))
    dependents = st.selectbox("Dependents", list(DEPENDENTS))
    education = st.selectbox("Education", list(EDUCATION))
    self_employed = st.selectbox("Self Employed", list(SELF_EMPLOYED))
    property_area = st.selectbox("Property Area", list(PROPERTY_AREA))

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=130, step=10)
    loan_term = st.selectbox(
        "Loan Amount Term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12]
    )
    credit_history = st.selectbox("Credit History", list(CREDIT_HISTORY))

if st.button("Predict", type="primary"):
    raw_input = {
        "Gender": GENDER[gender],
        "Married": MARRIED[married],
        "Dependents": DEPENDENTS[dependents],
        "Education": EDUCATION[education],
        "Self_Employed": SELF_EMPLOYED[self_employed],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": CREDIT_HISTORY[credit_history],
        "Property_Area": PROPERTY_AREA[property_area],
    }

    # Build dataframe in the exact column order used during training
    input_df = pd.DataFrame([raw_input])[expected_columns]

    pred = model.predict(input_df)[0]
    label = label_encoder.inverse_transform([pred])[0]  # 'Y' or 'N'
    proba = model.predict_proba(input_df)[0]
    approve_prob = proba[list(model.classes_).index(1)]

    if label == "Y":
        st.success(f"✅ Loan likely to be APPROVED (confidence: {approve_prob:.0%})")
    else:
        st.error(f"❌ Loan likely to be REJECTED (approval chance: {approve_prob:.0%})")