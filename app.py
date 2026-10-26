# Streamlit app: churn risk for one customer, using the same saved pipeline as the API.
# Run:  pip install -e .  then  streamlit run app.py

import pandas as pd
import streamlit as st

from churn.api import MODEL_DIR, ModelBundle


@st.cache_resource
def get_model() -> ModelBundle:
    return ModelBundle(MODEL_DIR)


model = get_model()

st.title("Customer Churn Risk")
st.caption(f"Model: {model.card['chosen_model']} · decision threshold "
           f"{model.threshold:.2f} · test ROC-AUC {model.card['test']['roc_auc']:.3f}")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
with col2:
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly * tenure))

if st.button("Predict"):
    customer = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }])
    prob = float(model.predict(customer)[0])
    reasons = model.reasons(customer)[0]

    st.metric("Churn probability", f"{prob * 100:.1f}%")
    st.write(f"Expected monthly revenue at risk: **${prob * monthly:,.2f}**")
    if prob >= model.threshold:
        st.error("At risk: above the model's decision threshold. Consider a retention offer.")
    else:
        st.success("Below the decision threshold: likely to stay.")
    if reasons:
        st.write("Main factors raising the risk: " + ", ".join(reasons))
