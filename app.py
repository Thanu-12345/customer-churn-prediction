import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load original dataset to reproduce training columns
original_data = pd.read_csv("data/customer_churn.csv")

original_data["Tenure_Group"] = pd.cut(
    original_data["Tenure_Months"],
    bins=[0, 12, 36, 100],
    labels=["New", "Medium", "Long-term"]
)

original_features = original_data.drop(
    columns=["Customer_ID", "Churn"]
)

original_encoded = pd.get_dummies(
    original_features,
    columns=original_features.select_dtypes(
        include=["object", "category"]
    ).columns,
    drop_first=True
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to predict the likelihood of churn."
)

st.subheader("Customer Information")

age = st.slider(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.slider(
    "Tenure (Months)",
    min_value=1,
    max_value=72,
    value=12
)

contract = st.selectbox(
    "Contract",
    ["Month-to-Month", "One Year", "Two Year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber Optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Credit Card",
        "Debit Card",
        "Bank Transfer",
        "UPI"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=20.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=100.0,
    max_value=8000.0,
    value=1000.0,
    step=50.0
)

if st.button("🔮 Predict Churn"):

    customer = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure_Months": tenure,
        "Contract": contract,
        "Internet_Service": internet,
        "Online_Security": security,
        "Tech_Support": support,
        "Payment_Method": payment,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges
    }])

    customer["Tenure_Group"] = pd.cut(
        customer["Tenure_Months"],
        bins=[0, 12, 36, 100],
        labels=["New", "Medium", "Long-term"]
    )

    customer_encoded = pd.get_dummies(
        customer,
        columns=customer.select_dtypes(
            include=["object", "category"]
        ).columns,
        drop_first=True
    )

    customer_encoded = customer_encoded.reindex(
        columns=original_encoded.columns,
        fill_value=0
    )

    customer_scaled = scaler.transform(
        customer_encoded
    )

    prediction = model.predict(
        customer_scaled
    )[0]

    probability = model.predict_proba(
        customer_scaled
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )