import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Customer details
customer = {
    "Age": 30,
    "Gender": "Female",
    "Tenure_Months": 8,
    "Contract": "Month-to-Month",
    "Internet_Service": "Fiber Optic",
    "Online_Security": "No",
    "Tech_Support": "No",
    "Payment_Method": "UPI",
    "Monthly_Charges": 110.00,
    "Total_Charges": 850.00
}

# Convert customer data to DataFrame
customer_df = pd.DataFrame([customer])

# Create the same tenure grouping used during analysis
customer_df["Tenure_Group"] = pd.cut(
    customer_df["Tenure_Months"],
    bins=[0, 12, 36, 100],
    labels=["New", "Medium", "Long-term"]
)

# One-hot encode categorical features
customer_encoded = pd.get_dummies(
    customer_df,
    columns=customer_df.select_dtypes(
        include=["object", "category"]
    ).columns,
    drop_first=True
)

# Load the original feature structure
original_data = pd.read_csv("data/customer_churn.csv")

original_data["Tenure_Group"] = pd.cut(
    original_data["Tenure_Months"],
    bins=[0, 12, 36, 100],
    labels=["New", "Medium", "Long-term"]
)

original_features = original_data.drop(columns=["Customer_ID", "Churn"])

original_encoded = pd.get_dummies(
    original_features,
    columns=original_features.select_dtypes(
        include=["object", "category"]
    ).columns,
    drop_first=True
)

# Match the columns used during training
customer_encoded = customer_encoded.reindex(
    columns=original_encoded.columns,
    fill_value=0
)

# Scale the customer data
customer_scaled = scaler.transform(customer_encoded)

# Make prediction
prediction = model.predict(customer_scaled)[0]
probability = model.predict_proba(customer_scaled)[0][1]

print("\nCustomer Churn Prediction")
print("-------------------------")

if prediction == 1:
    print("Prediction: CHURN")
else:
    print("Prediction: NOT CHURN")

print("Churn Probability:", round(probability * 100, 2), "%")