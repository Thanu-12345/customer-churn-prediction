import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

data = {
    "Customer_ID": [f"CUST{i:04d}" for i in range(1, n + 1)],
    "Age": np.random.randint(18, 71, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Tenure_Months": np.random.randint(1, 73, n),
    "Contract": np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        n,
        p=[0.55, 0.25, 0.20]
    ),
    "Internet_Service": np.random.choice(
        ["DSL", "Fiber Optic", "No"],
        n,
        p=[0.35, 0.50, 0.15]
    ),
    "Online_Security": np.random.choice(["Yes", "No"], n),
    "Tech_Support": np.random.choice(["Yes", "No"], n),
    "Payment_Method": np.random.choice(
        ["Credit Card", "Debit Card", "Bank Transfer", "UPI"],
        n
    ),
    "Monthly_Charges": np.round(np.random.uniform(20, 150, n), 2),
    "Total_Charges": np.round(np.random.uniform(100, 8000, n), 2),
}

df = pd.DataFrame(data)

# Generate churn probability based on customer characteristics
churn_probability = (
    0.15
    + (df["Contract"] == "Month-to-Month") * 0.25
    + (df["Internet_Service"] == "Fiber Optic") * 0.10
    + (df["Tech_Support"] == "No") * 0.08
    + (df["Online_Security"] == "No") * 0.07
    + (df["Tenure_Months"] < 12) * 0.15
    + (df["Monthly_Charges"] > 100) * 0.10
)

churn_probability = np.clip(churn_probability, 0, 0.85)

df["Churn"] = np.random.binomial(1, churn_probability)

df["Churn"] = df["Churn"].map({
    0: "No",
    1: "Yes"
})

df.to_csv("data/customer_churn.csv", index=False)

print("Customer churn dataset created successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nChurn Distribution:")
print(df["Churn"].value_counts())