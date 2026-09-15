# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn based on customer demographics, service usage, contract details, tenure, and billing information.

## Project Overview

Customer churn prediction helps businesses identify customers who may discontinue their services. This project uses a simulated telecom-style customer dataset and applies data analysis and machine learning to identify churn patterns and predict customer churn risk.

## Features

- Customer churn data analysis
- Exploratory Data Analysis (EDA)
- Churn rate analysis
- Churn analysis by contract and internet service
- Tenure-based churn analysis
- Monthly charges analysis
- Customer support and security analysis
- Feature importance analysis
- Logistic Regression model
- Model evaluation using:
  - Accuracy
  - Classification Report
  - Confusion Matrix
  - ROC-AUC
- Saved trained model and scaler
- Interactive Streamlit prediction application

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Logistic Regression
      ↓
Model Evaluation
      ↓
Churn Prediction

Dataset

The project uses a simulated telecom-style customer dataset containing 2,000 customer records.

Important features include:

Age
Gender
Tenure
Contract
Internet Service
Online Security
Tech Support
Payment Method
Monthly Charges
Total Charges
Churn

Note: The dataset is synthetically generated for educational and portfolio purposes and does not represent real customer data.

Model

The project uses Logistic Regression to classify customers into:

No Churn
Churn

The trained model and scaler are saved in the models directory using Joblib.

Streamlit Application

The project includes an interactive Streamlit application where users can enter customer information and receive:

Churn prediction
Churn probability
Churn risk level

Example output:

High Churn Risk
Churn Probability: 74.05%

The displayed probability is an example from the application and should not be interpreted as real-world business performance.

Project Structure
customer-churn-prediction/
│
├── data/
│   ├── customer_churn.csv
│   └── feature_importance.csv
│
├── images/
│
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│
├── create_dataset.py
├── predict_churn.py
├── app.py
├── customer_churn_analysis.ipynb
├── .gitignore
└── README.md
How to Run
1. Clone the repository
git clone https://github.com/Thanu-12345/customer-churn-prediction.git
2. Navigate to the project
cd customer-churn-prediction
3. Create a virtual environment
python -m venv venv
4. Activate the environment

Windows:

.\venv\Scripts\Activate.ps1
5. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter openpyxl
6. Run the Streamlit application
streamlit run app.py
Key Business Insights

The analysis identifies customer characteristics associated with higher churn risk, including:

Short customer tenure
Month-to-month contracts
Higher monthly charges
Lack of technical support
Lack of online security services
Certain internet service categories

These insights can help businesses identify customers who may require targeted retention strategies.

Skills Demonstrated
Python Programming
Data Cleaning
Exploratory Data Analysis
Data Visualization
Feature Engineering
Machine Learning
Classification
Model Evaluation
Feature Importance Analysis
Streamlit Application Development
Git & GitHub
Future Improvements
Compare multiple machine learning algorithms
Hyperparameter tuning
Cross-validation
Deploy the Streamlit application
Add customer retention recommendations
Use a real-world public churn dataset
Add interactive analytics dashboards
Author

Thanmayee

GitHub:
https://github.com/Thanu-12345