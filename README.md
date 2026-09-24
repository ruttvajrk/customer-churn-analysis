# Customer Churn Prediction and Business Analysis

Predicts which telecom customers are likely to leave, explains the main reasons, and estimates the monthly revenue at risk. Includes a small Streamlit app to check the churn risk of a single customer.

**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit

## Dataset
IBM Telco Customer Churn – 7,043 customers, 21 columns (demographics, services, contract, billing, churn).
Source: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (`Telco-Customer-Churn.csv`, included in this repo).

## Steps
1. Data cleaning – converted `TotalCharges` to numeric (11 blank values for new customers set to 0), dropped the ID column, encoded the target
2. Business analysis (EDA) – churn rate by contract, internet service, payment method, tenure; revenue lost to churn
3. Preprocessing – one-hot encoding, stratified 80/20 train-test split, scaling of numeric columns
4. Models – Logistic Regression, Decision Tree, Random Forest (class-balanced), compared on precision, recall, F1 and ROC-AUC
5. Tuning – GridSearchCV with 5-fold stratified cross-validation (scoring: F1)
6. Feature importance and business impact

## Key business findings
_Fill in / check against your notebook output._

| Finding | Value |
|---|---|
| Overall churn rate | 26.5% |
| Churn rate: month-to-month vs two-year contract | 42.7% vs 2.8% |
| Churn rate in first 12 months | 47.4% |
| Churn rate: fiber optic / electronic check | 41.9% / 45.3% |
| Monthly revenue from churned customers | $139,131 (30.5% of total) |

## Model results (test set, 1,409 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 |
| Decision Tree | 0.735 | 0.500 | 0.807 | 0.618 | 0.831 |
| Random Forest | 0.789 | 0.627 | 0.503 | 0.558 | 0.824 |
| **Tuned Logistic Regression** (C=0.1, balanced) | 0.74 | 0.51 | 0.78 | 0.62 | 0.841 |

Metrics are for the churn class. Accuracy alone is misleading here: predicting "no churn" for everyone would already give about 73%.
The tuned model catches 293 of 374 churners in the test set, covering 82.4% of their monthly revenue.

## Recommendations
1. Offer incentives to move month-to-month customers to longer contracts
2. Focus retention and onboarding on the first 12 months
3. Review fiber optic pricing and service quality
4. Promote automatic payment methods over electronic check

## Run it
```bash
pip install -r requirements.txt
jupyter notebook customer_churn_analysis.ipynb   # run all cells (also saves the model files)
streamlit run app.py                             # churn risk app
```

## Limitations
- Single snapshot of data, no time information
- Moderate precision: about half of the customers flagged as "will churn" would have stayed, so retention offers have a cost
