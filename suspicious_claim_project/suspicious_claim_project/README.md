# Suspicious Insurance Claim Identification

Base ML project that flags suspicious insurance claims using a Random Forest (supervised) and an Isolation Forest (unsupervised).

## Setup
pip install -r requirements.txt

## Run
python suspicious_claim_detection.py

Generates synthetic claims data, trains and evaluates the models, lists the top suspicious claims, and saves `claim_fraud_model.joblib`.
To use real data, replace `generate_data()` with `pd.read_csv("claims.csv")` (needs an `is_fraud` column).
