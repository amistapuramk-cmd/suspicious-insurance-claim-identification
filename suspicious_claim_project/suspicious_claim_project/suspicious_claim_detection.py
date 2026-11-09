"""
Suspicious Insurance Claim Identification - Base Project
---------------------------------------------------------
Steps:
  1. Generate a synthetic insurance-claims dataset (replace with a real CSV later)
  2. Preprocess the data
  3. Train a supervised model (Random Forest) to classify fraud / not fraud
  4. Train an unsupervised model (Isolation Forest) to flag anomalies
  5. Evaluate, rank the most suspicious claims, and predict on a new claim

Install:  pip install pandas numpy scikit-learn joblib
Run:      python suspicious_claim_detection.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. DATA  (synthetic; to use real data: df = pd.read_csv("claims.csv"))
# ---------------------------------------------------------------------------
def generate_data(n=5000):
    df = pd.DataFrame({
        "customer_age": rng.integers(18, 80, n),
        "policy_tenure_months": rng.integers(1, 120, n),
        "claim_type": rng.choice(["auto", "health", "property", "theft"], n,
                                 p=[0.4, 0.3, 0.2, 0.1]),
        "policy_coverage": rng.choice([50_000, 100_000, 250_000, 500_000], n),
        "num_prior_claims": rng.poisson(1.0, n),
        "police_report_filed": rng.choice([0, 1], n, p=[0.4, 0.6]),
        "num_witnesses": rng.poisson(1.2, n),
        "days_to_report": rng.integers(0, 60, n),
    })
    # Claim amount is a fraction of coverage
    df["claim_amount"] = (df["policy_coverage"] * rng.beta(2, 6, n)).round(2)

    # Fraud probability driven by common red flags
    score = (
        -4.0
        + 1.5 * (df["policy_tenure_months"] < 6)          # new policy
        + 1.8 * (df["claim_amount"] / df["policy_coverage"] > 0.5)  # near limit
        + 0.5 * df["num_prior_claims"]                     # frequent claimant
        + 1.0 * (df["police_report_filed"] == 0)
        + 0.8 * (df["num_witnesses"] == 0)
        + 0.03 * df["days_to_report"]                      # late reporting
        + 0.8 * (df["claim_type"] == "theft")
    )
    prob = 1 / (1 + np.exp(-score))
    df["is_fraud"] = (rng.random(n) < prob).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess(df, encoders=None):
    df = df.copy()
    # Engineered feature: how much of the coverage was claimed
    df["claim_to_coverage_ratio"] = df["claim_amount"] / df["policy_coverage"]

    fit = encoders is None
    encoders = encoders or {}
    for col in ["claim_type"]:
        if fit:
            encoders[col] = LabelEncoder().fit(df[col])
        df[col] = encoders[col].transform(df[col])
    return df, encoders


# ---------------------------------------------------------------------------
# 3 & 4. MODELS
# ---------------------------------------------------------------------------
def main():
    df = generate_data()
    print(f"Dataset: {df.shape[0]} claims | fraud rate: {df['is_fraud'].mean():.1%}\n")

    X_raw, y = df.drop(columns="is_fraud"), df["is_fraud"]
    X, encoders = preprocess(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # --- Supervised: Random Forest ---
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("=== Random Forest (supervised) ===")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Suspicious"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}\n")

    print("Top features:")
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(imp.head(6).round(3).to_string(), "\n")

    # --- Unsupervised: Isolation Forest (works without fraud labels) ---
    iso = IsolationForest(contamination=float(y.mean()), random_state=RANDOM_STATE)
    iso.fit(X_train)
    iso_pred = (iso.predict(X_test) == -1).astype(int)   # -1 = anomaly
    print("=== Isolation Forest (unsupervised) ===")
    print(classification_report(y_test, iso_pred, target_names=["Genuine", "Suspicious"]))

    # --- Rank the most suspicious claims in the test set ---
    ranked = X_raw.loc[X_test.index].copy()
    ranked["fraud_probability"] = y_prob
    ranked["actual_fraud"] = y_test
    print("Top 5 most suspicious claims:")
    print(ranked.sort_values("fraud_probability", ascending=False).head(5).to_string(), "\n")

    # --- Save model ---
    joblib.dump({"model": clf, "encoders": encoders, "columns": list(X.columns)},
                "claim_fraud_model.joblib")
    print("Model saved to claim_fraud_model.joblib")

    # --- Predict a single new claim ---
    new_claim = pd.DataFrame([{
        "customer_age": 34, "policy_tenure_months": 2, "claim_type": "theft",
        "policy_coverage": 100_000, "num_prior_claims": 3,
        "police_report_filed": 0, "num_witnesses": 0, "days_to_report": 30,
        "claim_amount": 85_000,
    }])
    print("\nNew claim fraud probability:", round(predict_claim(new_claim), 3))


def predict_claim(claim_df, model_path="claim_fraud_model.joblib"):
    """Return the fraud probability for one or more new claims."""
    bundle = joblib.load(model_path)
    Xn, _ = preprocess(claim_df, bundle["encoders"])
    return bundle["model"].predict_proba(Xn[bundle["columns"]])[:, 1][0]


if __name__ == "__main__":
    main()
