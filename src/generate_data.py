import os
import numpy as np
import pandas as pd

SEED = 42
N = 40000

rng = np.random.default_rng(SEED)
os.makedirs("data/raw", exist_ok=True)

customer_id = [f"CUST_{i:06d}" for i in range(1, N + 1)]
gender = rng.choice(["Male", "Female"], N)
age = rng.integers(18, 76, N)
senior_citizen = (age >= 60).astype(int)
partner = rng.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], N, p=[0.30, 0.70])
tenure = rng.integers(0, 73, N)
phone_service = rng.choice(["Yes", "No"], N, p=[0.90, 0.10])
internet_service = rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.50, 0.15])
online_security = rng.choice(["Yes", "No"], N, p=[0.35, 0.65])
online_backup = rng.choice(["Yes", "No"], N, p=[0.40, 0.60])
device_protection = rng.choice(["Yes", "No"], N, p=[0.40, 0.60])
tech_support = rng.choice(["Yes", "No"], N, p=[0.35, 0.65])
streaming_tv = rng.choice(["Yes", "No"], N, p=[0.45, 0.55])
streaming_movies = rng.choice(["Yes", "No"], N, p=[0.45, 0.55])
contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
paperless_billing = rng.choice(["Yes", "No"], N, p=[0.60, 0.40])
payment_method = rng.choice(
    ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
    N, p=[0.35, 0.25, 0.25, 0.15]
)
monthly_charges = np.round(rng.uniform(20, 150, N), 2)
total_charges = np.maximum(
    np.round(monthly_charges * tenure + rng.normal(0, 100, N), 2), 0
)
support_tickets = rng.poisson(2.5, N)
late_payments = rng.poisson(1.2, N)
satisfaction_score = rng.integers(1, 11, N)

risk = np.zeros(N)
risk += np.where(contract == "Month-to-month", 1.4, 0)
risk += np.where(contract == "One year", 0.3, 0)
risk += np.where(tenure < 6, 1.2, 0)
risk += np.where(tenure > 36, -0.8, 0)
risk += np.where(monthly_charges > 100, 0.8, 0)
risk += support_tickets * 0.15
risk += late_payments * 0.20
risk += np.where(satisfaction_score <= 4, 1.0, 0)
risk += np.where(satisfaction_score >= 8, -0.6, 0)
risk += np.where(internet_service == "Fiber optic", 0.35, 0)
risk += np.where(payment_method == "Electronic check", 0.45, 0)
risk += senior_citizen * 0.25

probability = 1 / (1 + np.exp(-(risk - 1.5)))
churn = rng.binomial(1, probability)

df = pd.DataFrame({
    "customer_id": customer_id,
    "gender": gender,
    "age": age,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "tenure": tenure,
    "phone_service": phone_service,
    "internet_service": internet_service,
    "online_security": online_security,
    "online_backup": online_backup,
    "device_protection": device_protection,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "streaming_movies": streaming_movies,
    "contract": contract,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "support_tickets": support_tickets,
    "late_payments": late_payments,
    "satisfaction_score": satisfaction_score,
    "churn": churn
})

for col in ["online_security", "online_backup", "tech_support", "total_charges"]:
    mask = rng.random(N) < 0.02
    df.loc[mask, col] = np.nan

path = "data/raw/customers_40000.csv"
df.to_csv(path, index=False)

print(f"Created {path}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Churn rate: {df.churn.mean():.2%}")
