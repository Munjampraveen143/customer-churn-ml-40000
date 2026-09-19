import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports", exist_ok=True)
df = pd.read_csv("data/raw/customers_40000.csv")

print(df.info())
print("\nMissing values:\n", df.isna().sum())
print("\nTarget distribution:\n", df["churn"].value_counts())
print("\nChurn rate:", df["churn"].mean())

plots = [
    ("contract", "Churn by Contract"),
    ("internet_service", "Churn by Internet Service"),
    ("payment_method", "Churn by Payment Method")
]

for column, title in plots:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, x=column, hue="churn", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(f"reports/{column}_churn.png", dpi=160)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="churn", y="monthly_charges", ax=ax)
ax.set_title("Monthly Charges vs Churn")
fig.tight_layout()
fig.savefig("reports/monthly_charges_churn.png", dpi=160)
plt.close(fig)

print("EDA charts saved in reports/")
