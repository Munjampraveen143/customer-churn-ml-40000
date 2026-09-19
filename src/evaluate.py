from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# Model results from verified evaluation
results = {
    "Accuracy": 0.7202,
    "Precision": 0.7475,
    "Recall": 0.8226,
    "F1 Score": 0.7833,
    "ROC-AUC": 0.7817
}


# ---------------------------------------------------------
# 1. Model Performance Chart
# ---------------------------------------------------------

metrics = list(results.keys())
values = list(results.values())

plt.figure(figsize=(9, 5))
bars = plt.bar(metrics, values)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Customer Churn Model Performance")
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(REPORTS_DIR / "model_performance.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------------

confusion_matrix = np.array([
    [1718, 1366],
    [872, 4044]
])

plt.figure(figsize=(7, 6))
plt.imshow(confusion_matrix)

plt.title("Customer Churn Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(confusion_matrix[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar(label="Number of Customers")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 3. Save Evaluation Summary
# ---------------------------------------------------------

summary = {
    "test_rows": 8000,
    "correct_predictions": 5762,
    "incorrect_predictions": 2238,
    "metrics": results,
    "confusion_matrix": confusion_matrix.tolist()
}

with open(REPORTS_DIR / "evaluation_summary.json", "w") as file:
    json.dump(summary, file, indent=4)


print("=" * 60)
print("CUSTOMER CHURN EVALUATION")
print("=" * 60)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print()
print("Correct predictions:", 5762)
print("Incorrect predictions:", 2238)

print()
print("Evaluation files created:")
print(REPORTS_DIR / "model_performance.png")
print(REPORTS_DIR / "confusion_matrix.png")
print(REPORTS_DIR / "evaluation_summary.json")

print("=" * 60)