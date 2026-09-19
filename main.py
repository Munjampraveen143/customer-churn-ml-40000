import subprocess
import sys

steps = [
    [sys.executable, "src/generate_data.py"],
    [sys.executable, "src/eda.py"],
    [sys.executable, "src/train.py"],
    [sys.executable, "src/evaluate.py"],
]

for command in steps:
    subprocess.run(command, check=True)

print("\nPipeline completed.")
print("Run API: uvicorn api.main:app --reload")
print("Run dashboard: streamlit run dashboard/app.py")
