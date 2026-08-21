import os
import yaml
from src.train import train

experiments = [
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2},
]

print("Bat dau chay lai 4 thi nghiem vao MLflow...")

for idx, params in enumerate(experiments, 1):
    print(f"\n--- Dang chay Thi nghiem {idx}: {params} ---")
    acc = train(params)
    print(f"-> Ket qua: Accuracy = {acc:.4f}")

# Luu lai bo tham so tot nhat vao params.yaml
best_params = experiments[-1]
with open("params.yaml", "w") as f:
    f.write("# Sieu tham so cho mo hinh RandomForestClassifier\n")
    f.write("# Thay doi cac gia tri nay giua cac lan chay de thi nghiem (Buoc 1)\n")
    yaml.dump(best_params, f)

print("\nHoan tat 4 thi nghiem! params.yaml da duoc cap nhat bo tham so tot nhat.")
