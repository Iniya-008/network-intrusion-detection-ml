import sys
import pandas as pd
import numpy as np
import json
import time
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, classification_report)

MODEL_NAME = sys.argv[1]  # "rf", "xgb", "lgb", or "mlp"

os.makedirs("results", exist_ok=True)
RESULTS_PATH = "results/model_results.json"

print(f"Loading capped train + full test data for model: {MODEL_NAME}")
X_train = pd.read_csv("dataset/X_train_capped.csv")
y_train = pd.read_csv("dataset/y_train_capped.csv")["Label"].values
X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv")["Label"].values

with open("dataset/label_mapping.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

print(f"Train: {X_train.shape}  Test: {X_test.shape}")

t0 = time.time()

if MODEL_NAME == "rf":
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    display_name = "Random Forest"

elif MODEL_NAME == "xgb":
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=150, max_depth=8, tree_method="hist",
                               n_jobs=-1, random_state=42, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    display_name = "XGBoost"

elif MODEL_NAME == "lgb":
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=150, max_depth=8, n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    display_name = "LightGBM"

elif MODEL_NAME == "mlp":
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=42)
    model.fit(X_train_s, y_train)
    X_test = X_test_s  # use scaled version for prediction
    display_name = "MLP"

else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

train_time = time.time() - t0
print(f"{display_name} trained in {train_time:.1f}s")

t0 = time.time()
y_pred = model.predict(X_test)
pred_time = time.time() - t0
print(f"Predicted in {pred_time:.1f}s")

acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

present = sorted(set(y_test) | set(y_pred))
names = [label_map[p] for p in present]
report = classification_report(y_test, y_pred, labels=present, target_names=names,
                                 zero_division=0, output_dict=True)

result = {
    "accuracy": acc,
    "precision_macro": prec_macro,
    "recall_macro": rec_macro,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
    "train_seconds": train_time,
    "predict_seconds": pred_time,
    "per_class_report": report,
}

if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results[display_name] = result
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{display_name} RESULTS")
print(f"  Accuracy        : {acc:.4f}")
print(f"  Precision(macro): {prec_macro:.4f}")
print(f"  Recall(macro)   : {rec_macro:.4f}")
print(f"  F1(macro)       : {f1_macro:.4f}")
print(f"  F1(weighted)    : {f1_weighted:.4f}")
print(f"\nSaved to {RESULTS_PATH}")
