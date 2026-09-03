import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix

with open("dataset/label_mapping.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
class_names = [label_map[i] for i in range(len(label_map))]

X_train = pd.read_csv("dataset/X_train_capped.csv")
y_train = pd.read_csv("dataset/y_train_capped.csv")["Label"].values
X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv")["Label"].values

# Retrain the two best models (fast - already know timing) to get predictions for confusion matrices
print("Training Random Forest (best model) for confusion matrix...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Training XGBoost (second best) for confusion matrix...")
xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=8, tree_method="hist",
                               n_jobs=-1, random_state=42, eval_metric="mlogloss")
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

def plot_confusion(y_true, y_pred, title, filename, normalize=True):
    present = sorted(set(y_true) | set(y_pred))
    names = [label_map[p] for p in present]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=110)
    plt.close()
    print(f"Saved: {filename}")

plot_confusion(y_test, y_pred_rf, "Random Forest — Confusion Matrix (row-normalized)",
                "graphs/05_confusion_matrix_rf.png")
plot_confusion(y_test, y_pred_xgb, "XGBoost — Confusion Matrix (row-normalized)",
                "graphs/06_confusion_matrix_xgb.png")

# Save the trained RF model for SHAP in the next step (avoid retraining again)
import joblib
joblib.dump(rf, "results/best_model_rf.pkl")
print("Saved trained Random Forest model to results/best_model_rf.pkl")

print("\nDone with Step 11-13.")
