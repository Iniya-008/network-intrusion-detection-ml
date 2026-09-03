import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Loading model and data...")
rf = joblib.load("results/best_model_rf.pkl")
X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv")["Label"].values

with open("dataset/label_mapping.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

# Use a small representative sample for SHAP - it is computationally
# expensive (especially for multi-class tree models), so a sample of
# a few hundred rows is standard practice for generating explanations.
print("Sampling test rows for SHAP (500 rows, stratified by class)...")
df = X_test.copy()
df["__y__"] = y_test
sample_parts = []
for cls, group in df.groupby("__y__"):
    n = min(len(group), max(5, 500 // len(label_map)))
    sample_parts.append(group.sample(n=n, random_state=42))
sample_df = pd.concat(sample_parts, axis=0)
y_sample = sample_df["__y__"].values
X_sample = sample_df.drop(columns="__y__")
print(f"SHAP sample size: {X_sample.shape[0]}")

print("Computing SHAP values (TreeExplainer, fast for tree models)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample, check_additivity=False)

# shap_values shape for multiclass: (n_samples, n_features, n_classes) in newer SHAP,
# or a list of arrays per class in older versions. Handle both.
if isinstance(shap_values, list):
    shap_values_arr = np.array(shap_values)  # (n_classes, n_samples, n_features)
else:
    shap_values_arr = np.transpose(shap_values, (2, 0, 1))  # -> (n_classes, n_samples, n_features)

print(f"SHAP values shape (classes, samples, features): {shap_values_arr.shape}")

# =========================================================
# 1) GLOBAL FEATURE IMPORTANCE (mean |SHAP| across all classes)
# =========================================================
print("\nGenerating global feature importance plot...")
mean_abs_shap_per_class = np.abs(shap_values_arr).mean(axis=1)  # (n_classes, n_features)
overall_importance = mean_abs_shap_per_class.mean(axis=0)  # (n_features,)

importance_df = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": overall_importance
}).sort_values("mean_abs_shap", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top15 = importance_df.head(15)
ax.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="#1C7293")
ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
ax.set_title("Global Feature Importance — Random Forest (SHAP)")
plt.tight_layout()
plt.savefig("graphs/07_shap_global_importance.png", dpi=110)
plt.close()
print("Saved: graphs/07_shap_global_importance.png")

importance_df.to_csv("results/shap_feature_importance.csv", index=False)
print("Saved: results/shap_feature_importance.csv")

# =========================================================
# 2) PER-CLASS TOP FEATURES for key attack types
#    (what SHAP says drives predictions of specific attacks)
# =========================================================
print("\nTop features per class:")
per_class_top = {}
for idx, cls_name in label_map.items():
    idx = int(idx)
    class_importance = np.abs(shap_values_arr[idx]).mean(axis=0)
    top_features = pd.Series(class_importance, index=X_sample.columns).sort_values(ascending=False).head(5)
    per_class_top[cls_name] = top_features.to_dict()
    print(f"\n  {cls_name}:")
    for feat, val in top_features.items():
        print(f"    {feat}: {val:.4f}")

with open("results/shap_per_class_top_features.json", "w") as f:
    json.dump(per_class_top, f, indent=2)
print("\nSaved: results/shap_per_class_top_features.json")

print("\nStep 14 (SHAP) complete.")
