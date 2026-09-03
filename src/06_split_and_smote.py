import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import gc
import json

DATA_PATH = "dataset/combined_selected_features.csv"

print("="*60)
print("STEP 7-9: TRAIN/TEST SPLIT + SMOTE (CLASS BALANCING)")
print("="*60)

# =========================================================
# 1) Load the feature-selected dataset from Step 6
# =========================================================
print("\n[1] Loading dataset (32 selected features)...")
df = pd.read_csv(DATA_PATH)
print(f"    Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

X = df.drop(columns=["Label", "Source_File"])
y = df["Label"]
del df
gc.collect()

print(f"    X shape: {X.shape}")
print(f"    y classes: {y.nunique()}")

# =========================================================
# 2) Encode text labels into numbers (models need numeric labels)
# =========================================================
print("\n[2] Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the mapping so we can always translate numbers back to
# real attack names later (for reports, confusion matrices, etc.)
label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
with open("dataset/label_mapping.json", "w") as f:
    json.dump(label_map, f, indent=2)
print(f"    Classes: {list(le.classes_)}")
print(f"    Saved mapping to dataset/label_mapping.json")

# =========================================================
# 3) STRATIFIED TRAIN/TEST SPLIT (80/20)
#    "Stratified" = keeps the same class proportions in both
#    train and test sets (so rare classes aren't accidentally
#    left out of the test set entirely).
#    Critically: this happens BEFORE any balancing, and the
#    test set stays untouched from here on - it must reflect
#    real-world (imbalanced) traffic.
# =========================================================
print("\n[3] Splitting into train (80%) / test (20%), stratified...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
del X, y_encoded
gc.collect()

print(f"    Train: {X_train.shape[0]:,} rows")
print(f"    Test : {X_test.shape[0]:,} rows")

print("\n    Class counts in TRAIN (before SMOTE):")
train_counts = pd.Series(y_train).value_counts().sort_index()
for cls_idx, cnt in train_counts.items():
    print(f"      {label_map[cls_idx]:30s}: {cnt:,}")

# Save the untouched test set now, before we do anything else to train
print("\n[4] Saving test set (untouched, real-world distribution)...")
X_test.to_csv("dataset/X_test.csv", index=False)
pd.Series(y_test, name="Label").to_csv("dataset/y_test.csv", index=False)
print("    Saved dataset/X_test.csv and dataset/y_test.csv")

# =========================================================
# 5) SMOTE - applied ONLY to the training data
#    Practical note: fully balancing every class to match
#    BENIGN (~1.6M) would mean generating millions of synthetic
#    rows for classes that only had ~10-20 real examples - that's
#    unrealistic and would overwhelm the model with fake data.
#    Instead, we bring only the severely under-represented classes
#    up to a reasonable floor (2,000 samples), leaving classes that
#    already have decent representation untouched. This is a common,
#    documented practical adjustment to standard SMOTE.
# =========================================================
print("\n[5] Applying SMOTE to training data only...")
FLOOR = 2000
class_counts = pd.Series(y_train).value_counts()
smote_targets = {cls: FLOOR for cls, cnt in class_counts.items() if cnt < FLOOR}

print(f"    Classes being oversampled up to {FLOOR} samples:")
for cls_idx, target in smote_targets.items():
    print(f"      {label_map[cls_idx]:30s}: {class_counts[cls_idx]:,} -> {target:,}")

# k_neighbors must be smaller than the smallest class being oversampled
min_class_size = min(class_counts[c] for c in smote_targets.keys())
k_neighbors = max(1, min(5, min_class_size - 1))
print(f"    Using k_neighbors={k_neighbors} (smallest class being oversampled has {min_class_size} real samples)")

smote = SMOTE(sampling_strategy=smote_targets, random_state=42, k_neighbors=k_neighbors)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\n    Training set before SMOTE: {X_train.shape[0]:,} rows")
print(f"    Training set after SMOTE : {X_train_bal.shape[0]:,} rows")

print("\n    Class counts in TRAIN (after SMOTE):")
bal_counts = pd.Series(y_train_bal).value_counts().sort_index()
for cls_idx, cnt in bal_counts.items():
    print(f"      {label_map[cls_idx]:30s}: {cnt:,}")

# =========================================================
# 6) Save the balanced training set
# =========================================================
print("\n[6] Saving balanced training set...")
X_train_bal.to_csv("dataset/X_train_smote.csv", index=False)
pd.Series(y_train_bal, name="Label").to_csv("dataset/y_train_smote.csv", index=False)
print("    Saved dataset/X_train_smote.csv and dataset/y_train_smote.csv")

print("\n" + "="*60)
print("STEP 7-9 COMPLETE")
print(f"  Test set (untouched)         : {X_test.shape[0]:,} rows")
print(f"  Train set (before SMOTE)     : {X_train.shape[0]:,} rows")
print(f"  Train set (after SMOTE)      : {X_train_bal.shape[0]:,} rows")
print(f"  Classes brought up to floor  : {len(smote_targets)}")
print("="*60)
