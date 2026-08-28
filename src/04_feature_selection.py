import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import gc

DATA_PATH = "dataset/combined_cleaned.csv"

print("="*60)
print("STEP 6: FEATURE SELECTION")
print("="*60)

# =========================================================
# 1) Load full data (needed for accurate variance check)
# =========================================================
print("\n[1] Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

non_feature_cols = ["Label", "Source_File"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]
print(f"    Candidate feature columns: {len(feature_cols)}")

# =========================================================
# 2) ZERO-VARIANCE FEATURE REMOVAL
#    A column with zero variance has the exact same value in
#    every single row - it carries no information for the model.
# =========================================================
print("\n[2] Checking for zero-variance features...")
variances = df[feature_cols].var(numeric_only=True)
zero_var_cols = variances[variances == 0].index.tolist()
print(f"    Zero-variance columns found: {len(zero_var_cols)}")
for c in zero_var_cols:
    print(f"      - {c}")

feature_cols = [c for c in feature_cols if c not in zero_var_cols]
print(f"    Remaining features after this step: {len(feature_cols)}")

# =========================================================
# 3) CORRELATION-BASED REDUNDANCY REMOVAL
#    If two features are near-perfectly correlated (>0.95),
#    they carry almost the same information - keep one, drop other.
# =========================================================
print("\n[3] Checking for highly correlated feature pairs (threshold 0.95)...")

# Use a sample for the correlation matrix - result is the same
# as full data for this purpose, and much faster/lighter on memory.
sample_for_corr = df[feature_cols].sample(n=min(300_000, len(df)), random_state=42)
corr_matrix = sample_for_corr.corr().abs()

# Walk through the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = set()
dropped_pairs = []
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and val > 0.95:
            # keep the one with higher variance (more informative spread), drop the other
            if row in to_drop or col in to_drop:
                continue
            if variances.get(row, 0) >= variances.get(col, 0):
                to_drop.add(col)
                dropped_pairs.append((row, col, round(val, 3)))
            else:
                to_drop.add(row)
                dropped_pairs.append((col, row, round(val, 3)))

print(f"    Redundant features to drop: {len(to_drop)}")
for kept, dropped, val in dropped_pairs[:20]:
    print(f"      kept '{kept}'  dropped '{dropped}'  (corr={val})")
if len(dropped_pairs) > 20:
    print(f"      ... and {len(dropped_pairs)-20} more pairs")

feature_cols = [c for c in feature_cols if c not in to_drop]
print(f"    Remaining features after this step: {len(feature_cols)}")

del sample_for_corr, corr_matrix, upper
gc.collect()

# =========================================================
# 4) VIF (Variance Inflation Factor)
#    Catches remaining multicollinearity that pairwise
#    correlation alone might miss (3+ features jointly redundant).
#    VIF > 10 is the common rule-of-thumb threshold for concern.
# =========================================================
print("\n[4] Computing VIF on remaining features (using a sample for speed)...")
vif_sample = df[feature_cols].sample(n=min(50_000, len(df)), random_state=42).copy()

# VIF requires no NaN/inf and no zero-variance columns (already handled)
# Standardize scale isn't required for VIF itself, but drop any constant
# columns that might have slipped through on this particular sample.
vif_sample = vif_sample.loc[:, vif_sample.var() > 0]

current_features = vif_sample.columns.tolist()
removed_for_vif = []

# Iteratively remove the single worst offender until all VIF < 10,
# capped at a reasonable number of iterations for speed.
max_iterations = 15
for i in range(max_iterations):
    X = vif_sample[current_features].values
    vifs = []
    for idx in range(X.shape[1]):
        try:
            v = variance_inflation_factor(X, idx)
        except Exception:
            v = np.nan
        vifs.append(v)
    vif_series = pd.Series(vifs, index=current_features)
    worst = vif_series.idxmax()
    worst_val = vif_series.max()

    if worst_val < 10 or np.isnan(worst_val):
        print(f"    All remaining features have VIF < 10 after {i} removals.")
        break

    current_features.remove(worst)
    removed_for_vif.append((worst, round(worst_val, 1)))
    print(f"    Iteration {i+1}: removing '{worst}' (VIF={worst_val:.1f})")

print(f"\n    Total removed for multicollinearity (VIF): {len(removed_for_vif)}")
feature_cols = current_features
print(f"    FINAL feature count: {len(feature_cols)}")

del vif_sample
gc.collect()

# =========================================================
# 5) Save the reduced dataset (full row count, fewer columns)
# =========================================================
print("\n[5] Saving reduced dataset with selected features...")
final_cols = feature_cols + non_feature_cols
reduced = df[final_cols]
reduced.to_csv("dataset/combined_selected_features.csv", index=False)
print(f"    Saved: dataset/combined_selected_features.csv")
print(f"    Shape: {reduced.shape[0]:,} rows x {reduced.shape[1]} columns")

# Save the feature list itself for documentation/reproducibility
with open("dataset/selected_features.txt", "w") as f:
    f.write("\n".join(feature_cols))
print(f"    Feature list saved: dataset/selected_features.txt")

print("\n" + "="*60)
print("STEP 6 COMPLETE")
print(f"  Started with : 78 candidate features")
print(f"  Zero-variance removed : {len(zero_var_cols)}")
print(f"  Correlation-redundant removed : {len(to_drop)}")
print(f"  VIF-multicollinear removed : {len(removed_for_vif)}")
print(f"  FINAL feature count : {len(feature_cols)}")
print("="*60)
