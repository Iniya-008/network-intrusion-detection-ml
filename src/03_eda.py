import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
import gc

plt.rcParams["figure.dpi"] = 110

DATA_PATH = "dataset/combined_cleaned.csv"
GRAPHS_DIR = "graphs"

# =========================================================
# 1) CLASS DISTRIBUTION (only needs the Label column - cheap)
# =========================================================
print("[1] Class distribution...")
labels_only = pd.read_csv(DATA_PATH, usecols=["Label"])
counts = labels_only["Label"].value_counts()
print(counts)

fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#1C7293")
ax.set_xscale("log")  # log scale because BENIGN dwarfs everything else
ax.set_xlabel("Number of flows (log scale)")
ax.set_ylabel("Traffic class")
ax.set_title("Class Distribution — CIC-IDS2017 (Cleaned)")
for i, v in enumerate(counts.values):
    ax.text(v, i, f" {v:,}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_class_distribution.png")
plt.close()
print("    Saved: graphs/01_class_distribution.png\n")

del labels_only
gc.collect()

# =========================================================
# 2) Load a SAMPLE for the heavier analyses (correlation,
#    feature distributions). Using all 2.5M rows for these
#    isn't necessary - a representative sample is standard
#    practice for EDA visuals and keeps memory safe.
# =========================================================
print("[2] Loading a stratified-ish sample for feature analysis...")
df = pd.read_csv(DATA_PATH)
sample = df.groupby("Label", group_keys=False)[df.columns.tolist()].apply(
    lambda g: g.sample(n=min(len(g), 15000), random_state=42)
)
print(f"    Sample size: {sample.shape[0]:,} rows (from {df.shape[0]:,} total)")
del df
gc.collect()

numeric_cols = sample.select_dtypes(include=np.number).columns.tolist()
print(f"    Numeric feature columns: {len(numeric_cols)}")

# =========================================================
# 3) FEATURE DISTRIBUTIONS (pick a few important, well-known features)
# =========================================================
print("\n[3] Feature distributions for key features...")
key_features = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s",
                 "Total Fwd Packets", "Total Backward Packets",
                 "Destination Port"]
key_features = [f for f in key_features if f in sample.columns]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    data = sample[feat]
    # clip extreme outliers just for readable plotting (99th percentile)
    upper = data.quantile(0.99)
    sns.histplot(data[data <= upper], bins=50, ax=axes[i], color="#065A82")
    axes[i].set_title(feat, fontsize=11)
    axes[i].set_xlabel("")
plt.suptitle("Key Feature Distributions (clipped at 99th percentile for readability)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_feature_distributions.png")
plt.close()
print("    Saved: graphs/02_feature_distributions.png")

# =========================================================
# 4) CORRELATION HEATMAP
# =========================================================
print("\n[4] Correlation heatmap...")
corr = sample[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
            xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.6})
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.xticks(fontsize=6, rotation=90)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/03_correlation_heatmap.png")
plt.close()
print("    Saved: graphs/03_correlation_heatmap.png")

# Flag highly correlated pairs (>0.95) - useful for feature selection later
print("\n    Highly correlated feature pairs (|corr| > 0.95):")
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        val = corr.iloc[i, j]
        if abs(val) > 0.95:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], round(val, 3)))
print(f"    Found {len(high_corr_pairs)} highly correlated pairs (sample shown below)")
for pair in high_corr_pairs[:15]:
    print(f"      {pair[0]}  <->  {pair[1]}   corr={pair[2]}")

# =========================================================
# 5) OUTLIER CHECK (boxplots for key features, by class BENIGN vs rest)
# =========================================================
print("\n[5] Outlier check (Benign vs Attack) for key features...")
sample["Is_Attack"] = sample["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    data = sample[[feat, "Is_Attack"]].copy()
    upper = data[feat].quantile(0.99)
    data = data[data[feat] <= upper]
    sns.boxplot(data=data, x="Is_Attack", y=feat, ax=axes[i], palette=["#21C89A", "#5c1a24"])
    axes[i].set_title(feat, fontsize=11)
    axes[i].set_xlabel("")
plt.suptitle("Benign vs Attack — Feature Spread (outliers visible as dots)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/04_outliers_benign_vs_attack.png")
plt.close()
print("    Saved: graphs/04_outliers_benign_vs_attack.png")

print("\n" + "="*60)
print("EDA COMPLETE. 4 graphs saved in graphs/ folder.")
print("="*60)
