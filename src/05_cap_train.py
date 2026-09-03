import pandas as pd
import numpy as np
import time

print("Loading full SMOTE training set...")
t0 = time.time()
X_train = pd.read_csv("dataset/X_train_smote.csv")
y_train = pd.read_csv("dataset/y_train_smote.csv")["Label"].values
print(f"Loaded in {time.time()-t0:.1f}s. Shape: {X_train.shape}")

PER_CLASS_CAP = 60_000
df = pd.DataFrame(X_train)
df["__y__"] = y_train
parts = []
for v, g in df.groupby("__y__"):
    if len(g) > PER_CLASS_CAP:
        g = g.sample(n=PER_CLASS_CAP, random_state=42)
    parts.append(g)
capped = pd.concat(parts, axis=0).sample(frac=1, random_state=42)
y_train_c = capped["__y__"].values
X_train_c = capped.drop(columns="__y__")

print(f"Capped shape: {X_train_c.shape}")
X_train_c.to_csv("dataset/X_train_capped.csv", index=False)
pd.Series(y_train_c, name="Label").to_csv("dataset/y_train_capped.csv", index=False)
print("Saved dataset/X_train_capped.csv and dataset/y_train_capped.csv")
