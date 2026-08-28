import pandas as pd
import numpy as np
import glob
import os

RAW_DIR = "raw_data/MachineLearningCVE"
OUT_PATH = "dataset/combined_raw.csv"

os.makedirs("dataset", exist_ok=True)

files = sorted(glob.glob(f"{RAW_DIR}/*.csv"))
print(f"Found {len(files)} files to combine.\n")

frames = []
for path in files:
    fname = os.path.basename(path)
    df = pd.read_csv(path)

    # Strip whitespace from column names (CIC-IDS2017 has leading spaces
    # on most column names, e.g. ' Destination Port'). This does NOT
    # touch the original file on disk - only the in-memory copy.
    df.columns = df.columns.str.strip()

    # Tag every row with which file/day it came from, so we can trace
    # any row back to its source later if something looks odd.
    df["Source_File"] = fname

    frames.append(df)
    print(f"Loaded {fname}: {df.shape[0]} rows")

# Sanity check: confirm every frame has the exact same columns before concatenating
col_sets = [tuple(sorted(f.columns)) for f in frames]
assert len(set(col_sets)) == 1, "Column mismatch between files - cannot safely combine!"
print("\nColumn check passed: all files share identical columns.")

combined = pd.concat(frames, axis=0, ignore_index=True)

print(f"\n=== COMBINED SHAPE ===")
print(combined.shape)

print(f"\n=== LABEL DISTRIBUTION (combined, untouched) ===")
print(combined["Label"].value_counts())

print(f"\n=== ROWS PER SOURCE FILE (sanity check) ===")
print(combined["Source_File"].value_counts())

combined.to_csv(OUT_PATH, index=False)
print(f"\nSaved combined (but still UNCLEANED) dataset to: {OUT_PATH}")
print("Original 8 CSV files in raw_data/ were not modified.")
