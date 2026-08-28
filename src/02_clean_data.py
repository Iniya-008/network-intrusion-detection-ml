import pandas as pd
import numpy as np
import glob
import os

RAW_DIR = "raw_data/MachineLearningCVE"
OUT_PATH = "dataset/combined_cleaned.csv"
os.makedirs("dataset", exist_ok=True)

files = sorted(glob.glob(f"{RAW_DIR}/*.csv"))
print(f"Found {len(files)} files.\n")

# ---- Label text fix map ----
# The raw files use a corrupted character (encoding issue) inside some
# Web Attack labels, e.g. "Web Attack \x96 Brute Force".
# We standardize every label so there's exactly ONE consistent name
# per attack type.
def fix_label(label):
    label = str(label).strip()
    # Replace the corrupted dash character with a normal " - "
    for bad_char in ["\x96", "\ufffd", "�"]:
        label = label.replace(bad_char, "-")
    # Collapse any weird multiple spaces
    label = " ".join(label.split())
    return label

total_before = 0
total_after_dupe = 0
total_after_clean = 0
first_write = True

for path in files:
    fname = os.path.basename(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Source_File"] = fname

    n0 = df.shape[0]
    total_before += n0

    # 1) Fix corrupted label text
    df["Label"] = df["Label"].apply(fix_label)

    # 2) Remove exact duplicate rows (within this file)
    df = df.drop_duplicates()
    n1 = df.shape[0]

    # 3) Replace infinite values with NaN, then drop rows with any NaN
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    n2 = df.shape[0]

    total_after_dupe += n1
    total_after_clean += n2

    print(f"{fname}: {n0:,} -> after dedupe {n1:,} -> after fixing missing/inf {n2:,}")

    # Append to the cleaned combined file, writing header only once
    df.to_csv(OUT_PATH, mode="w" if first_write else "a", header=first_write, index=False)
    first_write = False

    del df  # free memory before next file

print("\n" + "="*60)
print(f"TOTAL rows before cleaning : {total_before:,}")
print(f"TOTAL rows after removing duplicates : {total_after_dupe:,}")
print(f"TOTAL rows after fixing missing/infinite : {total_after_clean:,}")
print(f"Rows removed overall: {total_before - total_after_clean:,} ({(total_before-total_after_clean)/total_before*100:.2f}%)")
print(f"\nCleaned dataset written to: {OUT_PATH}")
print("Original 8 raw CSV files were NOT modified.")
