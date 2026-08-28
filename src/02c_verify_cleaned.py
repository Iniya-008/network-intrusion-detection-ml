import pandas as pd
import numpy as np

# Read only the Label and Source_File columns to keep memory low
df = pd.read_csv("dataset/combined_cleaned.csv", usecols=["Label", "Source_File"])

print("=== SHAPE CHECK ===")
print(f"Rows: {df.shape[0]:,}")

print("\n=== LABEL DISTRIBUTION (cleaned) ===")
print(df["Label"].value_counts())

print("\n=== CHECK: any leftover corrupted characters in labels? ===")
bad = df["Label"].str.contains(r"[\x96\ufffd]", regex=True, na=False)
print(f"Rows with corrupted label text remaining: {bad.sum()}")

print("\n=== CHECK: duplicate rows remaining (within full file) ===")
# Note: we removed duplicates PER FILE. It's possible (rare) that an
# identical row appears in two different day-files. Let's check.
full_check = pd.read_csv("dataset/combined_cleaned.csv")
full_check_no_source = full_check.drop(columns=["Source_File"])
cross_file_dupes = full_check_no_source.duplicated().sum()
print(f"Duplicate rows across the whole cleaned file: {cross_file_dupes:,}")
