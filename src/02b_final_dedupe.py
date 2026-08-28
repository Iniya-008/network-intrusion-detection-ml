import pandas as pd
import gc

print("Loading cleaned dataset for final cross-file dedupe pass...")
df = pd.read_csv("dataset/combined_cleaned.csv")
print(f"Before: {df.shape[0]:,} rows")

# Drop duplicates based on all columns EXCEPT Source_File
# (Source_File differing shouldn't count two identical flows as unique)
cols_to_check = [c for c in df.columns if c != "Source_File"]
before = df.shape[0]
df = df.drop_duplicates(subset=cols_to_check, keep="first")
after = df.shape[0]

print(f"After removing {before - after:,} cross-file duplicates: {after:,} rows")

df.to_csv("dataset/combined_cleaned.csv", index=False)
print("Overwrote dataset/combined_cleaned.csv with fully deduplicated data.")

del df
gc.collect()
