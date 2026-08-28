import pandas as pd
import numpy as np
import glob
import os

files = sorted(glob.glob("raw_data/MachineLearningCVE/*.csv"))

summary_rows = []

for path in files:
    fname = os.path.basename(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # only for inspection, not saved back

    n_rows, n_cols = df.shape
    label_counts = df["Label"].value_counts()
    n_labels = len(label_counts)
    n_missing = df.isnull().sum().sum()
    numeric_df = df.select_dtypes(include=np.number)
    n_inf = np.isinf(numeric_df).sum().sum()
    n_dupe = df.duplicated().sum()

    summary_rows.append({
        "file": fname,
        "rows": n_rows,
        "cols": n_cols,
        "n_label_types": n_labels,
        "labels": ", ".join(f"{k}:{v}" for k, v in label_counts.items()),
        "missing_cells": n_missing,
        "infinite_cells": n_inf,
        "duplicate_rows": n_dupe,
    })
    print(f"Done: {fname}  (rows={n_rows}, cols={n_cols}, labels={n_labels}, missing={n_missing}, inf={n_inf}, dupes={n_dupe})")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("summary_all_files.csv", index=False)

print("\n=== FULL LABEL BREAKDOWN PER FILE ===")
for row in summary_rows:
    print(f"\n{row['file']}:")
    print(f"  {row['labels']}")

print("\n=== TOTALS ===")
print(f"Total rows across all files: {summary_df['rows'].sum()}")
print(f"Total duplicate rows across all files: {summary_df['duplicate_rows'].sum()}")
print(f"Total missing cells across all files: {summary_df['missing_cells'].sum()}")
print(f"Total infinite cells across all files: {summary_df['infinite_cells'].sum()}")

# check column consistency across files
print("\n=== COLUMN CONSISTENCY CHECK ===")
col_sets = []
for path in files:
    cols = tuple(pd.read_csv(path, nrows=0).columns.str.strip())
    col_sets.append((os.path.basename(path), cols))

base_cols = col_sets[0][1]
all_match = True
for fname, cols in col_sets[1:]:
    if cols != base_cols:
        all_match = False
        print(f"MISMATCH in {fname}")
        print(f"  Diff: {set(cols) ^ set(base_cols)}")
print(f"All files have identical (stripped) column sets: {all_match}")
