import pandas as pd
import numpy as np

path = "raw_data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
df = pd.read_csv(path)

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMN NAMES ===")
for c in df.columns:
    print(repr(c))

print("\n=== DTYPES (unique) ===")
print(df.dtypes.value_counts())

print("\n=== LABEL DISTRIBUTION ===")
label_col = [c for c in df.columns if c.strip() == "Label"][0]
print(f"Label column found as: {repr(label_col)}")
print(df[label_col].value_counts())

print("\n=== MISSING VALUES (sum per column, only >0 shown) ===")
nulls = df.isnull().sum()
print(nulls[nulls > 0])
print(f"Total missing cells: {nulls.sum()}")

print("\n=== INFINITE VALUES ===")
numeric_df = df.select_dtypes(include=np.number)
inf_count = np.isinf(numeric_df).sum().sum()
print(f"Total infinite values: {inf_count}")

print("\n=== DUPLICATE ROWS ===")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n=== HEAD (first 3 rows, first 8 cols) ===")
print(df.iloc[:3, :8])
