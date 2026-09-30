import os
import glob
import pyarrow.parquet as pq

# Location of your original Parquet files
SOURCE_DIR = r"dataset\raw\MachinelearningCVE"

# Location where the CSV files will be created
OUTPUT_DIR = r"raw_data\MachineLearningCVE"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all Parquet files
files = glob.glob(os.path.join(SOURCE_DIR, "*.parquet"))

print(f"Found {len(files)} Parquet files.\n")

for path in files:
    filename = os.path.basename(path)

    # Change .parquet to .csv
    output_name = os.path.splitext(filename)[0] + ".csv"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    print(f"Converting: {filename}")

    # Read Parquet file in batches to avoid using too much RAM
    parquet_file = pq.ParquetFile(path)

    first_batch = True

    for batch in parquet_file.iter_batches(batch_size=100000):
        df = batch.to_pandas()

        df.to_csv(
            output_path,
            mode="w" if first_batch else "a",
            header=first_batch,
            index=False
        )

        first_batch = False

    print(f"Saved: {output_path}\n")

print("========================================")
print("CONVERSION COMPLETE")
print("========================================")