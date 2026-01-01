import pandas as pd
import glob

csv_files = glob.glob("*.csv")
print(f"Found {len(csv_files)} CSV files.\n")

for csv in csv_files:
    try:
        df = pd.read_csv(csv)
        print(f"--- {csv} ---")
        print("Columns:", df.columns.tolist())
        print()
    except Exception as e:
        print(f"--- {csv} ---")
        print(f"Error reading file: {e}")
        print()
