import kagglehub
import shutil
import os
import pandas as pd

print("Downloading dataset...")
path = kagglehub.dataset_download("shantanushukla2207/load-balancing-dataset")

raw_data_dir = "data/raw"
os.makedirs(raw_data_dir, exist_ok=True)

for file_name in os.listdir(path):
    source_file = os.path.join(path, file_name)
    destination_file = os.path.join(raw_data_dir, file_name)
    if os.path.isfile(source_file):
        shutil.copy(source_file, destination_file)
        print(f"Copied {file_name} to {raw_data_dir}/")

print("\nVerifying data...")
file_path = [f for f in os.listdir(path) if not f.startswith('.')][0]

if file_path.endswith('.csv'):
    df = pd.read_csv(os.path.join(raw_data_dir, file_path))
elif file_path.endswith('.xlsx'):
    df = pd.read_excel(os.path.join(raw_data_dir, file_path))
    # Convert and save as CSV
    csv_path = os.path.join(raw_data_dir, file_path.replace('.xlsx', '.csv'))
    df.to_csv(csv_path, index=False)
    print(f"Converted {file_path} to {csv_path}")
else:
    print(f"Unknown file type: {file_path}")
    df = None

if df is not None:
    print("First 5 records:\n", df.head())
