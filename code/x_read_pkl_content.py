import pandas as pd
import os

# Define file paths
folder_path = r"C:\Master\Data\Generated\Portfolios\demo"
file_names = ["pfml_cf_base.pkl", "tpf_cf_base.pkl"]

# Read and display each file's content
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    data = pd.read_pickle(file_path)
    print(f"\nContents of '{file_name}':")

    print(data.head())
    print(f"\nColumns: {data.columns.tolist()}")
    print(f"Number of rows: {len(data)}")
