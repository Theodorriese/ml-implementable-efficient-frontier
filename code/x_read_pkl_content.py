import pandas as pd
import os

# Define file paths
folder_path = r"C:\Master\Data\Generated\Portfolios\demo"
file_name = "ret_cf.pkl"

# Read and display each file's content

file_path = os.path.join(folder_path, file_name)

data = pd.read_pickle(file_path)
print(f"\nContents of '{file_name}':")

print(data.head())
print(f"\nColumns: {data.columns.tolist()}")
print(f"Number of rows: {len(data)}")
