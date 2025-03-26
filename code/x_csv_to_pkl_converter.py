import os
import pandas as pd

# Define input and output directories
data_folder = r"C:\Master"

files_to_convert = {
    "chars_processed.csv": None,
    "data_ret_processed.csv": None,
    "ff3_m.csv": {"skiprows": 3, "usecols": ["yyyymm", "RF"], "sep": ","},
}

# Convert each file to Pickle
for file, options in files_to_convert.items():
    csv_path = os.path.join(data_folder, file)
    pkl_path = os.path.join(data_folder, file.replace(".csv", ".pkl").replace(".xlsx", ".pkl"))

    # Check if the file exists before converting
    if os.path.exists(csv_path):
        try:
            # Handle CSV and Excel files differently
            if options == "excel":
                df = pd.read_excel(csv_path)
            elif isinstance(options, dict):
                df = pd.read_csv(csv_path, **options)
            else:
                df = pd.read_csv(csv_path)

            df.to_pickle(pkl_path)
            print(f"Converted {file} → {os.path.basename(pkl_path)}")
        except Exception as e:
            print(f"Error: Could not convert {file}. Reason: {e}")
    else:
        print(f"Warning: {file} not found in {data_folder}, skipping.")

print("\nAll conversions completed!")


