import os
import pandas as pd
import numpy as np

def save_obj_to_csv(obj, output_path_base, key_path=""):
    """Recursive function to handle nested dicts and export supported items to CSV."""
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(f"{output_path_base}_{key_path}.csv", index=False)
        print(f"Saved DataFrame: {key_path}")
    elif isinstance(obj, pd.Series):
        obj.to_frame().to_csv(f"{output_path_base}_{key_path}.csv")
        print(f"Saved Series: {key_path}")
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 1 or obj.ndim == 2:
            pd.DataFrame(obj).to_csv(f"{output_path_base}_{key_path}.csv", index=False)
            print(f"Saved ndarray: {key_path}")
        elif obj.ndim == 3:
            for i in range(obj.shape[0]):
                pd.DataFrame(obj[i]).to_csv(f"{output_path_base}_{key_path}_slice{i}.csv", index=False)
                print(f"Saved 3D ndarray slice: {key_path}_slice{i}")
    elif isinstance(obj, list):
        pd.DataFrame(obj).to_csv(f"{output_path_base}_{key_path}.csv", index=False)
        print(f"Saved list: {key_path}")
    elif isinstance(obj, dict):
        for sub_key, sub_val in obj.items():
            clean_key = str(sub_key).replace(" ", "_").replace(":", "").replace("/", "-")
            new_key_path = f"{key_path}_{clean_key}" if key_path else clean_key
            save_obj_to_csv(sub_val, output_path_base, new_key_path)
    else:
        print(f"Skipped unsupported type: {type(obj)} at {key_path}")

# --------- Configuration ---------
file_name = "model_1.pkl"
input_folder = r"C:\Master\Outputs"
output_folder = os.path.join(input_folder, "csv")  # Put all CSVs in /Outputs/csv
os.makedirs(output_folder, exist_ok=True)

full_path = os.path.join(input_folder, file_name)
output_path_base = os.path.join(output_folder, file_name.replace(".pkl", ""))

# --------- Run Conversion ---------
try:
    obj = pd.read_pickle(full_path)
    print(f"Loaded: {type(obj)}")
    save_obj_to_csv(obj, output_path_base)
except Exception as e:
    print(f"Error: {e}")
