import os
import pandas as pd
import numpy as np

# Input
file_name = "model_1.pkl"
input_folder = r"C:\Master\ml-implementable-efficient-frontier\code\Outputs"
output_folder = r"C:\Master\ml-implementable-efficient-frontier\code\Outputs"

# Setup paths
full_path = os.path.join(input_folder, file_name)
os.makedirs(output_folder, exist_ok=True)

# Load and handle
try:
    obj = pd.read_pickle(full_path)
    print(f"Loaded: {type(obj)}")

    if isinstance(obj, pd.DataFrame):
        obj.to_csv(os.path.join(output_folder, file_name.replace(".pkl", ".csv")), index=False)
        print("Saved DataFrame.")

    elif isinstance(obj, pd.Series):
        obj.to_frame().to_csv(os.path.join(output_folder, file_name.replace(".pkl", ".csv")))
        print("Saved Series.")

    elif isinstance(obj, dict):
        for k, v in obj.items():
            print(f"Processing key: {k} ({type(v)})")

            try:
                if isinstance(v, pd.DataFrame):
                    v.to_csv(os.path.join(output_folder, f"{file_name}_{k}.csv"), index=False)
                elif isinstance(v, pd.Series):
                    v.to_frame().to_csv(os.path.join(output_folder, f"{file_name}_{k}.csv"))
                elif isinstance(v, np.ndarray):
                    # Flatten if necessary
                    if v.ndim == 1:
                        pd.DataFrame(v).to_csv(os.path.join(output_folder, f"{file_name}_{k}.csv"), index=False)
                    elif v.ndim == 2:
                        pd.DataFrame(v).to_csv(os.path.join(output_folder, f"{file_name}_{k}.csv"), index=False)
                    elif v.ndim == 3:
                        for i in range(v.shape[0]):
                            slice_df = pd.DataFrame(v[i])
                            slice_df.to_csv(os.path.join(output_folder, f"{file_name}_{k}_slice{i}.csv"), index=False)
                elif isinstance(v, list):
                    pd.DataFrame(v).to_csv(os.path.join(output_folder, f"{file_name}_{k}.csv"), index=False)
                else:
                    print(f"Skipped {k} - unsupported type: {type(v)}")
            except Exception as e:
                print(f"Error processing key {k}: {e}")

    elif isinstance(obj, list):
        pd.DataFrame(obj).to_csv(os.path.join(output_folder, file_name.replace(".pkl", ".csv")), index=False)
        print("Saved list.")

    else:
        print("Unsupported object type.")

except Exception as e:
    print(f"Error converting {file_name}: {e}")
