import os
import pickle
import pandas as pd

# Full path to the .pkl file
pkl_path = r"C:\Users\theod\OneDrive - CBS - Copenhagen Business School\4. semester FIN - Master\settings.pkl"
output_folder = r"C:\Users\theod\OneDrive - CBS - Copenhagen Business School\4. semester FIN - Master"
os.makedirs(output_folder, exist_ok=True)

# Output CSV path
csv_path = os.path.join(output_folder, "settings_converted.csv")

# Load the object
with open(pkl_path, "rb") as f:
    settings = pickle.load(f)

# Save to CSV depending on type
if isinstance(settings, pd.DataFrame):
    settings.to_csv(csv_path, index=False)
elif isinstance(settings, dict):
    df = pd.DataFrame.from_dict(settings, orient="index").reset_index()
    df.columns = ['key', 'value']  # Optional
    df.to_csv(csv_path, index=False)
else:
    raise TypeError(f"Unsupported type for CSV export: {type(settings)}")

print(f"✅ Exported settings to: {csv_path}")
