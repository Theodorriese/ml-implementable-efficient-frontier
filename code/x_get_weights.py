import os
import pickle
import pandas as pd

# Define the directory and file paths
output_path = r"C:\Users\theod\OneDrive - CBS - Copenhagen Business School\4. semester FIN - Master"
pkl_path = os.path.join(output_path, "portfolio-ml.pkl")
csv_path = os.path.join(output_path, "portfolio-ml_.csv")


# Load the portfolio-ML results from the pickle file
with open(pkl_path, "rb") as f:
    pfml_results = pickle.load(f)

# Extract the weights (assumed to be under key "w")
weights = pfml_results.get("w")

# Convert to DataFrame if not already one
if not isinstance(weights, pd.DataFrame):
    try:
        weights = pd.DataFrame(weights)
    except Exception as e:
        print("Error converting weights to DataFrame:", e)

# Save the weights to CSV
weights.to_csv(csv_path)
print("Portfolio weights saved to:", csv_path)


# # Load the weights DataFrame
# with open(pkl_path, "rb") as f:
#     weights = pickle.load(f)
#
# # Show a preview
# print("✅ Loaded weights:")
# print(weights.head())
#
# # Save only the first 20,000 rows
# weights.head(40000).to_csv(csv_path, index=False)
# print("Export done.")

