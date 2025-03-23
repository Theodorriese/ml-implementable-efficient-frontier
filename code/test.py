import pandas as pd
import pickle
import os

# Specify the path where your .pkl file is stored
output_path = "C:\Master\Data\Generated\Portfolios\demo"  # Replace with your actual path

# Load the Portfolio-ML results
portfolio_ml_file = os.path.join(output_path, "portfolio-ml.pkl")

# Load the portfolio-ml.pkl file
try:
    pfml_df = pd.read_pickle(portfolio_ml_file)
    print("Loaded 'portfolio-ml.pkl' successfully.")
except Exception as e:
    print(f"Error loading 'portfolio-ml.pkl': {e}")

# Display the first few rows of the dataframe to inspect its structure
print("\nPreview of the 'portfolio-ml.pkl' content:")
print(pfml_df.head())

# Check the type of the loaded data to confirm what was saved
print(f"\nData type of loaded object: {type(pfml_df)}")

# If it's a DataFrame, show some basic statistics
if isinstance(pfml_df, pd.DataFrame):
    print("\nBasic statistics of the DataFrame:")
    print(pfml_df.describe())

# If you also want to check the 'hps.pkl' file
hps_file = os.path.join(output_path, "hps.pkl")

try:
    with open(hps_file, 'rb') as file:
        hps_data = pickle.load(file)
    print("\nLoaded 'hps.pkl' successfully.")
    print(f"Keys in 'hps.pkl': {list(hps_data.keys())}")
except Exception as e:
    print(f"Error loading 'hps.pkl': {e}")
