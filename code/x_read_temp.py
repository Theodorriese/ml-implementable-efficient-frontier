import os
import pandas as pd

# Path to covariance matrix file
file_path = r"C:\Master\Data\Generated\Portfolios\20250217-1121_WEALTH10000000000.0_GAMMA10_SIZEperc_low50_high100_min40_INDTrue\covariance_matrix.pkl"

# Load the file
try:
    cov_data = pd.read_pickle(file_path)

    # Extract `barra_cov`
    barra_cov = cov_data.get("barra_cov", {})

    # Select a sample date (earliest available)
    sample_date = next(iter(barra_cov.keys()))  # Get the first date in barra_cov
    sample_entry = barra_cov[sample_date]

    print(f"\n🔵 🔍 Examining `barra_cov` for Date: {sample_date}\n")

    # Loop through each key inside `barra_cov[sample_date]`
    for key, value in sample_entry.items():
        print(f"\n📌 Key: {key}")

        if isinstance(value, pd.DataFrame):
            print(f"   🔹 DataFrame - Shape: {value.shape}")
            print(value.head(), "\n" + "-"*80 + "\n")

        elif isinstance(value, dict):
            print(f"   🔹 Dictionary - Keys: {list(value.keys())[:5]} ...")  # Print first 5 keys

        elif isinstance(value, list):
            print(f"   🔹 List - Length: {len(value)}")
            print(value[:5])  # Print first 5 elements

        else:
            print(f"   🔹 Other Type ({type(value)}): {value}")

except Exception as e:
    print(f"❌ Error loading covariance matrix: {e}")
