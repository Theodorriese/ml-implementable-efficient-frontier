import pandas as pd
import os


def load_and_filter(file_path, gvkey_value):
    """
    Load a CSV file and filter rows where 'gvkey' matches a specific value.

    Args:
        file_path (str): The path to the CSV file.
        gvkey_value (int): The gvkey value to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Filter rows where 'gvkey' matches the specified value
    filtered_df = df[df['gvkey'] == gvkey_value]

    return filtered_df


# Path to your 'eu.csv' file
file_path = "C:/Master/eu.csv"

# gvkey to filter
gvkey_value = 212354

# Load and filter the file
filtered_data = load_and_filter(file_path, gvkey_value)

# Print the first few rows of the filtered data
print(filtered_data.head())

# Optionally, save to a new CSV file
output_path = "C:/Master/filtered_eu.csv"
filtered_data.to_csv(output_path, index=False)
print(f"Filtered data saved to {output_path}")
