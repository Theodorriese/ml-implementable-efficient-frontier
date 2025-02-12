import os
import pandas as pd

# Define the data folder and file paths
data_folder = r"C:\Master"
csv_file = os.path.join(data_folder, "usa_dsf.csv")
pkl_file = os.path.join(data_folder, "usa_dsf.pkl")

# Define the chunk size for reading
chunk_size = 10 ** 6

# Initialize an empty list to store chunks
chunks = []

try:
    print(f"Starting conversion of {csv_file} to {pkl_file} using chunks...")

    # Read the CSV file in chunks
    for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)):
        # Process each chunk: convert `date` to datetime
        chunk["date"] = pd.to_datetime(chunk["date"], format="%d/%m/%Y", errors="coerce")

        # Convert `RET` to numeric, coercing errors
        chunk["RET"] = pd.to_numeric(chunk["RET"], errors="coerce")

        # Drop rows with invalid or missing values in `RET` or `date`
        chunk = chunk.dropna(subset=["RET", "date"])

        # Append the processed chunk
        chunks.append(chunk)
        print(f"Processed chunk {i + 1} with {len(chunk)} rows.")

    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)

    # Save the final DataFrame to a Pickle file
    df.to_pickle(pkl_file)
    print(f"Successfully converted {csv_file} to {pkl_file}.")

except Exception as e:
    print(f"Error during conversion: {e}")
