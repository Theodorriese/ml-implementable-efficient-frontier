#%%
# New code v2 - Loops through all years

import os
import zipfile
import pandas as pd
from glob import glob
import re
import shutil

print('Running new code (loop)')

# Define main directories
main_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_TSV_History_US"
output_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_CSV_US"
extract_root = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_Extraction_US"

# Ensure output and extraction directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(extract_root, exist_ok=True)

# List of all relevant year folders (2002-2024)
year_folders = sorted([f for f in os.listdir(main_folder) if re.match(r"\d{4}(-\d{4})?_BuySideInstP_All_AmerEqty", f)])

# Required columns (with corrected spacing)
required_columns = [
    "DXL Identifier", "ISIN", "SEDOL", "CUSIP", "Stock Description", "Market Area (Country level)",
    "Record Type", "Open Loan Transactions", "SAF", "SAR", "DCBS", "Indicative Fee",
    "Indicative Rebate", "Indicative Fee 1 Day", "Indicative Fee 7 Day", "Indicative Rebate 1 Day", "Indicative Rebate 7 Day"
]

# Process each year folder
for year_folder in year_folders:
    year = year_folder.split("_")[0]  # Extract year (or year range)
    zip_folder = os.path.join(main_folder, year_folder)  # Folder with ZIP files
    extract_folder = os.path.join(extract_root, f"{year}_Extracted_US")  # Folder for extracted TSVs
    output_csv = os.path.join(output_folder, f"{year}_Shortfees_US.csv")  # Final merged file

    print(f"Processing {year_folder}...")

    # Ensure extraction folder exists
    os.makedirs(extract_folder, exist_ok=True)

    # Unzip all files into unique subfolders inside `extract_folder`
    for zip_path in glob(os.path.join(zip_folder, "*.zip")):
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        temp_extract_folder = os.path.join(extract_folder, zip_name)  # Each ZIP gets its own subfolder
        os.makedirs(temp_extract_folder, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_extract_folder)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is a corrupted ZIP file and will be skipped.")

    # Get all extracted TSV files
    tsv_files = glob(os.path.join(extract_folder, "**", "*.tsv"), recursive=True)

    # Merge data
    all_data = []
    date_pattern = re.compile(r"(\d{8})")  # Regex to extract YYYYMMDD

    for tsv_file in tsv_files:
        filename = os.path.basename(tsv_file)
        parent_folder = os.path.basename(os.path.dirname(tsv_file))  # Get the folder name

        try:
            # Extract the date from the **parent folder name**
            match = date_pattern.search(parent_folder)
            if match:
                date_str = match.group(1)  # Extract YYYYMMDD part
                date = pd.to_datetime(date_str, format="%Y%m%d")
            else:
                print(f"Warning: No date found in folder name {parent_folder}. Skipping file {filename}.")
                continue  # Skip files without a valid date

            # Read the TSV file in chunks (for large files)
            for chunk in pd.read_csv(tsv_file, sep="\t", usecols=lambda x: x in required_columns, chunksize=10000):
                chunk.insert(0, "Date", date)  # Assigns extracted date in the first column
                all_data.append(chunk)

        except Exception as e:
            print(f"Error processing {tsv_file}: {e}")

    # Concatenate all data and save to CSV
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"CSV file saved at: {output_csv}")
        shutil.rmtree(extract_folder, ignore_errors=True) # Delete extraction files in extraction folder in each loop
        print(f"Deleted extracted files for {year_folder}.")
    else:
        print(f"No valid TSV files found for {year_folder}. No CSV file was created.")

print("Processing complete for all years.")


#%%
print('Merging csv files...')
import os
import pandas as pd
from glob import glob

# Define folder paths
csv_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_CSV_US"
output_csv = os.path.join(csv_folder, "Shortfees_US_2002_2024.csv")

# Get all CSV files in the folder
csv_files = sorted(glob(os.path.join(csv_folder, "*.csv")))  # Sorted ensures files are read in order

# Merge data
all_data = []

print(f"Found {len(csv_files)} CSV files. Processing...")

for file in csv_files:
    print(f"Reading {file}...")

    try:
        # Read CSV file
        df = pd.read_csv(file, low_memory=False)

        # Apply filters
        df = df[(df["Record Type"] == 1) &  # Filter: 'Record Type' = 1
                (df["Market Area (Country level)"] == "USA Equity") &  # Filter: 'Market Area' = 'USA Equity'
                (df["DCBS"].notna())]  # Filter: 'DCBS' must be non-empty

        # Append filtered data
        all_data.append(df)

    except Exception as e:
        print(f"Error reading {file}: {e}")

# Concatenate all data
if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)

    # Sort by Date and ISIN
    merged_df = merged_df.sort_values(by=["Date", "ISIN"])

    # Save merged CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged CSV saved at: {output_csv}")

else:
    print("No valid data found. Merged CSV was not created.")


#%%
# Code to change csv encoding to utf-8-sig
# I had a bug with Excel not being able open the merged csv files
file_path = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_CSV_US\Shortfees_US_2002_2024.csv"
df = pd.read_csv(file_path, low_memory=False)
df.to_csv(file_path, index=False, encoding="utf-8-sig")  # Use utf-8-sig for Excel compatibility
print("Rewritten CSV with UTF-8 encoding.")


#%%




#%%
# New code V1 - single
print('Running new code...')
import os
import zipfile
import pandas as pd
from glob import glob
import re  

# Define the main directory
main_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_TSV_History_US"
output_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_CSV_US"
extract_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_Extraction_US"
zip_folder = os.path.join(main_folder, "2019_BuySideInstP_All_AmerEqty")  # Folder with ZIP files
output_csv = os.path.join(output_folder, "2019_Shortfees_US.csv")  # Final merged file
extract_folder = os.path.join(main_folder, "2019_Extracted_US")  # Folder for extracted TSVs

# Ensure the extraction folder exists
os.makedirs(extract_folder, exist_ok=True)

# Unzip all files into unique subfolders inside `extract_folder`
for zip_path in glob(os.path.join(zip_folder, "*.zip")):
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    temp_extract_folder = os.path.join(extract_folder, zip_name)  # Each ZIP gets its own subfolder
    os.makedirs(temp_extract_folder, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_folder)
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is a corrupted ZIP file and will be skipped.")

# Get all extracted TSV files
tsv_files = glob(os.path.join(extract_folder, "**", "*.tsv"), recursive=True)

# List of required columns
required_columns = [
    "DXL Identifier", "ISIN", "SEDOL", "CUSIP", "Stock Description", "Market Area (Country level)",
    "Record Type", "Open Loan Transactions", "SAF", "SAR", "DCBS", "IndicativeFee",
    "IndicativeRebate", "IndicativeFee1Day", "IndicativeFee7Day", "IndicativeRebate1Day", "IndicativeRebate7Day"
]

# Merge data
print('Merging Data...')
all_data = []
date_pattern = re.compile(r"(\d{8})")  # Regex to extract YYYYMMDD

for tsv_file in tsv_files:
    filename = os.path.basename(tsv_file)
    parent_folder = os.path.basename(os.path.dirname(tsv_file))  # Get the folder name

    try:
        # Extract the date from the **parent folder name**
        match = date_pattern.search(parent_folder)
        if match:
            date_str = match.group(1)  # Extract YYYYMMDD part
            date = pd.to_datetime(date_str, format="%Y%m%d")
        else:
            print(f"Warning: No date found in folder name {parent_folder}. Skipping file {filename}.")
            continue  # Skip files without a valid date

        # Read the TSV file in chunks (for large files)
        for chunk in pd.read_csv(tsv_file, sep="\t", usecols=lambda x: x in required_columns, chunksize=10000):
            chunk.insert(0, "Date", date)  # Assigns extracted date in the first column
            all_data.append(chunk)

    except Exception as e:
        print(f"Error processing {tsv_file}: {e}")

# Concatenate all data and save to CSV
print('Creating csv...')
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"CSV file saved at: {output_csv}")
else:
    print("No valid TSV files found. No CSV file was created.")

#%%














#%%
# Old code

import os
import zipfile
import pandas as pd
from glob import glob
import re  

# Define the main directory
main_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_TSV_History_US"
zip_folder = os.path.join(main_folder, "2022_BuySideInstP_All_AmerEqty")  # Folder with ZIP files
output_csv = os.path.join(main_folder, "2022_zips_merged.csv")  # Final merged file
extract_folder = os.path.join(main_folder, "extracted_files_2022")  # Folder for extracted TSVs

# Ensure the extraction folder exists
os.makedirs(extract_folder, exist_ok=True)

# Unzip all files into unique subfolders inside `extract_folder`
for zip_path in glob(os.path.join(zip_folder, "*.zip")):
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    temp_extract_folder = os.path.join(extract_folder, zip_name)  # Each ZIP gets its own subfolder
    os.makedirs(temp_extract_folder, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_folder)
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is a corrupted ZIP file and will be skipped.")

# Get all extracted TSV files
tsv_files = glob(os.path.join(extract_folder, "**", "*.tsv"), recursive=True)

# Merge data
all_data = []
date_pattern = re.compile(r"(\d{8})")  # Regex to extract YYYYMMDD

for tsv_file in tsv_files:
    filename = os.path.basename(tsv_file)
    parent_folder = os.path.basename(os.path.dirname(tsv_file))  # Get the folder name

    try:
        # Extract the date from the **parent folder name**
        match = date_pattern.search(parent_folder)
        if match:
            date_str = match.group(1)  # Extract YYYYMMDD part
            date = pd.to_datetime(date_str, format="%Y%m%d")
        else:
            print(f"Warning: No date found in folder name {parent_folder}. Skipping file {filename}.")
            continue  # Skip files without a valid date

        # Read the TSV file in chunks (for large files)
        for chunk in pd.read_csv(tsv_file, sep="\t", chunksize=10000):
            chunk.insert(0, "Date", date)  # Assigns extracted date in the first column
            all_data.append(chunk)

    except Exception as e:
        print(f"Error processing {tsv_file}: {e}")

# Concatenate all data and save to CSV
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"CSV file saved at: {output_csv}")
else:
    print("No valid TSV files found. No CSV file was created.")

#%%



#%%
# Checking data in dfs
import pandas as pd
csv_file_check = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\0_CSV_US\Shortfees_US_2002_2024.csv"
df = pd.read_csv(csv_file_check)
pd.set_option('display.max_columns', None)
print(df.head())

#%%

# Filter for Apple's ISIN
apple_isin = "US0378331005"
df_apple = df[df["ISIN"] == apple_isin]

# Display the first few rows of Apple's data
print(df_apple)
