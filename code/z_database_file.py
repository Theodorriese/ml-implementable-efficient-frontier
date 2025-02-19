#%%
import sqlite3
import pandas as pd
import os

#%%

# Define paths
#data_folder = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data"
data_folder = r"C:\Users\johan\OneDrive\CBS\0_Speciale FIN\Data"
database_path = os.path.join(data_folder, "financial_data.db")

# Connect to SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Function to import a CSV into SQLite
def import_csv_to_sqlite(csv_path, conn):
    table_name = os.path.splitext(os.path.basename(csv_path))[0]  # Use filename as table name
    print(f"Importing {csv_path} into table {table_name}...")

    # Read CSV in chunks (to handle large files)
    chunk_size = 10000  # Adjust based on your system's memory
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk.to_sql(table_name, conn, if_exists="append", index=False)

    print(f"Imported {csv_path} into {table_name}")

# Loop through all CSV files in the folder
for file in os.listdir(data_folder):
    if file.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(data_folder, file)
        import_csv_to_sqlite(file_path, conn)

# Commit and close
conn.commit()
conn.close()
print("All CSV files have been imported into financial_data.db")
#%%

# Creates (SQL) database file - faster computation with SQL than csv?
conn = sqlite3.connect(r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data\Master_data.db")

df = pd.read_sql("SELECT * FROM usa LIMIT 10", conn)  # Replace 'usa' with any table name
print(df)

conn.close()
#%%


