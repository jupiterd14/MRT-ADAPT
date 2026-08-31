# import_excel_to_db.py
import pandas as pd
import sqlite3
import os

# Find your Excel file (replace with actual filename)
excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
print(f"Found Excel files: {excel_files}")

# Select the file (adjust name as needed)
excel_file = excel_files[0]  # or put the exact name
print(f"Importing: {excel_file}")

# Read the Excel file
df = pd.read_excel(excel_file)

# Connect to your database
conn = sqlite3.connect('mrt.db')

# Import to a new table
df.to_sql('historical_congestion_data', conn, if_exists='replace', index=False)

print(f"✅ Imported {len(df)} records to database")
print(f"Columns: {df.columns.tolist()}")
print(f"First 5 rows:\n{df.head()}")

conn.close()