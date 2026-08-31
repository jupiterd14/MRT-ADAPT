<<<<<<< HEAD
# check_data.py
import pandas as pd
import os

data_folder = 'data_new_2025'

if os.path.exists(data_folder):
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    for file in csv_files[:2]:  # Check first 2 files
        file_path = os.path.join(data_folder, file)
        print(f"\n📄 {file}")
        print("="*50)
        
        df = pd.read_csv(file_path)
        print(f"Columns: {list(df.columns)}")
        print(f"First 5 rows:")
        print(df.head())
        print(f"\nSample of TotalPassenger values: {df['TotalPassenger'].head(10).tolist()}")
        print(f"Min: {df['TotalPassenger'].min()}, Max: {df['TotalPassenger'].max()}, Avg: {df['TotalPassenger'].mean():.0f}")
else:
=======
# check_data.py
import pandas as pd
import os

data_folder = 'data_new_2025'

if os.path.exists(data_folder):
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    for file in csv_files[:2]:  # Check first 2 files
        file_path = os.path.join(data_folder, file)
        print(f"\n📄 {file}")
        print("="*50)
        
        df = pd.read_csv(file_path)
        print(f"Columns: {list(df.columns)}")
        print(f"First 5 rows:")
        print(df.head())
        print(f"\nSample of TotalPassenger values: {df['TotalPassenger'].head(10).tolist()}")
        print(f"Min: {df['TotalPassenger'].min()}, Max: {df['TotalPassenger'].max()}, Avg: {df['TotalPassenger'].mean():.0f}")
else:
>>>>>>> 02b8ad29728558911c1b62c16f2773b41ee9fad7
    print(f"❌ Folder '{data_folder}' not found!")