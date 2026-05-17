import os

def create_project_structure():
    # Define the folder name
    data_folder = 'data_new_2025'
    
    # List of subfolders you might need (e.g., for raw data or processed results)
    subfolders = ['raw', 'processed', 'exports']
    
    # 1. Create the main directory if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Successfully created: {data_folder}")
    else:
        print(f"Folder '{data_folder}' already exists.")

    # 2. Create subdirectories
    for sub in subfolders:
        path = os.path.join(data_folder, sub)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"  Created subfolder: {path}")

    # 3. Create a dummy placeholder file 
    # (This helps Git see the folder if it's empty)
    placeholder = os.path.join(data_folder, '.gitkeep')
    with open(placeholder, 'w') as f:
        f.write("This file ensures Git tracks this folder.")
    
    print("\nData structure is ready! You can now place your Excel files in 'data_new_2025/raw'.")

if __name__ == "__main__":
    create_project_structure()