import os

# Define the folder structure
folders = [
    "data",
    "data/raw_data",
    "data/processed_data"
]

# Create folders if they don't exist
for folder in folders:
    try:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    except Exception as e:
        print(f"Error creating folder {folder}: {e}")
