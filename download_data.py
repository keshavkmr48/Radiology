import os
import zipfile
import kaggle

def download_dataset(dataset_name, dest_folder):
    """
    Downloads and extracts data from Kaggle.

    Parameters:
    dataset_name (str): Kaggle dataset path in the format 'username/dataset-name'.
    dest_folder (str): Destination folder to extract the dataset.
    """
    # Set up Kaggle API credentials
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    kaggle.api.dataset_download_files(dataset_name, path=dest_folder, unzip=True)
    print(f"Dataset downloaded and extracted to {dest_folder}")

# Example usage
download_dataset('tawsifurrahman/covid19-radiography-database', './data/raw_data')
