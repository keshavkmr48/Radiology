### Project Report

#### **Objective**
This project aims to classify radiology images into four categories (COVID, Lung Opacity, Normal, and Viral Pneumonia) using a deep learning model (ResNet18). The pipeline covers downloading data, preprocessing it, and training the model with early stopping and augmentation.

Medical images like radiology scans often contain subtle features (e.g., lung opacities or abnormalities). ResNet’s ability to learn hierarchical feature representations (low-level edges to high-level patterns) makes it ideal for such tasks. Its residual connections allow the model to retain key low-level features even in deeper layers, improving the ability to detect fine-grained details.

ResNet-18 is specifically fine-tuned here so that we can iterate faster during experimentations as it's a smaller model with fewer patterns compared to other variants of 50 and 101. 

---

### **Steps Performed**

#### 1. **Data Downloading (`download_data.py`)**
- **Purpose**: Fetch the dataset from Kaggle and save it locally in the `data/raw_data` folder.
- **Key Functions**:
  - **`download_dataset`**: Uses Kaggle API to download and extract datasets.
  - Ensures the destination folder exists before downloading.
  - Example dataset used: `'tawsifurrahman/covid19-radiography-database'`.
- **Output**: The dataset is saved in the `data/raw_data` directory with subfolders for each class (`COVID`, `Lung_Opacity`, etc.).

---

#### 2. **Data Processing (`process_data.py`)**
- **Purpose**: Process raw radiology images into a format suitable for training.
- **Dataset Class**:
  - **`RadiologyDataset`**:
    - Loads images and masks from `raw_data_dir`.
    - Converts images to grayscale and resizes them to 299x299 for ResNet and also normalized the image.
    - Saves processed images in the `processed_data_dir`.
    - Includes optional mask processing if `include_masks=True`.
    - Supports augmentations like random horizontal flips and rotations.
- **Key Functions**:
  - `prepare_dataloader`: Creates a dataset with transformations and processes data into tensors ready for training.
- **Output**: Processed data is saved in `data/processed_data`.

---

#### 3. **Model Training (`training.py`)**
- **Purpose**: Train a ResNet18 model on the processed dataset with early stopping and checkpointing.
- **Pipeline**:
  - **Data Loading**: Splits data into training (80%) and validation (20%) using stratified sampling. Further Split the validation dataset into Validation and Test dataset in 1:1 ratio. 
  - **Model Definition**: 
    - Uses pretrained ResNet18 with the last layer modified to classify into four categories.
    - Applies ResNet-specific normalization.
  - **Optimization**:
    - Uses CrossEntropyLoss for classification.
    - Adam optimizer with a learning rate of 0.001.
  - **Training Loop**:
    - Tracks training/validation loss and accuracy.
    - Saves the best-performing model to `models/best_model.pth`.
    - Implements early stopping with a patience of 5 epochs.
- **Output**:
  - Trained model stored in `models/best_model.pth`.
  - Intermediate checkpoints saved in `models/checkpoint.pth`.

---

### **Instructions for Reproducing the Project**

1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```
2. **Create Required Directories**:
    Run the script to create follwoing folders: data, data/raw_data, data/processed_data
   ```bash
   python create_folders.py


2. **Set Up the Environment**:
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure the Kaggle API is configured:
     - Download your Kaggle API key and save it as `kaggle.json` in your working directory.
     - Run the following command:
       ```bash
       export KAGGLE_CONFIG_DIR=.
       ```

3. **Download the Dataset**:
   Run the script to fetch the data:
   ```bash
   python download_data.py
   ```
   This will download and extract the dataset into `data/raw_data`.

4. **Process the Data**:
   Run the preprocessing script:
   ```bash
   python process_data.py
   ```
   This will convert raw images into processed format and save them in `data/processed_data`.

5. **Train the Model**:
   Start model training:
   ```bash
   python training.py
   ```
   - The script will train the model and save the best version in `models/best_model.pth`.
   - Early stopping will prevent overfitting.


---

### **Future Enhancements**
- Explore transfer learning with advanced architectures like EfficientNet.
- experiment with different hyper parameters for learning rate, batch size
- try to use learning rate schedulers (hyper parameter)


This report details all project steps and provides instructions to replicate the pipeline. Let me know if you'd like enhancements or specific sections!