import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from process_data import prepare_dataloader 
from sklearn.model_selection import StratifiedShuffleSplit

def train_model(
    raw_data_dir,
    processed_data_dir="./data/processed_data",
    model_save_dir="./models",
    num_epochs=10,
    batch_size=16,
    learning_rate=0.001,
    include_masks=False,
    augment=True,
    early_stopping_patience=5,
    checkpoint_path="./models/checkpoint.pth"
):
    # Prepare the full dataset (train + validation data) using the existing prepare_dataloader function
    full_dataset = prepare_dataloader(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        batch_size=batch_size,
        include_masks=include_masks,
        augment=augment
    )

    # Extract labels from the dataset
    labels = [item['label'] for item in full_dataset.data]  # Assuming each item has 'label'

        # Perform stratified sampling
    val_fraction = 0.2  # 20% validation data
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)

    # Get train and validation indices
    train_indices, val_indices = next(strat_split.split(range(len(full_dataset.data)), labels))

        # Subset the dataset using the indices
    train_dataset = Subset(full_dataset, train_indices)
    # val_dataset = Subset(full_dataset, val_indices)

    # Further split the validation dataset into valid and test subsets
    test_strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    # Extract labels for the validation dataset
    val_labels = [full_dataset.data[idx]['label'] for idx in val_indices]

    # Perform stratified sampling within val_dataset
    test_strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    test_indices_relative, valid_indices_relative = next(test_strat_split.split(range(len(val_indices)), val_labels))

    # Map relative indices back to the original validation indices
    test_indices = [val_indices[i] for i in test_indices_relative]
    valid_indices = [val_indices[i] for i in valid_indices_relative]

    # Create validation and test subsets
    valid_dataset = Subset(full_dataset, valid_indices)
    test_dataset = Subset(full_dataset, test_indices)



    # Create DataLoader for all training and validation and test datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    print("Training dataset Length:",len(train_dataloader.dataset))
    print("Test dataset Length:",len(test_dataloader.dataset))
    print("Validation dataset Length:",len(val_dataloader.dataset))


    # Initialize the model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  # 4 classes: COVID, Lung_Opacity, Normal, Viral Pneumonia

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)

    # Early stopping and checkpoint mechanism
    best_loss = float('inf')
    patience_counter = 0

    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct / total
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

        # Validation Phase
        model.eval()  # Switch model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # No need to compute gradients for validation
            for images, labels in tqdm(val_dataloader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss
        }
        torch.save(checkpoint, checkpoint_path)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model
            best_model_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Early stopping patience counter: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    return model, train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # Define paths
    raw_data_dir = "./data/raw_data/COVID-19_Radiography_Dataset"
    processed_data_dir = "./data/processed_data"
    model_save_dir = "./models"

    # Train the model
    model, train_dataloader, val_dataloader, test_dataloader=train_model(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        model_save_dir=model_save_dir,
        num_epochs=20,
        batch_size=32,
        learning_rate=0.001,
        include_masks=False,
        augment=True,
        early_stopping_patience=5,
        checkpoint_path="./models/checkpoint.pth"
    )
