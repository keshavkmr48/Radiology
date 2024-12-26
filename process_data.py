import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToPILImage

# Convert the tensor to a PIL image
to_pil = ToPILImage()


class RadiologyDataset(Dataset):
    def __init__(self, raw_data_dir, processed_data_dir, include_masks=False, transform=None):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.include_masks = include_masks
        self.transform = transform

        # Ensure processed data directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Process data if not already processed
        self.data = self.load_or_process_data()

    def load_or_process_data(self):
        categories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
        processed_data = []

        for idx, category in enumerate(categories):
            raw_image_dir = os.path.join(self.raw_data_dir, category, "images")
            raw_mask_dir = os.path.join(self.raw_data_dir, category, "masks") if self.include_masks else None

            processed_image_dir = os.path.join(self.processed_data_dir, category, "images")
            processed_mask_dir = os.path.join(self.processed_data_dir, category, "masks") if self.include_masks else None

            os.makedirs(processed_image_dir, exist_ok=True)
            if self.include_masks:
                os.makedirs(processed_mask_dir, exist_ok=True)

            for image_name in os.listdir(raw_image_dir):
                processed_image_path = os.path.join(processed_image_dir, image_name)
                raw_image_path = os.path.join(raw_image_dir, image_name)

                # Check if processed image exists
                if not os.path.exists(processed_image_path):
                    # Process and save the image
                    image = Image.open(raw_image_path).convert("L")  # Grayscale
                    if self.transform:
                        image = self.transform(image)
                    pil_image=to_pil(image)
                    pil_image.save(processed_image_path)

                # Process mask if applicable
                mask = None
                if self.include_masks:
                    raw_mask_path = os.path.join(raw_mask_dir, image_name)
                    processed_mask_path = os.path.join(processed_mask_dir, image_name)

                    if not os.path.exists(processed_mask_path) and os.path.exists(raw_mask_path):
                        mask = Image.open(raw_mask_path).convert("L")
                        if self.transform:
                            mask = self.transform(mask)
                        mask.save(processed_mask_path)

                # Append to dataset
                processed_data.append({
                    "image_path": processed_image_path,
                    "mask_path": processed_mask_path if self.include_masks else None,
                    "label": idx
                })

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample["image_path"])
        image = transforms.ToTensor()(image)

        if self.include_masks:
            mask = Image.open(sample["mask_path"])
            mask = transforms.ToTensor()(mask)
            return image, mask, sample["label"]

        return image, sample["label"]


def prepare_dataloader(
    raw_data_dir,
    processed_data_dir="./data/processed_data",
    batch_size=16,
    include_masks=False,
    augment=False
):
    # Define the transformations to convert grayscale to RGB
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((299, 299)),  # Resize to 299x299 for ResNet input
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
    ])
    
    # Add augmentations if requested
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transform  # Add the original transformation at the end
        ])

    dataset = RadiologyDataset(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        include_masks=include_masks,
        transform=transform
    )

    return dataset
