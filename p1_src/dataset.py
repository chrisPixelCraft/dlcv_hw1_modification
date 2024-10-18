import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import autoaugment, RandomErasing
from glob import glob


class MiniDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.image_files = [
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.img_dir, f))
        ]

    def __len__(self):
        return len(glob(os.path.join(self.img_dir, "*")))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class OfficeHomeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        label = self.data_frame.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_mini_datasets(data_dir):
    train_img_dir = os.path.join(data_dir, "train")
    # Don't use data augmentation for mini dataset, it's already optimized in paper
    train_dataset = MiniDataset(
        train_img_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomResizedCrop(128, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                autoaugment.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                RandomErasing(
                    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
                ),
            ]
        ),
    )
    return train_dataset


def get_office_home_datasets(data_dir):
    train_csv = os.path.join(data_dir, "train.csv")
    print(f"Training CSV file: {train_csv}")
    val_csv = os.path.join(data_dir, "val.csv")
    train_img_dir = os.path.join(data_dir, "train")
    val_img_dir = os.path.join(data_dir, "val")

    train_dataset = OfficeHomeDataset(
        csv_file=train_csv,
        img_dir=train_img_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomResizedCrop(128, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                autoaugment.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                RandomErasing(
                    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
                ),
            ]
        ),
    )
    val_dataset = OfficeHomeDataset(csv_file=val_csv, img_dir=val_img_dir)

    return train_dataset, val_dataset
