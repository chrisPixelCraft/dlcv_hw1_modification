import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import random
from torchvision.transforms.functional import hflip, vflip
from copy import deepcopy
import glob
# import albumentations as A
from albumentations.pytorch import ToTensorV2


class P2Dataset(Dataset):
    def __init__(self, path, transform, train=False, augmentation=False) -> None:
        super().__init__()

        self.Train = train
        self.Transform = transform
        self.Image_names = sorted(glob.glob(os.path.join(path, "*.jpg")))

        if self.Train:
            self.Mask_names = sorted(glob.glob(os.path.join(path, "*.png")))

        if augmentation:
            pass
            # print(f"Using data augmentation")
            # self.augmentation = A.Compose(
            #     [
            #         A.HorizontalFlip(p=0.5),
            #         A.Rotate(limit=30, p=0.5),
            #         A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0), p=0.5),
            #         A.Affine(
            #             scale=(0.8, 1.2),
            #             rotate=(-30, 30),
            #             shear=(-10, 10),
            #             translate_percent=(0.1, 0.1),
            #             p=0.5,
            #         ),
            #         A.RandomBrightnessContrast(p=0.1),
            #         #                 A.VerticalFlip(p=0.5),
            #         # A.ToGray(p=0.1),
            #         # A.CoarseDropout(max_holes=1, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fp=0.5),
            #         # A.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #         # 0.229, 0.224, 0.225]),
            #     ],
            #     additional_targets={"mask": "mask"},
            # )
        else:
            self.augmentation = None

    def __getitem__(self, idx):
        if self.Train:
            img = np.array(Image.open(self.Image_names[idx]))
            mask = np.array(Image.open(self.Mask_names[idx]))

            # 掩膜预处理
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

            raw_mask = deepcopy(mask)

            mask[raw_mask == 3] = 0  # (Cyan: 011) Urban land
            mask[raw_mask == 6] = 1  # (Yellow: 110) Agriculture land
            mask[raw_mask == 5] = 2  # (Purple: 101) Rangeland
            mask[raw_mask == 2] = 3  # (Green: 010) Forest land
            mask[raw_mask == 1] = 4  # (Blue: 001) Water
            mask[raw_mask == 7] = 5  # (White: 111) Barren land
            mask[raw_mask == 0] = 6  # (Black: 000) Unknown

            if self.augmentation:

                augmented = self.augmentation(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

            img = self.Transform(Image.fromarray(img))
            mask = torch.tensor(mask, dtype=torch.long)

            return img, mask
        else:
            img = Image.open(self.Image_names[idx])
            img = self.Transform(img)

            return img, os.path.basename(self.Image_names[idx])

    def __len__(self):
        return len(self.Image_names)
