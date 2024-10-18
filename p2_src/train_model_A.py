import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import torch.optim as optim
from model import FCN32s  # Assume you have this custom model class'
from mean_iou_evaluate import mean_iou_score
import numpy as np
from dataset import P2Dataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss(ignore_index=6)

    def forward(self, inputs, targets):
        ce_loss = self.loss(inputs, targets)
        exp_loss = torch.exp(-ce_loss)
        loss = self.alpha * (1 - exp_loss) ** self.gamma * ce_loss
        return loss


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device).float(), masks.to(device).long()
        masks = masks.squeeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_gt = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device).float(), masks.to(device).long()
            masks = masks.squeeze(1)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Calculate IoU
            pred = outputs.argmax(dim=1)

            pred = pred.detach().cpu().numpy().astype(np.int64)
            masks = masks.detach().cpu().numpy().astype(np.int64)

            all_preds.append(pred)
            all_gt.append(masks)

    mIoU = mean_iou_score(
        np.concatenate(all_preds, axis=0), np.concatenate(all_gt, axis=0)
    )

    return total_loss / len(val_loader), mIoU


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.0001
    num_epochs = 30

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 归一化
        ]
    )

    # visualize = standard_transforms.ToTensor()

    train_set = P2Dataset(
        os.path.join(parent_path, "hw1_data", "p2_data", "train"),
        train=True,
        transform=transform,
        augmentation=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True
    )

    val_dataset = P2Dataset(
        os.path.join(parent_path, "hw1_data", "p2_data", "validation"),
        train=True,
        transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    # Model
    # model = FCN32s().to(device)
    model = FCN32s().to(device)
    # model.copy_params_from_vgg16(models.vgg16(pretrained=True))

    # Loss and optimizer
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model_path = os.path.join(
        current_path,
        f"models_A_lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}",
    )
    os.makedirs(model_path, exist_ok=True)

    # Training loop
    best_iou = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_path, f"best_model_epoch_{epoch+1}_val_iou_{val_iou:.4f}.pt"
                ),
            )
            print(f"New best model saved! Epoch: {epoch+1}, Val IoU: {val_iou:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                },
                os.path.join(model_path, f"checkpoint_epoch_{epoch+1}.pt"),
            )


if __name__ == "__main__":
    main()
