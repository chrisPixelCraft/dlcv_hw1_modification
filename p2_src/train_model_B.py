import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from dataset import P2Dataset
from model import Deeplabv3_Resnet50_Model
import numpy as np
import imageio.v2 as imageio
from mean_iou_evaluate import mean_iou_score


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


def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(train_loader, desc="Training"):

        images, masks = images.to(device).float(), masks.to(device).long()
        #         masks = masks.squeeze(1)

        # check masks value
        # if (masks < 0).any() or (masks >= 7).any():
        #     continue

        optimizer.zero_grad()
        outputs = model(images)

        #         logits = outputs["out"]
        #         aux_logits = outputs["aux"]
        logits, aux_logits = outputs

        # masks = torch.squeeze(masks, 1)  # remove channel dimension
        loss = criterion(logits, masks) + criterion(aux_logits, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_gt = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device).long()
            #             masks = masks.squeeze(1)  # Remove channel dimension if present

            outputs = model(images)
            logits, aux_logits = outputs
            loss = criterion(logits, masks) + criterion(aux_logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_gt.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_gt = torch.cat(all_gt, dim=0).cpu().numpy()
    mIoU = mean_iou_score(all_preds, all_gt)

    return total_loss / len(val_loader), mIoU


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Available GPUs: {num_gpus}")

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

    #     train_loader = DataLoader(
    #         train_dataset, batch_size=batch_size, shuffle=True, num_workers=6
    #     )
    #     val_loader = DataLoader(
    #         val_dataset, batch_size=2 * batch_size, shuffle=False, num_workers=6
    #     )

    pretrain_model_path = os.path.join(
        current_path, "best_model_epoch_18_mIoU_0.7363.pt"
    )

    model = Deeplabv3_Resnet50_Model()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrain_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    #     model = nn.DataParallel(model) # this cause different data augmentation in 2 gpu with miou acc bottleneck, unless random seed

    criterion = FocalLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    # Calculate total steps
    total_steps = len(train_loader) * num_epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 5,  # Reduced from 10x to 5x
        total_steps=total_steps,
        pct_start=0.2,  # Reduced warmup phase
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10,  # Changed from 25
        final_div_factor=1000,  # Reduced from 10000
    )

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=learning_rate,
    #     steps_per_epoch=len(train_loader),
    #     epochs=num_epochs,
    # )
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model_path = os.path.join(
        current_path,
        f"models_B_lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}_pretrained",
    )
    os.makedirs(model_path, exist_ok=True)

    # Training loop
    best_iou = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, miou = evaluate(model, val_loader, criterion, device)
        #         scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {miou:.4f}, lr: {current_lr}"
        )

        # Save best model
        if miou > best_iou:
            best_iou = miou
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_path, f"best_model_epoch_{epoch+1}_mIoU_{miou:.4f}.pt"
                ),
            )
            print(f"New best model saved! Epoch: {epoch+1}, mIoU: {miou:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mIoU": miou,
                },
                f"{model_path}/checkpoint_epoch_{epoch+1}.pth",
            )


if __name__ == "__main__":
    main()
