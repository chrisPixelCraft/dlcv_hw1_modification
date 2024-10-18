import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from model import ImageClassifier
from dataset import get_mini_datasets
import os
from tqdm import tqdm
from byol_pytorch import BYOL
from torchvision import models


def main(setting):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # print(f"Current directory: {current_dir}")
    # print(f"Parent directory: {parent_dir}")
    data_dir = os.path.join(parent_dir, "hw1_data", "p1_data", "mini")
    # print(f"Data directory: {data_dir}")
    train_dataset = get_mini_datasets(data_dir)

    lr = 0.005
    bs = 16
    num_epochs = 70

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)

    resnet = models.resnet50(weights=None)

    learner = BYOL(resnet, image_size=128, hidden_layer="avgpool").to(device)

    optimizer = torch.optim.SGD(
        learner.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    # optimizer = torch.optim.Adam(learner.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float("inf")
    save_dir = f"./models_{setting}_s1_lr_{lr}_bs_{bs}_ep_{num_epochs}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch.to(device)  # Move images to the same device as the model
            loss = learner(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        scheduler.step(avg_loss)

        # model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        # torch.save(resnet.state_dict(), model_path)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": resnet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(
                save_dir, f"best_model_{epoch+1}_{best_loss:.4f}.pt"
            )
            torch.save(resnet.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    print("Training completed.")
    return resnet


if __name__ == "__main__":
    model_C = main("C")
