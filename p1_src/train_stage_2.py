import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from model import ImageClassifier
from dataset import get_office_home_datasets
import os
from tqdm import tqdm
from torchvision import models
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def train(model, train_loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # print(outputs)
            _, predicted = outputs.max(1)
            # print(predicted)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            # Get the output of the second last layer
            feature = model.get_features(images)
            features.append(feature.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def visualize_tsne(features, labels, epoch, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis"
    )
    plt.colorbar(scatter)
    plt.title(f"t-SNE visualization - Epoch {epoch}")
    plt.savefig(save_path)
    plt.close()


def main(setting):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # print(f"Current directory: {current_dir}")
    # print(f"Parent directory: {parent_dir}")
    data_dir = os.path.join(parent_dir, "hw1_data", "p1_data", "office")
    # print(f"Data directory: {data_dir}")
    train_dataset, val_dataset = get_office_home_datasets(data_dir)

    num_epochs = 200
    best_accuracy = 0.0
    lr = 0.0001
    bs = 16

    if setting == "A":
        pretrained_model_number = "None"
    elif setting == "C" or setting == "E":
        pretrained_model_number = 70
    elif setting == "B" or setting == "D":
        pretrained_model_number = "SL"

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2)
    # print(train_loader)

    # Load pre-trained weights
    if setting == "A":
        pretrained_model_path = os.path.join(
            current_dir, "resnet_weight_none_backbone.pt"
        )
    elif setting == "C" or setting == "E":
        pretrained_model_path = os.path.join(
            current_dir, f"best_model_{pretrained_model_number}_BYOL.pt"
        )
    elif setting == "B" or setting == "D":
        pretrained_model_path = os.path.join(
            parent_dir, "hw1_data", "p1_data", "pretrain_model_SL.pt"
        )
    else:
        raise ValueError("Invalid setting. Choose 'A', 'B', 'C', 'D', 'E'.")
    print(f"Pretrained model path: {pretrained_model_path}")
    pretrained_dict = torch.load(pretrained_model_path)

    # Create a temporary ResNet50 model to load and filter the pretrained weights
    temp_model = models.resnet50(weights=None)
    temp_model_dict = temp_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in temp_model_dict}

    # Create the ImageClassifier with the filtered pretrained weights
    model = ImageClassifier(
        num_classes=65, pretrained_backbone=pretrained_dict, freeze_backbone=False
    )
    # temp_model_dict.update(pretrained_dict)
    # temp_model.load_state_dict(temp_model_dict)
    # model = temp_model
    # model.fc = nn.Linear(model.fc.in_features, 65)
    model.to(device)

    if setting == "A" or setting == "B" or setting == "C":
        # Fine-tune the entire model
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif setting == "D" or setting == "E":
        # Fix backbone, train classifier only
        model.freeze_backbone = True
        optimizer = optim.Adam(model.additional_layers.parameters(), lr=lr)
        # optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum =0.9)
    else:
        raise ValueError("Invalid setting. Choose 'A', 'B', 'C', 'D', 'E'.")

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Add scheduler
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # Add CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0
    )
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Create a directory to save models
    save_dir = os.path.join(
        current_dir,
        f"models_{pretrained_model_number}_{setting}_s2_lr_{lr}_bs_{bs}_ep{num_epochs}",
    )
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, scheduler, device)
        accuracy = validate(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}, Train Loss: {train_loss:.4f}"
        )

        # Extract features and visualize t-SNE for first and last epochs
        if epoch == 0 or epoch == num_epochs - 1:
            features, labels = extract_features(model, train_loader, device)
            visualize_tsne(
                features, labels, epoch + 1, f"tsne_visualization_epoch_{epoch+1}.png"
            )

        # Step the scheduler
        scheduler.step()

        # Save the model after each epoch
        # torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pt'))

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir, f"best_model_epoch_{epoch+1}_{best_accuracy:.4f}.pt"
                ),
            )
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    return model


if __name__ == "__main__":
    # model_A = main('A')
    # model_B = main('B')
    model_C = main("C")
    # model_D = main("D")
    # model_E = main("E")
