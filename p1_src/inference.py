import sys
import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

from model import ImageClassifier  # Your model definition
from tqdm import tqdm
import torchvision
import torch.nn as nn


def load_model(device, checkpoint_path):
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1000), nn.Dropout(p=0.5), nn.Linear(1000, 65)
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "model_state_dict"
        ]
    )
    return model.to(device)


def main(img_csv, img_dir, output_csv):
    # model = ImageClassifier()
    # # model.load_state_dict(torch.load("p1_src/best_model_epoch_144_0.5172.pt"))
    # model.load_state_dict(
    #     torch.load("p1_src/best_model_C.pth")
    # )  # acc with C is 0.5172 at epoch 144
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ImageClassifier()
    # model.load_state_dict(torch.load("p1_src/best_model_epoch_144_0.5172.pt"))
    checkpoint_path = "best_model_C.pt"
    # model = load_model(device, checkpoint_path)
    model = ImageClassifier()
    model.load_state_dict(torch.load("best_model_C.pt", map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    df = pd.read_csv(img_csv)
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    predictions = []

    for index, row in tqdm(
        list(df.iterrows()), total=len(df), desc="Inference Progress"
    ):
        img_path = os.path.join(img_dir, row["filename"])
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        predictions.append(
            {"id": row["id"], "filename": row["filename"], "label": predicted.item()}
        )

    df = pd.DataFrame(predictions).to_csv(output_csv, index=False)
    # Add this line at the end of the main function
    compare_accuracy(img_csv, output_csv)


def compare_accuracy(img_csv, output_csv):
    # Read the original and predicted CSV files
    original_df = pd.read_csv(img_csv)
    predicted_df = pd.read_csv(output_csv)

    # Merge the dataframes on 'id' and 'filename'
    merged_df = pd.merge(original_df, predicted_df, on=["id", "filename"])

    # Calculate accuracy
    correct_predictions = (merged_df["label_x"] == merged_df["label_y"]).sum()
    total_predictions = len(merged_df)
    accuracy = correct_predictions / total_predictions

    print(f"Inference Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    img_csv, img_dir, output_csv = sys.argv[1:4]
    main(img_csv, img_dir, output_csv)
