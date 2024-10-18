import imageio
import numpy as np
import torch
from torchvision import transforms
import os
import sys
from dataset import P2Dataset
from model import Deeplabv3_Resnet50_Model
from tqdm import tqdm


def pred2image(batch_preds, batch_names, out_path):
    # batch_preds = (b, H, W)
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]
        pred_img[np.where(pred == 1)] = [255, 255, 0]
        pred_img[np.where(pred == 2)] = [255, 0, 255]
        pred_img[np.where(pred == 3)] = [0, 255, 0]
        pred_img[np.where(pred == 4)] = [0, 0, 255]
        pred_img[np.where(pred == 5)] = [255, 255, 255]
        pred_img[np.where(pred == 6)] = [0, 0, 0]
        imageio.imwrite(os.path.join(out_path, name.replace(".jpg", ".png")), pred_img)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Deeplabv3_Resnet50_Model()
model.load_state_dict(torch.load("p2_best_model.pt", map_location=device))
model = model.to(device)
model.eval()

input_dir = sys.argv[1]
output_dir = sys.argv[2]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_dataset = P2Dataset(
    input_dir,
    transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
    train=False,
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

os.makedirs(output_dir, exist_ok=True)

for batch, file in tqdm(test_loader, desc="Inference"):
    with torch.no_grad():
        batch = batch.to(device)
        out, _ = model(batch)
    pred = out.argmax(dim=1)
    pred2image(pred, file, output_dir)
