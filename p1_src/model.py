import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Backbone(nn.Module):
    def __init__(self, num_classes=65, pretrained_backbone=None, freeze_backbone=False):
        super(Backbone, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.fc = nn.Linear(self.backbone.fc.in_features, 1000)

        model_dict = self.backbone.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_backbone.items() if k in model_dict}
        if pretrained_backbone is not None:
            model_dict.update(pretrained_backbone)
            self.backbone.load_state_dict(model_dict)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        # x = self.fc(x)
        return x


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=65, pretrained_backbone=None, freeze_backbone=False):
        super(ImageClassifier, self).__init__()

        self.backbone = Backbone(num_classes, pretrained_backbone, freeze_backbone)
        # self.fc = nn.Linear(self.backbone.in_features, num_classes)

        self.additional_layers = nn.Sequential(
            # plus the fc in backbone and 1 fc here, we only have 2 fc layers (avoid overfitting)
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def get_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        x = self.backbone(x)
        x = self.additional_layers(x)
        return x
