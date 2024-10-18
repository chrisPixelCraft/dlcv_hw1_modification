import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)
from torchvision.models import ResNet50_Weights, ResNet101_Weights


class FCN32s(nn.Module):
    def __init__(self, num_classes=7):
        super(FCN32s, self).__init__()

        # Load pretrained VGG16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Use features from VGG16 (excluding fully connected layers and last max pooling)
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(
                512, 4096, kernel_size=7
            ),  # No padding to reduce spatial dimensions
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )

        # Transposed convolution for 32x upsampling
        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        input_size = x.size()[2:]  # Store original input size
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x)

        # Crop to original input size
        x = x[:, :, : input_size[0], : input_size[1]]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Deeplabv3_Resnet50_Model(nn.Module):
    def __init__(self):
        super(Deeplabv3_Resnet50_Model, self).__init__()
        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.DEFAULT,
        )

        self.model.classifier[4] = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(256, 7, 1, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x["out"], x["aux"]
        # return output["out"]
