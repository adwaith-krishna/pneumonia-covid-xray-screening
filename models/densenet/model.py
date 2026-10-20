import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class DenseNet121CXR(nn.Module):
    """
    DenseNet-121 adapted for multi-class Chest X-Ray screening.
    Supports feature extraction for interpretability methods (Grad-CAM).
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Retrieve backbone output feature count (1024 for DenseNet-121)
        in_features = self.backbone.classifier.in_features

        # Replace classification head with custom dropout + linear projection
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_cam_target_layer(self) -> nn.Module:
        """
        Returns the final convolutional feature layer of DenseNet-121
        (features.norm5) for Grad-CAM computation.
        """
        return self.backbone.features.norm5


def build_densenet121(
    num_classes: int = 3,
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> DenseNet121CXR:
    """
    Factory function to instantiate DenseNet121CXR.
    """
    return DenseNet121CXR(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )


def get_densenet_cam_target_layer(model: DenseNet121CXR) -> nn.Module:
    """
    Convenience accessor for Grad-CAM target layer.
    """
    return model.get_cam_target_layer()
