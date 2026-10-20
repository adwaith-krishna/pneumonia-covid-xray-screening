import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3LargeCXR(nn.Module):
    """
    MobileNetV3-Large adapted for multi-class Chest X-Ray screening.
    Lightweight, ultra-fast inference with inverted residual blocks.
    Supports feature extraction for interpretability methods (Grad-CAM).
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)

        # Replace classification projection layer with custom head
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[2] = nn.Dropout(p=drop_rate)
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_cam_target_layer(self) -> nn.Module:
        """
        Returns the final 1x1 expansion convolutional block of MobileNetV3-Large
        (backbone.features[-1]) for Grad-CAM computation.
        """
        return self.backbone.features[-1]


def build_mobilenet_v3(
    num_classes: int = 3,
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> MobileNetV3LargeCXR:
    """
    Factory function to instantiate MobileNetV3LargeCXR.
    """
    return MobileNetV3LargeCXR(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )


def get_mobilenet_cam_target_layer(model: MobileNetV3LargeCXR) -> nn.Module:
    """
    Convenience accessor for Grad-CAM target layer.
    """
    return model.get_cam_target_layer()
