import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtTinyCXR(nn.Module):
    """
    ConvNeXt-Tiny adapted for multi-class Chest X-Ray screening.
    Supports feature extraction for interpretability methods (Grad-CAM).
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)

        # Retrieve backbone classifier projection input feature count (768 for ConvNeXt-Tiny)
        in_features = self.backbone.classifier[2].in_features

        # Replace classification projection with custom dropout + linear head
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_cam_target_layer(self) -> nn.Module:
        """
        Returns the final stage feature block of ConvNeXt-Tiny
        (backbone.features[-1]) for Grad-CAM computation.
        """
        return self.backbone.features[-1]


def build_convnext_tiny(
    num_classes: int = 3,
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> ConvNeXtTinyCXR:
    """
    Factory function to instantiate ConvNeXtTinyCXR.
    """
    return ConvNeXtTinyCXR(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )


def get_convnext_cam_target_layer(model: ConvNeXtTinyCXR) -> nn.Module:
    """
    Convenience accessor for Grad-CAM target layer.
    """
    return model.get_cam_target_layer()
