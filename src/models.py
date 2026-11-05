"""The two kinds of models in the project.

1. ``PedCNN`` - my own small CNN trained from scratch. The classifier head
   (hidden size, activation, number of hidden layers, weight init) is
   configurable, so I can run the three hyperparameter configs from the task.
2. ``mobilenet_v2_transfer`` - MobileNetV2 pretrained on ImageNet with a new
   3-class head (transfer learning).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def init_weights(module: nn.Module, method: str) -> None:
    """``XavierUniform`` or ``RandomNormal`` (the names used in the task text)."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if method == "XavierUniform":
            nn.init.xavier_uniform_(module.weight)
        elif method == "RandomNormal":
            nn.init.normal_(module.weight, mean=0.0, std=0.05)
        else:
            raise ValueError(f"unknown init {method}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def conv_block(c_in: int, c_out: int, activation: str, batchnorm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not batchnorm)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(c_out))
    layers += [ACTIVATIONS[activation](), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class PedCNN(nn.Module):
    """4 conv blocks (32-64-128-128 filters) -> global average pool -> dense head."""

    def __init__(
        self,
        num_classes: int = 3,
        activation: str = "relu",
        hidden_size: int = 64,
        hidden_layers: int = 1,
        init: str = "XavierUniform",
        batchnorm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        chans = [3, 32, 64, 128, 128]
        self.features = nn.Sequential(
            *[conv_block(chans[i], chans[i + 1], activation, batchnorm) for i in range(4)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        head: list[nn.Module] = [nn.Flatten()]
        width = chans[-1]
        for _ in range(hidden_layers):
            head += [nn.Linear(width, hidden_size), ACTIVATIONS[activation]()]
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            width = hidden_size
        head.append(nn.Linear(width, num_classes))
        self.classifier = nn.Sequential(*head)
        self.apply(lambda m: init_weights(m, init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def mobilenet_v2_transfer(num_classes: int = 3, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 (ImageNet weights) with a new classifier for my 3 classes."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.last_channel, num_classes))
    return net


def set_backbone_trainable(net: nn.Module, trainable: bool, last_blocks: int | None = None) -> None:
    """Freeze or unfreeze MobileNetV2 features. ``last_blocks`` unfreezes only the last N blocks."""
    blocks = list(net.features.children())
    for i, block in enumerate(blocks):
        on = trainable and (last_blocks is None or i >= len(blocks) - last_blocks)
        for p in block.parameters():
            p.requires_grad = on


def count_params(net: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable
