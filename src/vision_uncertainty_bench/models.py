"""Reference neural-network architectures for CIFAR-10 experiments."""

import torch
from torch import nn


class SmallCifarCNN(nn.Module):
    """A compact convolutional classifier that returns unnormalised logits."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalised class-score vector for each input image."""
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width].")

        if images.shape[1] != 3:
            raise ValueError("images must have three colour channels.")

        return self.classifier(self.features(images))