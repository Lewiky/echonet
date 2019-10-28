from typing import NamedTuple
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batch_2d_conv1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(
            # In a pooling operation, the input channels == output channels
            in_channels=self.conv1.out_channels,
            out_channels=64,
            kernel_size=(5,5),
            padding=(2,2),
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batch_2d_conv2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = nn.Linear(4096, 1024)
        self.initialise_layer(self.fc1)
        self.batch_1d_fc1 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.batch_2d_conv1(self.conv1(images)))
        x = self.pool1(x)
        x = F.relu(self.batch_2d_conv2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.batch_1d_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
