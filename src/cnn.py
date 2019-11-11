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
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float = 0.5):
        super().__init__()
        self.input_shape = ImageShape(
            height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout= nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(43,21)
        )
        self.initialise_layer(self.conv1)
        self.batch1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(43,21)
        )
        self.initialise_layer(self.conv2)
        self.batch2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1,1))
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(22,11))
        self.initialise_layer(self.conv3)
        self.batch3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1,1))
        self.initialise_layer(self.conv4)
        self.batch4 = nn.BatchNorm2d(self.conv3.out_channels)
        self.fc1 = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc1)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)zf
        x = F.sigmoid(x)
        x = self.fc2(x)
        return F.softmax(x)

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)