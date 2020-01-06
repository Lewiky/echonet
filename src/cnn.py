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
        self.stride = (1,1)

        self.dropout= nn.Dropout(dropout)

        # 1st Conv. Layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=self.stride,
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv1)
        self.batch1 = nn.BatchNorm2d(self.conv1.out_channels)
        params = sum(p.numel() for p in self.conv1.parameters() if p.requires_grad)
        print(f"Number of params layer 1: {params}")

        # 2nd Conv. Layer
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=self.stride,
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv2)
        self.batch2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        params = sum(p.numel() for p in self.conv2.parameters() if p.requires_grad)
        print(f"Number of params layer 2: {params}")

        # 3rd Conv. layer
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=self.stride,
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv3)
        self.batch3 = nn.BatchNorm2d(self.conv3.out_channels)

        params = sum(p.numel() for p in self.conv3.parameters() if p.requires_grad)
        print(f"Number of params layer 3: {params}")

        # 4th conv. layer
        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        self.initialise_layer(self.conv4)
        self.batch4 = nn.BatchNorm2d(self.conv3.out_channels)

        params = sum(p.numel() for p in self.conv4.parameters() if p.requires_grad)
        print(f"Number of params layer 4: {params}")
        #self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2), padding=(1,1))

        self.fc1 = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc1)
        params = sum(p.numel() for p in self.fc1.parameters() if p.requires_grad)
        print(f"Number of params layer 5: {params}")

        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of params: {params}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.batch1(F.relu(self.conv1(images)))
        x = self.batch2(self.dropout(self.pool1(F.relu(self.conv2(x)))))
        x = self.batch3(F.relu(self.conv3(x)))
        x = self.batch4(self.dropout(F.relu(self.conv4(x))))
        #x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(torch.sigmoid(self.fc1(x)))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            if type(layer.bias) != type(None):
                nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class MLMC_CNN(CNN):
   def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float = 0.5):
       CNN.__init__(self, height, width, channels, class_count, dropout)
       # Tweak size of FC1 layer
       self.fc1 = nn.Linear(26048, 1024) 
