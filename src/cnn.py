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
        self.stride = (2,2)

        self.dropout= nn.Dropout(dropout)

        # 1st Conv. Layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=self.stride,
            padding=(43,21),
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
            padding=(43,21),
            bias=False
        )
        self.initialise_layer(self.conv2)
        self.batch2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2), padding=(1,1))

        params = sum(p.numel() for p in self.conv2.parameters() if p.requires_grad)
        print(f"Number of params layer 2: {params}")

        # 3rd Conv. layer
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=self.stride,
            padding=(22,11),
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
            stride=self.stride,
            padding=(1,1), 
            bias=False
        )
        self.initialise_layer(self.conv4)
        self.batch4 = nn.BatchNorm2d(self.conv3.out_channels)

        params = sum(p.numel() for p in self.conv4.parameters() if p.requires_grad)
        print(f"Number of params layer 4: {params}")
        #self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2), padding=(1,1))

        self.fc1 = nn.Linear(15488, 1024, bias=False)
        self.initialise_layer(self.fc1)
        params = sum(p.numel() for p in self.fc1.parameters() if p.requires_grad)
        print(f"Number of params layer 5: {params}")

        self.fc2 = nn.Linear(1024, 10, bias=False)
        self.initialise_layer(self.fc2)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of params: {params}")

        self.softy = nn.Softmax(dim=1)

    # Batch norm should come after relu: https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
    # Pooling can come before or after activation function: https://stackoverflow.com/questions/35543428/activation-function-after-pooling-layer-or-convolutional-layer
    # dropout: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout 
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = F.relu(x)
        x = self.batch1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.batch2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.batch4(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.softy(x)

        return x 

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            if type(layer.bias) != type(None):
                nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
