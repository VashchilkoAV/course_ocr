from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride, is_proj):
        super().__init__()

        self.main_forward = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

        if is_proj:
            self.skip_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1, padding=0)
        else:
            self.skip_connection = lambda x: x


    def forward(self, x):
        main = self.main_forward(x)
        skip = self.skip_connection(x)
        return main + skip


class Resnet34GrayscaleFeat(nn.Module):
    """
    Версия resnet, которую надо написать с нуля.
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self, num_classes=10):
        # полносверточная сеть, архитектуру можно найти в
        # https://arxiv.org/pdf/1512.03385.pdf, Table1
        super().__init__()
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, stride=2, kernel_size=7, padding=3),
            nn.MaxPool2d(stride=2, kernel_size=2),

            ResidualBlock(3, 64, 64, 1, False),
            ResidualBlock(3, 64, 64, 1, False),
            ResidualBlock(3, 64, 64, 1, False),

            ResidualBlock(3, 64, 128, 2, True),
            ResidualBlock(3, 128, 128, 1, False),
            ResidualBlock(3, 128, 128, 1, False),
            ResidualBlock(3, 128, 128, 1, False),

            ResidualBlock(3, 128, 256, 2, True),
            ResidualBlock(3, 256, 256, 1, False),
            ResidualBlock(3, 256, 256, 1, False),
            ResidualBlock(3, 256, 256, 1, False),
            ResidualBlock(3, 256, 256, 1, False),
            ResidualBlock(3, 256, 256, 1, False),

            ResidualBlock(3, 256, 512, 2, True),
            ResidualBlock(3, 512, 512, 1, False),
            ResidualBlock(3, 512, 512, 1, False),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_backbone(x)
        x = self.avgpool(x)

        y = self.fc(x)

        return y, x
        
