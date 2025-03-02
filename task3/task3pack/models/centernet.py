"""
Здесь находится backbone на основе resnet-18, в статье "Objects as Points" он описан в
5.Implementation details/Resnet и в Figure 6-b.
"""
from typing import Tuple

import torch
from torch import nn
from torchvision.models import resnet18, resnet34
import torch.nn.functional as F

from .argmax import SpatialSoftArgmax

class HeadlessPretrainedResnet18Encoder(nn.Module):
    """
    Предобученная на imagenet версия resnet, у которой
    нет avg-pool и fc слоев.
    Принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        super().__init__()
        md = resnet18(pretrained=True)
        # все, кроме avgpool и fc
        self.md = nn.Sequential(
            md.conv1,
            md.bn1,
            md.relu,
            md.maxpool,
            md.layer1,
            md.layer2,
            md.layer3,
            md.layer4
        )

    def forward(self, x):
        return self.md(x)


class HeadlessPretrainedResnet34Encoder(nn.Module):
    """
    Предобученная на imagenet версия resnet, у которой
    нет avg-pool и fc слоев.
    Принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        super().__init__()
        md = resnet34(pretrained=True)
        # все, кроме avgpool и fc
        self.md = nn.Sequential(
            md.conv1,
            md.bn1,
            md.relu,
            md.maxpool,
            md.layer1,
            md.layer2,
            md.layer3,
            md.layer4
        )

    def forward(self, x):
        return self.md(x)


class HeadlessResnet18Encoder(nn.Module):
    """
    Версия resnet, которую надо написать с нуля.
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        # полносверточная сеть, архитектуру можно найти в
        # https://arxiv.org/pdf/1512.03385.pdf, Table1
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, stride=2, kernel_size=7, padding=3),
            nn.MaxPool2d(stride=2, kernel_size=2),

            ResidualBlock(3, 64, 64, 1, False),
            ResidualBlock(3, 64, 64, 1, False),

            ResidualBlock(3, 64, 128, 2, True),
            ResidualBlock(3, 128, 128, 1, False),

            ResidualBlock(3, 128, 256, 2, True),
            ResidualBlock(3, 256, 256, 1, False),

            ResidualBlock(3, 256, 512, 2, True),
            ResidualBlock(3, 512, 512, 1, False)
        )
        

    def forward(self, x):
        return self.net(x)



class UpscaleTwiceLayer(nn.Module):
    """
    Слой, повышающий height и width в 2 раза.
    В реализации из "Objects as Points" используются Transposed Convolutions с
    отсылкой по деталям к https://arxiv.org/pdf/1804.06208.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=1):
        super().__init__()
        
        self.layer = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=2, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


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


class ResnetBackbone(nn.Module):
    """
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, C, H/R, W/R], где R = 4.
    C может быть выбрано разным, в конструкторе ниже C = 64.
    """
    def __init__(self, pretrained: str = "resnet18", out_channels=64):
        super().__init__()
        # downscale - fully-convolutional сеть, снижающая размерность в 32 раза
        if pretrained == "resnet18":
            self.downscale = HeadlessPretrainedResnet18Encoder()
        elif pretrained == "resnet34":
            self.downscale = HeadlessPretrainedResnet34Encoder()
        else:
            self.downscale = HeadlessResnet18Encoder()

        # upscale - fully-convolutional сеть из UpscaleTwiceLayer слоев, повышающая размерность в 2^3 раз
        downscale_channels = 512 # выход resnet
        channels = [downscale_channels, 256, 128, out_channels]
        layers_up = [
            UpscaleTwiceLayer(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ]
        self.upscale = nn.Sequential(*layers_up)

    def forward(self, x):
        x = self.downscale(x)
        x = self.upscale(x)
        return x
    

class CenterNetHead(nn.Module):
    """
    Принимает на вход тензор из Backbone input[B, K, W/R, H/R], где
    - B = batch_size
    - K = количество каналов (в ResnetBackbone K = 64)
    - H, W = размеры изображения на вход Backbone
    - R = output stride, т.е. во сколько раз featuremap меньше, чем исходное изображение
      (в ResnetBackbone R = 4)

    Возвращает тензора [B, C, W/R, H/R]:
    - первые C каналов: probs[B, С, W/R, H/R] - вероятности от 0 до 1
    """
    def __init__(self, k_in_channels=64, c_classes: int = 2):
        super().__init__()
        self.c_classes = c_classes

        self.soft_argmax = SpatialSoftArgmax()
        
        self.probs_head = nn.Sequential(
            nn.Conv2d(k_in_channels, k_in_channels,
                kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(k_in_channels, c_classes, 
                kernel_size=1, stride=1, 
                padding=0),
            #nn.Sigmoid()
            )



    def forward(self, input_t: torch.Tensor):
        class_heatmap = self.probs_head(input_t)

        #x_coordmap = torch.Tensor([[idx for _ in range(class_heatmap.shape[-2])] for idx in range(class_heatmap.shape[-1])]).to(class_heatmap.device)
        #y_coordmap = torch.Tensor([[idx for _ in range(class_heatmap.shape[-1])] for idx in range(class_heatmap.shape[-2])]).to(class_heatmap.device)

        #x_mass_center = (class_heatmap * x_coordmap).sum(dim=[-1, -2]) / class_heatmap.sum(dim=[-1, -2])# / class_heatmap.shape[-2]
        #y_mass_center = (class_heatmap * y_coordmap).sum(dim=[-1, -2]) / class_heatmap.sum(dim=[-1, -2])# / class_heatmap.shape[-1]

        #return torch.cat([x_mass_center[..., None], y_mass_center[..., None]], dim=-1)

        return self.soft_argmax(class_heatmap)


class CenterNet(nn.Module):
    """
    Детектор объектов из статьи 'Objects as Points': https://arxiv.org/pdf/1904.07850.pdf
    """
    def __init__(self, pretrained="resnet18", head_kwargs={}):
        super().__init__()
        self.backbone = ResnetBackbone(pretrained)
        self.head = CenterNetHead(**head_kwargs)

    def forward(self, input_t):
        x = input_t
        x = self.backbone(x)
        x = self.head(x)
        return x