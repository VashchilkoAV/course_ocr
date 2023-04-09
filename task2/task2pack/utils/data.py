from collections import defaultdict
from typing import Optional, Callable, List, Any, Tuple

import numpy as np
from PIL import Image
import cv2

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F


import torch

from torch.utils.data import Dataset, DataLoader
from torch import nn

class HWDBDataset(Dataset):
    def __init__(self, helper, transforms=None):
        self.helper = helper
        self.transforms=transforms
    
    def __len__(self):
        return self.helper.size()
    
    def __getitem__(self, idx):
        img, label = self.helper.get_item(idx)
        tensor_img = torch.FloatTensor((cv2.resize(img, (32, 32)) - 127.5) / 255.)[None, ...]
        if self.transforms is not None:
            tensor_img = self.transforms(tensor_img)
        return tensor_img, label