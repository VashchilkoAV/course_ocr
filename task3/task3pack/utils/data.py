from collections import defaultdict
from typing import Optional, Callable, List, Any, Tuple

import numpy as np
from PIL import Image
import cv2
import pandas
import os

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F

class SegmentationDataset(VisionDataset):
    def __init__(self, images_path, csv_header, csv_encoding='utf-16', csv_path=None, image_transforms=None, target_transforms=None, is_test=False, shape=[512, 512]):
        if not is_test:
            self.markup = pandas.read_csv(csv_path, header=None, names=csv_header, encoding=csv_encoding)
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.images_path = images_path
        self.is_test=is_test
        self.shape = shape

    def __len__(self):
        if self.is_test:
            return len(os.listdir(self.images_path))
        else:
            return len(self.markup)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.is_test:
            image_name = os.listdir(self.images_path)[index]
            image = Image.open(self.images_path / image_name)
            
        else:
            #return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512)
            
            image_name = self.markup.loc[index, 'path']
            coords = self.markup.loc[index, ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].to_numpy(dtype=int).reshape(4, 2)
            if not (self.images_path / image_name).exists():
                return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512)
            
            image = Image.open(self.images_path / image_name)

            target = np.zeros((image.size[1], image.size[0]))
            target = torch.FloatTensor(cv2.fillConvexPoly(target, coords, (1, )))

            
        if self.image_transforms is not None:
            image = self.image_transforms(image)
            
            
        if not self.is_test:
            if self.target_transforms is not None:
                target = self.target_transforms(target[None, ...])

            return image, target
        
        return image, image_name
    

def convert_segm_to_quadr(pred, image_size):
    pred = torch.where(pred > 0.99, 1., 0.)
    nonzero = pred[0].nonzero().flip(dims=[1])

    if nonzero.shape[0] == 0:
        return torch.zeros(4, 2)

    x = nonzero[:, 1]
    y = nonzero[:, 0]

    result = []
    result.append(nonzero[torch.argmin(x+y)][None, ...] / image_size[0])
    result.append(nonzero[torch.argmax(-x+y)][None, ...] / image_size[0])
    result.append(nonzero[torch.argmax(x+y)][None, ...] / image_size[0])
    result.append(nonzero[torch.argmax(x-y)][None, ...] / image_size[0])
    result = torch.cat(result)
    return result