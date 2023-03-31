from collections import defaultdict
from typing import Optional, Callable, List, Any, Tuple

import numpy as np
from PIL import Image
import cv2

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F


class HeatmapDataset(VisionDataset):
    def __init__(self, data_packs, split='train', transforms=None, realative=True, output_size=(128, 128)):
        self.data_packs = data_packs
        self.indices = []
        self.transforms = transforms
        self.output_size = output_size
        self.split = split
        self.relative = realative

        for dp_idx, dp in enumerate(data_packs):
            for im_idx, im in enumerate(dp):
                if im.is_test_split() and split == 'test':
                    self.indices.append((dp_idx, im_idx))
                elif not im.is_test_split() and split == 'train':
                    self.indices.append((dp_idx, im_idx))
    

    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        dp_idx, im_idx = self.indices[index]
        image = np.array(self.data_packs[dp_idx][im_idx].image.convert('RGB'))
        target = torch.FloatTensor(self.data_packs[dp_idx][im_idx].quadrangle)
        
        if not self.relative:
            target *= self.output_size[0]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target
    
    def get_key(self, index: int) -> str:
        dp_idx, im_idx = self.indices[index]
        
        return self.data_packs[dp_idx][im_idx].unique_key
    

class SegmentationDataset(VisionDataset):
    def __init__(self, data_packs, split='train', image_transforms=None, target_transforms=None):
        self.data_packs = data_packs
        self.indices = []
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.split = split

        for dp_idx, dp in enumerate(data_packs):
            for im_idx, im in enumerate(dp):
                if im.is_test_split() and split == 'test':
                    self.indices.append((dp_idx, im_idx))
                elif not im.is_test_split() and split == 'train':
                    self.indices.append((dp_idx, im_idx))

    def __len__(self):
        return len(self.indices)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        dp_idx, im_idx = self.indices[index]
        image = np.array(self.data_packs[dp_idx][im_idx].image.convert('RGB'))

        target = np.zeros(np.array(self.data_packs[dp_idx][im_idx].image).shape[:2])
        target = torch.FloatTensor(cv2.fillConvexPoly(target, np.array(self.data_packs[dp_idx][im_idx].gt_data['quad']), (1, )))

        if self.image_transforms is not None:
            image = self.image_transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target[None, ...])

        return image, target
    
    def get_key(self, index: int) -> str:
        dp_idx, im_idx = self.indices[index]
        
        return self.data_packs[dp_idx][im_idx].unique_key
    

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