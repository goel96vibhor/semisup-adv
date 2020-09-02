import math
import os
import random
import sys
from torch.utils.data import DataLoader,Dataset
from os import path
import torch
# from scipy.misc import toimage
import torchvision
sys.path.append(path.abspath('../CIFAR_modified/code'))

import CIFAR_modified
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from CIFAR_modified.code import utils
from PIL import Image
# from custom_folder_loader import ImageFolder

class Empty_Dataset(Dataset):
    
    def __init__(self, root, download = False, train=False, transform=None, target_transform=None):

            self.transform = transform
            self.target_transform = target_transform
            self.train = train
            # print(self.transform)
            self.data = []
            self.targets = []

            

    def __getitem__(self, index):

            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            # img = Image.fromarray(img)

            if self.transform is not None:
                  # print("transforming image")
                  # print(img[1,:])
                  # print(img.dtype)
                  img = self.transform(img)
                  # print(self.transform)
                  # print(img[1,:])
                  # print(img.dtype)
            if self.target_transform is not None:
                  target = self.target_transform(target)

            return img, target

            # return self.data[index],self.targets[index]
    def __len__(self):
            return len(self.data)
  
