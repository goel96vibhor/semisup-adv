"""
Datasets with unlabeled (or pseudo-labeled) data
"""

from torchvision.datasets import CIFAR10, SVHN, MNIST
from torch.utils.data import Sampler, Dataset
import torch
import numpy as np# from PIL import Image
import cifar_own
import qmnist_own
import os
from PIL import Image
import pickle
# from qmnist import QMNIST
import logging
from torchvision import transforms
DATASETS = ['cifar10', 'svhn', 'cifar_own']
from diff_distribution_dataload_helper import *
from dataset_utils.benrecht_cifar10 import BenRecht_cifar10_dataset
from dataset_utils.tinyimages_80mn_loader import TinyImages

def to_tensor(x):
    t = torch.Tensor(x).transpose(2,0).transpose(1,2) / 255
    t -= torch.Tensor([0.4914, 0.4822, 0.4465]).reshape(3, 1 ,1)
    t /= torch.Tensor([0.2470, 0.2435, 0.2616]).reshape(3, 1 ,1)
    return t

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])

class TICifarDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 extend_dataset,
                 ti_indices_map,
                 targets,
                 ti_start_index = 0,
            #      base_targets,
            #      extend_targets,
                 train=False, 
                 transform = None,
                 used_targets = None,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""
        logger = logging.getLogger()
        self.base_dataset = base_dataset
        self.extend_dataset = extend_dataset
        self.ti_indices_map = ti_indices_map
        self.transform = transform
      #   self.base_targets = base_targets
      #   self.extend_targets = extend_targets
        self.targets = targets
        self.used_targets = used_targets
        self.base_dataset_size = len(base_dataset)
        self.ti_start_index = ti_start_index
        print("Base dataset size for TI CIFAR dataset %d, extend dataset size %d, ti indices map size %d, targets size %d, ti start index %d" 
                              %(self.base_dataset_size, len(self.extend_dataset), len(self.ti_indices_map), len(self.targets), self.ti_start_index))
        self.train = train
        
    @property
    def data(self):
        return self.base_dataset.data

#     @data.setter
#     def data(self, value):
#         self.dataset.data = value

#     @property
#     def targets(self):
#         return self.dataset.targets

#     @targets.setter
#     def targets(self, value):
#         self.dataset.targets = value

    def __len__(self):
        return len(self.base_dataset) + len(self.ti_indices_map)

    def __getitem__(self, item):
        if item >= self.base_dataset_size:
            # print('returning extend item')
            # item = item % self.base_dataset_size
            # extend_tuple = torch.tensor(self.extend_dataset[self.ti_indices_map[item-self.base_dataset_size]-self.ti_start_index].transpose([2,0,1]), dtype= torch.float32)
            extend_tuple = self.extend_dataset[self.ti_indices_map[item-self.base_dataset_size]-self.ti_start_index]
            extend_tuple = Image.fromarray(extend_tuple)
            train_transform = torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  # torchvision.transforms.Normalize(mean, std),
            ])
            if item < self.base_dataset_size+80:
              extend_tuple.save('selection_model/for_view/ti/'+str(item)+'.png')
            # # print(extend_tuple.dtype)
            if self.transform is not None:
                  extend_tuple = train_transform(extend_tuple)
            # extend_tuple = to_tensor(extend_tuple)
            # # print(extend_tuple.shape)
            # # print(extend_tuple.dtype)
            return (extend_tuple, self.targets[item], item)
            # return self.extend_dataset[self.ti_indices_map[item-self.base_dataset_size]]   # because torchvision is annoying
            # print(self.extend_dataset[self.ti_indices_map[item-self.base_dataset_size]])
      #   else:
      #       print('returning base item')
      #   print(self.base_dataset[item][0].shape)
      #   print(self.base_dataset[item][0].dtype)
        return self.base_dataset[item] 

#     def __repr__(self):
#         fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Training: {}\n'.format(self.train)
#         fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str


