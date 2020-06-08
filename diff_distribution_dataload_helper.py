import math
import os
import random
import sys
from torch.utils.data import DataLoader,Dataset
from os import path
import torch
from scipy.misc import toimage

sys.path.append(path.abspath('CIFAR_modified/code'))

import CIFAR_modified
from matplotlib import pyplot as plt
import numpy as np

from CIFAR_modified.code import utils

class my_dataset(Dataset):
    def __init__(self,data,targets):
        self.data=data
        self.targets=targets          
    def __getitem__(self, index):
        return self.data[index],self.targets[index]
    def __len__(self):
        return len(self.data)

def get_new_distribution_loader():
      cifar_label_names = utils.cifar10_label_names

      version = 'v4'
      images, labels = utils.load_new_test_data(version)
      images = images.transpose(0, 3, 1, 2)
      num_images = images.shape[0]
      images = images.astype(np.float32)/255.0
      # new_images = []
      # for i in range(images.shape[0]):
      #       mod_image = toimage(images[i])
      #       new_images.append(mod_image)
      labels = labels.astype(np.int64)
      print('\nLoaded version "{}" of the CIFAR-10.1 dataset.'.format(version))
      print('There are {} images in the dataset.'.format(num_images))


      image_index = random.randrange(num_images)
      # plt.figure()
      # plt.imshow(images[image_index,:,:,:])
      # plt.show()
      print(images.dtype)
      dl_kwargs = {'num_workers': 1, 'pin_memory': True}
      # test_loader = torch.utils.data.DataLoader(dataset = my_dataset(images, labels),
      #                                         batch_size=200,
      #                                         shuffle=False, **dl_kwargs)
      print('Class "{}"'.format(cifar_label_names[labels[image_index]]))
      return my_dataset(images, labels)
      


if __name__ == '__main__':
      get_new_distribution_loader()

