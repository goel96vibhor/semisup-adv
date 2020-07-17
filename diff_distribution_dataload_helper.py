import math
import os
import random
import sys
from torch.utils.data import DataLoader,Dataset
from os import path
import torch
# from scipy.misc import toimage
import torchvision
sys.path.append(path.abspath('CIFAR_modified/code'))

import CIFAR_modified
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from CIFAR_modified.code import utils
from custom_folder_loader import ImageFolder

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
      

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_cinic_dataset(train = False):

      cinic_directory = 'CINIC_dataset'
      cinic_mean = [0.47889522, 0.47227842, 0.43047404]
      cinic_std = [0.24205776, 0.23828046, 0.25874835]
      if train:
            cinic_dataset = ImageFolder(cinic_directory + '/train',
                        transform=transforms.Compose([
                        transforms.ToTensor()
                        # ,transforms.Normalize(mean=cinic_mean,std=cinic_std)
                        ]))
            
      else:            
            cinic_dataset = ImageFolder(cinic_directory + '/test',
                        # None, extensions = '.png',
                        transform=transforms.Compose([
                        transforms.ToTensor()
                        # ,transforms.Normalize(mean=cinic_mean,std=cinic_std)
                        ]))
                        
      # print(str(cinic_dataset[0:8][1]))
      # print(cinic_dataset.dtype)
      # targets = [x[1] for x in cinic_dataset]
      # print(targets[0:5])
      print(cinic_dataset.samples[0])
      print(cinic_dataset.__dict__.keys())
      return cinic_dataset


if __name__ == '__main__':
      get_new_distribution_loader()

