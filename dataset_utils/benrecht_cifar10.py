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

class BenRecht_cifar10_dataset(Dataset):
    
    def __init__(self, root, download = False, train=False, transform=None, target_transform=None):

            self.get_new_distribution_loader()
            self.transform = transform
            self.target_transform = target_transform
            self.train = train

            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2470, 0.2435, 0.2616])

    def __getitem__(self, index):

            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            # img = Image.fromarray(img)

            if self.transform is not None:
                  img = self.transform(img)

            if self.target_transform is not None:
                  target = self.target_transform(target)

            return img, target

            # return self.data[index],self.targets[index]
    def __len__(self):
            return len(self.data)

    def get_new_distribution_loader(self):
            cifar_label_names = utils.cifar10_label_names

            version = 'v4'
            self.data, self.targets = utils.load_new_test_data(version)
            self.data = self.data.transpose(0, 1, 2, 3)

            # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            # self.data = self.data.transpose((0, 2, 3, 1))

            num_images = self.data.shape[0]
            
            # new_images = []
            # for i in range(images.shape[0]):
            #       mod_image = toimage(images[i])
            #       new_images.append(mod_image)
            self.targets = self.targets.astype(np.int64)
            print('\nLoaded version "{}" of the CIFAR-10.1 dataset.'.format(version))
            print('There are {} images in the dataset.'.format(num_images))


            image_index = random.randrange(num_images)
            # plt.figure()
            # plt.imshow(images[image_index,:,:,:])
            # plt.show()
            print(self.data.dtype)
            dl_kwargs = {'num_workers': 1, 'pin_memory': True}
            # test_loader = torch.utils.data.DataLoader(dataset = my_dataset(images, self.targets),
            #                                         batch_size=200,
            #                                         shuffle=False, **dl_kwargs)
            print('Class "{}"'.format(cifar_label_names[self.targets[image_index]]))
      
