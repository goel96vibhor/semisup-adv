import math
import os
import random
import sys
from torch.utils.data import DataLoader,Dataset
from os import path
import torch
# from scipy.misc import toimage
import pickle
import torchvision
sys.path.append(path.abspath('CIFAR_modified/code'))
import logging
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

def get_cinic_dataset(train = False, valid = False, load_from_checkpoint = True, cinic_type = 0):
      # print("Loading CINIC dataset using custom folder loader")
      if cinic_type == 0:
            cinic_directory = 'CINIC_dataset/cinic_imagenet'
      elif cinic_type == 1:
            cinic_directory = 'CINIC_dataset/cinic_both'
      else :
            cinic_directory = 'CINIC_dataset/cinic_cifar10'
      cinic_mean = [0.47889522, 0.47227842, 0.43047404]
      cinic_std = [0.24205776, 0.23828046, 0.25874835]
      assert not(train and valid), 'Both train and valid cannot be true'
      if train:
            train_valid_test = 0
      elif valid:
            train_valid_test = 1
      else:
            train_valid_test = 2            

      if load_from_checkpoint:
            cinic_dataset = pickle_load(cinic_directory, train_valid_test)
      else:      
            if train:
                  cinic_dataset = ImageFolder(cinic_directory, train_valid_test = 0,  
                              transform=transforms.Compose([
                              transforms.ToTensor()
                              # ,transforms.Normalize(mean=cinic_mean,std=cinic_std)
                              ]))
            elif valid:            
                  cinic_dataset = ImageFolder(cinic_directory, train_valid_test = 1, 
                              # None, extensions = '.png',
                              transform=transforms.Compose([
                              transforms.ToTensor()
                              # ,transforms.Normalize(mean=cinic_mean,std=cinic_std)
                              ]))      
            else:            
                  cinic_dataset = ImageFolder(cinic_directory, train_valid_test = 2, 
                              # None, extensions = '.png',
                              transform=transforms.Compose([
                              transforms.ToTensor()
                              # ,transforms.Normalize(mean=cinic_mean,std=cinic_std)
                              ]))

            pickle_dump(cinic_dataset, cinic_directory, train_valid_test)                              
                        
      # print(str(cinic_dataset[0:8][1]))
      # print(cinic_dataset.dtype)
      # targets = [x[1] for x in cinic_dataset]
      # print(targets[0:5])
      print(cinic_dataset.__dict__.keys())
      print(cinic_dataset.samples[0])
      
      return cinic_dataset



def pickle_load(dest_path, train_valid_test = 0):
        dest_path = dest_path + '_checkpoint'
        if train_valid_test == 0:
              dest_path = os.path.join(dest_path, 'train.pickle')
        elif train_valid_test == 1:
              dest_path = os.path.join(dest_path, 'valid.pickle')                         
        else:
              dest_path = os.path.join(dest_path, 'test.pickle')    
        assert os.path.exists(dest_path), 'Pickle file %s does not exist' %(dest_path)
        with open(dest_path, 'rb') as handle:
              temp_dict = pickle.load(handle)                    
       
        return temp_dict  


def pickle_dump(obj, dest_path, train_valid_test):
        dest_path = dest_path + '_checkpoint'
        if not os.path.exists(dest_path):
              os.makedirs(dest_path)
        if train_valid_test == 0:
              dest_path = os.path.join(dest_path, 'train.pickle')
        elif train_valid_test == 1:
              dest_path = os.path.join(dest_path, 'valid.pickle')                         
        else:
              dest_path = os.path.join(dest_path, 'test.pickle')    
        
        with open(dest_path, 'wb') as handle:
              pickle.dump(obj, handle)      

        logging.info("Dumped cinic file into pickle .. %s" %(dest_path))        

if __name__ == '__main__':
      get_new_distribution_loader()

