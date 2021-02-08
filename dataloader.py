"""
Based on code from https://github.com/hysts/pytorch_shake_shake
"""

import numpy as np
import torch
import torchvision
import os
import pickle
from torch.utils import data
import pdb
from torchvision import transforms
from ti_cifar_dataset import *
import cifar_own
# from torchvision.datasets import ImageFolder
from imagefolder_loader import ImageFolder
from dataset_utils.tinyimages_80mn_loader import TinyImages
from datasets import SemiSupervisedDataset


def get_loader(batch_size, num_workers, use_gpu):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    dataset_dir = 'data'
    train_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader


# """ Function to load data for cifar0 vs TI classifier
# """
# def get_cifar10_vs_ti_loader(batch_size, num_workers, use_gpu,
#                              cifar_fraction=0.5, dataset_dir='data', custom_testset = None,
#                              logger=None):

#     # Normalization values for CIFAR-10
#     mean = np.array([0.4914, 0.4822, 0.4465])
#     std = np.array([0.2470, 0.2435, 0.2616])

#     train_transform = torchvision.transforms.Compose([
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean, std),
#     ])
#     test_transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#       #   torchvision.transforms.Normalize(mean, std),
#     ])

#     train_dataset = torchvision.datasets.CIFAR10(
#         dataset_dir, train=True, transform=train_transform, download=True)
#     if custom_testset is None:
#           test_dataset = torchvision.datasets.CIFAR10(
#                   dataset_dir, train=False, transform=test_transform, download=True)
#     else:
#           logger.info("using custom testset")
#           test_dataset = custom_testset


#     # Reading tinyimages and appropriate train/test indices
#     logger.info('Reading tiny images')
# #     ti_path = os.path.join(dataset_dir, 'tiny_images.bin')
# #     ti_data = np.memmap(ti_path, mode='r', dtype='uint8', order='F',
# #                         shape=(32, 32, 3, 79302017)).transpose([3, 0, 1, 2])
    
# #     logger.info('Size of tiny images {}'.format(ti_data.size))
# #     ti_indices_path = os.path.join(dataset_dir,
# #                                    'ti_vs_cifar_inds.pickle')
# #     with open(ti_indices_path, 'rb') as f:
# #         ti_indices = pickle.load(f)
# #     logger.info('Loaded TI indices')
    
# #     for dataset, name in zip((train_dataset, test_dataset), ('train', 'test')):
# #         dataset.data = np.concatenate((dataset.data, ti_data[ti_indices[name]]))
# #         # All tinyimages are given label 10
# #         dataset.targets.extend([10] * len(ti_indices[name]))

# #     logger.info('Calling train sampler')
# #     # Balancing training batches with CIFAR10 and TI
# #     train_sampler = BalancedSampler(
# #         train_dataset.targets, batch_size,
# #         balanced_fraction=cifar_fraction,
# #         num_batches=int(50000 / (batch_size * cifar_fraction)),
# #         label_to_balance=10, 
# #         logger=logger)
    
#     logger.info('Created train sampler')
#     train_loader = data.DataLoader(
#         train_dataset,
#       #   batch_sampler=train_sampler,
#         num_workers=num_workers,
#         pin_memory=use_gpu,
#     )

#     logger.info('Created train loader')
#     test_loader = data.DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         pin_memory=use_gpu,
#         drop_last=False,
#     )
#     logger.info('Created test loader')
#     return train_loader, test_loader


""" Function to load data for cifar10 vs TI classifier
"""
def get_cifar10_vs_ti_loader(batch_size, num_workers, use_gpu, num_images,
                            cifar_fraction=0.5, dataset_dir='data',
                            logger=None, even_odd=-1, load_ti_head_tail=-1,
                            use_ti_data_for_training=1, random_split_version=2,
                            ti_start_index=0):
    # Normalization values for CIFAR-10
    # num_images = 79302017
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(32, padding=4),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean, std),
    ])

    cifar_train_dataset = cifar_own.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True, 
        even_odd=even_odd, random_split_version=random_split_version)
    cifar_test_dataset = cifar_own.CIFAR10(
        dataset_dir, train=False, transform=test_transform, download=True)

    # Reading tinyimages and appropriate train/test indices
    if use_ti_data_for_training: 
        logger.info('Reading tiny images')
        if load_ti_head_tail == 0:
            ti_start_index += num_images
            ti_path = os.path.join(dataset_dir, 'tiny_250k_tail_v' + str(random_split_version) + '.bin')
        elif load_ti_head_tail == 1:
            # ti_start_index = 0
            ti_path = os.path.join(dataset_dir, 'tiny_250k_head_v'+ str(random_split_version) +'.bin')
        else:
            # ti_start_index = 0
            ti_path = os.path.join(dataset_dir, 'tiny_images.bin')
        
        ti_data = np.memmap(ti_path, mode='r', dtype='uint8', order='F',
                                shape=(32, 32, 3, num_images)).transpose([3, 0, 1, 2])
        
        logger.info('Size of tiny images {} loaded fom file {}'.format(ti_data.shape, ti_path))
        ti_indices_path = os.path.join(dataset_dir,
                                        'ti_vs_cifar_inds_v'+ str(random_split_version) +'.pickle')
        logger.info('Loaded TI indices from file %s' %(ti_indices_path))
        with open(ti_indices_path, 'rb') as f:
            ti_indices = pickle.load(f)
        # logger.info('Loaded TI indices with size %d' %(ti_indices.size))
        logger.info("Min of ti indixes train %d max %d" %(min(ti_indices['train']), max(ti_indices['train'])))
        if load_ti_head_tail >= 0:
            # ti_data = ti_data[ti_data%2==even_odd]
            if load_ti_head_tail == 0:
                ti_indices['train'] = ti_indices['train'][ti_indices['train']>= ti_start_index]
            else:
                ti_indices['train'] = ti_indices['train'][ti_indices['train']< ti_start_index + num_images]
        
        i = 0
        logger.info("ti train size %d, ti test size %d" %(ti_indices['train'].size, ti_indices['test'].size))

        cifar_train_dataset.targets.extend([10] * ti_indices['train'].size)
        cifar_test_dataset.targets.extend([10] * ti_indices['test'].size)
        
        train_dataset = TICifarDataset(cifar_train_dataset, ti_data, ti_indices['train'], cifar_train_dataset.targets, train = True, ti_start_index = ti_start_index,
                                        transform = train_transform)
        test_dataset = TICifarDataset(cifar_test_dataset, ti_data, ti_indices['test'], cifar_test_dataset.targets, train = False, ti_start_index = ti_start_index,
                                        transform = test_transform)
        logger.info('Calling train sampler')
        train_sampler = BalancedSampler(
                            train_dataset.targets, batch_size,
                            balanced_fraction=cifar_fraction,
                            num_batches=int(len(cifar_train_dataset) / (batch_size * cifar_fraction)),
                            label_to_balance=10, 
                            logger=logger
                        )
        logger.info('Created train sampler')
        train_loader = data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=use_gpu,
        )
    else:
        logger.info("not using TI data for training")
        train_dataset = cifar_train_dataset
        test_dataset = cifar_test_dataset
        train_sampler = None
        train_loader = data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            num_workers=num_workers,
            shuffle = True,
            pin_memory=use_gpu,
        )

    logger.info('Created train loader')
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    logger.info('Created test loader')
    return train_loader, test_loader


def get_tinyimages_loader(
        batch_size,
        dataset_dir='data/unlabeled_datasets/80M_Tiny_Images/tiny_50k.bin', 
        logger=None,
        num_images=50000):
    testset = SemiSupervisedDataset(
        base_dataset='tinyimages',
        dataset_dir=dataset_dir,
        logger=logger,
        num_images=num_images
    )
    test_loader = data.DataLoader(
        testset, 
        batch_size=batch_size
    )
    logger.info('Created tiny images test loader.')
    return test_loader


def get_cinic_dataset_loader(batch_size, num_workers, use_gpu):
    cinic_directory = 'data/CINIC_dataset'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    cinic_train = torch.utils.data.DataLoader(
        ImageFolder(cinic_directory + '/train',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean,std=cinic_std)
                    ])),
        
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    cinic_test = torch.utils.data.DataLoader(
        ImageFolder(cinic_directory + '/test',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=cinic_mean,std=cinic_std)
                    ])),

        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )

    return cinic_train, cinic_test

class BalancedSampler(data.Sampler):
    def __init__(self, labels, batch_size,
                 balanced_fraction=0.5,
                 num_batches=None,
                 label_to_balance=-1, 
                 logger=None):
        logger.info('Inside balanced sampler')
        self.minority_inds = [i for (i, label) in enumerate(labels)
                              if label != label_to_balance]
        self.majority_inds = [i for (i, label) in enumerate(labels)
                              if label == label_to_balance]
        self.batch_size = batch_size
        balanced_batch_size = int(batch_size * balanced_fraction)
        self.minority_batch_size = batch_size - balanced_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.minority_inds) / self.minority_batch_size))

        super().__init__(labels)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            minority_inds_shuffled = [self.minority_inds[i]
                                      for i in
                                      torch.randperm(len(self.minority_inds))]
            # Cycling through permutation of minority indices
            for sup_k in range(0, len(self.minority_inds),
                               self.minority_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = minority_inds_shuffled[
                        sup_k:(sup_k + self.minority_batch_size)]
                # Appending with random majority indices
                if self.minority_batch_size < self.batch_size:
                    batch.extend(
                        [self.majority_inds[i] for i in
                         torch.randint(high=len(self.majority_inds),
                                       size=(self.batch_size - len(batch),),
                                       dtype=torch.int64)])
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
