"""
Datasets with unlabeled (or pseudo-labeled) data
"""

from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import Sampler, Dataset
import torch
import numpy as np
import cifar_own
import os
import pickle

import logging

DATASETS = ['cifar10', 'svhn', 'cifar_own']


class SemiSupervisedDataset(Dataset):
    def __init__(self,
                 base_dataset='cifar10',
                 take_amount=None,
                 take_amount_seed=13,
                 add_svhn_extra=False,
                 aux_data_filename=None,
                 add_aux_labels=False,
                 aux_take_amount=None,
                 train=False, custom_dataset = None, 
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        if base_dataset == 'cifar10':
            self.dataset = CIFAR10(train=train, **kwargs)
        elif base_dataset == 'cifar_own':
            print("Using own cifar implementation")
            self.dataset = cifar_own.CIFAR10(train=train, **kwargs)
        elif base_dataset == 'svhn':
            if train:
                self.dataset = SVHN(split='train', **kwargs)
            else:
                self.dataset = SVHN(split='test', **kwargs)
            # because torchvision is annoying
            self.dataset.targets = self.dataset.labels
            self.targets = list(self.targets)

            if train and add_svhn_extra:
                svhn_extra = SVHN(split='extra', **kwargs)
                self.data = np.concatenate([self.data, svhn_extra.data])
                self.targets.extend(svhn_extra.labels)
        elif base_dataset == 'custom' and custom_dataset != None:
            self.dataset = custom_dataset
        else:
            raise ValueError('Dataset %s not supported' % base_dataset)
        self.base_dataset = base_dataset
        self.train = train
        self.sup_indices = list(range(len(self.targets)))
        print("Printing length of targets: %s sup indices: %s" %(len(self.targets), len(self.sup_indices)))
        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices),
                                             take_amount, replace=False).astype(int)
            #     print(take_inds.dtype)
                np.random.set_state(rng_state)

                logger = logging.getLogger()
                logger.info('Randomly taking only %d/%d examples from training'
                            ' set, seed=%d, indices=%s',
                            take_amount, len(self.sup_indices),
                            take_amount_seed, take_inds)
                self.targets = np.array(self.targets)[take_inds]
                self.data = np.array(self.data)[take_inds]
                self.sup_indices = list(range(len(self.targets)))

            print("CIFAR data: targets after filtering: %s sup indices: %s" %(len(self.targets), len(self.sup_indices)))
            self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = os.path.join(kwargs['root'], aux_data_filename)
                print("Loading data from %s" % aux_path)
                with open(aux_path, 'rb') as f:
                    aux = pickle.load(f)
                aux_data = aux['data']
                aux_targets = aux['extrapolated_targets']
                orig_len = len(self.data)
                self.unsup_indices.extend(range(orig_len, orig_len+len(aux_data)))

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(self.unsup_indices),
                                                 aux_take_amount, replace=False)
                    np.random.set_state(rng_state)
                    logger = logging.getLogger()
                    logger.info(
                        'Randomly taking only %d/%d examples from aux data'
                        ' set, seed=%d, indices=%s',
                        aux_take_amount, len(self.unsup_indices),
                        take_amount_seed, take_inds)
                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]
                    self.unsup_indices = list(range(orig_len, orig_len+len(aux_data)))
            
                self.data = np.concatenate([self.data, aux_data])
                if not add_aux_labels:
                    self.targets = np.concatenate([ self.targets, extend([-1] * len(aux_data))])
                else:
                    self.targets = np.concatenate([self.targets, aux_targets])
                print("Unlabeled data: targets after filtering: %s unsup indices: %s" %(len(self.targets), len(self.unsup_indices)))
                print(self.data.dtype)
                print(self.targets.dtype)
                # note that we use unsup indices to track the labeled datapoints
                # whose labels are "fake"
                

            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Class of data: %s", self.data[1].dtype)
            logger.info("Value of data: %s", self.data[1, 1, 1, :])
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item] + (item,)

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(Sampler):
    """Balanced sampling from the labeled and unlabeled data"""
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5,
                 num_batches=None):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds
        print(self.unsup_inds[0:20])
        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size
        
        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.sup_inds) / self.sup_batch_size))

        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in
                                  torch.randint(high=len(self.unsup_inds),
                                                size=(
                                                    self.batch_size - len(
                                                        batch),),
                                                dtype=torch.int64)])
                # this shuffle operation is very important, without it
                # batch-norm / DataParallel hell ensues
                np.random.shuffle(batch)
                yield batch
            #     print(batch)
                batch_counter += 1
        
    def __len__(self):
        return self.num_batches
