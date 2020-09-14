import numpy as np
import torch
from bisect import bisect_left

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, train = False, transform=None, target_transform = None, exclude_cifar=False):

        data_file = open('data/unlabeled_datasets/80M_Tiny_Images/tiny_500k.bin', "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F"), 0

        self.load_image = load_image
        self.offset = 0     # offset index
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.exclude_cifar = exclude_cifar
        print(self.transform)
        if exclude_cifar:
            self.cifar_idxs = []
            with open('data/unlabeled_datasets/80M_Tiny_Images/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search

        def load_tinyimages():
            data = []
            targets = []

            # from PIL import Image

            for idx in range(500000):
                dt, tgt = self.load_image(idx)

                # if idx < 5:
                #     print('dt shape:', np.shape(dt))
                #     img = Image.fromarray(dt)
                #     img.save('test_{}.png'.format(idx))

                data.append(dt)
                targets.append(tgt)
            #     if idx<10:
            #           print(dt)
            #           print(tgt)
            return np.asarray(data), np.asarray(targets)
            # return data, targets

        self.data, self.targets = load_tinyimages()
        print('Shape of data in loader:', np.shape(self.data))

    def __getitem__(self, index):
        index = (index + self.offset) % 499999

        if self.exclude_cifar:
            print("Excluding cifar")
            while self.in_cifar(index):
                index = np.random.randint(500000)

      #   img = self.load_image(index)
        
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target  # 0 is the class

    def __len__(self):
        return 500000


    
