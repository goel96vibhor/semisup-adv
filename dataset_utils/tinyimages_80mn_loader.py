import numpy as np
import torch
from bisect import bisect_left


class TinyImages(torch.utils.data.Dataset):

    def __init__(self, train=False, 
                transform=None, target_transform=None, 
                exclude_cifar=False,
                dataset_dir='data/unlabeled_datasets/80M_Tiny_Images/tiny_50k.bin',
                logger=None, num_images=50000):

        data_file = open(dataset_dir, "rb")
        self.num_images = num_images

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
            for idx in range(self.num_images):
                dt, tgt = self.load_image(idx)
                # if idx < 5:
                #     print('dt shape:', np.shape(dt))
                #     img = Image.fromarray(dt)
                #     img.save('test_rohish_{}.png'.format(idx))
                data.append(dt)
                targets.append(tgt)
            return np.asarray(data), np.asarray(targets)    # torch.tensor(data), torch.tensor(targets)

        self.data, self.targets = load_tinyimages()
        logger.info(f'Shape of data in tinyimages loader: {np.shape(self.data)}, {np.shape(self.targets)}')

    def __getitem__(self, index):
        index = (index + self.offset) % (self.num_images - 1)

        if self.exclude_cifar:
            print("Excluding cifar")
            while self.in_cifar(index):
                index = np.random.randint(self.num_images)

        # img = self.load_image(index)
        
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target  # 0 is the class

    def __len__(self):
        return self.num_images
