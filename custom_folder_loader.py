from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import pickle
import numpy as np
import logging
from pathlib import Path
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if os.path.islink(path):
          path = os.readlink(path)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    print("Inside make dataset ..................")
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                  #   print(fname)
                    if fname == 'n04308273_4242.png':
                          print("-------------------Found file -------------------")
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

#     def __new__(self, cls, *args, **kwargs):
#         print("inside new")
#         print(kwargs)
#         print(args)
#         if kwargs['load_from_checkpoint']:
            
#             dest_path = args[0] + '_checkpoint'
#             print("Loading from checkpoint from path %s" %(dest_path))
#             if kwargs['train_valid_test'] == 0:
#                   dest_path = os.path.join(dest_path, 'train.pickle')
#             elif kwargs['train_valid_test'] == 1:
#                   dest_path = os.path.join(dest_path, 'valid.pickle')                         
#             else:
#                   dest_path = os.path.join(dest_path, 'test.pickle') 
#             with open(dest_path, 'rb') as f:
#                inst = pickle.load(f)
#             if not isinstance(inst, cls):
#                raise TypeError('Unpickled object is not of type {}'.format(cls))
#             print("Loaded from checkpoint .. %s" %(dest_path))
#         else:
#             inst = super(DatasetFolder, cls).__new__(cls, *args, **kwargs)
#         return inst

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, train_valid_test = 0, ):
      super(DatasetFolder, self).__init__(root, transform=transform,
                                          target_transform=target_transform)
      self.root = root
      self.train_valid_test = train_valid_test
      self.base_folder = root

      if self.train_valid_test == 0:
            self.root = os.path.join(self.root, 'train')
      elif self.train_valid_test == 1:
            self.root = os.path.join(self.root, 'valid')                         
      else:
            self.root = os.path.join(self.root, 'test')   


      classes, class_to_idx = self._find_classes(self.root)
      samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
      if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                  msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

      self.loader = loader
      self.extensions = extensions

      self.classes = classes
      self.class_to_idx = class_to_idx
      #   samples = samples[:10000]
      self.samples = samples
      self.targets = [s[1] for s in samples]
      
      self.data = []
      print(len(self.targets))
      count = 0
      for (path, target) in samples:
            entry = self.loader(path)
            if path.endswith('n04308273_4242.png'):
                  print("Converted file with shape ........")
                  # print(entry.shape)
            self.data.append(entry)
            count +=1
            if(count % 10000==0):
                  print(count)

      #   print(self.data[1].shape)
      self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
      self.imgs = self.samples
      #     .reshape(-1, 3, 32, 32)
      print(self.data.shape)
      # self.dump(self.base_folder)
      # self.load(self.base_folder, train_valid_test)
      #   self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
      #   print(self.data.shape)


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
      #   path, target = self.samples[index]
        sample = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)


#     def dump(self, dest_path):
#         dest_path = dest_path + '_checkpoint'
#         if not os.path.exists(dest_path):
#               os.makedirs(dest_path)
#         if self.train_valid_test == 0:
#               dest_path = os.path.join(dest_path, 'train.pickle')
#         elif self.train_valid_test == 1:
#               dest_path = os.path.join(dest_path, 'valid.pickle')                         
#         else:
#               dest_path = os.path.join(dest_path, 'test.pickle')    
        
#         with open(dest_path, 'wb') as handle:
#               pickle.dump(self, handle)      

#         logging.info("Dumped cinic file into pickle .. %s" %(dest_path))


#     def load(self, dest_path, train_valid_test = 0):
#         dest_path = dest_path + '_checkpoint'
#         if train_valid_test == 0:
#               dest_path = os.path.join(dest_path, 'train.pickle')
#         elif train_valid_test == 1:
#               dest_path = os.path.join(dest_path, 'valid.pickle')                         
#         else:
#               dest_path = os.path.join(dest_path, 'test.pickle')    

#         with open(dest_path, 'rb') as handle:
#               temp_dict = pickle.load(handle)                    

#       #   self.__dict__.update(tmp_dict)       
#         return temp_dict     
        

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.s  



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')





class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, train_valid_test = 0):
        
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          train_valid_test = train_valid_test,                                           
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        


    