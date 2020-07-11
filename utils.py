"""
Utilities. Partially based on code from
https://github.com/modestyachts/CIFAR-10.1
"""
import io
import json
import os
import pickle

import numpy as np
import pathlib
import logging
from models.wideresnet import WideResNet
from models.shake_shake import ShakeNet
from models.cifar_resnet import ResNet
from models.preact_resnet import PreActResNet18
from models.mnist_resnet import mnist_resnet18, mnist_resnet34, mnist_resnet50, mnist_resnet101
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Module

cifar10_label_names = ['airplane', 'automobile', 'bird',
                       'cat', 'deer', 'dog', 'frog', 'horse',
                       'ship', 'truck']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_detector_model(args):

    detector_model = get_model(args.detector_model,
                      num_classes=11,
                      normalize_input=False)
    detector_model = torch.nn.DataParallel(detector_model.cuda())
    checkpoint = torch.load(args.detector_model_path, map_location=torch.device(device))
    detector_model.load_state_dict(checkpoint['state_dict'])
    logging.info("Loaded detector model with epoch %d, accuracy %0.4f" %(checkpoint['epoch'], checkpoint['accuracy_vs']))
    return detector_model

def get_model(name, num_classes=10, normalize_input=False):
    name_parts = name.split('-')
    if name_parts[0] == 'wrn':
        depth = int(name_parts[1])
        widen = int(name_parts[2])
        model = WideResNet(
            depth=depth, num_classes=num_classes, widen_factor=widen)
        
    elif name_parts[0] == 'ss':
        model = ShakeNet(dict(depth=int(name_parts[1]),
                              base_channels=int(name_parts[2]),
                              shake_forward=True, shake_backward=True,
                              shake_image=True, input_shape=(1, 3, 32, 32),
                              n_classes=num_classes,
                              ))
    elif name_parts[0] == 'resnet':
        logging.info("using model resnet")
        model = ResNet(num_classes=num_classes, depth=int(name_parts[1]))
    elif name_parts[0] == 'PreActResnet18':
        logging.info("using model preactresnet")
        model = PreActResNet18()
    elif name_parts[0] == 'mnistresnet':
            logging.info("using model mnistresnet")
            kwargs = {'num_classes': 10}
            pretrained = 'mnist'
            if name == 'mnistresnet-18':
                  model = mnist_resnet18()
                  print("using model mnist_resnet18")
            elif name == 'mnistresnet-34':
                  model = mnist_resnet34()
                  print("using model mnist_resnet34")
            elif name == 'mnistresnet-50':
                  model = mnist_resnet50()
                  print("using model mnist_resnet50")
            else:
                  model = mnist_resnet101()
                  print("using model mnist_resnet101")
    else:
        raise ValueError('Could not parse model name %s' % name)

    if normalize_input:
        model = Sequential(NormalizeInput(), model)

    return model

def calculate_tensor_percentile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
#     print("k for percentile: %d" %(k))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def calc_indiv_loss(y, targets):
    temp = F.softmax(y)
    loss = [-torch.log(temp[i][targets[i].item()]) for i in range(y.size(0))]
#     loss = F.cross_entropy(y, targets, reduction = 'None')
    return torch.stack(loss)

class NormalizeInput(Module):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

# TODO: decide whether to remove all code below this line
# Should we add some description of how these files
# were obtained?
def load_tinyimage_subset(other_data_path,
                          version_string='v7'):
    image_data_filename = 'tinyimage_subset_data'
    if version_string != '':
        image_data_filename += '_' + version_string
    image_data_filename += '.pickle'
    image_data_filepath = os.path.abspath(os.path.join(other_data_path, image_data_filename))
    indices_filename = 'tinyimage_subset_indices'
    if version_string != '':
        indices_filename += '_' + version_string
    indices_filename += '.json'
    indices_filepath = os.path.abspath(os.path.join(other_data_path, indices_filename))
    print('Loading indices from file {}'.format(indices_filepath))
    assert pathlib.Path(indices_filepath).is_file()
    print('Loading image data from file {}'.format(image_data_filepath))
    assert pathlib.Path(image_data_filepath).is_file()
    with open(indices_filepath, 'r') as f:
        indices = json.load(f)
    with open(image_data_filepath, 'rb') as f:
        image_data = pickle.load(f)
    num_entries = 0
    for kw, kw_indices in indices.items():
        for entry in kw_indices:
            assert entry['tinyimage_index'] in image_data
            num_entries += 1
    assert num_entries == len(image_data)
    return indices, image_data


def load_cifar10_by_keyword(unique_keywords=True, version_string='v7'):
    cifar10_keywords = load_cifar10_keywords(unique_keywords=unique_keywords,
                                             lists_for_unique=True,
                                             version_string=version_string)
    cifar10_by_keyword = {}
    for ii, keyword_entries in enumerate(cifar10_keywords):
        for entry in keyword_entries:
            cur_keyword = entry['nn_keyword']
            if not cur_keyword in cifar10_by_keyword:
                cifar10_by_keyword[cur_keyword] = []
            cifar10_by_keyword[cur_keyword].append(ii)
    return cifar10_by_keyword


def load_cifar10_keywords(other_data_path, 
                          unique_keywords=True,
                          lists_for_unique=False,
                          version_string='v7'):
    filename = 'cifar10_keywords'
    if unique_keywords:
        filename += '_unique'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    keywords_filepath = os.path.abspath(os.path.join(other_data_path, filename))
    print('Loading keywords from file {}'.format(keywords_filepath))
    assert pathlib.Path(keywords_filepath).is_file()
    with open(keywords_filepath, 'r') as f:
        cifar10_keywords = json.load(f)
    if unique_keywords and lists_for_unique:
        result = []
        for entry in cifar10_keywords:
            result.append([entry])
    else:
        result = cifar10_keywords
    assert len(result) == 60000
    return result


def load_distances_to_cifar10(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    filename = 'tinyimage_cifar10_distances'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    filepath = os.path.abspath(os.path.join(data_path, filename))
    print('Loading distances from file {}'.format(filepath))
    assert pathlib.Path(filepath).is_file()
    with open(filepath, 'r') as f:
        tmp = json.load(f)
    if version_string == 'v4':
        assert len(tmp) == 372131
    elif version_string == 'v6':
        assert len(tmp) == 1646248
    elif version_string == 'v7':
        assert len(tmp) == 589711
    result = {}
    for k, v in tmp.items():
        result[int(k)] = v
    return result


def load_new_test_data_indices(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../data/')
    ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
    ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
    print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
    assert pathlib.Path(ti_indices_filepath).is_file()
    with open(ti_indices_filepath, 'r') as f:
        tinyimage_indices = json.load(f)
    assert type(tinyimage_indices) is list
    if version_string == 'v6' or version_string == 'v7':
        assert len(tinyimage_indices) == 2000
    elif version_string == 'v4':
        assert len(tinyimage_indices) == 2021
    return tinyimage_indices


def dump_pretrained_example_losses_to_file(model_dir, pretrained_epochs, model_name, loss_info):
      pretrained_model = get_model(model_name)
      assert os.path.isdir(model_dir), 'Error: no checkpoint directory found!'
      assert os.path.isdir(model_dir + '/' + model_name), 'Error: no checkpoint directory found!'
      
      model_file = model_dir + '/' + model_name + '/' + 'unlabloss_' + str(pretrained_epochs) + '.pickle'
      # assert os.path.isfile(model_file), 'Error: no checkpoint file found!'
      # loss_info = dict()
      # loss_info['model_name'] = model_name
      # loss_info['dataset_length'] = example_losses.size(0)
      # loss_info['examples_losses'] = example_losses
      # loss_info['pretrained_acc'] = pretrained_acc
      # loss_info['pretrained_epochs'] = pretrained_epochs
      with open(model_file, 'wb') as f:
            pickle.dump(loss_info, f)
      print("Dumped losses for %d train examples to file %s" %(loss_info['example_cross_ent_losses'].size(0), model_file))

def load_pretrained_example_losses_from_file(model_dir, model_name, epoch_no = None):
      pretrained_model = get_model(model_name)
      assert os.path.isdir(model_dir), 'Error: no checkpoint directory found!'
      assert os.path.isdir(model_dir + '/' + model_name), 'Error: no checkpoint directory found!'
      model_file = model_dir + '/' + model_name + '/' + 'unlabloss_' + str(epoch_no) + '.pickle'
      assert os.path.isfile(model_file), 'Error: no example loss file found! --- %s' %(model_file)
      with open(model_file, 'rb') as f:
            loss_info = pickle.load(f)
      model_name = loss_info['model_name']
      train_dataset_size = loss_info['dataset_length']
      example_cross_ent_losses = loss_info['example_cross_ent_losses']
      example_multi_margin_losses = loss_info['example_multi_margin_losses']
      example_labels = loss_info['example_labels']
      example_outputs = loss_info['example_outputs']
      pretrained_acc = loss_info['pretrained_acc']
      pretrained_epochs = loss_info['pretrained_epochs']
      multimarginloss_margin = loss_info['multimarginloss_margin']
      # print(multimarginloss_margin)
      assert train_dataset_size == example_cross_ent_losses.size(0), 'Error: size of input example loaded from file does not match'
      print("Loaded losses for %d == %d train examples from file %s" %(train_dataset_size, example_cross_ent_losses.size(0), model_file))

      return example_cross_ent_losses, example_multi_margin_losses, pretrained_acc, pretrained_epochs, example_outputs


