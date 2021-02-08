"""
Utilities. Partially based on code from
https://github.com/modestyachts/CIFAR-10.1
"""
import io
import json
import os
import pickle

import numpy as np
import pandas as pd
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

logger = logging.getLogger()
cifar10_label_names = ['airplane', 'automobile', 'bird',
                       'cat', 'deer', 'dog', 'frog', 'horse',
                       'ship', 'truck']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_detector_model(args):
    detector_model = get_model(args.detector_model,
                      num_classes=args.n_classes,
                      normalize_input=False)
    # detector_model = torch.nn.DataParallel(detector_model).cuda()
    checkpoint = torch.load(args.detector_model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    def strip_data_parallel(s):
        if s.startswith('module.1'):
            return s[len('module.1.'):]
        elif s.startswith('module.0'):
            return None
        elif s.startswith('module'):
            return s[len('module.'):]
        else:
            return s
    new_state_dict = {}
    for k,v in state_dict.items():
        k_new = strip_data_parallel(k)
        if k_new:
            new_state_dict[k_new] = v
    # state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    detector_model.load_state_dict(new_state_dict)
    detector_model = torch.nn.DataParallel(detector_model).cuda()
    #logging.info("Loaded detector model with epoch %d, accuracy %0.4f" %(checkpoint['epoch'], checkpoint['accuracy_vs']))
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
        logging.info("Using model resnet")
        model = ResNet(num_classes=num_classes, depth=int(name_parts[1]))
    elif name_parts[0] == 'PreActResnet18':
        logging.info("Using model preactresnet")
        model = PreActResNet18()
    elif name_parts[0] == 'mnistresnet':
            logging.info("Using model mnistresnet")
            kwargs = {'num_classes': 10}
            pretrained = 'mnist'
            if name == 'mnistresnet-18':
                model = mnist_resnet18()
                logging.info("Using model mnist_resnet18")
            elif name == 'mnistresnet-34':
                model = mnist_resnet34()
                logging.info("Using model mnist_resnet34")
            elif name == 'mnistresnet-50':
                model = mnist_resnet50()
                logging.info("Using model mnist_resnet50")
            else:
                model = mnist_resnet101()
                logging.info("Using model mnist_resnet101")
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
    # loss = F.cross_entropy(y, targets, reduction = 'None')
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
    print('Loaded indices from file {} with size {}'.format(indices_filepath,num_entries))
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
    logger.info("Loaded losses for %d == %d train examples from file %s" %(train_dataset_size, example_cross_ent_losses.size(0), model_file))

    return example_cross_ent_losses, example_multi_margin_losses, pretrained_acc, pretrained_epochs, example_outputs


def plot_histogram(cifar_conf_vals, noncifar_conf_vals, dataset):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, constrained_layout=True)

    axs[0].hist(cifar_conf_vals, bins=100)
    axs[0].set_title('P1')
    axs[0].set_xlabel('Confidence')
    axs[0].set_ylabel('Number of examples')

    axs[1].hist(noncifar_conf_vals, bins=100)
    axs[1].set_title('Non-CIFAR')
    axs[1].set_xlabel('Confidence')
    axs[1].set_ylabel('Number of examples')

    fig.suptitle('Histogram for {} Dataset confidence distribution for R subset'.format(dataset), fontsize=14)
    # plt.show()
    plt.savefig('{}_hist.png'.format(dataset))


def plot_R_histogram():
    import matplotlib.pyplot as plt

    df = pd.read_csv('cifar10-vs-ti/tinyimages.csv')

    print(df)

    confs, is_c10 = df.iloc[:, 2:13], df.iloc[:, -2]
    print(confs.shape)
    print(confs)

    last_less_2 = confs.iloc[:, -1] < .2
    conf_less_2 = confs[last_less_2]

    class_confs_less_2 = conf_less_2.iloc[:, :-1]

    p1_confs = class_confs_less_2.max(axis=1)
    last_confs = conf_less_2.iloc[:, -1]


    plot_histogram(p1_confs, last_confs, 'tinyimages')

def plot_W_histogram():
    import matplotlib.pyplot as plt

    df = pd.read_csv('cifar10-vs-ti/tinyimages.csv')

    print(df)

    confs, is_c10 = df.iloc[:, 2:13], df.iloc[:, -2]
    print(confs.shape)
    print(confs)

    any_more_8 = confs.iloc[:, :-1].max(axis=1) > .8
    conf_more_8 = confs[any_more_8]

    p1_confs = conf_more_8.iloc[:, :-1].max(axis=1)
    last_confs = conf_more_8.iloc[:, -1]

    plot_histogram(p1_confs, last_confs, 'tinyimages')


if __name__ == "__main__":
    plot_W_histogram()
