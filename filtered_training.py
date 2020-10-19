"""
Main robust self-training script. Based loosely on code from
https://github.com/yaodongyu/TRADES

"""


import os
import sys
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datetime import datetime
import pandas as pd
import numpy as np

from utils import *

from losses import *
from datasets import SemiSupervisedDataset, SemiSupervisedSampler
from attack_pgd import pgd
from smoothing import quick_smoothing
from scipy.stats import norm
from autoaugment import CIFAR10Policy
from cutout import Cutout

import logging


# ----------------------------- CONFIGURATION ----------------------------------
parser = argparse.ArgumentParser(
    description='PyTorch TRADES Adversarial Training')

# Dataset config
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar_own','svhn', 'qmnist', 'qmnist_own','mnist', 
                        'benrecht_cifar10', 'cinic10', 'cinic10_v2']
                  , help='The dataset to use for training)')
parser.add_argument('--data_dir', default='data', type=str, help='Directory where datasets are located')
parser.add_argument('--svhn_extra', action='store_true', default=False, help='Adds the extra SVHN data')
parser.add_argument('--extend_svhn', default=0, type=int, help='Whether to add supervised svhn data while training')
parser.add_argument('--extend_svhn_fraction', default=0.5, type=float, help='What fraction of svhn data while training')
# Model config
parser.add_argument('--model', '-m', default='wrn-28-10', type=str, help='Name of the model (see utils.get_model)')
parser.add_argument('--model_dir', default='./rst_augmented', help='Directory of model for saving checkpoint')
parser.add_argument('--test_name', default='', help='Test name to give proper subdirectory to model for saving checkpoint')
parser.add_argument('--overwrite', action='store_true', default=False, help='Cancels the run if an appropriate checkpoint is found')
parser.add_argument('--normalize_input', action='store_true', default=False, help='Apply standard CIFAR normalization first thing '
                         'in the network (as part of the model, not in the data fetching pipline)')
# detector model config
parser.add_argument('--detector-model', default='wrn-28-10', type=str, help='Name of the detector model (see utils.get_model)')
parser.add_argument('--use-detector-training', default=0, type=int, help='Use detector model for natural shift generation')
# Logging and checkpointing
parser.add_argument('--log_interval', type=int, default=25, help='Number of batches between logging of training status')
parser.add_argument('--save_freq', default=25, type=int, help='Checkpoint save frequency (in epochs)')

# Generic training configs
parser.add_argument('--seed', type=int, default=1, help='Random seed. Note: fixing the random seed does not give complete reproducibility. See '
                         'https://pytorch.org/docs/stable/notes/randomness.html')

parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='Input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=500, metavar='N', help='Input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='Number of epochs to train. Note: we arbitrarily define an epoch as a pass through 50K datapoints. This is convenient for '
                         'comparison with standard CIFAR-10 training configurations.')

# Eval config
parser.add_argument('--eval_freq', default=1, type=int, help='Eval frequency (in epochs)')
parser.add_argument('--train_eval_batches', default=None, type=int, help='Maximum number for batches in training set eval')
parser.add_argument('--eval_attack_batches', default=1, type=int, help='Number of eval batches to attack with PGD or certify '
                         'with randomized smoothing')

# Pre train configs
parser.add_argument('--pretrained_model_dir', default='pretrained_models', type=str, help='Model directory for pretrained models')
parser.add_argument('--pretrained_model_name', default='PreActResnet18' ,type=str, help='name for the pretrained-model')
parser.add_argument('--pretrained_epochs', default=14,type=int, help='number of epochs for the pretrained-model')

#Filtering configs
parser.add_argument('--unsup_std_deviations', type=float, default=1.0, help='Number of std to consider')
parser.add_argument('--filter_unsup_data', default=1, type=int, help='Whether to filter unsupervised data')
parser.add_argument('--use_two_detector_filtering', default=1, type=int, help='Whether to filter unsupervised data using two detector')
parser.add_argument('--use_example_weighing', default=0, type=int, help='Whether to use example weighing for detectors using two detector')
parser.add_argument('--use_example_sampling', default=1, type=int, help='Whether to sample unlabled examples for detectors using two detector')
parser.add_argument('--example_weight_alpha', type=float, default=1.0, help='hyperparamter for example weights')
parser.add_argument('--random_split_version', type=int, default=2, help='Version of random split')
parser.add_argument('--use_distrib_selection', default=0, type=int, help='Whether to use distribution using pair of two detector')
parser.add_argument('--use_distrib_concatenation', default=0, type=int, help='Whether to use distribution using pair of two detector')
# Optimizer config
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='Learning rate')
parser.add_argument('--lr_schedule', type=str, default='cosine', choices=('trades', 'trades_fixed', 'cosine', 'wrn'), help='Learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--nesterov', action='store_true', default=True, help='Use extragrdient steps')

# Adversarial / stability training config
parser.add_argument('--use_adv_training', default=0, type=int, help='whether to use adversarial training')
parser.add_argument('--loss', default='trades', type=str, choices=('trades', 'noise', 'shift'), help='Which loss to use: TRADES-like KL regularization '
                         'or noise augmentation')

parser.add_argument('--distance', '-d', default='l_inf', type=str, help='Metric for attack model: l_inf uses adversarial '
                         'training and l_2 uses stability training and randomized smoothing certification', choices=['l_inf', 'l_2'])
parser.add_argument('--epsilon', default=0.031, type=float, help='Adversarial perturbation size (takes the role of sigma for stability training)')
parser.add_argument('--pgd_num_steps', default=10, type=int, help='number of pgd steps in adversarial training')
parser.add_argument('--pgd_step_size', default=0.007, help='pgd steps size in adversarial training', type=float)
parser.add_argument('--beta', default=6.0, type=float, help='stability regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--w_1', default=1.0, type=float, help='detector loss weight')
parser.add_argument('--w_2', default=1.0, type=float, help='main loss weight')
# Semi-supervised training configuration
parser.add_argument('--aux_data_filename', default='ti_500K_pseudo_labeled.pickle', type=str,
                    help='Path to pickle file containing unlabeled data and pseudo-labels used for RST')
parser.add_argument('--unsup_fraction', default=0.5, type=float,
                    help='Fraction of unlabeled examples in each batch; implicitly sets the weight of unlabeled data in the '
                         'loss. If set to -1, batches are sampled from a single pool')
parser.add_argument('--train_take_amount', default=None, type=int, help='Number of random aux examples to retain. None retains all aux data.')                         
parser.add_argument('--aux_take_amount', default=None, type=int, help='Number of random aux examples to retain. '
                         'None retains all aux data.')
parser.add_argument('--remove_pseudo_labels', action='store_true', default=False, help='Performs training without pseudo-labels (rVAT)')
parser.add_argument('--entropy_weight', type=float, default=0.0, help='Weight on entropy loss')

# Additional aggressive data augmentation
parser.add_argument('--autoaugment', action='store_true', default=False, help='Use autoaugment for data augmentation')
parser.add_argument('--cutout', action='store_true', default=False, help='Use cutout for data augmentation')

args = parser.parse_args()


def get_filtered_indices(args, example_outputs, random_split_version = 1):
      weights = None
      print("random pslit version %d" %(random_split_version))
      if args.use_two_detector_filtering:
            if (not (args.use_distrib_selection or args.use_distrib_concatenation)) or (random_split_version == 1):
                  det1_unlab_file = 'selection_model/testing_head_0.1/unlabeled_percy_500k.csv'
                  det2_unlab_file = 'selection_model/testing_tail_0.1/unlabeled_percy_500k.csv'            
            else:
                  det1_unlab_file = 'selection_model/testing_head_0.1_v'+str(args.random_split_version) + '_1Mstart/unlabeled_percy_500k.csv'
                  det2_unlab_file = 'selection_model/testing_tail_0.1_v'+str(args.random_split_version) + '_1Mstart/unlabeled_percy_500k.csv'
            logging.info('Filtering unsup indices using two detector predictions from file %s' %(det1_unlab_file))
            det1_unlab_df = pd.read_csv(det1_unlab_file)
            det2_unlab_df = pd.read_csv(det2_unlab_file)
            det1_unlab_df = det1_unlab_df.set_index(['0']).sort_index()
            det2_unlab_df = det2_unlab_df.set_index(['0']).sort_index()
            det1_unlab_scores = det1_unlab_df.iloc[:,1:12]
            det2_unlab_scores = det2_unlab_df.iloc[:,1:12]
            det1_unlab_preds = det1_unlab_df.iloc[:,22]
            det2_unlab_preds = det2_unlab_df.iloc[:,22]
            det1_unlab_targets = det1_unlab_df.iloc[:,0]
            det2_unlab_targets = det2_unlab_df.iloc[:,0]
            d1_d2_unlab = (det1_unlab_preds == det2_unlab_preds)
            x1 = (det1_unlab_preds != 10)
            y1 = (det2_unlab_preds != 10)
            x2 = (det1_unlab_preds == 10)
            y2 = (det2_unlab_preds == 10)

            # mask = (x1&y2) | (x2&y1)
            # mask = x1|y1
            mask = (det1_unlab_preds !=  det2_unlab_preds)
            indices = det1_unlab_preds.index[mask]
            indices = torch.tensor(indices, dtype = torch.int)
            logging.info("Filtered indices from unlabeled softmax confidences with count %s" %(indices.shape))
            if args.use_example_weighing or args.use_example_sampling:
                  # weights = torch.ones(indices.shape)
                  # weights = torch.tensor((abs(det1_unlab_scores-det2_unlab_scores))[mask].iloc[:,10].values)
                  weights = torch.tensor((abs(det1_unlab_scores-det2_unlab_scores))[mask].sum(axis=1).values)                  
                  weights = torch.exp(args.example_weight_alpha*weights)
                  # print(weights.shape)
                  # print(indices.shape)
                  assert weights.shape == indices.shape , "shapes of indices and weights are not equal"
                  logging.info("Using example weighing for training with total sum %0.4f" %(float(torch.sum(weights))))
      else:      
            logging.info('Filtering unsup indices using thresholding')
            soft_out = F.softmax(example_outputs)
            soft_max = torch.max(soft_out, dim=1).values
            mu, std = norm.fit(soft_max)
            upper_limit = mu
            lower_limit = mu - (args.unsup_std_deviations*std)
            # upper_limit = mu + args.unsup_std_deviations*std
            # lower_limit = mu - args.unsup_std_deviations*std
            indices = ((soft_max < upper_limit) & (soft_max > lower_limit)).nonzero().view(-1)
            logging.info("Filtered indices from unlabeled softmax confidences with upper limit %0.4f, lower limit %0.4f, count %s" %(upper_limit, lower_limit, indices.shape))
      if weights is None:
            weights = torch.ones(indices.shape, dtype = torch.float64)
      return indices, weights



# ------------------------------ OUTPUT SETUP ----------------------------------

model_dir = args.model_dir

if args.test_name == '':
      sub_dir = str('_'.join(sys.argv[2::2])) +'_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
else:
      sub_dir = args.test_name

model_dir = model_dir + '/' + sub_dir + '/' + args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Robust self-training')
logging.info('Args: %s', args)

if not args.overwrite:
    final_checkpoint_path = os.path.join(
        model_dir, 'checkpoint-epoch{}.pt'.format(args.epochs))
    if os.path.exists(final_checkpoint_path):
        logging.info('Appropriate checkpoint found - quitting!')
        sys.exit(0)
# ------------------------------------------------------------------------------

# ------------------------------- CUDA SETUP -----------------------------------
# should provide some improved performance
cudnn.benchmark = True
# useful setting for debugging
# cudnn.benchmark = False
# cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
# ------------------------------------------------------------------------------

# --------------------------- DATA AUGMENTATION --------------------------------
if args.dataset == 'cifar10' or args.dataset == 'cinic10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
elif args.dataset == 'svhn':
    # the WRN paper does no augmentation on SVHN
    # obviously flipping is a bad idea, and it makes some sense not to
    # crop because there are a lot of distractor digits in the edges of the
    # image
    transform_train = transforms.ToTensor()

if args.autoaugment or args.cutout:
    assert (args.dataset == 'cifar10')
    transform_list = [
        transforms.RandomCrop(32, padding=4, fill=128),
        # fill parameter needs torchvision installed from source
        transforms.RandomHorizontalFlip()]
    if args.autoaugment:
        transform_list.append(CIFAR10Policy())
    transform_list.append(transforms.ToTensor())
    if args.cutout:
        transform_list.append(Cutout(n_holes=1, length=16))

    transform_train = transforms.Compose(transform_list)
    logger.info('Applying aggressive training augmentation: %s'
                % transform_train)

transform_test = transforms.Compose([
    transforms.ToTensor()])
# ------------------------------------------------------------------------------

# ----------------- DATASET WITH AUX PSEUDO-LABELED DATA -----------------------
trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                 add_svhn_extra=args.svhn_extra,
                                 root=args.data_dir, train=True,
                                 extend_svhn = args.extend_svhn,
                                 extend_svhn_fraction = args.extend_svhn_fraction,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount)
example_weights = torch.ones(len(trainset.sup_indices) + len(trainset.unsup_indices), dtype=torch.float64)
total_unsup_size = len(trainset.unsup_indices)
example_weights_2 = None
example_probabilties = None
example_probabilties_2 = None
if args.use_distrib_selection:
      trainset_2 = SemiSupervisedDataset(base_dataset=args.dataset,
                                 add_svhn_extra=args.svhn_extra,
                                 root=args.data_dir, train=True,
                                 extend_svhn = args.extend_svhn,
                                 extend_svhn_fraction = args.extend_svhn_fraction,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount)
      example_weights_2 = torch.ones(len(trainset.sup_indices) + len(trainset.unsup_indices), dtype=torch.float64)                                 
# num_batches=50000 enforces the definition of an "epoch" as passing through 50K
# datapoints
# TODO: make sure that this code works also when trainset.unsup_indices=[]

if args.filter_unsup_data:
      example_cross_ent_losses, example_multi_margin_losses, pretrained_acc, pretrained_epochs, example_outputs = load_pretrained_example_losses_from_file(
                  args.pretrained_model_dir, args.pretrained_model_name, args.pretrained_epochs)
      filtered_unsup_indices, filtered_unsup_weights = get_filtered_indices(args, example_outputs, random_split_version = 1)
      logger.info("Filtered indices obtained of size %d" %(filtered_unsup_indices.shape[0]))
      print(filtered_unsup_indices[0:10])
      trainset.unsup_indices = torch.add(filtered_unsup_indices, len(trainset.sup_indices)).tolist()
      example_weights[trainset.unsup_indices] = filtered_unsup_weights

      if args.use_distrib_selection:
            assert args.random_split_version != 1, "Random split version should be other than 1"
            assert args.use_distrib_concatenation !=1, "Use one of distribution selction or concatenation"
            filtered_unsup_indices_2, filtered_unsup_weights_2 = get_filtered_indices(args, example_outputs, random_split_version = args.random_split_version)
            logger.info("Filtered indices obtained of size %d" %(filtered_unsup_indices_2.shape[0]))
            print(filtered_unsup_indices_2[0:10])
            trainset_2.unsup_indices = torch.add(filtered_unsup_indices_2, len(trainset_2.sup_indices)).tolist()
            example_weights_2[trainset_2.unsup_indices] = filtered_unsup_weights_2
            example_weights_2 = example_weights_2.cuda()
      elif args.use_distrib_concatenation:
            assert args.random_split_version != 1, "Random split version should be other than 1"
            filtered_unsup_indices_2, filtered_unsup_weights_2 = get_filtered_indices(args, example_outputs, random_split_version = args.random_split_version)
            logger.info("Filtered indices obtained of size %d" %(filtered_unsup_indices_2.shape[0]))
            unsup_indices_2 = torch.add(filtered_unsup_indices_2, len(trainset.sup_indices)).tolist()
            combined = torch.cat((torch.tensor(trainset.unsup_indices), torch.tensor(unsup_indices_2)))
            uniques, counts = combined.unique(return_counts=True)
            trainset.unsup_indices = uniques[counts >= 1].tolist()
            intersection = uniques[counts > 1]
            logger.info("Intersection size %d " %(intersection.shape[0]))
            print(intersection[0:5])
            print(example_weights[intersection[0:5]])
            print(filtered_unsup_weights_2[0:10], unsup_indices_2[0:10])
            example_weights[unsup_indices_2] = example_weights[unsup_indices_2] + (torch.tensor(filtered_unsup_weights_2) - 1)
            print(example_weights[intersection[0:5]])
            logger.info("Final filtered indices obtained after concatenation: %d, Sum weights_1 %0.4f weights_2 %0.4f Concat %0.4f"
                  %(len(trainset.unsup_indices), torch.sum(filtered_unsup_weights), torch.sum(filtered_unsup_weights_2), torch.sum(example_weights[trainset.unsup_indices])))

if args.use_example_sampling:
      assert args.use_example_weighing == False, 'Can do only one of example sampling or weighing'
      logger.info("Using example sampling instead of weighing")
      example_probabilties = example_weights[trainset.unsup_indices]
      example_weights = torch.ones(len(trainset.sup_indices) + total_unsup_size, dtype=torch.float64)     
      logger.info("Example weights shape %s" %(str(example_weights.shape)))

      # print(trainset.unsup_indices[0:10])
example_weights = example_weights.cuda()
epoch_datapoint_count = trainset.__len__()
if args.train_take_amount is not None:
      epoch_datapoint_count = args.train_take_amount
# elif args.extend_svhn:
#       epoch_datapoint_count += 73257
logger.info("epoch datapoints count: %d" %(epoch_datapoint_count))
train_batch_sampler = SemiSupervisedSampler(
    trainset.sup_indices, trainset.unsup_indices,
    args.batch_size, args.unsup_fraction,
    num_batches=int(np.ceil(epoch_datapoint_count / args.batch_size)), unsup_probabilities = example_probabilties)
epoch_size = len(train_batch_sampler) * args.batch_size

logger.info("Epoch size: %d" %(epoch_size))

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)
logger.info("Created train loader 1 with size %d" %(len(trainset)))
testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                root=args.data_dir, train=False,
                                download=True,
                              #   extend_svhn = args.extend_svhn,
                                transform=transform_test)
test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)

trainset_eval = SemiSupervisedDataset(
    base_dataset=args.dataset,
    add_svhn_extra=args.svhn_extra,
#     extend_svhn = args.extend_svhn,
    root=args.data_dir, train=True,
    download=True, transform=transform_train)

eval_train_loader = DataLoader(trainset_eval, batch_size=args.test_batch_size,
                               shuffle=True, **kwargs)

eval_test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                              shuffle=False, **kwargs)
train_loader_2 = None
if args.use_distrib_selection:
      if args.use_example_sampling:
            example_probabilties_2 = example_weights_2[trainset.unsup_indices_2]
            example_weights_2 = torch.ones(len(trainset.sup_indices) + total_unsup_size, dtype=torch.float64)
            example_weights_2 = example_weights_2.cuda()
      train_batch_sampler_2 = SemiSupervisedSampler(
            trainset_2.sup_indices, trainset_2.unsup_indices,
            args.batch_size, args.unsup_fraction,
            num_batches=int(np.ceil(epoch_datapoint_count / args.batch_size)), unsup_probabilities = example_probabilties_2)
      train_loader_2 = DataLoader(trainset_2, batch_sampler=train_batch_sampler_2, **kwargs)
      logger.info("Created train loader 2 with size %d" %(len(trainset_2)))

# ------------------------------------------------------------------------------

# ----------------------- TRAIN AND EVAL FUNCTIONS -----------------------------
def train(args, model, device, train_loader, optimizer, epoch, detector_model = None, example_weights = None, train_loader_2 = None, example_weights_2 = None):
    model.train()
    train_metrics = []
    epsilon = args.epsilon
    total_main_loss = 0
    total_detector_loss = 0
    total_count = 0.01
#     if train_loader_2 is None:
#           train_loader_2 = train_loader
    loss_1_back_count = 0
    loss_2_back_count = 0
    data_2 = None
    target_2 = None
    indexes_2 = None
    for batch_idx, (data, target, indexes) in enumerate(train_loader):
#     for batch_idx, ((data, target, indexes), (data_2, target_2, indexes_2)) in enumerate(zip(train_loader, train_loader_2)):
        data, target = data.to(device), target.to(device)
        if args.use_distrib_selection:
            data_2, target_2 = data_2.to(device), target_2.to(device)
        optimizer.zero_grad()

        # calculate robust loss
        if args.loss == 'shift':
            # The TRADES KL-robustness regularization term proposed by
            # Zhang et al., with some additional features
            assert detector_model != None, 'Error: no detector model found!'
            (loss, natural_loss, robust_loss,
             entropy_loss_unlabeled, main_loss, detector_loss) = shift_loss(
                model=model,
                detector_model = detector_model,
                x_natural=data,
                y=target,
                optimizer=optimizer,
                w_1 = args.w_1, 
                w_2 = args.w_2,
                step_size=args.pgd_step_size,
                epsilon=epsilon,
                perturb_steps=args.pgd_num_steps,
                beta=args.beta,
                distance=args.distance,
                adversarial=args.distance == 'l_inf',
                entropy_weight=args.entropy_weight)

            total_main_loss += main_loss.item()
            total_detector_loss += detector_loss.item()
            total_count += 1
        elif not args.use_adv_training:
            (loss, natural_loss, robust_loss,
             entropy_loss_unlabeled) = trades_non_adv_loss(
                model=model,
                x_natural=data,
                y=target,
                optimizer=optimizer,
                step_size=args.pgd_step_size,
                epsilon=epsilon,
                beta=args.beta,
                distance=args.distance,
                entropy_weight=args.entropy_weight,
                example_weights = example_weights, 
                indexes = indexes)
            
            if args.use_distrib_selection:
                  assert example_weights_2 is not None, "Weights for example none for second version"
                  optimizer.zero_grad()
                  (loss_2, natural_loss, robust_loss, entropy_loss_unlabeled) = trades_non_adv_loss( model=model, x_natural=data_2, y=target_2, optimizer=optimizer,
                                    step_size=args.pgd_step_size, epsilon=epsilon, beta=args.beta, distance=args.distance, entropy_weight=args.entropy_weight,
                                    example_weights = example_weights_2, indexes = indexes_2)                              
        elif args.loss == 'trades':
            # The TRADES KL-robustness regularization term proposed by
            # Zhang et al., with some additional features
            (loss, natural_loss, robust_loss,
             entropy_loss_unlabeled) = trades_loss(
                model=model,
                x_natural=data,
                y=target,
                optimizer=optimizer,
                step_size=args.pgd_step_size,
                epsilon=epsilon,
                perturb_steps=args.pgd_num_steps,
                beta=args.beta,
                distance=args.distance,
                adversarial=args.distance == 'l_inf',
                entropy_weight=args.entropy_weight)

        elif args.loss == 'noise':
            # Augmenting the input with random noise as in Cohen et al.
            assert (args.distance == 'l_2')
            loss = noise_loss(model=model, x_natural=data,
                              y=target, clamp_x=True, epsilon=epsilon)
            entropy_loss_unlabeled = torch.Tensor([0.])
            natural_loss = robust_loss = loss

        if args.use_distrib_selection and loss_2 > loss:
            #   logging.info("back from loss 2")
              loss_2_back_count = loss_2_back_count + 1 
              loss_2.backward()
        else:
            #   logging.info("back from loss 1")
              loss_1_back_count = loss_1_back_count + 1 
              loss.backward()
        optimizer.step()

        train_metrics.append(dict(
            epoch=epoch,
            loss=loss.item(),
            natural_loss=natural_loss.item(),
            robust_loss=robust_loss.item(),
            entropy_loss_unlabeled=entropy_loss_unlabeled.item()))

        # print progress
        if batch_idx % args.log_interval == 0:
            logging.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMain Loss: {:.6f}\tDetector Loss: {:.6f}\tloss1 count: {:d}\tloss2 count: {:d}'.format(
                    epoch, batch_idx * len(data), epoch_size, 100. * batch_idx / len(train_loader), 
                    loss.item(), total_main_loss/total_count, total_detector_loss/total_count, loss_1_back_count, loss_2_back_count))

    return train_metrics


def eval(args, model, device, eval_set, loader):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, indexes) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data, target = data[target != -1], target[target != -1]
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx < args.eval_attack_batches:
                if args.distance == 'l_2':
                    # run coarse certification
                    incorrect_clean, incorrect_rob = quick_smoothing(
                        model, data, target,
                        sigma=args.epsilon,
                        eps=args.epsilon,
                        num_smooth=100, batch_size=1000)
                    pass
                elif args.distance == 'l_inf':
                    # run medium-strength gradient attack
                    is_correct_clean, is_correct_rob, _ = pgd(
                        model, data, target,
                        epsilon=args.epsilon,
                        num_steps=2 * args.pgd_num_steps,
                        step_size=args.pgd_step_size,
                        random_start=False)
                    incorrect_clean = (1-is_correct_clean).sum()
                    incorrect_rob = (1-np.prod(is_correct_rob, axis=1)).sum()
                else:
                    raise ValueError('No support for distance %s',
                                     args.distance)
                adv_correct_clean += (len(data) - int(incorrect_clean))
                adv_correct += (len(data) - int(incorrect_rob))
                adv_total += len(data)
            total += len(data)
            if ((eval_set == 'train') and
                    (batch_idx + 1 == args.train_eval_batches)):
                break
    loss /= total
    accuracy = correct / total
    if adv_total > 0:
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.

    eval_data = dict(loss=loss, accuracy=accuracy,
                     robust_accuracy=robust_accuracy,
                     robust_clean_accuracy=robust_clean_accuracy)
    eval_data = {eval_set + '_' + k: v for k, v in eval_data.items()}
    logging.info(
        '{}: Clean loss: {:.4f}, '
        'Clean accuracy: {}/{} ({:.2f}%), '
        '{} clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            correct, total, 100.0 * accuracy,
            'Smoothing' if args.distance == 'l_2' else 'PGD',
            adv_correct_clean, adv_total, 100.0 * robust_clean_accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))

    return eval_data

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    schedule = args.lr_schedule
    # schedule from TRADES repo (different from paper due to bug there)
    if schedule == 'trades':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
    # schedule as in TRADES paper
    elif schedule == 'trades_fixed':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.9 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    # cosine schedule
    elif schedule == 'cosine':
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    # schedule as in WRN paper
    elif schedule == 'wrn':
        if epoch >= 0.3 * args.epochs:
              lr = args.lr * 0.2
        if epoch >= 0.6 * args.epochs:
            lr = args.lr * 0.2 * 0.2
        if epoch >= 0.8 * args.epochs:
            lr = args.lr * 0.2 * 0.2 * 0.2
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# ------------------------------------------------------------------------------

# ----------------------------- TRAINING LOOP ----------------------------------
def main():
    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    num_classes = 10
    if args.extend_svhn:
          num_classes = 20
    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=args.normalize_input)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    detector_model = None
    if args.use_detector_training:
        detector_model = load_detector_model(args)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)
        logger.info('Setting learning rate to %g' % lr)
        # adversarial training
        train_data = train(args, model, device, train_loader, optimizer, epoch, detector_model = detector_model, 
                              example_weights = example_weights, train_loader_2 = train_loader_2, example_weights_2 = example_weights_2)
        train_df = train_df.append(pd.DataFrame(train_data), ignore_index=True)

        # evaluation on natural examples
        logging.info(120 * '=')
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            eval_data = {'epoch': int(epoch)}
            eval_data.update(
                eval(args, model, device, 'train', eval_train_loader))
            eval_data.update(
                eval(args, model, device, 'test', eval_test_loader))
            eval_df = eval_df.append(pd.Series(eval_data), ignore_index=True)
            logging.info(120 * '=')

        # save stats
        train_df.to_csv(os.path.join(model_dir, 'stats_train.csv'))
        eval_df.to_csv(os.path.join(model_dir, 'stats_eval.csv'))

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(dict(num_classes=num_classes,
                            state_dict=model.state_dict(),
                            normalize_input=args.normalize_input),
                       os.path.join(model_dir,
                                    'checkpoint-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir,
                                    'opt-checkpoint_epoch{}.tar'.format(epoch)))
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
