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
import time
from utils import *

from losses import *
from datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS
from attack_pgd import pgd
from smoothing import quick_smoothing

from autoaugment import CIFAR10Policy
from cutout import Cutout

import logging
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# ----------------------------- CONFIGURATION ----------------------------------
parser = argparse.ArgumentParser(
    description='PyTorch TRADES Adversarial Training')

# Dataset config
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS, help='The dataset to use for training)')
parser.add_argument('--data_dir', default='data', type=str,
                    help='Directory where datasets are located')
parser.add_argument('--svhn_extra', action='store_true', default=False,
                    help='Adds the extra SVHN data')

# Model config
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='Name of the model (see utils.get_model)')
parser.add_argument('--model_dir', default='./rst_augmented',
                    help='Directory of model for saving checkpoint')
parser.add_argument('--test_name', default='',
                    help='Test name to give proper subdirectory to model for saving checkpoint')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Cancels the run if an appropriate checkpoint is found')
parser.add_argument('--normalize_input', action='store_true', default=False,
                    help='Apply standard CIFAR normalization first thing '
                         'in the network (as part of the model, not in the data'
                         ' fetching pipline)')

# Logging and checkpointing
parser.add_argument('--log_interval', type=int, default=5,
                    help='Number of batches between logging of training status')
parser.add_argument('--save_freq', default=5, type=int,
                    help='Checkpoint save frequency (in epochs)')

# Generic training configs
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed. '
                         'Note: fixing the random seed does not give complete '
                         'reproducibility. See '
                         'https://pytorch.org/docs/stable/notes/randomness.html')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=500, metavar='N',
                    help='Input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='Number of epochs to train. '
                         'Note: we arbitrarily define an epoch as a pass '
                         'through 50K datapoints. This is convenient for '
                         'comparison with standard CIFAR-10 training '
                         'configurations.')

# Eval config
parser.add_argument('--eval_freq', default=1, type=int,
                    help='Eval frequency (in epochs)')
parser.add_argument('--train_eval_batches', default=None, type=int,
                    help='Maximum number for batches in training set eval')
parser.add_argument('--eval_attack_batches', default=1, type=int,
                    help='Number of eval batches to attack with PGD or certify '
                         'with randomized smoothing')
parser.add_argument('--distance', '-d', default='l_inf', type=str,
                    help='Metric for attack model: l_inf uses adversarial '
                         'training and l_2 uses stability training and '
                         'randomized smoothing certification',
                    choices=['l_inf', 'l_2'])      
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='Adversarial perturbation size (takes the role of'
                         ' sigma for stability training)')
parser.add_argument('--pgd_num_steps', default=10, type=int,
                    help='number of pgd steps in adversarial training')
parser.add_argument('--pgd_step_size', default=0.007,
                    help='pgd steps size in adversarial training', type=float)             
parser.add_argument('--loss-percent', default=50, type=float, help='Loss percent for filtering examples for adversarial training')                                                    

# Optimizer config
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='Learning rate')
parser.add_argument('--lr_schedule', type=str, default='cosine',
                    choices=('trades', 'trades_fixed', 'cosine', 'wrn'),
                    help='Learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='Use extragrdient steps')

# Pre train configs
parser.add_argument('--pretrained_model_dir', default='pretrained_models', type=str, help='Model directory for pretrained models')
parser.add_argument('--pretrained_model_name', default='PreActResnet18' ,type=str, help='name for the pretrained-model')
parser.add_argument('--pretrained_epochs', default=14,type=int, help='number of epochs for the pretrained-model')

# Semi-supervised training configuration
parser.add_argument('--aux_data_filename', default='ti_500K_pseudo_labeled.pickle', type=str,
                    help='Path to pickle file containing unlabeled data and '
                         'pseudo-labels used for RST')

parser.add_argument('--unsup_fraction', default=0.5, type=float,
                    help='Fraction of unlabeled examples in each batch; '
                         'implicitly sets the weight of unlabeled data in the '
                         'loss. If set to -1, batches are sampled from a '
                         'single pool')
parser.add_argument('--train_take_amount', default=0, type=int, help='Number of random aux examples to retain. None retains all aux data.')                  
parser.add_argument('--aux_take_amount', default=None, type=int, help='Number of random aux examples to retain. None retains all aux data.')

parser.add_argument('--remove_pseudo_labels', action='store_true',
                    default=False,
                    help='Performs training without pseudo-labels (rVAT)')
parser.add_argument('--entropy_weight', type=float,
                    default=0.0, help='Weight on entropy loss')

# Additional aggressive data augmentation
parser.add_argument('--autoaugment', action='store_true', default=False,
                    help='Use autoaugment for data augmentation')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='Use cutout for data augmentation')

args = parser.parse_args()

# ------------------------------ OUTPUT SETUP ----------------------------------


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.pretrained_model_dir + '/' + args.pretrained_model_name,'unlabloss_' + str(args.pretrained_epochs) + '.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Robust self-training')
logging.info('Args: %s', args)


# ------------------------------------------------------------------------------

# ------------------------------- CUDA SETUP -----------------------------------
# should provide some improved performance
cudnn.benchmark = True
# useful setting for debugging
# cudnn.benchmark = False
# cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if use_cuda else 'cpu')
# ------------------------------------------------------------------------------

# --------------------------- DATA AUGMENTATION --------------------------------
if args.dataset == 'cifar10' or args.dataset == 'cifar_own':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
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
# take amount zero to just load unlabeled data

trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                 add_svhn_extra=args.svhn_extra,
                                 root=args.data_dir, train=True,
                                 take_amount = args.train_take_amount,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount)

# num_batches=50000 enforces the definition of an "epoch" as passing through 50K
# datapoints
# TODO: make sure that this code works also when trainset.unsup_indices=[]
epoch_datapoint_count = 500000
print("epoch datapoints count: %d" %(epoch_datapoint_count))
# train_batch_sampler = SemiSupervisedSampler(
#     trainset.sup_indices, trainset.unsup_indices,
#     args.batch_size, args.unsup_fraction,
#     num_batches=int(np.ceil(epoch_datapoint_count / args.batch_size)))
# epoch_size = len(train_batch_sampler) * args.batch_size
# 
# print("Epoch size: %d" %(epoch_size))

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)

train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = False, **kwargs)

testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                root=args.data_dir, train=False,
                                download=True,
                                transform=transform_test)
test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)

trainset_eval = SemiSupervisedDataset(
    base_dataset=args.dataset,
    add_svhn_extra=args.svhn_extra,
    root=args.data_dir, train=True,
    download=True, transform=transform_train)

eval_train_loader = DataLoader(trainset_eval, batch_size=args.test_batch_size,
                               shuffle=True, **kwargs)

eval_test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                              shuffle=False, **kwargs)
# ------------------------------------------------------------------------------




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
            if batch_idx % 100 == 0:
                  print(batch_idx)
                  print(data[:,:,:,1])
                  print(target)
                  print(pred.view_as(target))
                  print(output)
                  print()
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
                    is_correct_clean, is_correct_rob = pgd(
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


def load_pretrained_model(model_dir, model_name, epoch_no = None):

      pretrained_model = get_model(model_name)
      assert os.path.isdir(model_dir), 'Error: no checkpoint directory found!'
      assert os.path.isdir(model_dir + '/' + model_name), 'Error: no checkpoint directory found!'
      model_file = 'ckpt.pth'
      if epoch_no is not None:
            model_file = 'ckpt_' + str(epoch_no) + '.pth'
      print("Loading model from file: %s ........." %(model_file))
      assert os.path.isfile(model_dir + '/' + model_name + '/' + model_file), 'Error: no checkpoint file found!'
      checkpoint = torch.load(model_dir + '/' + model_name + '/' + model_file, map_location=torch.device(device))
      pretrained_model = pretrained_model.to(device)
      if device == 'cuda':
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            cudnn.benchmark = True
      else:
            pretrained_model = torch.nn.DataParallel(pretrained_model)
      pretrained_model.load_state_dict(checkpoint['net'])  
      best_acc = checkpoint['acc']
      start_epoch = checkpoint['epoch']

      return pretrained_model, best_acc, start_epoch



def calculate_pretrained_model_losses(dataloader, pretrained_model, criterion, trainset_length, multi_margin_loss_margin = 3):
      with torch.no_grad():
        start_epoch_time = time.time()
        print("Calculating individual losses for pretrained model ....")
        train_loss = 0
        train_acc = 0
        train_n = 0 
        example_cross_ent_losses = torch.flatten(torch.empty(0,1)).to(device)
        example_outputs = torch.flatten(torch.empty(0,1)).to(device)
        example_labels = torch.flatten(torch.empty(0,1)).long().to(device)
        example_multi_margin_losses = torch.flatten(torch.empty(0,1)).to(device)
        ce_criterion = nn.CrossEntropyLoss()
        mm_criterion = nn.MultiMarginLoss()
        pretrained_model.eval()

      #   example_cross_ent_losses = torch.zeros(trainset_length).to(device)
        used_indices = torch.zeros(trainset_length).to(device)
      #   pretrained_model.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = inputs[targets != -1], targets[target != -1]
            outputs = pretrained_model(inputs)
            # print(outputs.shape)
            # print(indexes)
            indiv_loss = calc_indiv_loss(outputs, targets)

            indiv_mm_loss = F.multi_margin_loss(outputs, targets, reduction = 'none', margin = multi_margin_loss_margin)
            indiv_ce_loss = F.cross_entropy(outputs, targets, reduction = 'none')
            example_cross_ent_losses = torch.cat([example_cross_ent_losses, indiv_loss], dim=0)
            example_multi_margin_losses = torch.cat([example_multi_margin_losses, indiv_mm_loss], dim=0)
            example_outputs = torch.cat([example_outputs, outputs], dim=0)
            example_labels = torch.cat([example_labels, targets], dim=0)
            # example_cross_ent_losses.data[indexes] = indiv_loss
            used_indices.data[indexes] = 1
            # print(example_cross_ent_losses.shape)
            if batch_idx % 100 == 0:
                  print(batch_idx)
                  # print(targets)
                  # print(outputs.max(1)[1])
                  # print(outputs)
                  # print()
            train_loss += torch.mean(indiv_ce_loss).item() * targets.size(0)
            train_acc += (outputs.max(1)[1] == targets).sum().item()
            train_n += targets.size(0)
        epoch_time = time.time()
        print('Result on pretrained model: Time: %.2f \t \t Loss: %.4f \t Accuracy: %.4f' %(epoch_time - start_epoch_time, train_loss/train_n, train_acc/train_n))
      print("used indices: %d" %(torch.sum(used_indices)))
      return example_cross_ent_losses.detach().cpu(), example_multi_margin_losses.detach().cpu(), example_outputs.detach().cpu(), example_labels.detach()



# ----------------------------- TRAINING LOOP ----------------------------------
def main():
    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    num_classes = 10
    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=args.normalize_input)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    criterion = F.cross_entropy
    trainset_length = epoch_datapoint_count
    pretrained_model, pretrained_acc, pretrained_epochs = load_pretrained_model(args.pretrained_model_dir, args.pretrained_model_name, args.pretrained_epochs)
    eval_data = {}
#     eval_data.update(eval(args, pretrained_model, device, 'train', train_loader))

    
#     
#     print("Loaded pretrained model. Calculating losses ......")
    example_cross_ent_losses,  example_multi_margin_losses, example_outputs, example_labels = calculate_pretrained_model_losses(
                                          train_loader, pretrained_model, criterion, trainset_length)
    ce_loss_thresh = np.percentile(example_cross_ent_losses, args.loss_percent)
    mm_loss_thresh = np.percentile(example_multi_margin_losses, args.loss_percent)
    print("Loaded pretrained model wth acc %0.2f and epochs %d, loss threshold %0.8f, %0.8f" %(pretrained_acc, pretrained_epochs, ce_loss_thresh, mm_loss_thresh))                                       
#     loss_info = {
#                 'model_name': args.pretrained_model_name, 'dataset_length': example_cross_ent_losses.size(0), 'example_cross_ent_losses': example_cross_ent_losses, 
#                 'pretrained_acc': pretrained_acc, 'pretrained_epochs': pretrained_epochs, 'example_multi_margin_losses': example_multi_margin_losses, 
#                 'example_outputs': example_outputs, 'example_labels': example_labels, 'multimarginloss_margin': 3
#           }
# #     assert example_cross_ent_losses.size() == example_multi_margin_losses.size(), 'CE and MM loss sizes dont match'
#     dump_pretrained_example_losses_to_file(args.pretrained_model_dir, pretrained_epochs, args.pretrained_model_name, loss_info)
#     example_cross_ent_losses, example_multi_margin_losses, pretrained_acc, pretrained_epochs = load_pretrained_example_losses_from_file(
#           args.pretrained_model_dir, args.pretrained_model_name, args.pretrained_epochs)
#     print("Pretrained model had acc %0.2f and epochs %d, loss threshold %0.8f, %0.8f" %(pretrained_acc, pretrained_epochs, ce_loss_thresh, mm_loss_thresh))

#     for epoch in range(1, args.epochs + 1):
#         # adjust learning rate for SGD
#         lr = adjust_learning_rate(optimizer, epoch)
#         logger.info('Setting learning rate to %g' % lr)
#         # adversarial training
#         train_data = train(args, model, device, train_loader, optimizer, epoch)
#         train_df = train_df.append(pd.DataFrame(train_data), ignore_index=True)

#         # evaluation on natural examples
#         logging.info(120 * '=')
#         if epoch % args.eval_freq == 0 or epoch == args.epochs:
#             eval_data = {'epoch': int(epoch)}
#             eval_data.update(
#                 eval(args, model, device, 'train', eval_train_loader))
#             eval_data.update(
#                 eval(args, model, device, 'test', eval_test_loader))
#             eval_df = eval_df.append(pd.Series(eval_data), ignore_index=True)
#             logging.info(120 * '=')

#         # save stats
#         train_df.to_csv(os.path.join(model_dir, 'stats_train.csv'))
#         eval_df.to_csv(os.path.join(model_dir, 'stats_eval.csv'))

#         # save checkpoint
#         if epoch % args.save_freq == 0 or epoch == args.epochs:
#             torch.save(dict(num_classes=num_classes,
#                             state_dict=model.state_dict(),
#                             normalize_input=args.normalize_input),
#                        os.path.join(model_dir,
#                                     'checkpoint-epoch{}.pt'.format(epoch)))
#             torch.save(optimizer.state_dict(),
#                        os.path.join(model_dir,
#                                     'opt-checkpoint_epoch{}.tar'.format(epoch)))
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
