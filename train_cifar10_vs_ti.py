"""
Train data sourcing model. Based on code from
https://github.com/hysts/pytorch_shake_shake
"""
import argparse
from collections import OrderedDict
import importlib
import json
import logging
import pathlib
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from utils import *
from dataloader import *
from datasets import SemiSupervisedDataset, DATASETS
from diff_distribution_dataload_helper import get_new_distribution_loader
import pdb
import pandas as pd
from dataloader import get_cifar10_vs_ti_loader

torch.backends.cudnn.benchmark = True

# logging.basicConfig(
#     format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
#     datefmt='%Y/%m/%d %H:%M:%S',
#     level=logging.INFO)
# logger = logging.getLogger(__name__)

global_step = 0
use_cuda = torch.cuda.is_available()


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def mean_std_normalize(input, mean, std):
    input = input.transpose(-1,-3).transpose(-2,-3).cuda()
    assert input.shape[-1] == mean.shape[-1], "last input dimension does not match mean dimension"
    assert input.shape[-1] == std.shape[-1], "last input dimension does not match std dimension"
    mean = mean.repeat(*list(input.shape[:-1]), 1).cuda()
    std = std.repeat(*list(input.shape[:-1]), 1).cuda()
    output = input.sub(mean).div(std)
    output = output.transpose(-1,-3).transpose(-2,-1)
    return output

def load_base_model(args):
        checkpoint = torch.load(args.base_model_path)
        state_dict = checkpoint.get('state_dict', checkpoint)
        num_classes = checkpoint.get('num_classes', args.base_num_classes)
        normalize_input = checkpoint.get('normalize_input', False)
        print("checking if input normalized")
        print(normalize_input)
        logging.info("using %s model for evaluation from path %s" %(args.base_model, args.base_model_path))
        base_model = get_model(args.base_model, num_classes=num_classes, normalize_input=normalize_input)
        if use_cuda:
            base_model = torch.nn.DataParallel(base_model).cuda()
            cudnn.benchmark = True
            def strip_data_parallel(s):
                  if s.startswith('module.1'):
                        return 'module.' + s[len('module.1.'):]
                  elif s.startswith('module.0'):
                        return None
                  else:
                        return s
            
            if not all([k.startswith('module') for k in state_dict]):
                  state_dict = {'module.' + k: v for k, v in state_dict.items()}
            new_state_dict = {}
            for k,v in state_dict.items():
                  k_new = strip_data_parallel(k)
                  if k_new:
                        new_state_dict[k_new] = v
            state_dict = new_state_dict            
            # state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}      
        else:
            def strip_data_parallel(s):
                  if s.startswith('module.1'):
                        return s[len('module.1.'):]
                  elif s.startswith('module.0'):
                        return None
                  if s.startswith('module'):
                        return s[len('module.'):]
                  else:
                        return s
            state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict)
        return base_model


def parse_args():
    parser = argparse.ArgumentParser()
    # model config
    # parser.add_argument('--model', type=str, default='wrn-28-10')
    parser.add_argument('--dataset', type=str, default='custom', help='The dataset', 
                              choices=['cifar10', 'svhn', 'custom', 'cinic10', 'benrecht_cifar10', 'tinyimages', 'unlabeled_percy_500k'])
    # detector model config
    parser.add_argument('--detector-model', default='wrn-28-10', type=str, help='Name of the detector model (see utils.get_model)')
    parser.add_argument('--use-old-detector', default=0, type=int, help='Use detector model for evaluation')
    parser.add_argument('--detector_model_path', default = 'selection_model/selection_model.pth', type = str, help='Model for attack evaluation')
    parser.add_argument('--n_classes', type=int, default=11, help='Number of classes for detector model')
    parser.add_argument('--random_split_version', type=int, default=2, help='Version of random split')

    # base model configs
    parser.add_argument('--also-use-base-model', default=0, type=int, help='Use base model for confusion matrix evaluation')
    parser.add_argument('--base_model_path', help='Base Model path')
    parser.add_argument('--base_model', '-bm', default='resnet-20', type=str, help='Name of the base model')
    parser.add_argument('--base_num_classes', type=int, default=10, help='Number of classes for base model')
    parser.add_argument('--base_normalize', type=int, default=0, help='Normalze input for base model')
    # run config
    parser.add_argument('--output_dir', default='selection_model',type=str, required=True)
    parser.add_argument('--test_name', default='', help='Test name to give proper subdirectory to model for saving checkpoint')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--store_to_dataframe', default=0, type=int, help='Store confidences to dataframe')
    # Semi-supervised training configuration
    parser.add_argument('--aux_data_filename', default='ti_500K_pseudo_labeled.pickle', type=str,
                    help='Path to pickle file containing unlabeled data and pseudo-labels used for RST')
    parser.add_argument('--train_take_amount', default=None, type=int, help='Number of random aux examples to retain. None retains all aux data.')                         
    parser.add_argument('--aux_take_amount', default=None, type=int, help='Number of random aux examples to retain. '
                         'None retains all aux data.')
    parser.add_argument('--remove_pseudo_labels', action='store_true', default=False, help='Performs training without pseudo-labels (rVAT)')
    parser.add_argument('--entropy_weight', type=float, default=0.0, help='Weight on entropy loss')
    # optim config
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--lr_min', type=float, default=0)

    #train configs
    parser.add_argument('--num_images', type=int, help='Number of images in dataset')
    parser.add_argument('--even_odd', type=int, default = 0, help='Filter train, test data for even odd indices')
    parser.add_argument('--ti_start_index', type=int, default=0, help='Starting index of image')
    parser.add_argument('--load_ti_head_tail', type=int, default = 0, help='Load ti head tail indices')
    parser.add_argument('--class11_weight', type=float, default=0.1)
    parser.add_argument('--use_ti_data_for_training', default=1, type=int, help='Whether to use ti data for training')   
    args = parser.parse_args()

    # 10 CIFAR10 classes and one non-CIFAR10 class
    model_config = OrderedDict([
        # ('name', args.model),
        ('n_classes', args.n_classes),
        ('detector_model_name', args.detector_model), 
        ('use_old_detector', args.use_old_detector), 
        ('detector_model_path', args.detector_model_path)
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('lr_min', args.lr_min),
        ('cifar10_fraction', 0.5)
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10VsTinyImages'),
        ('dataset_dir', args.data_dir),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('outdir', args.output_dir),
        ('num_workers', args.num_workers),
        ('device', args.device),
        ('save_freq', args.save_freq),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config, args


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


def get_cosine_annealing_scheduler(optimizer, optim_config):
    total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            optim_config['lr_min'] / optim_config['base_lr']))

    return scheduler


def train(epoch, model, optimizer, scheduler, criterion, train_loader,
          run_config):
    global global_step

    logging.info('Train {}'.format(epoch))

    model.train()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    accuracy_c10_meter = AverageMeter()
    accuracy_c10_v_ti_meter = AverageMeter()
    start = time.time()
    class_counts = np.zeros(11)
    for step, (data, targets, index) in enumerate(train_loader):
        global_step += 1

        scheduler.step()

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)
        unique_targets = np.array(targets.unique(return_counts=True)[0].cpu())
        unique_counts = np.array(targets.unique(return_counts=True)[1].cpu())
        class_counts[unique_targets] = class_counts[unique_targets] + unique_counts 
        if step == 0:
                print(data[1,:])
                print(outputs[1,:])
                print(preds)
                # print(indexes)
                print(targets)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)
        
        is_c10 = targets != 10
        num_c10 = is_c10.float().sum().item()
        # Computing cifar10 accuracy
        if num_c10 > 0:
            _, preds_c10 = torch.max(outputs[is_c10, :10], dim=1)
            correct_c10_ = preds_c10.eq(targets[is_c10]).sum().item()
            accuracy_c10_meter.update(correct_c10_ / num_c10, num_c10)

        # Computing cifar10 vs. ti accuracy
        correct_c10_v_ti_ = (preds != 10).float().eq(
            is_c10.float()).sum().item()
        accuracy_c10_v_ti_meter.update(correct_c10_v_ti_ / num, num)


        if step % 100 == 0:
            logging.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f}) '
                        'C10 Acc {:.4f} ({:.4f}) '
                        'Vs Acc {:.4f} ({:.4f})'.format(
                epoch,
                step,
                len(train_loader),
                loss_meter.val,
                loss_meter.avg,
                accuracy_meter.val,
                accuracy_meter.avg,
                accuracy_c10_meter.val,
                accuracy_c10_meter.avg,
                accuracy_c10_v_ti_meter.val,
                accuracy_c10_v_ti_meter.avg
            ))

    elapsed = time.time() - start
    logging.info('Target class count: '+str(class_counts))
    logging.info('Elapsed {:.2f}'.format(elapsed))

    train_log = OrderedDict({
        'epoch':
            epoch,
        'train':
            OrderedDict({
                'loss': loss_meter.avg,
                'accuracy': accuracy_meter.avg,
                'accuracy_c10': accuracy_c10_meter.avg,
                'accuracy_vs': accuracy_c10_v_ti_meter.avg,
                'time': elapsed,
            }),
    })
    return train_log


def test(args, epoch, model, criterion, test_loader, run_config, mean, std, base_model = None, dataframe_file = None):
    logging.info('Test {}'.format(epoch))
    dataset = args.dataset
    model.eval()
    if base_model != None:
        base_model.eval()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    correct_c10_meter = AverageMeter()
    correct_c10_v_ti_meter = AverageMeter()
    correct_on_predc10_meter = AverageMeter()
    pseudocorrect_on_predti_meter = AverageMeter()
    start = time.time()
    count_total = 0
    c10_correct_total = 0
    c10_count_total = 0
    ti_count_total = 0
    ti_correct_total = 0

    total = 0
    vs_correct_total = 0

    predc10_correct_total = 0
    predc10_count_total = 0
    predti_pseudocorrect_total = 0 
    predti_count_total = 0

    base_c10_correct_total = 0
    base_predc10_correct_total = 0
    base_predti_correct_total = 0 
    base_c10_count_total = 0

    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        cifar_conf = []
        noncifar_conf = []

        noncifar_all_confs = []

        id_list = []
        df = pd.DataFrame()
        for step, (data, targets, indexes) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            id_list = np.array(indexes)
            target_list = targets.cpu().detach().numpy()
            # TODO: This is hacky rn. See the right way to load TinyImages
            # if dataset == 'tinyimages':
            #     data = data.transpose(1, 3).type(torch.FloatTensor)
            # print(data.shape)
            # print(tuple(data.shape))
            # print(torch.transpose(data,1,3).view(-1,*tuple(data_shape[2:])).shape)
            # outputs = model(normalize_func(tensor=data.squeeze(1)).reshape(data_shape))
            outputs = model(mean_std_normalize(data, mean, std))
            loss = criterion(outputs, targets)
            outputs = softmax(outputs)

            conf, preds = torch.max(outputs, dim=1)

            if base_model != None:
                if args.base_normalize:
                      base_outputs = base_model(mean_std_normalize(data, mean, std))
                else:
                      base_outputs = base_model(data)
                base_outputs = softmax(base_outputs)
                _, base_preds = torch.max(base_outputs, dim=1)

            if step == 0:
                print(data[1,:])
                print(outputs[1,:])
                print(preds)
                # print(indexes)
                print(targets)

            if step%100 == 0:
                  print(step) 

            # is_pred_c10 = preds != 10
            is_predc10 = preds != 10
            is_pred_nonc10 = preds == 10
            cifar_conf.extend(conf[is_predc10].tolist())
            noncifar_conf.extend(conf[is_pred_nonc10].tolist())

            if len(noncifar_all_confs) < 30:
                noncifar_all_confs.extend(outputs[is_pred_nonc10].tolist())

            loss_ = loss.item()
            num = data.size(0)
            loss_meter.update(loss_, num)

            is_c10 = targets != 10
            # cifar10 accuracy 
            if is_c10.float().sum() > 0:
                _, preds_c10 = torch.max(outputs[is_c10, :10], dim=1)
                correct_c10_ = preds_c10.eq(targets[is_c10]).sum().item()
                if base_model != None:
                    _, base_preds_c10 = torch.max(base_outputs[is_c10, :10], dim=1)
                    base_c10_correct_total += base_preds_c10.eq(targets[is_c10]).sum().item() 
                    base_c10_count_total += is_c10.sum()              
                    if step == 0:
                        print("-----------------------------------------------------")
                        print(base_preds_c10)   
                        print(preds_c10)
                        print(targets)
                c10_correct_total += correct_c10_
                c10_count_total += is_c10.sum()
                correct_c10_meter.update(correct_c10_, 1)
                
            # cifar10 vs. TI accuracy
            correct_c10_v_ti_ = (is_predc10).eq(is_c10).sum().item()
            correct_c10_v_ti_meter.update(correct_c10_v_ti_, 1)
            total += len(targets)
            vs_correct_total += correct_c10_v_ti_
            # print("Step %d, batch size %d, correct_c10_vs_ti_count %d" %(step, len(targets), correct_c10_v_ti_))
            

            if is_predc10.float().sum() > 0:
                _, preds_on_predc10 = torch.max(outputs[is_predc10, :10], dim=1)
                correct_on_predc10_ = preds_on_predc10.eq(targets[is_predc10]).sum().item()
                if base_model != None:
                    _, base_preds_on_predc10 = torch.max(base_outputs[is_predc10, :10], dim=1)
                    base_predc10_correct_total += base_preds_on_predc10.eq(targets[is_predc10]).sum().item()

                predc10_correct_total += correct_on_predc10_
                predc10_count_total += is_predc10.sum()
                correct_on_predc10_meter.update(correct_on_predc10_, 1)

            is_predti = preds == 10
            if is_predti.float().sum() > 0:
                _, preds_on_predti = torch.max(outputs[is_predti, :10], dim=1)
                pseudocorrect_on_predti_ = preds_on_predti.eq(targets[is_predti]).sum().item()
                if base_model != None:
                    _, base_preds_on_predti = torch.max(base_outputs[is_predti, :10], dim=1)
                    base_predti_correct_total += base_preds_on_predti.eq(targets[is_predti]).sum().item()
                predti_pseudocorrect_total += pseudocorrect_on_predti_
                predti_count_total += is_predti.sum()
                pseudocorrect_on_predti_meter.update(pseudocorrect_on_predti_, 1)
            if args.store_to_dataframe:
                  batch_df = pd.DataFrame(np.column_stack([id_list, target_list, outputs.cpu().detach().numpy(), base_outputs.cpu().detach().numpy(), 
                                                preds.cpu().detach().numpy(), base_preds.cpu().detach().numpy(),
                                                is_c10.cpu().detach().numpy(),is_predc10.cpu().detach().numpy(),
                                                is_predti.cpu().detach().numpy()]))                                              
                  # print("Batch %d, batch df shape %s" %(step, str(batch_df.shape)))
                  df = df.append(batch_df)                                          

    test_targets = np.array(test_loader.dataset.targets)
    accuracy_c10 = ((c10_correct_total * 1.0) /
                   (c10_count_total*1.0))
    accuracy_vs = ((correct_c10_v_ti_meter.sum*1.0) / total)

    logging.info('Epoch {} Loss {:.4f} Accuracy inside C10 {:.4f}'
                ' C10-vs-TI {:.4f}'.format(
        epoch, loss_meter.avg, accuracy_c10, accuracy_vs))
    logging.info('Cifar10 correct {} Cifar10 sum {} c10-vs-ti correct {},'
                ' C10-vs-TI-sum {}'.format(
        c10_correct_total, c10_count_total, correct_c10_v_ti_meter.sum, total))
    logging.info('Cifar10 correct %d, cifar 10 count %d, predicted c10 correct %d, predicted c10 count %d, predicted ti pseudo correct %d ' \
                                                      'predicted ti count %d' %(c10_correct_total, c10_count_total, predc10_correct_total,
                                                      predc10_count_total, predti_pseudocorrect_total, predti_count_total))
    if base_model != None:
            logging.info('base cifar10 correct %d, base predicted c10 correct %d, base predicted TI correct %d' 
                        %(base_c10_correct_total, base_predc10_correct_total, base_predti_correct_total))

    logging.info('CIFAR count: {}, Non-CIFAR count: {}'.format(len(cifar_conf), len(noncifar_conf)))
    elapsed = time.time() - start
    if args.store_to_dataframe:
            df.to_csv(dataframe_file, index = False)
#     plot_histogram(cifar_conf, noncifar_conf, dataset)

    # print('Non cifar probabilities:')
    # print(noncifar_all_confs)

    test_log = OrderedDict({
        'epoch':
            epoch,
        'test':
            OrderedDict({
                'loss': loss_meter.avg,
                'accuracy_c10': accuracy_c10,
                'accuracy_vs': accuracy_vs,
                'time': elapsed,
            }),
    })
    return test_log


def main():
    # parse command line arguments
    config, args = parse_args()

    
    output_dir = args.output_dir
    if args.test_name != '':
            output_dir = output_dir + '/' + args.test_name

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if config['model_config']['use_old_detector']:
          output_file = args.dataset + '.log'
    else:
          output_file = 'training.log'
    logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s | %(message)s",
      handlers=[
            logging.FileHandler(os.path.join(output_dir, output_file)),
            logging.StreamHandler()
    ])
    logger = logging.getLogger()
    dataframe_file = output_dir + '/' + args.dataset + '.csv'

    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
#     outdir = pathlib.Path(run_config['outdir'])
#     outdir.mkdir(exist_ok=True, parents=True)
    save_freq = run_config['save_freq']

    # save config as json file in output directory


    outpath = os.path.join(output_dir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)
    custom_testset = None
    # if args.dataset == 'custom':
    #     custom_dataset = get_new_distribution_loader()
    #     print("custom dataset loaded ....")
    #     transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    #     mean = torch.tensor([0.4914, 0.4822, 0.4465])
    #     std = torch.tensor([
    # 0.2470, 0.2435, 0.2616])
    #     custom_testset = SemiSupervisedDataset(base_dataset=args.dataset,
    #                                       train=False, root='data',
    #                                       download=True,
    #                                       custom_dataset = custom_dataset,
    #                                       transform=transform_test)
    #     mean, std = 

    # data loaders
    
    # model
    model = get_model(config['model_config']['detector_model_name'],
                      num_classes=config['model_config']['n_classes'],
                      normalize_input=True)
    model = torch.nn.DataParallel(model.cuda())
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))
    if args.n_classes == 11:
          weight = torch.Tensor([1] * 10 + [args.class11_weight])
    else:
          weight = torch.Tensor([1]* args.n_classes)
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    weight=weight).cuda()
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    if args.also_use_base_model:
            base_model = load_base_model(args)
    else:
            base_model = None
    if config['model_config']['use_old_detector']:
            logging.info("using old detector model for evaluation")
            model = load_detector_model(args)
            
            dl_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            if args.dataset == 'benrecht_cifar10' or args.dataset == 'cifar10' or args.dataset == 'cinic10':
                  #   custom_dataset = get_new_distribution_loader()
                  print("custom dataset loaded ....")
                  transform_test = transforms.Compose([transforms.ToTensor(), ])
                  
                  testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                                      train=False, root='data',
                                                      download=True,
                                                      transform=transform_test)
                  trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                                      train=True, root='data',
                                                      download=True,
                                                      transform=transform_test)                                  
                  test_loader = torch.utils.data.DataLoader(testset,
                                                      batch_size=args.batch_size,
                                                      shuffle=False, **dl_kwargs)
                  train_loader = torch.utils.data.DataLoader(trainset,
                                                      batch_size=args.batch_size,
                                                      shuffle=True, **dl_kwargs)                                                                          
            elif args.dataset == 'unlabeled_percy_500k':
                  print('Loading unlabeled dataset:', args.dataset, '...')
                  transform_train = transforms.Compose([transforms.ToTensor(), ])
                  trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                 root=args.data_dir, train=True,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount) 
                  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
                  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                      shuffle=True, **kwargs)
                  
                  test_loader = train_loader
            elif args.dataset == 'tinyimages':
                  test_loader, _ = get_cifar10_vs_ti_loader(
                                                optim_config['batch_size'],
                                                run_config['num_workers'],
                                                run_config['device'] != 'cpu',
                                                args.num_images,
                                                optim_config['cifar10_fraction'],
                                                dataset_dir=data_config['dataset_dir'], 
                                                even_odd = args.even_odd,
                                                load_ti_head_tail = args.load_ti_head_tail,
                                                random_split_version = args.random_split_version, 
                                                ti_start_index = args.ti_start_index,
                                                logger=logger)

            # normalize_func  = transforms.Normalize(mean.unsqueeze(0),std.unsqueeze(0))

            logger.info('Instantiated data loaders')
            test(args, 0, model, criterion, test_loader, run_config, mean, std, base_model = base_model, dataframe_file = dataframe_file)

    else:
            train_loader, test_loader = get_cifar10_vs_ti_loader(
                                                optim_config['batch_size'],
                                                run_config['num_workers'],
                                                run_config['device'] != 'cpu',
                                                args.num_images,
                                                optim_config['cifar10_fraction'],
                                                dataset_dir=data_config['dataset_dir'], 
                                                even_odd = args.even_odd,
                                                load_ti_head_tail = args.load_ti_head_tail,
                                                use_ti_data_for_training = args.use_ti_data_for_training,
                                                random_split_version = args.random_split_version, 
                                                ti_start_index = args.ti_start_index,
                                                logger=logger)
            # optimizer
            optim_config['steps_per_epoch'] = len(train_loader)
            optimizer = torch.optim.SGD(
                  model.parameters(),
                  lr=optim_config['base_lr'],
                  momentum=optim_config['momentum'],
                  weight_decay=optim_config['weight_decay'],
                  nesterov=optim_config['nesterov'])
            scheduler = get_cosine_annealing_scheduler(optimizer, optim_config)

            # run test before start training
            #     test(args, 0, model, criterion, test_loader, run_config, mean, std, base_model = base_model, dataframe_file = dataframe_file)

            epoch_logs = []
            if args.even_odd >= 0:
                  if args.even_odd:
                        suffix = 'head'
                  else:
                        suffix = 'tail' 
            else:
                  suffix = ''              
            for epoch in range(1, optim_config['epochs'] + 1):
                  train_log = train(epoch, model, optimizer, scheduler, criterion,
                                    train_loader, run_config)
                  test_log = test(args, epoch, model, criterion, test_loader, run_config, mean, std, base_model = base_model, dataframe_file = dataframe_file)

                  epoch_log = train_log.copy()
                  epoch_log.update(test_log)
                  epoch_logs.append(epoch_log)
                  # with open(os.path.join(output_dir, 'log.json'), 'w') as fout:
                  #       json.dump(epoch_logs, fout, indent=2)

                  if epoch % save_freq == 0 or epoch == optim_config['epochs']:
                        state = OrderedDict([
                        ('config', config),
                        ('state_dict', model.state_dict()),
                        ('optimizer', optimizer.state_dict()),
                        ('epoch', epoch),
                        ('accuracy_vs', test_log['test']['accuracy_vs']),
                        ])

                        model_path = os.path.join(output_dir,('model_state_epoch_%s_%d.pth' % (suffix, epoch)))
                        torch.save(state, model_path)
                        print("Saved model for path %s" %(model_path))
                  
            
            test(args, 0, model, criterion, test_loader, run_config, mean, std, base_model = base_model, dataframe_file = dataframe_file)

if __name__ == '__main__':
    main()
