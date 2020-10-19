"""
Robust training losses. Based on code from
https://github.com/yaodongyu/TRADES
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import pdb


def entropy_loss(unlabeled_logits):
    unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
    return -(unlabeled_probs * F.log_softmax(unlabeled_logits, dim=1)).sum(
        dim=1).mean(dim=0)


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                adversarial=True,
                distance='inf',
                entropy_weight=0):
    """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""
    if beta == 0:
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y)
        inf = torch.Tensor([np.inf])
        zero = torch.Tensor([0.])
        return loss, loss, inf, zero

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()  # moving to eval mode to freeze batchnorm stats
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    if adversarial:
        if distance == 'l_inf':
            x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()

            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                                  x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError('No support for distance %s in adversarial '
                             'training' % distance)
    else:
        if distance == 'l_2':
            x_adv = x_adv + epsilon * torch.randn_like(x_adv)
        else:
            raise ValueError('No support for distance %s in stability '
                             'training' % distance)

    model.train()  # moving to train mode to update batchnorm stats

    # zero gradient
    optimizer.zero_grad()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    logits_adv = F.log_softmax(model(x_adv), dim=1)
    logits = model(x_natural)

    loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
    p_natural = F.softmax(logits, dim=1)
    loss_robust = criterion_kl(
        logits_adv, p_natural) / batch_size

    loss = loss_natural + beta * loss_robust

    is_unlabeled = (y == -1)
    if torch.sum(is_unlabeled) > 0:
        logits_unlabeled = logits[is_unlabeled]
        loss_entropy_unlabeled = entropy_loss(logits_unlabeled)
        loss = loss + entropy_weight * loss_entropy_unlabeled
    else:
        loss_entropy_unlabeled = torch.tensor(0)

    return loss, loss_natural, loss_robust, loss_entropy_unlabeled


def trades_non_adv_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,    
                epsilon=0.031,
                beta=1.0,
                distance='inf', 
                entropy_weight=0, 
                example_weights = None, 
                indexes = None):
    """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""
    if beta == 0:
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y)
        inf = torch.Tensor([np.inf])
        zero = torch.Tensor([0.])
        return loss, loss, inf, zero

    # define KL-loss
#     criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()  # moving to eval mode to freeze batchnorm stats
    batch_size = len(x_natural)
    # generate adversarial example
#     x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
#     if adversarial:
#         if distance == 'l_inf':
#             x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()

#             for _ in range(perturb_steps):
#                 x_adv.requires_grad_()
#                 with torch.enable_grad():
#                     loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
#                                            F.softmax(model(x_natural), dim=1))
#                 grad = torch.autograd.grad(loss_kl, [x_adv])[0]
#                 x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#                 x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
#                                   x_natural + epsilon)
#                 x_adv = torch.clamp(x_adv, 0.0, 1.0)
#         else:
#             raise ValueError('No support for distance %s in adversarial '
#                              'training' % distance)
#     else:
#         if distance == 'l_2':
#             x_adv = x_adv + epsilon * torch.randn_like(x_adv)
#         else:
#             raise ValueError('No support for distance %s in stability '
#                              'training' % distance)

    model.train()  # moving to train mode to update batchnorm stats

    # zero gradient
    optimizer.zero_grad()

#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     logits_adv = F.log_softmax(model(x_adv), dim=1)
    logits = model(x_natural)
    if example_weights is None:
      loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
    else:
      assert indexes is not None, "indexes can not be None"
      loss_natural = F.cross_entropy(logits, y, ignore_index=-1, reduction = 'none')
      loss_natural = torch.mean(torch.mul(loss_natural, example_weights[indexes]))
      # if 1 in indexes:
      #       print("printing example weights for batch")
      #       print(example_weights[indexes])
      #       print(indexes)
#     p_natural = F.softmax(logits, dim=1)
#     loss_robust = criterion_kl(
#         logits_adv, p_natural) / batch_size

#     loss = loss_natural + beta * loss_robust
    loss_robust = torch.tensor(0)
    loss = loss_natural
    is_unlabeled = (y == -1)
    if torch.sum(is_unlabeled) > 0:
        logits_unlabeled = logits[is_unlabeled]
        loss_entropy_unlabeled = entropy_loss(logits_unlabeled)
        loss = loss + entropy_weight * loss_entropy_unlabeled
    else:
        loss_entropy_unlabeled = torch.tensor(0)

    return loss, loss_natural, loss_robust, loss_entropy_unlabeled



def noise_loss(model,
               x_natural,
               y,
               epsilon=0.25,
               clamp_x=True):
    """Augmenting the input with random noise as in Cohen et al."""
    # logits_natural = model(x_natural)
    x_noise = x_natural + epsilon * torch.randn_like(x_natural)
    if clamp_x:
        x_noise = x_noise.clamp(0.0, 1.0)
    logits_noise = model(x_noise)
    loss = F.cross_entropy(logits_noise, y, ignore_index=-1)
    return loss


def shift_loss(model,
                detector_model, 
                x_natural,
                y,
                optimizer,
                w_1 = 1.0, w_2 = 1.0,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                adversarial=True,
                distance='inf',
                entropy_weight=0):
    """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""
    if beta == 0:
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y)
        inf = torch.Tensor([np.inf])
        zero = torch.Tensor([0.])
        return loss, loss, inf, zero

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()  # moving to eval mode to freeze batchnorm stats
    detector_model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    if adversarial:
        if distance == 'l_inf':
            x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()

            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_main = F.cross_entropy(F.log_softmax(model(x_adv), dim=1), y)
                    loss_detector = F.cross_entropy(F.log_softmax(detector_model(x_adv), dim=1), y)
                  #   loss_main = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1))
                  #   loss_detector = criterion_kl(F.log_softmax(detector_model(x_adv), dim=1), F.softmax(detector_model(x_natural), dim=1))
                  #   print("main loss %s %0.6f" %(str(loss_main.shape), torch.mean(loss_main)))
                  #   print("detector loss %s %0.6f" %(str(loss_detector.shape), torch.mean(loss_detector)))
                    loss_adv = w_2 * loss_main - w_1 * loss_detector
                grad = torch.autograd.grad(loss_adv, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                                  x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError('No support for distance %s in adversarial '
                             'training' % distance)
    else:
        if distance == 'l_2':
            x_adv = x_adv + epsilon * torch.randn_like(x_adv)
        else:
            raise ValueError('No support for distance %s in stability '
                             'training' % distance)

    model.train()  # moving to train mode to update batchnorm stats

    # zero gradient
    optimizer.zero_grad()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     logits_adv = F.log_softmax(model(x_adv), dim=1)
#     logits = model(x_natural)
    logits = model(x_adv)
    loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
#     p_natural = F.softmax(logits, dim=1)
#     loss_robust = criterion_kl(
#         logits_adv, p_natural) / batch_size
    loss_robust = torch.tensor(0)

#     loss = loss_natural + beta * loss_robust
    loss = loss_natural

    is_unlabeled = (y == -1)
    if torch.sum(is_unlabeled) > 0:
        logits_unlabeled = logits[is_unlabeled]
        loss_entropy_unlabeled = entropy_loss(logits_unlabeled)
      #   loss = loss + entropy_weight * loss_entropy_unlabeled
    else:
        loss_entropy_unlabeled = torch.tensor(0)

    return loss, loss_natural, loss_robust, loss_entropy_unlabeled, loss_main, loss_detector




