'''Various Loss Functions for training the robust model' '''
import torch
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


# 1. PGD
class LinfPGDAttack(object):
    '''Perturb a batch of images using Linf PGD attack
    https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/blob/master/pgd_adversarial_training.py
    '''
    epsilon = 8./255.
    alpha = epsilon/4

    def __init__(self, model, k):
        self.model = model
        self.k = k

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x

# 2. Interpolated PGD
def mixup_data(x, y, mixup_alpha = 1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Compute the mixup loss given criterion and predicted values'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# def kl_divergence_loss(logits, class_labels, num_classes=100):
#     # Apply a softmax to the logits to obtain the predicted distribution
#     pred_distribution = F.softmax(logits, dim=1)

#     # Convert class labels to one-hot encoding
#     one_hot_targets = F.one_hot(class_labels, num_classes)

#     # Calculate the KL Divergence loss
#     kl_loss = torch.sum(one_hot_targets * (torch.log(one_hot_targets) - torch.log(pred_distribution)))

#     return kl_loss

# Contrastive Loss
def contrastive_loss(logits, margin=1.0):
    positive_pair_loss = torch.norm(logits[0] - logits[1], p=2)
    negative_pair_loss = torch.clamp(margin - torch.norm(logits[0] - logits[2], p=2), min=0)
    return 0.5 * (positive_pair_loss + negative_pair_loss)

#Triplet Loss
def triplet_loss(logits, targets, margin=0.2):
    d_pos = torch.norm(logits[0] - logits[1], p=2)
    d_neg = torch.norm(logits[0] - logits[2], p=2)
    return torch.relu(d_pos - d_neg + margin)


# cross-entropy loss with contrastive loss
def cross_entropy_with_contrastive(output, target, margin=1.0, contrastive_weight=0.01):
    ce_loss = F.cross_entropy(output, target)
    cont_loss = contrastive_weight * contrastive_loss(output, margin=margin)
    total_loss = ce_loss + cont_loss
    return total_loss


def cross_entropy_with_triplet(output, target, margin=0.2, triplet_weight=0.01):
    ce_loss = F.cross_entropy(output, target)
    triplet_loss_val = triplet_weight * triplet_loss(output, target)
    total_loss = ce_loss + triplet_loss_val
    return total_loss



# cross-entropy loss with KL divergence
def cross_entropy_with_kl(output, target, kl_weight=0.01):
    ce_loss = F.cross_entropy(output, target)
    kl_loss = kl_weight * F.kl_div(F.log_softmax(output, dim=1), F.softmax(torch.ones_like(output), dim=1), reduction='batchmean')
    total_loss = ce_loss + kl_loss
    return total_loss



##### TRADES #####
def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss







