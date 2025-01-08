'''Various Loss Functions for training the robust model' '''
import torch
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



# 1. PGD
class LinfPGDAttack(object):
    '''Perturb a batch of images using Linf PGD attack
    https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/blob/master/pgd_adversarial_training.py
    '''
    epsilon = 0.001
    k = 2
    alpha = 0.001

    def __init__(self, model):
        self.model = model

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







