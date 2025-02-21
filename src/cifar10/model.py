# Import
import logging
import numpy as np
import os
import time

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler

from robustbench.model_zoo.architectures.resnet import ResNet18

# local imports
from src.losses import LinfPGDAttack, cross_entropy_with_contrastive, cross_entropy_with_kl, cross_entropy_with_triplet, mixup_data, mixup_criterion, trades_loss


def init_model(DEVICE, args, num_classes):
    # ==> MODEL <==
    
    if args['model'] == 'resnet18':
        model = ResNet18()
        # model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    elif args['model'] == 'resnet50':
        model = torchvision.models.resnet50(weights='IMAGENET1K_V2')

    if args['learning'] == 'tl':
        print('Transfer Learning (only fc layer is trainable)')
        for param in model.parameters():
            param.requires_grad = False
    elif args['learning'] == 'full':
        print('Full Training (all parameters are trainable))')
        for param in model.parameters():
            param.requires_grad = True

    # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    print(str(model)) # print/log model architecture
    # To GPU
    model = model.to(DEVICE)
    # ===

    # LOSS FUNCTION
    if args['loss_function'] == 'cross_entropy':
        loss = nn.CrossEntropyLoss()
    elif args['loss_function'] == 'nll':
        loss = nn.NLLLoss()
    elif args['loss_function'] == 'cross_entropy_with_kl':
        loss = cross_entropy_with_kl
    elif args['loss_function'] == 'cross_entropy_with_contrastive':
        loss = cross_entropy_with_contrastive
    elif args['loss_function'] == 'cross_entropy_with_triplet':
        loss = cross_entropy_with_triplet
    else:
        raise Exception("Loss function not supported")
    # ===

    # OPTIMIZER
    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                                   lr=args["learning_rate"],
                                   momentum=args["momentum"],
                                   weight_decay=0.0002)
    elif args['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                                    lr=args["learning_rate"])
    else:
        raise Exception("Optimizer not supported")
    # ===

    # Decay LR by a factor of 0.1 every 10 epochs
    if args['num_epochs'] > 50:
        sched = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        sched = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return model, loss, optimizer, sched    

