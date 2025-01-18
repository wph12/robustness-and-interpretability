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
from src.losses import LinfPGDAttack, cross_entropy_with_contrastive, cross_entropy_with_kl, cross_entropy_with_triplet, mixup_data, mixup_criterion




def train_model(model, criterion, optimizer, scheduler,
                label, dataloaders, dataset_sizes, device, args, run_id,
                num_epochs=25):
    logger = logging.getLogger(run_id)
    logger.info(str(model)) # print/log model architecture
    logger.info("USING DEVICE: ", device) # print/log model architecture


    since = time.time()

    best_model_params_path = os.path.join(f'logs/{label}/{run_id}_best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    if args['adversarial'] == 'pgd':
        adversary = LinfPGDAttack(model)
    if args['adversarial'] == 'ipgd':
        adversary = LinfPGDAttack(model)

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            benign_loss = 0
            adv_loss = 0
            benign_correct = 0
            adv_correct = 0
            total = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # use adversarial perturbation to obtain output
                    if args['adversarial'] == 'none':
                        outputs = model(inputs)
                        # predication and loss computation
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    elif args['adversarial'] in ['pgd', 'pgdL2']:
                        adv = adversary.perturb(inputs, labels)
                        outputs = model(adv)
                        # predication and loss computation
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    elif args['adversarial'] == 'ipgd':
                        benign_inputs, benign_targets_a, benign_targets_b, benign_lam = mixup_data(
                            inputs, labels)
                        benign_outputs = model(benign_inputs)
                        loss1 = mixup_criterion(
                            criterion, benign_outputs, benign_targets_a, benign_targets_b, benign_lam)
                        benign_loss += loss1.item()

                        _, predicted = benign_outputs.max(1)
                        preds = predicted  # to compute the accuracy during training
                        benign_correct += (benign_lam * predicted.eq(benign_targets_a).sum().float() +
                                          (1 - benign_lam) * predicted.eq(benign_targets_b).sum().float())

                        adv = adversary.perturb(inputs, labels)
                        adv_inputs, adv_targets_a, adv_targets_b, adv_lam = mixup_data(
                            adv, labels)
                        adv_outputs = model(adv_inputs)
                        loss2 = mixup_criterion(
                            criterion, adv_outputs, adv_targets_a, adv_targets_b, adv_lam)
                        adv_loss += loss2.item()

                        _, predicted = adv_outputs.max(1)
                        adv_correct += (adv_lam * predicted.eq(adv_targets_a).sum().float() +
                                       (1 - adv_lam) * predicted.eq(adv_targets_b).sum().float())
                        
                        loss = (loss1 + loss2) / 2
                        

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(
                f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        logger.info('==================')
        time_elapsed = time.time() - since
        logger.info(
            f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, best_model_params_path


# def load_model(model, num_classes, model_path, DEVICE):
#     '''load model from saved model parameters.'''
#     # ==> MODEL <==
#     if model == 'resnet18':
#         model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
#         # model = ResNet18()
#     elif model == 'resnet50':
#         model = torchvision.models.resnet50(weights='IMAGENET1K_V2')

#     for param in model.parameters():
#         param.requires_grad = False

#     # Parameters of newly constructed modules have requires_grad=True by default
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, num_classes)

#     # load pretrained model
#     model.load_state_dict(torch.load(model_path))
#     # To GPU
#     model = model.to(DEVICE)

#     return model


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
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
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

    # Decay LR by a factor of 0.1 every 7 epochs
    if args['num_epochs'] > 50:
        sched = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        sched = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return model, loss, optimizer, sched    

