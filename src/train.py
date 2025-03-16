# Import
import logging
import os
import time
import torch


from robustbench.model_zoo.architectures.resnet import ResNet18

# local imports
from src.losses import LinfPGDAttack, cross_entropy_with_contrastive, cross_entropy_with_kl, cross_entropy_with_triplet, mixup_data, mixup_criterion, trades_loss


def train_model(model, criterion, optimizer, scheduler,
                label, dataloaders, dataset_sizes, device, args, run_id, epsilon, dont_train = False):
    logger = logging.getLogger(run_id)
    logger.info(str(model)) # print/log model architecture
    logger.info("USING DEVICE: %s", device)

    num_epochs = args['num_epochs']
    since = time.time()

    best_model_params_path = os.path.join(f'logs/{label}/{run_id}_best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    if(dont_train):
        logger.info("NOT ACTUALLY TRAINING")
        return model, best_model_params_path

    best_acc = 0.0

    if args['adversarial'] == 'pgd':
        adversary = LinfPGDAttack(model, args, epsilon)
    if args['adversarial'] == 'ipgd':
        adversary = LinfPGDAttack(model, args, epsilon)

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
                    
                    elif args['adversarial'] == 'TRADES':
                        outputs = model(inputs)
                        # predication and loss computation
                        _, preds = torch.max(outputs, 1)
                        
                        loss = trades_loss(model=model,
                           x_natural=inputs,
                           y=labels,
                           optimizer=optimizer)
                        
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

            # greedily deep copy the model
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
