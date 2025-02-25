import logging
import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader 
import configparser



def load_cifar10_data(args):
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_dir = config["paths"]["data_dir"]
    data_dir = os.path.expanduser(data_dir)  # expand ~ if used
    print("Data directory", data_dir)

    train_xform = [
        # T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
    ]
    if args['data_xform'] == "augmix":
        train_xform += [T.AugMix()]
    train_xform += [
        T.ToTensor(),
        # T.Normalize(mean=[0.4914, 0.4822, 0.4465],
        #             std=[0.247, 0.243, 0.261])
    ]

    data_transforms = {
        'train': T.Compose(train_xform),
        'val': T.Compose([
            # T.Resize((224, 224)),
            T.ToTensor(),
            # T.Normalize(mean=[0.4914, 0.4822, 0.4465],
            #             std=[0.247, 0.243, 0.261])
        ]),
    }
    print(f'Data transform:\n{data_transforms}')

    # Load CIFAR data
    if args['dataset'] == 'cifar10':
        full_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=data_transforms['train'])
        val_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=data_transforms['val'])
    # elif args['dataset'] == 'cifar100':
    #     full_dataset = datasets.CIFAR100(
    #         root=data_dir, train=True, download=True, transform=data_transforms['train'])
    #     val_dataset = datasets.CIFAR100(
    #         root=data_dir, train=False, download=True, transform=data_transforms['val'])
    else:
        raise Exception("Dataset not supported")

    dataloaders = {
        'train': DataLoader(
            full_dataset, batch_size=args['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(
            val_dataset, batch_size=args['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(
            val_dataset, batch_size = args['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True),
    }
    dataset_sizes = {'train': len(full_dataset), 'val': len(val_dataset), 'test': len(full_dataset)}
    return dataloaders, dataset_sizes, data_transforms
