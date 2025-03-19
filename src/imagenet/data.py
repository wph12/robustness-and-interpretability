import logging
import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader 
import configparser
import random


def load_imagenet_data(args):
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_dir = config["paths"]["data_dir"]
    data_dir = os.path.expanduser(data_dir)  # expand ~ if used
    print("Data directory", data_dir)

    train_xform = []
    if args['data_xform'] == "augmix":
        train_xform += [T.AugMix()]
        
        
    train_xform += [
        T.RandomHorizontalFlip(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': T.Compose(train_xform),
        'val': T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ]),
    }
    print(f'Data transform:\n{data_transforms}')


    train_dataset = datasets.ImageNet(
        root=data_dir, split= 'train', transform=data_transforms['train'])
    val_dataset = datasets.ImageNet(
        root=data_dir, split='val', transform=data_transforms['val'])
    n_samples = 512
    random.seed(42)
    random_indices = random.sample(range(len(val_dataset)), k=n_samples)
    subset = torch.utils.data.Subset(val_dataset, random_indices)


    dataloaders = {
        'train': DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(
            val_dataset, batch_size=args['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(
            val_dataset, batch_size = args['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True),
        'mini': DataLoader(
            subset, batch_size = args['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(train_dataset),'mini': len(subset)}
    return dataloaders, dataset_sizes, data_transforms