import logging
import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader 

def load_data(args):
    if args['dataset'] == 'cifar10':
        return load_cifar_data(args)
    elif args['dataset'] == 'cifar100':
        return load_cifar_data(args)
    elif args['dataset'] == 'hymenoptera':
        return load_hymenoptera_data(args)
    else:
        raise Exception("Dataset not supported")


def load_cifar_data(args):    
    train_xform = [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
    ]
    if args['data_xform'] == "augmix":
        train_xform += [T.AugMix()]
    train_xform += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': T.Compose(train_xform),
        'val': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ]),
    }
    print(f'Data transform:\n{data_transforms}')

    # Load CIFAR data
    if args['dataset'] == 'cifar10':
        data_dir = 'data/cifar10'
        full_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=data_transforms['train'])
        val_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=data_transforms['val'])
    elif args['dataset'] == 'cifar100':
        data_dir = 'data/cifar100'
        full_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=data_transforms['train'])
        val_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=data_transforms['val'])
    else:
        raise Exception("Dataset not supported")
    class_names = full_dataset.classes

    dataloaders = {
        'train': DataLoader(
            full_dataset, batch_size=args['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(
            val_dataset, batch_size=args['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True),
    }
    dataset_sizes = {'train': len(full_dataset), 'val': len(val_dataset)}
    return dataloaders, dataset_sizes, class_names


def load_hymenoptera_data(args):
    # Data augmentation and normalization for training
    # Just normalization for validation
    train_xform = [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ]
    if args['data_xform'] == 'augmix':
        train_xform += [T.AugMix()]
    train_xform += [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': T.Compose(train_xform),
        'val': T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print(f'Data transform:\n{data_transforms}')

    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(
        image_datasets[x], batch_size=args['batch_size'], shuffle=True, 
        num_workers=4,  pin_memory=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
