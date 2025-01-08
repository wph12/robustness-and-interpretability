import logging
import sys

import torch
import yaml

from src.cifar10.data import load_data
from src.cifar10.test_baseline import test_model
from src.cifar10.train import init_model, train_model
from src.utils import get_label, init_log, make_training_deterministic, get_run_id


if __name__ == "__main__":
    # read hyper-parameters from a config file
    config_file = sys.argv[1]
    label = get_label(config_file)
    args = yaml.safe_load(open(config_file))

    run_id = get_run_id()

    init_log(args, label, config_file, run_id)

    make_training_deterministic(0)  # set seed for reproducibility

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, dataset_sizes, class_names = load_data(args)
    logging.info(f'Len of Class: {len(class_names)}')

    model_conv, criterion, optimizer_conv, exp_lr_scheduler = init_model(
        DEVICE, args, len(class_names))

    model_conv = train_model(
        model_conv, criterion, optimizer_conv, exp_lr_scheduler, label,
        dataloaders, dataset_sizes, DEVICE, args, run_id, num_epochs=args['num_epochs'])
    
    # Test the model and log the accuracy for reporting
    test_model(model_conv, dataloaders['val'], criterion, DEVICE)