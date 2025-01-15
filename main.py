import logging
import sys
import argparse
import os

import torch
import yaml

from src.cifar10.data import load_data
from src.cifar10.test_baseline import test_model
from src.cifar10.test_robust import autoattack_test, autoattack_benchmark
from src.cifar10.train import init_model, train_model
from src.utils import get_label, init_log, make_training_deterministic, get_run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required = True) #config file location, specifying training 
    parser.add_argument('--state', type=str, required=False) #loads state dictionary. if not provided, will just train new model
    parser.add_argument('--test-standard', action="store_true")
    parser.add_argument('--test-robust', action="store_true")
    parser.add_argument('--test-interpretable', action="store_true")


    parsed_args = parser.parse_args()

    # read hyper-parameters from a config file
    config_file = parsed_args.config
    label = get_label(config_file)
    args = yaml.safe_load(open(config_file))
    
    make_training_deterministic(0)  # set seed for reproducibility

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #creates model dataset
    dataloaders, dataset_sizes, class_names = load_data(args)
    print(f'Len of Class: {len(class_names)}')
    
    #intializes model architecture
    model_conv, criterion, optimizer_conv, exp_lr_scheduler = init_model(
        DEVICE, args, len(class_names))

    model_path = ''
    run_id = ''

    if parsed_args.state:
        model_path = parsed_args.state

        #gets run id and inits log
        run_id = os.path.basename(model_path).split("_")[0]
        print("run id: ", run_id)

        logger = init_log(args, label, config_file, run_id)

        model_conv.load_state_dict(torch.load(model_path))

    else:
        run_id = get_run_id()
        print("run id: ", run_id)

        logger = init_log(args, label, config_file, run_id)

        model_conv, model_path = train_model(
            model_conv, criterion, optimizer_conv, exp_lr_scheduler, label,
            dataloaders, dataset_sizes, DEVICE, args, run_id, num_epochs=args['num_epochs'])
    
    if parsed_args.test_standard:
        print("starting standard test")
        # Test the model and log the accuracy for reporting
        test_model(model_conv, dataloaders['val'], criterion, DEVICE, run_id)

    if parsed_args.test_robust:
        print("starting robust test")
        # autoattack_test(model_conv, dataloaders['val'], model_path, args['batch_size'])
        autoattack_benchmark(model_conv, run_id)

    if parsed_args.test_interpretable: 
        pass