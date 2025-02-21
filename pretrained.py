import logging
import sys
import argparse
import os

import torch
import yaml

from src.cifar10.data import load_data
from src.test_baseline import test_model
from src.test_interpretability import integrated_gradient, saliency
from src.test_robust import autoattack_test, autoattack_benchmark
from src.utils import get_label, init_log

from robustbench.utils import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/cifar10/resnet18/_baseline.ym') #just config file for cifar10
    parser.add_argument('--id', type=str, required=True) #robustbench model id
    parser.add_argument('--test-robust', action="store_true")
    parser.add_argument('--test-interpretable', action="store_true")


    parsed_args = parser.parse_args()

    # read hyper-parameters from a config file
    config_file = parsed_args.config
    run_id = parsed_args.id

    args = yaml.safe_load(open(config_file))
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #creates model dataset
    dataloaders, dataset_sizes, class_names = load_data(args)
    print(f'Len of Class: {len(class_names)}')
    
    #intializes model architecture
    model_conv = load_model(model_name=run_id,
                dataset='cifar10',
                threat_model='Linf',
            )
    
    logger = init_log(args, 'pretrained', config_file, run_id)


    #set to eval mode
    model_conv.eval()


    if parsed_args.test_robust:
        print("starting robust test")
        # autoattack_test(model_conv, dataloaders['val'], model_path, args['batch_size'])
        autoattack_benchmark(model_conv, run_id, DEVICE)

    if parsed_args.test_interpretable: 
        print("starting interpretability test")
        saliency(model_conv,dataloaders['val'], run_id)