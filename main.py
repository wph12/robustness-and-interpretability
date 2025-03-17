import logging
import sys
import argparse
import os

import torch
import yaml
import torchvision
from robustbench.utils import load_model

from src.cifar10.data import load_cifar10_data
from src.imagenet.data import load_imagenet_data
from src.model import init_model

from src.test_baseline import test_model
from src.test_interpretability import interpretability_metrics
from src.test_robust import autoattack_test, autoattack_benchmark
from src.train import train_model
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

    #creates dataset and specifies adversarial epsilon ball
    eps = 8./255.
    if(args['dataset'] == 'cifar10'):
        dataloaders, dataset_sizes, data_transforms = load_cifar10_data(args)
    elif(args['dataset'] == 'imagenet'):
        dataloaders, dataset_sizes, data_transforms = load_imagenet_data(args)
        eps = 4./255.
    else:
        raise Exception("Dataset not supported")
    print("CURRENT EPS: ", eps)

    #intializes model architecture
    model_conv, criterion, optimizer_conv, lr_scheduler = init_model(
        DEVICE, args)

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

        no_training_needed = False
        if(args['dataset'] == 'imagenet' and args['adversarial'] == 'none'):
            no_training_needed = True

        model_conv, model_path = train_model(
            model_conv, criterion, optimizer_conv, lr_scheduler, label,
            dataloaders, dataset_sizes, DEVICE, args, run_id, epsilon = eps, dont_train=no_training_needed)

    #set to eval mode
    model_conv.eval()

    if parsed_args.test_standard:
        print("starting standard test")
        # Test the model and log the accuracy for reporting
        test_model(model_conv, dataloaders['test'], criterion, DEVICE, run_id)

    if parsed_args.test_robust:
        print("starting robust test, with epsilon being", eps)
        
        if(args['dataset'] == 'cifar10'):
            autoattack_benchmark(model_conv, run_id, DEVICE, 
                             dataset=args['dataset'], 
                             preprocessing=data_transforms['val'], 
                             eps=eps)
        elif(args['dataset'] == 'imagenet'):
            autoattack_benchmark(model_conv, run_id, DEVICE, 
                             dataset=args['dataset'], 
                             preprocessing=data_transforms['val'], 
                             eps=eps,
                             num_examples=4096)
        
    if parsed_args.test_interpretable: 
        print("starting interpretability test")
        #imagenet pretrained proxy
        imagenet_proxy = torchvision.models.resnet34()
        imagenet_proxy.eval()
        
        # cifar pretrained proxy
        cifar_proxy = load_model(model_name='Standard',
                   dataset='cifar10',
                   threat_model='Linf')
        cifar_proxy.eval()

        if(args['dataset'] == 'imagenet'):
                interpretability_metrics(model_conv, imagenet_proxy, dataloaders['mini'], run_id, xai_method ='sal',
                                    use_ground_truth= True,
                                    use_alignment= False,
                                    use_infidelity=False, 
                                    use_max_sensitivity=False, 
                                    use_sparseness = False, 
                                    use_road= True)
        else: 
            interpretability_metrics(model_conv, cifar_proxy, dataloaders['test'], run_id, xai_method ='sal',
                                    use_ground_truth= True,
                                    use_alignment= True,
                                    use_infidelity=False, 
                                    use_max_sensitivity=True, 
                                    use_sparseness = True, 
                                    use_road= False)
        