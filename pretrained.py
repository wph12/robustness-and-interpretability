import logging
import sys
import argparse
import os

import torch
import yaml

from src.cifar10.data import load_cifar10_data 
from src.imagenet.data import load_imagenet_data
from src.test_interpretability import interpretability_metrics
from src.test_robust import autoattack_benchmark
from src.utils import init_log
import torchvision
from src.test_baseline import test_model
from robustbench.utils import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required = True) #config file
    parser.add_argument('--id', type=str, required=True) #robustbench model id
    parser.add_argument('--test-standard', action="store_true")
    parser.add_argument('--test-robust', action="store_true")
    parser.add_argument('--test-interpretable', action="store_true")


    parsed_args = parser.parse_args()

    # read hyper-parameters from a config file
    config_file = parsed_args.config
    model_id = parsed_args.id

    args = yaml.safe_load(open(config_file))
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #creates dataset and specifies adversarial epsilon ball
    eps = 8./255.
    if(args['dataset'] == 'cifar10'):
        dataloaders, dataset_sizes, data_transforms = load_cifar10_data(args)
    elif(args['dataset'] == 'imagenet'):
        dataloaders, dataset_sizes, data_transforms = load_imagenet_data(args, normalize= False)
        eps = 4./255.
    else:
        raise Exception("Dataset not supported")
    print("CURRENT EPS: ", eps)

    
    #intializes model architecture
    model_conv = load_model(model_name=model_id,
                dataset=args['dataset'],
                threat_model='Linf',
            )
    
    logger = init_log(args, 'pretrained', config_file, model_id)



    #set to eval mode
    model_conv.to(DEVICE)
    model_conv.eval()


    if parsed_args.test_standard:
        print("starting standard test")
        # Test the model and log the accuracy for reporting
        test_model(model_conv, dataloaders['test'], torch.nn.CrossEntropyLoss(), DEVICE, "pretrained")

    if parsed_args.test_robust:
        print("starting robust test, with epsilon being", eps)
        
        if(args['dataset'] == 'cifar10'):
            autoattack_benchmark(model_conv, model_id, DEVICE, 
                             dataset=args['dataset'], 
                             preprocessing=data_transforms['val'], 
                             eps=eps)
        elif(args['dataset'] == 'imagenet'):
            autoattack_benchmark(model_conv, model_id, DEVICE, 
                             dataset=args['dataset'], 
                             preprocessing=data_transforms['val'], 
                             eps=eps,
                             num_examples=4096)
        
    if parsed_args.test_interpretable: 
        print("starting interpretability test")
        #imagenet pretrained proxy
        imagenet_proxy = load_model(model_name='Standard_R50',
                dataset='imagenet',
                threat_model='Linf')
        imagenet_proxy.eval()
        
        # cifar pretrained proxy
        cifar_proxy = load_model(model_name='Standard',
                   dataset='cifar10',
                   threat_model='Linf')
        cifar_proxy.eval()

        if(args['dataset'] == 'imagenet'):
            interpretability_metrics(model_conv, imagenet_proxy, dataloaders['mini'], 'pretrained', xai_method ='occlusion',
                                use_ground_truth= True,
                                use_alignment= False,
                                use_infidelity=False, 
                                use_max_sensitivity=True, 
                                use_sparseness = True, 
                                use_road= True)
        else: 
            interpretability_metrics(model_conv, cifar_proxy, dataloaders['mini'], 'pretrained', xai_method ='occlusion',
                                    use_ground_truth= True,
                                    use_alignment= False,
                                    use_infidelity=False, 
                                    use_max_sensitivity=False, 
                                    use_sparseness = False, 
                                    use_road= True)
        

        