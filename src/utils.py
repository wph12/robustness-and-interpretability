import logging
import os
import random
import uuid

import numpy as np
import torch


def make_training_deterministic(seed: int = 0):
    '''Set random seed for reproducibility'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_run_id():
    '''Generate a unique ID for the experiment run'''
    return uuid.uuid4().hex[:8]


def get_label(config_file):
    '''Generate a label for the experiment run'''
    # label based on config file from first folder onwards
    return config_file[config_file.find('/') + 1:]


def init_log(args, label, config_file, run_id):
    '''Initialize logging'''
    # Create directory for logs
    os.makedirs(f'./logs/{label}', exist_ok=True)
    log_file = f"./logs/{label}/{run_id}.log"
    print("log file location: ", log_file)
    
    # Configure logger
    logging.basicConfig(filename=log_file, filemode= 'a', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.addHandler(console)

    # Log the hyperparameters
    logger.info('Using config file: ' + config_file)
    logger.info("Logging Hyperparameters:")
    for key, value in args.items():
        logger.info(f"\t{key}: {value}")
    
    return logger