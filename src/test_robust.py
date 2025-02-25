# uses auto-attack: https://github.com/fra31/auto-attack

import configparser
import torch
import os
import logging
from autoattack import AutoAttack
from robustbench.eval import benchmark


def autoattack_test(model, test_loader, model_path, batch_size, norm= 'Linf', epsilon= 8./255.):
    log_folder_path = os.path.dirname(model_path)
    run_id = os.path.basename(model_path).split("_")[0]
    log_path = os.path.join(log_folder_path, '{}_autoattack_log.txt'.format(run_id))
    
    adversary = AutoAttack(model, norm=norm, eps=epsilon, log_path=log_path, version = 'standard')
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    

    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test,
            bs=batch_size, state_path=None)
        
        torch.save({'adv_complete': adv_complete}, '{}/{}_{}_{}_1_{}_eps_{:.5f}.pth'.format(
            log_folder_path, run_id, 'aa', 'standard', adv_complete.shape[0], epsilon))
        

def autoattack_benchmark(model, run_id, device, dataset, preprocessing, eps):
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_dir = config["paths"]["data_dir"]
    data_dir = os.path.expanduser(data_dir)  # expand ~ if used
    print("data directory: ", data_dir)

    logger = logging.getLogger(run_id)
    logger.info('==================<Autoattack Benchmark>==================')
    clean_acc, robust_acc = benchmark(model,
                                    dataset= dataset,
                                    threat_model='Linf',
                                    device = device,
                                    eps = eps,
                                    data_dir = "./data",
                                    preprocessing = preprocessing)
    logger.info(f'Clean Accuracy: {clean_acc:.4f} Robust Accuracy: {robust_acc:.4f}')
    


