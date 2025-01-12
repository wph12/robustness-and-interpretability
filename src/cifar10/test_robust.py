# uses auto-attack: https://github.com/fra31/auto-attack

import torch
import os
from autoattack import AutoAttack


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