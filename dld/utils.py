import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

def get_device(gpu):
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        if gpu >= 0:
            # When specifying GPUs
            device_str = "cuda:{}".format(gpu)
        else:
            device_str = "cuda"
    else:
        device_str = "cpu"
    device = torch.device(device_str)

    if use_cuda:
        cudnn.benchmark = True
    
    return device, use_cuda


def fix_state_dict(state_dict):
    """ The state_dict of the model trained with parallel=True has the key name as "module.***"
        Thus, return those without the "module." part"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.' of DataParallel
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def save_result(save_dir, classify_type, results, for_test=False):
    """ Save the results of train/validation in 10 Fold """
    keys = results.keys()

    lines = []
    for key in keys:
        value = results[key]
        line = "{}={}".format(key, value)
        lines.append(line + "\n")

    if for_test:
        file_name = save_dir + "/result_ct{}_test.txt".format(classify_type)
    else:
        file_name = save_dir + "/result_ct{}.txt".format(classify_type)
    
    f = open(file_name, "w")
    f.writelines(lines)
    f.close()


def save_predictions(save_dir, classify_type, fold, labels, preds):
    """ Save predicted values """
    file_name = save_dir + "/preds_ct{}_{}_test.csv".format(classify_type, fold)
    
    lines = []
    lines.append("label,prob\n")
    
    for label,pred in zip(labels, preds):
        line = "{},{:.5f}\n".format(int(label),pred)
        lines.append(line)

    f = open(file_name, "w")
    f.writelines(lines)
    f.close()


def save_pfi_result(save_dir, classify_type, mask_name,
                    base_accuracies, importances):
    """ Save the PFI results"""

    average_importances = np.mean(importances, axis=1)
    flat_importances = np.array(importances).ravel()
    
    base_accuracies_str = ','.join(map(str,base_accuracies))
    flat_importances_str = ','.join(map(str,flat_importances))
    average_importances_str = ','.join(map(str,average_importances))
    
    lines = []
    
    lines.append(f'base_accuracy={base_accuracies_str}\n')
    lines.append(f'importance={flat_importances_str}\n')
    lines.append(f'average_importance={average_importances_str}')
    
    file_name = save_dir + f'/pfi_ct{classify_type}_{mask_name}.txt'
    
    f = open(file_name, "w")
    f.writelines(lines)
    f.close()
    

def get_test_subject_ids(test_subjects):
    """ Make a string of the test_subjects option into an ID array """
    if len(test_subjects) == 0:
        return []
    else:
        return test_subjects.split(',')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fix_run_seed(seed):
    """ Fix random seeds """
    # Fix random seeds in Numpy
    np.random.seed(seed)
    
    # Fix random seeds in Pytorch
    torch.manual_seed(seed)
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        # Even if "benchmark = True" during "get_device()", it is False here
        #cudnn.benchmark = False
