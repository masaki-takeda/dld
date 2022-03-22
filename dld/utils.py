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
        
