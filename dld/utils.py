import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

def get_device(gpu):
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        if gpu >= 0:
            # GPUを指定している場合
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
    """ parallel=Trueで学習されたモデルのstate_dictはkey名画module.***といった名前になっているので
    module.の部分をカットした物を返す. """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.' of DataParallel
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def save_result(save_dir, classify_type, results, for_test=False):
    """ 10 Foldの学習/評価結果を保存する """
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
    """ optionのtest_subjects文字列から、実際のtest被験者ID配列にする """
    if len(test_subjects) == 0:
        return []
    else:
        return test_subjects.split(',')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fix_run_seed(seed):
    """ 乱数を固定 """
    # Numpyの乱数固定
    np.random.seed(seed)
    
    # Pytorchの乱数固定
    torch.manual_seed(seed)
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        # 先にget_device()時にbenchmark = Trueにしていても、ここでFalseにしている
        #cudnn.benchmark = False
        
