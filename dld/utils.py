import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc

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


def save_raw_result(save_dir, classify_type, metrics_df, data_type):
    """ Save the results of train/validation/test in full folds """
    assert data_type == "train" or \
        data_type == "validation" or \
        data_type == "test"

    file_name = save_dir + "/result_ct{}_raw_{}.csv".format(classify_type, data_type)
    metrics_df.to_csv(file_name, index=True)
    

def save_aggregated_result(save_dir, classify_type, aggregated_df, for_test):
    """ Save the aggreaged results of train/validation/test """
    
    if for_test:
        file_name = save_dir + "/result_ct{}_test.csv".format(classify_type)
    else:
        file_name = save_dir + "/result_ct{}.csv".format(classify_type)

    aggregated_df.to_csv(file_name, index=True)


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
                    base_scores, importances):
    """ Save the PFI results"""
    
    average_importances = np.mean(importances, axis=1)
    flat_importances = np.array(importances).ravel()
    
    base_scores_str = ','.join(map(str,base_scores))
    flat_importances_str = ','.join(map(str,flat_importances))
    average_importances_str = ','.join(map(str,average_importances))
    
    lines = []
    
    lines.append(f'base_score={base_scores_str}\n')
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
        

def calc_metrics(labels, preds):
    """ Collect evaluation metrics """
    roc_auc = roc_auc_score(labels, preds)
    
    precisions, recalls, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recalls, precisions)
    
    binarised_preds = (np.array(preds) > 0.5)
    
    # Accuracy is multiplied by 100 for percentage notation
    accuracy = accuracy_score(labels, binarised_preds) * 100.0

    f1 = f1_score(labels, binarised_preds)    

    # Precision
    # (true positives) / (predicted as positive)
    precision = precision_score(labels, binarised_preds)

    # Recall = TPR
    # true positives / (labeled as positive) 
    recall = recall_score(labels, binarised_preds)

    inv_labels = 1 - np.array(labels)
    inv_binarised_preds = 1 - binarised_preds

    # Precision for negative (NPV)
    # (true negative) / (predicted as negative)
    n_precisoin = precision_score(inv_labels, inv_binarised_preds)
    
    # Recall for negative (Specificity)
    # (true negatives) / (labeled as negative)
    n_recall= recall_score(inv_labels, inv_binarised_preds)
    
    # F1 for negative
    n_f1 = f1_score(inv_labels, inv_binarised_preds)

    # PR_AUC for negative
    n_precisions, n_recalls, _ = precision_recall_curve(inv_labels, 1.0 - np.array(preds))
    n_pr_auc = auc(n_recalls, n_precisions)

    metrics = {
        'accuracy'    : accuracy,
        'roc_auc'     : roc_auc,
        'pr_auc'      : pr_auc,
        'f1'          : f1,
        'precision'   : precision,
        'recall'      : recall,
        'n_precisoin' : n_precisoin,
        'n_recall'    : n_recall,
        'n_f1'        : n_f1,
        'n_pr_auc'    : n_pr_auc,
    }
    
    return metrics
