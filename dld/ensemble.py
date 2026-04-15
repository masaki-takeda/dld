import argparse
import pandas as pd
import numpy as np
import os

from dataset import CLASSIFY_TYPE_MAX

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc


def calc_non_binarized_metrics(probs, labels):
    roc_auc = roc_auc_score(labels, probs)
    
    precisions, recalls, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recalls, precisions)

    inv_labels = 1 - labels
    n_precisions, n_recalls, _ = precision_recall_curve(inv_labels, 1.0 - probs)
    n_pr_auc = auc(n_recalls, n_precisions)
    
    return roc_auc, pr_auc, n_pr_auc


def calc_binarized_metrics(binarised_probs, labels):
    # Accuracy is multiplied by 100 for percentage notation
    accuracy = accuracy_score(labels, binarised_probs) * 100.0
    
    f1 = f1_score(labels, binarised_probs)
    
    # Precision
    # (true positives) / (predicted as positive)
    precision = precision_score(labels, binarised_probs)
    
    # Recall = TPR
    # true positives / (labeled as positive) 
    recall = recall_score(labels, binarised_probs)
    
    inv_labels = 1 - labels
    inv_binarised_probs = 1 - binarised_probs
    
    # Precision for negative (NPV)
    # (true negative) / (predicted as negative)
    n_precisoin = precision_score(inv_labels, inv_binarised_probs)
    
    # Recall for negative (Specificity)
    # (true negatives) / (labeled as negative)
    n_recall= recall_score(inv_labels, inv_binarised_probs)
    
    # F1 for negative
    n_f1 = f1_score(inv_labels, inv_binarised_probs)
    
    return accuracy, f1, precision, recall, n_precisoin, n_recall, n_f1


def calc_metrics(all_probs, labels, classify_type):
    # all_probs = (9, 728), etc.

    # mean ensemble
    mean_probs = np.mean(all_probs, axis=0)

    # non binalized metrics
    mean_roc_auc, mean_pr_auc, mean_n_pr_auc = calc_non_binarized_metrics(mean_probs, labels)

    # binalized metrics
    mean_binary = (mean_probs > 0.5).astype(np.int32)
    
    out = calc_binarized_metrics(mean_binary, labels)
    mean_accuracy, mean_f1, mean_precision, mean_recall, \
        mean_n_precisoin, mean_n_recall, mean_n_f1 = out
    
    # vote ensemble
    fold_size = all_probs.shape[0]
    if fold_size % 2 == 0:
        print(f'[WARNING] Voting with even fold size: {fold_size}')
    
    vote_binary = (np.mean((all_probs > 0.5).astype(np.int32), axis=0) > 0.5).astype(np.int32)
    
    out = calc_binarized_metrics(vote_binary, labels)
    vote_accuracy, vote_f1, vote_precision, vote_recall, \
        vote_n_precisoin, vote_n_recall, vote_n_f1 = out
    
    metrics = ['accuracy','roc_auc','pr_auc','f1','precision','recall',
               'n_precisoin','n_recall','n_f1','n_pr_auc']
    
    d = {
        'accuracy'    : [mean_accuracy,    vote_accuracy],
        'roc_auc'     : [mean_roc_auc,     np.nan],
        'pr_auc'      : [mean_pr_auc,      np.nan],
        'f1'          : [mean_f1,          vote_f1],
        'precision'   : [mean_precision,   vote_precision],
        'recall'      : [mean_recall,      vote_recall],
        'n_precisoin' : [mean_n_precisoin, vote_n_precisoin],
        'n_recall'    : [mean_n_recall,    vote_n_recall],
        'n_f1'        : [mean_n_f1,        vote_n_f1],
        'n_pr_auc'    : [mean_n_pr_auc,    np.nan],
    }
    return pd.DataFrame(d, index=[f'mean_ensemble_ct{classify_type}',
                                  f'vote_ensemble_ct{classify_type}'])
    

def calc_ensemble_accuracies(save_dir, classify_type):
    if not os.path.exists(save_dir + f'/preds_ct{classify_type}_0_test.csv'):
        return None
    
    all_probs = []
    for fold in range(9):
        path = save_dir + f'/preds_ct{classify_type}_{fold}_test.csv'
        if not os.path.exists(path):
            break
        df = pd.read_csv(path)
        labels = df.label.values
        probs = df.prob.values
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)

    ensemble_metrics_df = calc_metrics(all_probs, labels, classify_type)
    
    normal_test_result_path = save_dir + f'/result_ct{classify_type}_test.csv'
    normal_metrics_df = pd.read_csv(normal_test_result_path, index_col=0)

    normal_metrics_df.index = [f'non_ensemble_ct{classify_type}_mean',
                               f'non_ensemble_ct{classify_type}_std']

    return pd.concat([ensemble_metrics_df, normal_metrics_df])

    
def process_ensemble(save_dir):
    dfs = []
    
    for classify_type in range(CLASSIFY_TYPE_MAX):
        df = calc_ensemble_accuracies(save_dir, classify_type)
        if df is not None:
            dfs.append(df)

    file_path = save_dir + '/ensemble_test.csv'

    print(f'Output result csv to: {file_path}')

    df = pd.concat(dfs)
    df.to_csv(file_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="saved")
    args = parser.parse_args()
    
    process_ensemble(args.save_dir)
