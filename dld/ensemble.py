import argparse
import pandas as pd
import numpy as np
import os

def calc_ensemble_accuracies(save_dir, classify_type):
    if not os.path.exists(save_dir + f'/preds_ct{classify_type}_0_test.csv'):
        return None
    
    all_probs = []
    for fold in range(9):
        df = pd.read_csv(save_dir + f'/preds_ct{classify_type}_{fold}_test.csv')
        labels = df.label.values
        probs = df.prob.values
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    mean_prob = np.mean(all_probs, axis=0)
    mean_prob_binary = (mean_prob > 0.5).astype(np.int32)
    
    mean_ensemble_accuracy = np.sum(mean_prob_binary == labels) / len(labels)
    
    vote_binary = (np.mean((all_probs > 0.5).astype(np.int32), axis=0) > 0.5).astype(np.int32)
    vote_ensemble_accuracy = np.sum(vote_binary == labels) / len(labels)
    
    all_probs_binay = (all_probs > 0.5).astype(np.int32)   
    normal_accuracies = (all_probs_binay == labels).astype(np.int32).mean(axis=1)
    
    normal_accuracy_mean = np.mean(normal_accuracies)
    normal_accuracy_std = np.std(normal_accuracies, ddof=1)
    
    return mean_ensemble_accuracy, vote_ensemble_accuracy, normal_accuracy_mean, normal_accuracy_std


def process_ensemble(save_dir):
    lines = []
    
    for classify_type in range(3):
        out = calc_ensemble_accuracies(save_dir, classify_type)
        if out is not None:
            mean_ensemble_accuracy, vote_ensemble_accuracy, normal_accuracy_mean, normal_accuracy_std = out
            lines.append(f'ct{classify_type}_mean_ensemble_accuracy={mean_ensemble_accuracy:.5f}\n')
            lines.append(f'ct{classify_type}_vote_ensemble_accuracy={vote_ensemble_accuracy:.5f}\n')
            lines.append(f'ct{classify_type}_normal_accuracy={normal_accuracy_mean:.5f} +- {normal_accuracy_std:.5f}\n')

    file_path = save_dir + '/ensemble_test.txt'

    f = open(file_path, "w")
    f.writelines(lines)
    f.close()    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="saved")
    args = parser.parse_args()
    
    process_ensemble(args.save_dir)
