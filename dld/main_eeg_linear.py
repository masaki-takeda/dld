import os
import math
from pathlib import Path
import argparse
from collections import OrderedDict
from distutils.util import strtobool

import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import BrainDataset, FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, CLASSIFY_ALL
from dataset import DATA_TYPE_TRAIN, DATA_TYPE_VALIDATION, DATA_TYPE_TEST
from utils import save_result
import options


def get_windowed_data(x,
                      window_size,
                      stride_size):
    sample_size = x.shape[0]
    ch_size     = x.shape[1]
    frame_size  = x.shape[2]
    
    step_size = int(math.ceil( (frame_size - (window_size - 1)) / stride_size ))
    ret = np.empty([step_size, sample_size, ch_size], dtype=x.dtype)
    
    for i in range(step_size):
        start_idndex = i * stride_size
        indices = [start_idndex + j for j in range(window_size)]
        step_x = x[:,:,indices]
        #(14129, 63, window_size)
        mean_step_x = np.mean(step_x, axis=2)
        #(14129, 63)
        ret[i] = mean_step_x
        
    ret = ret.transpose(1,2,0)
        
    return ret


def get_data(data_dir,
             data_type,
             classify_type,
             fold,
             window_size,
             stride_size):
    
    test_subject_ids = [
        'TM_200716_01',
        'TM_200720_01',
        'TM_200721_01',
        'TM_200722_01',
        'TM_200727_01'
    ]
    
    subjects_per_fold = 5
    dataset = BrainDataset(data_type=data_type,
                  classify_type=classify_type,
                  data_seed=0,
                  use_fmri=False,
                  use_eeg=True,
                  data_dir=data_dir,
                  eeg_normalize_type='pre',
                  eeg_frame_type='normal',
                  average_trial_size=9,
                  average_repeat_size=9,
                  fold=fold,
                  test_subjects=test_subject_ids,
                  subjects_per_fold=subjects_per_fold)
    
    x = np.array([sample['eeg_data'] for sample in dataset])    
    # (14129, 63, 250)
    y = np.array([int(sample['label'][0]) for sample in dataset])
    # (14129,)
    
    x = get_windowed_data(x, window_size=window_size, stride_size=stride_size)
    
    return x, y


def train_fold(args, classify_type, fold):
    train_x, train_y = get_data(args.data_dir, DATA_TYPE_TRAIN, classify_type, fold,
                                window_size=args.window,
                                stride_size=args.stride)
    valid_x, valid_y = get_data(args.data_dir, DATA_TYPE_VALIDATION, classify_type, fold,
                                window_size=args.window, 
                                stride_size=args.stride)
    
    if args.debug:
        train_x = train_x[:1000]
        train_y = train_y[:1000]
    
    train_accuracies = []
    valid_accuracies = []

    models = []
    
    step_size = train_x.shape[2]

    for i in tqdm(range(step_size)):
        model = svm.LinearSVC(loss='hinge', 
                       C=args.c,
                       class_weight='balanced', 
                       random_state=0,
                       max_iter=args.max_iter)
        
        train_x_i = train_x[:,:,i]
        valid_x_i = valid_x[:,:,i]

        model.fit(train_x_i, train_y)
        models.append(model)

        train_pred = model.predict(train_x_i) # (14129,)
        valid_pred = model.predict(valid_x_i) # (1668,)

        train_accuracy = accuracy_score(train_y, train_pred)
        valid_accuracy = accuracy_score(valid_y, valid_pred)

        train_accuracies.append(train_accuracy)    
        valid_accuracies.append(valid_accuracy)
        
    model_path = Path(args.save_dir) / f'model_ct{classify_type}_{fold}'
    np.save(model_path, models)
    
    return train_accuracies, valid_accuracies


def train_all_folds(args, classify_type):
    all_train_accuracies = []
    all_valid_accuracies = []

    for fold in range(args.fold_size):
        out = train_fold(args, classify_type, fold)
        train_accuracies, valid_accuracies = out

        all_train_accuracies.append(train_accuracies)
        all_valid_accuracies.append(valid_accuracies)

    columns = [f'fold{i}' for i in range(args.fold_size)]

    train_df = pd.DataFrame(np.array(all_train_accuracies).T, columns=columns)
    valid_df = pd.DataFrame(np.array(all_valid_accuracies).T, columns=columns)
    
    max_validation_accurcy_mean = np.mean(np.max(all_valid_accuracies, axis=1))
    max_validation_accurcy_std  = np.std(np.max(all_valid_accuracies, axis=1), ddof=1)

    results = OrderedDict([
        ('max_validation_accurcy_mean', max_validation_accurcy_mean),
        ('max_validation_accurcy_std', max_validation_accurcy_std),
    ])
    
    save_result(args.save_dir, classify_type, results)

    train_df.to_csv(Path(args.save_dir) / f'result_ct{classify_type}_train.csv')
    valid_df.to_csv(Path(args.save_dir) / f'result_ct{classify_type}_validation.csv')


def test_fold(args, classify_type, fold):
    test_x, test_y = get_data(args.data_dir, DATA_TYPE_TEST, classify_type, fold,
                              window_size=args.window, 
                              stride_size=args.stride)                              
    
    model_path = Path(args.save_dir) / f'model_ct{classify_type}_{fold}.npy'
    models = np.load(model_path, allow_pickle=True)
    
    test_accuracies = []
    
    test_preds = []
    
    step_size = test_x.shape[2]

    for i in tqdm(range(step_size)):
        model = models[i]
        test_pred = model.predict(test_x[:,:,i])
        # (1468)
        test_preds.append(test_pred)
        test_accuracy = accuracy_score(test_y, test_pred)
        test_accuracies.append(test_accuracy)
        
    test_y = np.array(test_y)
    test_preds = np.array(test_preds)
    
    test_preds = test_preds.transpose([1,0])
        
    return test_accuracies, test_y, test_preds


def test_all_folds(args, classify_type):
    all_test_accuracies = []
    all_test_preds = []

    for fold in range(args.fold_size):
        out = test_fold(args, classify_type, fold)
        test_accuracies, test_y, test_preds = out

        all_test_accuracies.append(test_accuracies)
        all_test_preds.append(test_preds)

    all_test_preds = np.array(all_test_preds)
    ensemble_test_preds = (all_test_preds.mean(axis=0) > 1 / len(all_test_preds)).astype(np.float32)

    step_size = ensemble_test_preds.shape[1]
    ensembletest_accuracies = []

    for i in tqdm(range(step_size)):
        ensemble_test_pred = ensemble_test_preds[:,i]                
        test_accuracy = accuracy_score(test_y, ensemble_test_pred) 
        ensembletest_accuracies.append(test_accuracy)

    ensembletest_accuracies = np.array(ensembletest_accuracies)

    all_test_accuracies.append(ensembletest_accuracies)

    columns = [f'fold{i}' for i in range(args.fold_size)]
    columns.append('ensemble')

    ret_df = pd.DataFrame(np.array(all_test_accuracies).T, columns=columns)
    
    ret_df.to_csv(Path(args.save_dir) / f'result_ct{classify_type}_test.csv')


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_dir', type=str,
                        default='saved_linear')
    parser.add_argument('--classify_type', type=int,
                        default=-1) # FACE_PLACE=0, MALE_FEMALE=1, ARTIFICAL_NATURL=2, ALL=-1
    parser.add_argument('--c', type=float,
                        default=1.0)
    parser.add_argument('--max_iter', type=int,
                        default=100000)
    parser.add_argument('--window', type=int,
                        default=1)
    parser.add_argument('--stride', type=int,
                        default=1)    
    parser.add_argument('--fold_size', type=int,
                        default=9)
    parser.add_argument('--data_dir', type=str,
                        default='/data2/DLD/Data_Converted_EEG_ICAed/')
    parser.add_argument('--test', type=strtobool,
                        default='false')
    parser.add_argument('--debug', type=strtobool,
                        default='false')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    options.save_args(args)

    if args.test == False:
        # Train
        if args.classify_type == CLASSIFY_ALL:
            train_all_folds(args, classify_type=FACE_OBJECT)
            train_all_folds(args, classify_type=MALE_FEMALE)
            train_all_folds(args, classify_type=ARTIFICIAL_NATURAL)
        else:
            train_all_folds(args, classify_type=args.classify_type)
    else:
        # Test
        if args.classify_type == CLASSIFY_ALL:
            test_all_folds(args, classify_type=FACE_OBJECT)
            test_all_folds(args, classify_type=MALE_FEMALE)
            test_all_folds(args, classify_type=ARTIFICIAL_NATURAL)
        else:
            test_all_folds(args, classify_type=args.classify_type)

    

if __name__ == '__main__':
    main()
