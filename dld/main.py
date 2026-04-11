import sys
import argparse
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from dataset import get_dataset, FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, FRONT_SIDE, SMALL_LARGE, CLASSIFY_ALL
from dataset import DATA_TYPE_TRAIN, DATA_TYPE_VALIDATION, DATA_TYPE_TEST
from dataset import COMBINE_TYPE_EEG, COMBINE_TYPE_FMRI, COMBINE_TYPE_COMBINED
from model import get_eeg_model, get_fmri_model, get_combined_model
from logger import Logger
import options
from early_stopping import EarlyStopping
from utils import get_device, fix_state_dict, save_raw_result, save_aggregated_result, save_predictions, fix_run_seed, calc_metrics, save_pfi_result



def merge_state_dict(part_state_dict, combined_state_dict):
    for name, param in part_state_dict.items():
        if name not in combined_state_dict:
            continue
        combined_state_dict[name].copy_(param)


def load_pretrained_models(model,
                           device,
                           classify_type,
                           fold,
                           fix_weights,
                           preload_fmri_dir,
                           preload_eeg_dir):
    fmri_model_path = "{}/model_ct{}_{}.pt".format(preload_fmri_dir, classify_type, fold)
    eeg_model_path  = "{}/model_ct{}_{}.pt".format(preload_eeg_dir, classify_type, fold)
    combined_state = model.state_dict()

    fmri_state = torch.load(fmri_model_path, map_location=device)
    eeg_state  = torch.load(eeg_model_path,  map_location=device)

    # Remove "module." from the key name in state_dict when training with data_parallel
    fmri_state = fix_state_dict(fmri_state)
    eeg_state = fix_state_dict(fmri_state)
    
    merge_state_dict(fmri_state, combined_state)
    merge_state_dict(eeg_state,  combined_state)

    model.load_state_dict(combined_state)

    if fix_weights:
        model.fix_preloads()


def get_model(combine_type, args, fold, device):
    if combine_type == COMBINE_TYPE_EEG:
        model = get_eeg_model(args.model_type,
                              args.parallel,
                              args.kernel_size,
                              args.level_size,
                              args.level_hidden_size,
                              args.residual,
                              args.eeg_duration_type,
                              device)
    else:
        if args.fmri_frame_type == 'normal' or args.fmri_frame_type == 'average':
            fmri_ch_size = 1
        else:
            fmri_ch_size = 3

        if combine_type == COMBINE_TYPE_FMRI:
            model = get_fmri_model(fmri_ch_size,
                                   args.parallel,
                                   device)
        else:
            model = get_combined_model(args.model_type,
                                       fmri_ch_size,
                                       args.kernel_size,
                                       args.level_size,
                                       args.level_hidden_size,
                                       args.residual,
                                       args.combined_hidden_size,
                                       args.combined_layer_size,
                                       args.parallel,
                                       device)
            if not args.test:
                # Loading of pretrained model
                if args.preload_fmri_dir is not None or args.preload_eeg_dir is not None:
                    # If either preload_fmri_dir or preload_eeg_dir is None, prelaod will not be done
                    load_pretrained_models(model,
                                           device,
                                           args.classify_type,
                                           fold,
                                           args.fix_preloads,
                                           args.preload_fmri_dir,
                                           args.preload_eeg_dir)
    return model


class CombinedOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        
    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


def get_optimizer(combine_type, args, model):
    lr = args.lr
    weight_decay = args.weight_decay
    
    if combine_type == COMBINE_TYPE_COMBINED:
        lr_eeg = args.lr_eeg
        lr_fmri = args.lr_fmri
        if lr_eeg is None:
            # Use lr if not specified
            lr_eeg = lr
        if lr_fmri is None:
            # Use lr if not specified
            lr_fmri = lr
        
        weight_decay_eeg = args.weight_decay_eeg
        weight_decay_fmri = args.weight_decay_fmri
        if weight_decay_eeg is None:
            # Use weight_decay if not specified
            weight_decay_eeg = weight_decay
        if weight_decay_fmri is None:
            # Use weight_decay if not specified
            weight_decay_fmri = weight_decay
        
        optimizer_fc = optim.Adam(model.parameters_fc(),
                                  lr=lr,
                                  weight_decay=weight_decay)
        optimizer_eeg = optim.Adam(model.parameters_eeg(),
                                   lr=lr_eeg,
                                   weight_decay=weight_decay_eeg)
        optimizer_fmri = optim.Adam(model.parameters_fmri(),
                                    lr=lr_fmri,
                                    weight_decay=weight_decay_fmri)
        optimizer = CombinedOptimizer([optimizer_fc, optimizer_eeg, optimizer_fmri])
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    return optimizer


def train_epoch(combine_type, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    
    threshold = torch.Tensor([0.5]).to(device)

    recorded_labels = []
    recorded_preds = []
    
    count = epoch * len(train_loader)
    
    pbar = tqdm(total=len(train_loader))
    
    for batch_idx, sample_batched in enumerate(train_loader):
        if combine_type != COMBINE_TYPE_FMRI:
            data_e = sample_batched['eeg_data']
            data_e = data_e.to(device)
            
        if combine_type != COMBINE_TYPE_EEG:
            data_f = sample_batched['fmri_data']
            data_f = data_f.to(device)
            
        label  = sample_batched['label']
        label = label.to(device)
        
        optimizer.zero_grad()
        
        if combine_type == COMBINE_TYPE_EEG:
            output = model(data_e)
        elif combine_type == COMBINE_TYPE_FMRI:
            output = model(data_f)
        else:
            output = model(data_f, data_e)

        diverged = torch.any(torch.isnan(output))
        if diverged:
            print('>>> diverged')
            return -np.inf
        
        loss = F.binary_cross_entropy(output, label)
        loss.backward()
        
        optimizer.step()

        recorded_labels += list(label.cpu().detach().numpy().reshape(-1))
        recorded_preds += list(output.cpu().detach().numpy().reshape(-1))
        
        count += 1
        
        logger.log("loss/train", loss.item(), count)
        
        pbar.update()
        
    metrics = calc_metrics(recorded_labels, recorded_preds)
    
    pbar.close()
    
    logger.log("accuracy/train", metrics['accuracy'], epoch)
    return metrics


def eval_epoch(combine_type,
               model,
               device,
               validation_loader,
               epoch,
               logger,
               record_result=False):
    
    model.eval()
    
    validation_loss = 0
    correct = 0
    
    threshold = torch.Tensor([0.5]).to(device)
    
    recorded_labels = []
    recorded_preds = []

    with torch.no_grad():
        for sample_batched in validation_loader:
            if combine_type != COMBINE_TYPE_FMRI:
                data_e = sample_batched['eeg_data']
                data_e = data_e.to(device)
            
            if combine_type != COMBINE_TYPE_EEG:
                data_f = sample_batched['fmri_data']
                data_f = data_f.to(device)

            label  = sample_batched['label']
            label = label.to(device)

            if combine_type == COMBINE_TYPE_EEG:
                output = model(data_e)
            elif combine_type == COMBINE_TYPE_FMRI:
                output = model(data_f)
            else:
                output = model(data_f, data_e)
            
            recorded_labels += list(label.cpu().detach().numpy().reshape(-1))
            recorded_preds += list(output.cpu().detach().numpy().reshape(-1))

            diverged = torch.any(torch.isnan(output))
            if diverged:
                # TODO:
                print('>>> diverged')
                return -np.inf

            validation_loss = F.binary_cross_entropy(output,
                                                     label,
                                                     reduction='sum').item()
            
            # Sum up batch loss
            result = (output > threshold).float() * 1
            pred = torch.sum(result == label).item()
            correct += pred
            
    validation_loss /= len(validation_loader.dataset)

    metrics = calc_metrics(recorded_labels, recorded_preds)
    
    if logger is not None:
        logger.log("loss/validation", validation_loss, epoch)
        logger.log("accuracy/validation", metrics['accuracy'], epoch)

    if record_result:
        return metrics, (recorded_labels, recorded_preds)
    else:
        return metrics


def train_fold(combine_type, args, classify_type, fold):
    device, use_cuda = get_device(args.gpu)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    train_loader = DataLoader(get_dataset(combine_type=combine_type,
                                          data_type=DATA_TYPE_TRAIN,
                                          classify_type=classify_type,
                                          fold=fold,
                                          args=args),
                              batch_size=args.batch_size,
                              shuffle=True,
                              **kwargs)
    
    validation_loader = DataLoader(get_dataset(combine_type=combine_type,
                                               data_type=DATA_TYPE_VALIDATION,
                                               classify_type=classify_type,
                                               fold=fold,
                                               args=args),
                             batch_size=args.batch_size,
                             shuffle=True,
                             **kwargs)
    
    if args.run_seed >= 0:
        # Fix random seeds at runtime
        fix_run_seed(args.run_seed + fold)
        
    model = get_model(combine_type, args, fold, device)
    
    optimizer = get_optimizer(combine_type, args, model)
    
    save_dir = args.save_dir
    log_path = os.path.join(save_dir, "log/log_ct{}_{}".format(classify_type, fold))
    logger = Logger(log_path)

    # No ignore epochs for preloaded model
    if combine_type == COMBINE_TYPE_COMBINED:
        ignore_epochs = 0
    else:
        ignore_epochs = 5

    early_stopping = EarlyStopping(patience=args.patience,
                                   ignore_epochs=ignore_epochs,
                                   save_dir=save_dir,
                                   fold=fold,
                                   classify_type=classify_type,
                                   metric=args.early_stopping_metric,
                                   debug=args.debug)

    if args.epochs == 0:
        # In the case of no training, only save the model
        early_stopping.save(model)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(combine_type,
                                     model,
                                     device,
                                     train_loader,
                                     optimizer,
                                     epoch,
                                     logger)
        print('{}: Train: {}: {:.2f}'.format(
            epoch, args.early_stopping_metric, train_metrics[args.early_stopping_metric]))
        
        validation_metrics = eval_epoch(combine_type,
                                        model,
                                        device,
                                        validation_loader,
                                        epoch,
                                        logger)
        print('{}: Validation: {}: {:.2f}'.format(
            epoch, args.early_stopping_metric, validation_metrics[args.early_stopping_metric]))
            
        
        if args.early_stopping:
            stop = early_stopping.check_stopping(validation_metrics, train_metrics, epoch, model)
            if stop:
                print("stop early: Max Validation {}: {:.2f}".format(
                    args.early_stopping_metric,
                    early_stopping.best_score))
                break

        logger.flush()

    logger.close()
    
    if args.early_stopping:
        return early_stopping.accompanied_train_metrics, early_stopping.best_validation_metrics
    else:
        return train_metrics, validation_metrics


def train_full_folds(combine_type, args, classify_type):
    print("start ten fold training: classify_type={}".format(classify_type))
    
    all_train_metrics = []
    all_validation_metrics = []

    for fold in range(args.fold_size):
        print("train fold: {}".format(fold))
        train_metrics, validation_metrics = train_fold(combine_type, args, classify_type, fold)
        
        all_train_metrics.append(train_metrics)
        all_validation_metrics.append(validation_metrics)

    train_metrics_df = pd.DataFrame(all_train_metrics)
    validation_metrics_df = pd.DataFrame(all_validation_metrics)

    aggregated_df = pd.DataFrame([train_metrics_df.mean(), 
                                 validation_metrics_df.mean(),
                                 train_metrics_df.std(ddof=1),              
                                 validation_metrics_df.std(ddof=1)],             
                                index=['train_mean', 
                                       'validation_mean', 
                                       'train_std',
                                       'validation_std'])

    print(aggregated_df)

    save_raw_result(args.save_dir, classify_type, train_metrics_df, 'train')
    save_raw_result(args.save_dir, classify_type, validation_metrics_df, 'validation')
    save_aggregated_result(args.save_dir, classify_type, aggregated_df, for_test=False)
    
    if args.epochs == 0:
        validation_score = None
    else:
        # Use early stopping metric (default = 'accuracy') for the Optuna optimization.
        validation_score = aggregated_df[args.early_stopping_metric]['validation_mean']

    # The output value is used in optimize.py
    return validation_score


def test_fold(combine_type, args, classify_type, fold):
    device, use_cuda = get_device(args.gpu)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    test_loader = DataLoader(get_dataset(combine_type=combine_type,
                                         data_type=DATA_TYPE_TEST,
                                         classify_type=classify_type,
                                         fold=fold,
                                         args=args),
                             batch_size=args.batch_size,
                             shuffle=False,
                             **kwargs)
    
    if args.run_seed >= 0:
        # Fix random seeds at runtime
        fix_run_seed(args.run_seed + fold)
    
    model = get_model(combine_type, args, fold, device)
    
    model_path  = "{}/model_ct{}_{}.pt".format(args.save_dir, classify_type, fold)
    state = torch.load(model_path, device)
    state = fix_state_dict(state)
    
    model.load_state_dict(state)
    
    test_metrics, (recorded_labels, recorded_preds) = eval_epoch(
        combine_type,
        model,
        device,
        test_loader,
        0,
        None,
        record_result=True)
    
    return test_metrics, (recorded_labels, recorded_preds)


def test_pfi_fold(combine_type, args, classify_type, fold):
    device, use_cuda = get_device(args.gpu)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # Explicitly ignore fmri mask option.
    base_test_loader = DataLoader(get_dataset(combine_type=combine_type,
                                              data_type=DATA_TYPE_TEST,
                                              classify_type=classify_type,
                                              fold=fold,
                                              args=args,
                                              ignore_fmri_mask_arg=True),
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  **kwargs)
    
    if args.run_seed >= 0:
        # Fix random seeds at runtime
        fix_run_seed(args.run_seed + fold)
    
    model = get_model(combine_type, args, fold, device)
    
    model_path  = "{}/model_ct{}_{}.pt".format(args.save_dir, classify_type, fold)
    state = torch.load(model_path, device)
    state = fix_state_dict(state)
    
    model.load_state_dict(state)

    target_metric = 'accuracy'
    
    test_metrics = eval_epoch(
        combine_type,
        model,
        device,
        base_test_loader,
        0,
        None,
        record_result=False)

    base_score = test_metrics[target_metric]
    
    importances = []
    
    for pfi_shuffle_index in range(args.pfi_shuffle_size):
        pfi_test_loader = DataLoader(get_dataset(combine_type=combine_type,
                                                 data_type=DATA_TYPE_TEST,
                                                 classify_type=classify_type,
                                                 fold=fold,
                                                 args=args,
                                                 pfi_seed=pfi_shuffle_index), # Set PFI shuffle seed
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     **kwargs)
        
        pfi_test_metrics = eval_epoch(
            combine_type,
            model,
            device,
            pfi_test_loader,
            0,
            None,
            record_result=False)
        
        pfi_test_score = pfi_test_metrics[target_metric]
        importance = base_score - pfi_test_score
        importances.append(importance)

    return base_score, importances


def test_full_folds(combine_type, args, classify_type):
    print("start ten fold test: classify_type={}".format(classify_type))
    
    all_test_metrics = []
    
    for fold in range(args.fold_size):
        print("test fold: {}".format(fold))
        test_metrics, (recorded_labels, recorded_preds) = test_fold(
            combine_type, args, classify_type, fold)
        all_test_metrics.append(test_metrics)
        # Save predicted values
        save_predictions(args.save_dir, classify_type, fold, recorded_labels, recorded_preds)

    test_metrics_df = pd.DataFrame(all_test_metrics)
    
    aggregated_df = pd.DataFrame([test_metrics_df.mean(), 
                                 test_metrics_df.std(ddof=1)],
                                index=['test_mean', 
                                       'test_std'])
    print(aggregated_df)
    
    save_raw_result(args.save_dir, classify_type, test_metrics_df, 'test')
    save_aggregated_result(args.save_dir, classify_type, aggregated_df, for_test=True)


def test_pfi(combine_type, args, classify_type):
    print("start PFI calculation: classify_type={}".format(classify_type))
    
    base_scores = []
    all_importances = []
    
    for fold in range(args.fold_size):
        print("test PFI fold: {}".format(fold))
        base_score, importances = test_pfi_fold(combine_type, args, classify_type, fold)
        base_scores.append(base_score)
        all_importances.append(importances)
    
    save_pfi_result(args.save_dir, classify_type,
                    args.fmri_mask,
                    base_scores, all_importances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combine_type", type=str, default='')
    main_args = parser.parse_args(sys.argv[1:2])
    
    if main_args.combine_type == "eeg":
        combine_type = COMBINE_TYPE_EEG
    elif main_args.combine_type == "fmri":
        combine_type = COMBINE_TYPE_FMRI
    elif main_args.combine_type == "combined":
        combine_type = COMBINE_TYPE_COMBINED
    else:
        assert False
    
    if combine_type == COMBINE_TYPE_EEG:
        args = options.get_eeg_args()
    elif combine_type == COMBINE_TYPE_FMRI:
        args = options.get_fmri_args()
    elif combine_type == COMBINE_TYPE_COMBINED:
        args = options.get_combined_args()
    
    options.save_args(args)
    
    if not args.test:
        assert args.pfi_shuffle_size == 0
        
        # Train
        if args.classify_type == CLASSIFY_ALL:
            train_full_folds(combine_type, args, classify_type=FACE_OBJECT)
            train_full_folds(combine_type, args, classify_type=MALE_FEMALE)
            train_full_folds(combine_type, args, classify_type=ARTIFICIAL_NATURAL)
            train_full_folds(combine_type, args, classify_type=FRONT_SIDE)
            train_full_folds(combine_type, args, classify_type=SMALL_LARGE)
        else:
            train_full_folds(combine_type, args, classify_type=args.classify_type)
    else:
        if args.pfi_shuffle_size == 0:
            # Test
            if args.classify_type == CLASSIFY_ALL:
                test_full_folds(combine_type, args, classify_type=FACE_OBJECT)
                test_full_folds(combine_type, args, classify_type=MALE_FEMALE)
                test_full_folds(combine_type, args, classify_type=ARTIFICIAL_NATURAL)
                test_full_folds(combine_type, args, classify_type=FRONT_SIDE)
                test_full_folds(combine_type, args, classify_type=SMALL_LARGE)
            else:
                test_full_folds(combine_type, args, classify_type=args.classify_type)
        else:
            # PFI calculation
            if args.classify_type == CLASSIFY_ALL:
                test_pfi(combine_type, args, classify_type=FACE_OBJECT)
                test_pfi(combine_type, args, classify_type=MALE_FEMALE)
                test_pfi(combine_type, args, classify_type=ARTIFICIAL_NATURAL)
                test_pfi(combine_type, args, classify_type=FRONT_SIDE)
                test_pfi(combine_type, args, classify_type=SMALL_LARGE)
            else:
                test_pfi(combine_type, args, classify_type=args.classify_type)


if __name__ == '__main__':
    main()
