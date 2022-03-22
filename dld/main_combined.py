import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from dataset import BrainDataset, FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, CLASSIFY_ALL
from dataset import DATA_TYPE_TRAIN, DATA_TYPE_VALIDATION, DATA_TYPE_TEST
from model import get_combined_model
from logger import Logger
import options
import export
from early_stopping import EarlyStopping
from utils import get_device, fix_state_dict, save_result, get_test_subject_ids, fix_run_seed


def train_epoch(model, device, train_loader, optimizers, epoch, logger):
    model.train()

    threshold = torch.Tensor([0.5]).to(device)
    
    correct = 0
    count = epoch * len(train_loader)

    pbar = tqdm(total=len(train_loader))
    
    for batch_idx, sample_batched in enumerate(train_loader):
        data_f = sample_batched['fmri_data']
        data_e = sample_batched['eeg_data']
        label  = sample_batched['label']
        data_f, data_e, label = data_f.to(device), data_e.to(device), label.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()
        output = model(data_f, data_e)
        loss = F.binary_cross_entropy(output, label)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        result = (output > threshold).float() * 1
        pred = torch.sum(result == label).item()
        correct += pred

        count += 1
        logger.log("f_loss/train", loss.item(), count)

        pbar.update()

    accuracy = 100.0 * correct / len(train_loader.dataset)

    pbar.close()
    
    logger.log("f_accuracy/train", accuracy, epoch)
    return accuracy


def eval_epoch(model, device, validation_loader, epoch, logger):
    model.eval()
    validation_loss = 0
    correct = 0

    threshold = torch.Tensor([0.5]).to(device)
    
    with torch.no_grad():
        for sample_batched in validation_loader:
            data_f = sample_batched['fmri_data']
            data_e = sample_batched['eeg_data']
            label  = sample_batched['label']            
            data_f, data_e, label = data_f.to(device), data_e.to(device), label.to(device)
            
            output = model(data_f, data_e)
            validation_loss = F.binary_cross_entropy(output,
                                               label,
                                               reduction='sum').item()
            # sum up batch loss
            result = (output > threshold).float() * 1
            pred = torch.sum(result == label).item()
            correct += pred

    validation_loss /= len(validation_loader.dataset)
    
    accuracy = 100.0 * correct / len(validation_loader.dataset)

    if logger != None:
        logger.log("f_loss/validation", validation_loss, epoch)
        logger.log("f_accuracy/validation", accuracy, epoch)
    return accuracy


def merge_state_dict(part_state_dict, combined_state_dict):
    for name, param in part_state_dict.items():
        if name not in combined_state_dict:
            continue
        combined_state_dict[name].copy_(param)


def load_pretrained_models(model, device, classify_type, fold, fix_weights,
                           preload_fmri_dir, preload_eeg_dir):
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


def train_fold(args, classify_type, fold):
    device, use_cuda = get_device(args.gpu)
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_subject_ids = get_test_subject_ids(args.test_subjects)

    train_loader = DataLoader(BrainDataset(data_type=DATA_TYPE_TRAIN,
                                           classify_type=classify_type,
                                           data_seed=args.data_seed,
                                           use_fmri=True,
                                           use_eeg=True,
                                           data_dir=args.data_dir,
                                           fmri_frame_type=args.fmri_frame_type,
                                           eeg_normalize_type=args.eeg_normalize_type,
                                           eeg_frame_type=args.eeg_frame_type,
                                           use_smooth=args.smooth,
                                           average_trial_size=args.average_trial_size,
                                           average_repeat_size=args.average_repeat_size,
                                           fold=fold,
                                           test_subjects=test_subject_ids,
                                           subjects_per_fold=args.subjects_per_fold,
                                           debug=args.debug),
                              batch_size=args.batch_size,
                              shuffle=True,
                              **kwargs)
    validation_loader = DataLoader(BrainDataset(data_type=DATA_TYPE_VALIDATION,
                                                classify_type=classify_type,
                                                data_seed=args.data_seed,
                                                use_fmri=True,
                                                use_eeg=True,
                                                data_dir=args.data_dir,
                                                fmri_frame_type=args.fmri_frame_type,
                                                eeg_normalize_type=args.eeg_normalize_type,
                                                eeg_frame_type=args.eeg_frame_type,
                                                use_smooth=args.smooth,
                                                average_trial_size=args.average_trial_size,
                                                average_repeat_size=args.average_repeat_size,
                                                fold=fold,
                                                test_subjects=test_subject_ids,
                                                subjects_per_fold=args.subjects_per_fold,
                                                debug=args.debug),
                             batch_size=args.batch_size,
                             shuffle=True,
                             **kwargs)

    if args.run_seed >= 0:
        # Fix random seeds at runtime
        fix_run_seed(args.run_seed + fold)    

    fmri_ch_size = train_loader.dataset.fmri_ch_size
    
    model = get_combined_model(args.model_type, fmri_ch_size,
                               args.kernel_size,
                               args.level_size,
                               args.level_hidden_size,
                               args.residual,
                               args.combined_hidden_size,
                               args.combined_layer_size,
                               args.parallel,
                               device)
    
    # Loading of pretrained model
    if args.preload_fmri_dir is not None or args.preload_eeg_dir is not None:
        # If either preload_fmri_dir or preload_eeg_dir is None, prelaod will not be done
        load_pretrained_models(model, device, classify_type, fold, args.fix_preloads,
                               args.preload_fmri_dir, args.preload_eeg_dir)

    lr = args.lr
    weight_decay = args.weight_decay

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

    optimizers = [optimizer_fc, optimizer_eeg, optimizer_fmri]

    save_dir = args.save_dir
    log_path = os.path.join(save_dir, "log/log_ct{}_{}".format(classify_type, fold))
    logger = Logger(log_path)

    #ignore_epochs = 10
    # No ignore epochs for preloaded model
    ignore_epochs = 0

    early_stopping = EarlyStopping(patience=args.patience,
                                   ignore_epochs=ignore_epochs,
                                   save_dir=save_dir,
                                   fold=fold,
                                   classify_type=classify_type,
                                   debug=args.debug)

    if args.epochs == 0:
        # In the case of no training, only save the model
        early_stopping.save(model)

    for epoch in range(args.epochs):
        train_accuracy = train_epoch(model, device, train_loader, optimizers, epoch, logger)
        print('{}: Train: Accuracy: {:.0f}%'.format(epoch, train_accuracy))
        
        validation_accuracy  = eval_epoch(model, device, validation_loader, epoch, logger)
        print('{}: Validation:  Accuracy: {:.0f}%'.format(epoch, validation_accuracy))

        if args.early_stopping:
            stop = early_stopping.check_stopping(validation_accuracy, train_accuracy, epoch, model)
            if stop:
                print("stop early: Max Validation Accuracy: {:.0f}%".format(
                    early_stopping.max_validation_accuracy))
                break

        logger.flush()

    logger.close()

    if args.early_stopping:
        return early_stopping.accompanied_train_accuracy, early_stopping.max_validation_accuracy
    else:
        return train_accuracy, validation_accuracy


def train_ten_folds(args, classify_type):
    train_accuracies = []
    validation_accuracies = []

    for fold in range(args.fold_size):
        print("train fold: {}".format(fold))
        train_accuracy, validation_accuracy = train_fold(args, classify_type, fold)

        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

    train_accurcy_mean = np.mean(train_accuracies)
    train_accurcy_std = np.std(train_accuracies)
    validation_accurcy_mean = np.mean(validation_accuracies)
    validation_accurcy_std = np.std(validation_accuracies)

    train_result_str = ""
    validation_result_str = ""
    for train_accuracy in train_accuracies:
        train_result_str += "{:.2f}, ".format(train_accuracy)
    for validation_accuracy in validation_accuracies:
        validation_result_str += "{:.2f}, ".format(validation_accuracy)

    print(train_result_str)
    print(validation_result_str)

    results = OrderedDict([
        ("train_accurcy_mean",train_accurcy_mean),
        ("validation_accurcy_mean",validation_accurcy_mean),
        ("train_accurcy_std",train_accurcy_std),
        ("validation_accurcy_std",validation_accurcy_std),
        ("train_result",train_result_str),
        ("validation_result",validation_result_str)])
    
    save_result(args.save_dir, classify_type, results)


def test_fold(args, classify_type, fold):
    device, use_cuda = get_device(args.gpu)
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_subject_ids = get_test_subject_ids(args.test_subjects)

    test_loader = DataLoader(BrainDataset(data_type=DATA_TYPE_TEST,
                                          classify_type=classify_type,
                                          data_seed=args.data_seed,
                                          use_fmri=True,
                                          use_eeg=True,
                                          data_dir=args.data_dir,
                                          fmri_frame_type=args.fmri_frame_type,
                                          eeg_normalize_type=args.eeg_normalize_type,
                                          eeg_frame_type=args.eeg_frame_type,
                                          use_smooth=args.smooth,
                                          average_trial_size=args.average_trial_size,
                                          average_repeat_size=args.average_repeat_size,
                                          fold=fold,
                                          test_subjects=test_subject_ids,
                                          subjects_per_fold=args.subjects_per_fold,
                                          debug=args.debug),
                             batch_size=args.batch_size,
                             shuffle=True,
                             **kwargs)
    
    if args.run_seed >= 0:
        # Fix random seeds at runtime
        fix_run_seed(args.run_seed + fold)

    fmri_ch_size = test_loader.dataset.fmri_ch_size
    
    model = get_combined_model(args.model_type, fmri_ch_size,
                               args.kernel_size,
                               args.level_size,
                               args.level_hidden_size,
                               args.residual,
                               args.combined_hidden_size,
                               args.combined_layer_size,
                               args.parallel,
                               device)
    
    # No need to load a pretrained model here
    
    model_path  = "{}/model_ct{}_{}.pt".format(args.save_dir, classify_type, fold)
    state = torch.load(model_path, map_location=device)
    state = fix_state_dict(state)

    model.load_state_dict(state)

    test_accuracy = eval_epoch(model, device, test_loader, 0, None)

    return test_accuracy


def test_ten_folds(args, classify_type):
    print("start ten fold test: classify_type={}".format(classify_type))
    
    test_accuracies = []
    
    for fold in range(args.fold_size):
        print("test fold: {}".format(fold))
        test_accuracy = test_fold(args, classify_type, fold)
        test_accuracies.append(test_accuracy)

    test_accurcy_mean = np.mean(test_accuracies)
    test_accurcy_std = np.std(test_accuracies)

    test_result_str = ""
    for test_accuracy in test_accuracies:
        test_result_str += "{:.2f}, ".format(test_accuracy)

    print(test_result_str)

    results = OrderedDict([
        ("test_accurcy_mean",test_accurcy_mean),
        ("test_accurcy_std",test_accurcy_std),
        ("test_result",test_result_str)])

    save_result(args.save_dir, classify_type, results, for_test=True)


def main():
    args = options.get_combined_args()
    options.save_args(args)

    if args.test == False:
        # Train
        if args.classify_type == CLASSIFY_ALL:
            train_ten_folds(args, classify_type=FACE_OBJECT)
            train_ten_folds(args, classify_type=MALE_FEMALE)
            train_ten_folds(args, classify_type=ARTIFICIAL_NATURAL)
            title = "exp: {}".format(args.save_dir)
            export.export_results(args.save_dir, title)
        else:
            train_ten_folds(args, classify_type=args.classify_type)
    else:
        # Test
        if args.classify_type == CLASSIFY_ALL:
            test_ten_folds(args, classify_type=FACE_OBJECT)
            test_ten_folds(args, classify_type=MALE_FEMALE)
            test_ten_folds(args, classify_type=ARTIFICIAL_NATURAL)
        else:
            test_ten_folds(args, classify_type=args.classify_type)


if __name__ == '__main__':
    main()
