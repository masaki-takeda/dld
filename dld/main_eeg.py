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
from model import get_eeg_model
from logger import Logger
import options
import export
from early_stopping import EarlyStopping
from utils import get_device, fix_state_dict, save_result, get_test_subject_ids, fix_run_seed


def train_epoch(model, device, train_loader, optimizer, epoch, logger,
                transpose_input=False, use_state=False):
    model.train()

    threshold = torch.Tensor([0.5]).to(device)
    
    correct = 0
    count = epoch * len(train_loader)

    pbar = tqdm(total=len(train_loader))
    
    for batch_idx, sample_batched in enumerate(train_loader):
        data  = sample_batched['eeg_data'] # (batch_size, x_dim, seq_length)
        label = sample_batched['label']
        if transpose_input:
            # Replace x_dim and seq in RNN
            # TODO: Refactor by putting it in forward()
            data = torch.transpose(data, 1, 2) # (batch_size, seq_length, x_dim)
        
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()

        if use_state:
            # Use state in RNN
            # TODO: Refactor by putting it in forward()
            state = model.init_state(data.shape[0], device)
            output = model(data, state)
        else:
            output = model(data)
            
        loss = F.binary_cross_entropy(output, label)
        loss.backward()
        optimizer.step()

        result = (output > threshold).float() * 1
        pred = torch.sum(result == label).item()
        correct += pred

        count += 1
        logger.log("e_loss/train", loss.item(), count)

        pbar.update()

    accuracy = 100.0 * correct / len(train_loader.dataset)

    pbar.close()
    
    logger.log("e_accuracy/train", accuracy, epoch)
    return accuracy


def eval_epoch(model, device, validation_loader, epoch, logger,
               transpose_input=False, use_state=False):
    model.eval()
    validation_loss = 0
    correct = 0

    threshold = torch.Tensor([0.5]).to(device)
    
    with torch.no_grad():
        for sample_batched in validation_loader:
            data   = sample_batched['eeg_data']
            label  = sample_batched['label']
            if transpose_input:
                data = torch.transpose(data, 1, 2)
            
            data, label = data.to(device), label.to(device)

            if use_state:
                state = model.init_state(data.shape[0], device)
                output = model(data, state)
            else:
                output = model(data)
            
            validation_loss = F.binary_cross_entropy(output,
                                                     label,
                                                     reduction='sum').item()
            # Sum up batch loss
            result = (output > threshold).float() * 1
            pred = torch.sum(result == label).item()
            correct += pred

    validation_loss /= len(validation_loader.dataset)

    accuracy = 100.0 * correct / len(validation_loader.dataset)

    if logger is not None:
        logger.log("e_loss/validation", validation_loss, epoch)
        logger.log("e_accuracy/validation", accuracy, epoch)
    return accuracy


def train_fold(args, classify_type, fold):
    device, use_cuda = get_device(args.gpu)
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_subject_ids = get_test_subject_ids(args.test_subjects)
    
    train_loader = DataLoader(BrainDataset(data_type=DATA_TYPE_TRAIN,
                                           classify_type=classify_type,
                                           data_seed=args.data_seed,
                                           use_fmri=False,
                                           use_eeg=True,
                                           data_dir=args.data_dir,
                                           eeg_normalize_type=args.eeg_normalize_type,
                                           eeg_frame_type=args.eeg_frame_type,
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
                                                use_fmri=False,
                                                use_eeg=True,
                                                data_dir=args.data_dir,
                                                eeg_normalize_type=args.eeg_normalize_type,
                                                eeg_frame_type=args.eeg_frame_type,
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
    
    model = get_eeg_model(args.model_type, args.parallel,
                          args.kernel_size, args.level_size, args.level_hidden_size,
                          args.residual, device)
    
    optimizer = optim.Adam(model.parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay)

    save_dir = args.save_dir
    log_path = os.path.join(save_dir, "log/log_ct{}_{}".format(classify_type, fold))
    logger = Logger(log_path)

    early_stopping = EarlyStopping(patience=args.patience,
                                   ignore_epochs=5,
                                   save_dir=save_dir,
                                   fold=fold,
                                   classify_type=classify_type,
                                   debug=args.debug)
    
    transpose_input = False
    use_state = False

    if args.epochs == 0:
        # In the case of no training, only save the model
        early_stopping.save(model)

    for epoch in range(args.epochs):
        train_accuracy = train_epoch(model, device, train_loader, optimizer, epoch, logger,
                                     transpose_input, use_state)
        print('{}: Train: Accuracy: {:.0f}%'.format(epoch, train_accuracy))
        
        validation_accuracy  = eval_epoch(model, device, validation_loader, epoch, logger,
                                          transpose_input, use_state)
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
    print("start ten fold training: classify_type={}".format(classify_type))
    
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
                                          use_fmri=False,
                                          use_eeg=True,
                                          data_dir=args.data_dir,
                                          eeg_normalize_type=args.eeg_normalize_type,
                                          eeg_frame_type=args.eeg_frame_type,
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

    model = get_eeg_model(args.model_type, args.parallel,
                          args.kernel_size, args.level_size, args.level_hidden_size,
                          args.residual, device)
    
    model_path  = "{}/model_ct{}_{}.pt".format(args.save_dir, classify_type, fold)
    state = torch.load(model_path, device)
    state = fix_state_dict(state)

    model.load_state_dict(state)

    transpose_input = False
    use_state = False
    test_accuracy = eval_epoch(model, device, test_loader, 0, None,
                               transpose_input, use_state)

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
    args = options.get_eeg_args()
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
