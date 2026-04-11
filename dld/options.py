import argparse
import os
from datetime import datetime as dt
import json
from distutils.util import strtobool

def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combine_type", type=str)
    # Random seed for the dataset
    parser.add_argument("--data_seed", type=int,
                        default=0)
    # Run seed
    parser.add_argument("--run_seed", type=int,
                        default=-1)
    # Save directory
    parser.add_argument("--save_dir", type=str,
                        default="saved")
    # Classify types
    # FACE_PLACE=0, MALE_FEMALE=1, ARTIFICAL_NATURL=2, FRONT_SIDE=3, SMALL_LARGE=4, ALL=-1    
    parser.add_argument("--classify_type", type=int,
                        default=-1)
    parser.add_argument("--desc", type=str,
                        default="description of the experiment")
    parser.add_argument("--early_stopping", type=strtobool,
                        default="true")
    # Early stopping metric
    # 'accuracy', 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall',
    # 'n_precision', 'n_recall', 'n_f1', 'n_pr_auc'
    parser.add_argument("--early_stopping_metric", type=str,
                        default="accuracy") 
    parser.add_argument("--parallel", type=strtobool,
                        default="false")
    parser.add_argument("--data_dir", type=str,
                        default="./data")
    parser.add_argument("--eeg_normalize_type", type=str,
                        default="normal") # "normal", "pre" or "none"
    parser.add_argument("--fmri_frame_type", type=str, # "normal", "avarage", "three"
                        default="normal")
    parser.add_argument("--fmri_offset_tr", type=int,
                        default=2) # 1,2,3
    parser.add_argument("--gpu", type=int,
                        default=-1)
    parser.add_argument("--eeg_frame_type", type=str,
                        default="normal")
    parser.add_argument("--eeg_duration_type", type=str,
                        default="normal") # "normal", "short", "long"
    # Whether to use smoothed fMRI data
    parser.add_argument("--smooth", type=strtobool,
                        default="true")
    parser.add_argument("--test_subjects", type=str,
                        default="TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01")
    parser.add_argument("--test", type=strtobool,
                        default="false")
    parser.add_argument("--fold_size", type=int,
                        default=9)
    parser.add_argument("--subjects_per_fold", type=int,
                        default=5)
    parser.add_argument("--patience", type=int,
                        default=20)
    parser.add_argument("--average_trial_size", type=int,
                        default=0)
    parser.add_argument("--average_repeat_size", type=int,
                        default=0)
    parser.add_argument("--kernel_size", type=int,
                        default=3) # The kernel size for STNN and TCN
    parser.add_argument("--level_size", type=int,
                        default=-1) # The level size for TCN
    parser.add_argument("--level_hidden_size", type=int,
                        default=63) # Number of the output channels for TCN
    parser.add_argument("--residual", type=strtobool,
                        default="true")
    parser.add_argument("--unmatched", type=strtobool,
                        default="false")
    parser.add_argument("--fmri_mask", type=str,
                        default=None)
    parser.add_argument("--pfi_shuffle_size", type=int,
                        default=0)
    parser.add_argument("--debug", type=strtobool,
                        default="false")
    return parser


def get_eeg_args():
    parser = get_common_parser()
    parser.add_argument("--model_type", type=str,
                        default="tcn1")
    parser.add_argument("--batch_size", type=int,
                        default=10)
    parser.add_argument("--lr", type=float,
                        default=0.001)
    parser.add_argument("--weight_decay", type=float,
                        default=0.0)
    parser.add_argument("--epochs", type=int,
                        default=100)
    args = parser.parse_args()
    return args


def get_fmri_args():
    parser = get_common_parser()
    
    parser.add_argument("--batch_size", type=int,
                        default=10)
    parser.add_argument("--lr", type=float,
                        default=0.001)
    parser.add_argument("--weight_decay", type=float,
                        default=0.0)
    parser.add_argument("--epochs", type=int,
                        default=100)
    args = parser.parse_args()
    return args


def get_combined_args():
    parser = get_common_parser()

    parser.add_argument("--model_type", type=str,
                        default="combined_tcn1")
    parser.add_argument("--batch_size", type=int,
                        default=10)
    parser.add_argument("--lr", type=float,
                        default=0.001)
    parser.add_argument("--weight_decay", type=float,
                        default=0.0)
    parser.add_argument("--lr_eeg", type=float,
                        default=None)
    parser.add_argument("--weight_decay_eeg", type=float,
                        default=None)
    parser.add_argument("--lr_fmri", type=float,
                        default=None)
    parser.add_argument("--weight_decay_fmri", type=float,
                        default=None)
    parser.add_argument("--epochs", type=int,
                        default=100)
    parser.add_argument("--fix_preloads", type=strtobool,
                        default="true")
    parser.add_argument("--preload_eeg_dir", type=str,
                        default=None)
    parser.add_argument("--preload_fmri_dir", type=str,
                        default=None)
    parser.add_argument("--combined_hidden_size", type=int,
                        default=128)
    parser.add_argument("--combined_layer_size", type=int,
                        default=0)
    args = parser.parse_args()
    return args


def get_grad_cam_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str,
                        default="saved")
    parser.add_argument("--debug", type=strtobool,
                        default="false")
    args = parser.parse_args()
    return args


def get_optimize_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=strtobool,
                        default="false")
    args = parser.parse_args()
    return args
    

def save_args(args):
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'options.json')

    args_dic = vars(args)
    
    args_dic['date'] = dt.now().strftime('%Y-%m-%d %H:%M')
    
    with open(file_path, 'w') as f:
        json.dump(args_dic, f, indent=2)
    

class Arguments:
    """ Overidable arguments wrapper """
    def __init__(self, args):
        if type(args) == dict:
            args_dic = args
        else:
            args_dic = vars(args).copy()
        
        for name,value in args_dic.items():
            setattr(self, name, value)
    
    def override_param(self, name, value):
        if name in self.__dict__:
            setattr(self, name, value)
        else:
            assert False
            
    def override_params(self, param_dict):
        for name, value in param_dict.items():
            self.override_param(name, value)
            
    def __contains__(self, name):
        return name in self.__dict__

    
def load_args(save_dir):
    file_path = os.path.join(save_dir, 'options.json')
    
    with open(file_path) as f:
        args_dic = json.load(f)
        
    return Arguments(args_dic)


if __name__ == '__main__':
    args = get_eeg_args()
    save_args(args)
    args = load_args(args.save_dir)
