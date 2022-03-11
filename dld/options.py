import argparse
import os
import re
from datetime import datetime as dt
from distutils.util import strtobool

def get_common_parser():
    parser = argparse.ArgumentParser()
    # dataset乱数seed
    parser.add_argument("--data_seed", type=int,
                        default=0)
    # 実行時seed
    parser.add_argument("--run_seed", type=int,
                        default=-1)
    # 保存ディレクトリ
    parser.add_argument("--save_dir", type=str,
                        default="saved")
    # 分類タイプ
    parser.add_argument("--classify_type", type=int,
                        default=-1) # FACE_PLACE=0, MALE_FEMALE=1, ARTIFICAL_NATURL=2, ALL=-1
    parser.add_argument("--desc", type=str,
                        default="description of the experiment")
    parser.add_argument("--early_stopping", type=strtobool,
                        default="true")
    parser.add_argument("--parallel", type=strtobool,
                        default="false")
    parser.add_argument("--data_dir", type=str,
                        default="./data")
    parser.add_argument("--eeg_normalize_type", type=str,
                        default="normal") # "normal", "pre" or "none"
    parser.add_argument("--fmri_frame_type", type=str, # "normal", "avarage", "three"
                        default="normal")
    parser.add_argument("--gpu", type=int,
                        default=-1)
    parser.add_argument("--eeg_frame_type", type=str,
                        default="filter") # "normal", "filter", "ft"
    # fMRIにてsmoothingをしたデータを利用するかどうか
    parser.add_argument("--smooth", type=strtobool,
                        default="true")
    parser.add_argument("--test_subjects", type=str,
                        default="TM_191008_01,TM_191009_01")
    parser.add_argument("--test", type=strtobool,
                        default="false")
    parser.add_argument("--fold_size", type=int,
                        default=10)
    parser.add_argument("--subjects_per_fold", type=int,
                        default=4)
    parser.add_argument("--patience", type=int,
                        default=20)
    parser.add_argument("--average_trial_size", type=int,
                        default=0)
    parser.add_argument("--average_repeat_size", type=int,
                        default=0)
    parser.add_argument("--kernel_size", type=int,
                        default=3) # STNN,TCN用のカーネルサイズ指定
    parser.add_argument("--level_size", type=int,
                        default=-1) # TCN用のレベルサイズ
    parser.add_argument("--level_hidden_size", type=int,
                        default=63) # TCNのOutputのch数
    parser.add_argument("--residual", type=strtobool,
                        default="true")
    parser.add_argument("--debug", type=strtobool,
                        default="false")
    return parser


def get_eeg_args():
    parser = get_common_parser()
    parser.add_argument("--model_type", type=str,
                        default="model1") # model1, model2, rnn1, convrnn1, filter1, filter2, stnn1
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
    parser.add_argument("--data_dir", type=str,
                        default="./data")
    parser.add_argument("--save_dir", type=str,
                        default="saved")
    parser.add_argument("--classify_type", type=int,
                        default=-1) # FACE_PLACE=0, MALE_FEMALE=1, ARTIFICAL_NATURL=2, ALL=-1
    parser.add_argument("--eeg_normalize_type", type=str,
                        default="normal") # "normal", "pre" or "none"
    parser.add_argument("--fmri_frame_type", type=str, 
                        default="normal") # "normal", "avarage", "three"
    parser.add_argument("--eeg_frame_type", type=str,
                        default="filter") # "normal", "filter", "ft"
    parser.add_argument("--smooth", type=strtobool,
                        default="true")
    parser.add_argument("--model_type", type=str,
                        default="")
    parser.add_argument("--gpu", type=int,
                        default=-1)
    parser.add_argument("--data_seed", type=int,
                        default=0)
    parser.add_argument("--test", type=strtobool,
                        default="true")
    parser.add_argument("--test_subjects", type=str,
                        default="TM_191008_01,TM_191009_01")
    parser.add_argument("--fold_size", type=int,
                        default=10)
    parser.add_argument("--subjects_per_fold", type=int,
                        default=4)
    parser.add_argument("--average_trial_size", type=int,
                        default=0)
    parser.add_argument("--average_repeat_size", type=int,
                        default=0)
    parser.add_argument("--kernel_size", type=int,
                        default=3) # STNN,TCN用のカーネルサイズ指定
    parser.add_argument("--level_size", type=int,
                        default=-1) # TCN用のレベルサイズ
    parser.add_argument("--level_hidden_size", type=int,
                        default=63) # TCNのOutputのch数
    parser.add_argument("--residual", type=strtobool,
                        default="true")
    parser.add_argument("--debug", type=strtobool,
                        default="false")
    args = parser.parse_args()
    return args


def save_args(args):
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_path = os.path.join(save_dir, "options.txt")
    lines = []

    time_str = dt.now().strftime('# %Y-%m-%d %H:%M')
    lines.append("{}\n".format(time_str))
    args_str = str(args)
    # Strip head "Namespace" string
    args_str = re.sub(r'^Namespace\(', '', args_str)
    args_str = re.sub(r'\)$', '', args_str)
    
    lines.append("{}\n".format(args_str))
    
    f = open(file_path, "w")
    f.writelines(lines)
    f.close()


if __name__ == '__main__':
    args = get_eeg_args()
    save_args(args)
