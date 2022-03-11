import numpy as np
import pandas as pd
import os
import argparse
from distutils.util import strtobool

from behavior import Behavior
from eeg import EEG
from fmri import FMRI


def preprocess_eeg(src_base, dst_base, behaviors, normalize_type, frame_type):
    print("start preparing eeg")
    eeg_datas = []

    for behavior in behaviors:
        eeg = EEG(src_base, behavior, normalize_type=normalize_type,
                  frame_type=frame_type)
        eeg_datas.append(eeg.data)

    all_eeg_datas = np.vstack(eeg_datas)
    # (1734, 63, 375) or (1734, 5, 63, 375)

    if normalize_type == "normal":
        output_file_path = os.path.join(dst_base, "final_eeg_data") # npzファイル名
    elif normalize_type == "pre":
        output_file_path = os.path.join(dst_base, "final_eeg_data_pre") # npzファイル名
    else:
        output_file_path = os.path.join(dst_base, "final_eeg_data_none") # npzファイル名

    if frame_type == "filter":
        # フィルタ利用の場合は後ろに"_filter"を付加したファイル名にする
        output_file_path = "{}_filter".format(output_file_path)
    elif frame_type == "ft":
        # FTの場合は後ろに"_ft"を付加したファイル名にする
        output_file_path = "{}_ft".format(output_file_path)
    
    np.savez(output_file_path,
             eeg_data=all_eeg_datas)


def preprocess_fmri(src_base, dst_base, behaviors, frame_type, smooth):
    print("start preparing fmri")

    start_frame_index = 0
    for behavior in behaviors:
        fmri = FMRI(src_base,
                    behavior,
                    frame_type=frame_type,
                    smooth=smooth)
        print("date={}, subject={}, run={}".format(behavior.date, behavior.subject, behavior.run))
        
        frame_size = fmri.frame_size

        fmri.export(dst_base=dst_base,
                    start_frame_index=start_frame_index)
        start_frame_index += frame_size
        del fmri


def save_aggregated_behavior_data(dst_base, behaviors, debug):
    categories = []
    identities = []
    sub_categories = []
    angles = []
    subjects = []

    for behavior in behaviors:
        for trial in behavior.trials:
            # datasetにおいては、パラメータをすべて0始まりの値にするために、1を引いておく
            categories.append(trial.category-1)
            identities.append(trial.identity-1)
            sub_categories.append(trial.sub_category-1)
            angles.append(trial.angle-1)
            # 各trailの属するsubject id(TM_191008_01など)を保存しておく.
            subjects.append(behavior.subject_id)

    output_file_path = os.path.join(dst_base, "final_behavior_data") # npzファイル名
    if debug:
        output_file_path = output_file_path + "_debug"
    np.savez_compressed(output_file_path,
                        category=np.array(categories, dtype=np.int32),
                        sub_category=np.array(sub_categories, dtype=np.int32),
                        identity=np.array(identities, dtype=np.int32),
                        angle=np.array(angles, dtype=np.int32),
                        subject=subjects)


def collect_behaviors(src_base, debug):
    # 実験データ
    df = pd.read_csv("./experiment_data/experiments.csv")

    column_size = len(df.columns)

    print("start preparing behaviors")
    
    behaviors = []

    for index, row in df.iterrows():
        valid   = int(row['valid']) != 0
        date    = int(row['date']) # 19008など
        subject = int(row['subject']) # 
        run     = int(row['run'])
        reject_trials = []

        if debug and date > 191009:
            # デバッグ時は被験者を最初の数人だけを利用するようにする
            break
        
        for i in range(4, column_size):
            if not np.isnan(row[i]):
                reject_trial = int(row[i])
                # これは1始まりの値になっている
                reject_trials.append(reject_trial)
                
        if valid:
            # Behaviorインスタンスを作成
            behavior = Behavior(src_base, date, subject, run, reject_trials)
            behaviors.append(behavior)
        else:
            # TODO: 一部欠けているデータ等をskipしている
            print("skipping invalid behavior: date={}, subject={}, run={}".format(
                date, subject, run))
        
    return behaviors


def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri", type=strtobool,
                        default="true")
    parser.add_argument("--eeg", type=strtobool,
                        default="true")
    parser.add_argument("--smooth", type=strtobool,
                        default="true")
    parser.add_argument("--behavior", type=strtobool,
                        default="true")
    parser.add_argument("--eeg_normalize_type", type=str,
                        default="normal") # normal, pre, none
    parser.add_argument("--src_base", type=str,
                        default="/data1/DLD/Data_Prepared")
    parser.add_argument("--dst_base", type=str,
                        default="./data")
    parser.add_argument("--fmri_frame_type", type=str,
                        default="normal") # normal, avarage, three
    parser.add_argument("--eeg_frame_type", type=str,
                        default="filter") # normal, filter, ft
    parser.add_argument("--debug", type=strtobool,
                        default="false")
    args = parser.parse_args()

    # prepardデータの場所
    src_base = args.src_base
    # データの書き出し場所
    dst_base = args.dst_base

    if not os.path.exists(src_base):
        os.makedirs(src_base)
    if not os.path.exists(dst_base):
        os.makedirs(dst_base)

    # Behaviorは(保存するしないにかかわらず)、experiments.csvから毎回ロードする.
    behaviors = collect_behaviors(src_base, args.debug)

    if args.eeg:
        preprocess_eeg(src_base, dst_base, behaviors, args.eeg_normalize_type,
                       frame_type=args.eeg_frame_type)
        
    if args.fmri:
        preprocess_fmri(src_base, dst_base, behaviors,
                        frame_type=args.fmri_frame_type,
                        smooth=args.smooth)
        
    if args.behavior:
        # Behaviorを保存する
        save_aggregated_behavior_data(dst_base, behaviors, args.debug)


if __name__ == '__main__':
    preprocess()
