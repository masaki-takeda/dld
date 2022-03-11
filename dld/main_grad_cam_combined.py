import numpy as np
import argparse
import torch
import os
from distutils.util import strtobool
from tqdm import tqdm

from dataset import BrainDataset, FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, CLASSIFY_ALL
from dataset import DATA_TYPE_TRAIN, DATA_TYPE_VALIDATION, DATA_TYPE_TEST
from model import get_combined_model
from options import get_grad_cam_args
from utils import get_device, fix_state_dict, save_result, get_test_subject_ids
from utils import sigmoid



def get_combined_cam(model, dataset, device, index):
    sample = dataset[index]

    s_data_f = sample['fmri_data']
    s_data_e = sample['eeg_data'] # (5, 63, 250)
    s_label  = sample['label'] # (1,)
    
    # バッチの次元を先頭に足す
    b_data_f = np.expand_dims(s_data_f, axis=0)
    b_data_e = np.expand_dims(s_data_e, axis=0)
    b_label  = np.expand_dims(s_label, axis=0)
    
    # テンソルに変換
    b_data_f = torch.Tensor(b_data_f)
    b_data_e = torch.Tensor(b_data_e)
    b_label  = torch.Tensor(b_label)
    
    # デバイスに転送
    b_data_f = b_data_f.to(device)
    b_data_e = b_data_e.to(device)
    b_label = b_label.to(device)
    
    model.eval()
    model.zero_grad()
    out = model.forward_grad_cam(b_data_f, b_data_e) # (1,1) logitの値
    out = torch.sum(out) # (,) スカラーに
    
    logit = out.cpu().detach().numpy()
    
    # sigmoidを通して出力確率を出す
    predicted_prob = sigmoid(logit)
    
    if logit > 0:
        predicted_label = 1
    else:
        predicted_label = 0
    
    if b_label[0][0] == 0:
        # 教師ラベルがnegativeの場合
        out = -out
    
    out.backward()
    
    f_cam_grads, e_cam_grads = model.get_cam_gradients()
    # (1, 32, 6, 7, 6), (1, 32, 3, 28)
    f_cam_features, e_cam_features = model.get_cam_features()
    # (1, 32, 6, 7, 6), (1, 32, 3, 28)
    
    f_cam_grad = f_cam_grads.cpu().numpy()[0] # (32, 6, 7, 6)
    e_cam_grad = e_cam_grads.cpu().numpy()[0] # (32, 3, 28)

    f_feature = f_cam_features.data.cpu().numpy()[0] # (32, 6, 7, 6)
    e_feature = e_cam_features.data.cpu().numpy()[0] # (32, 3, 28)
    
    # Channel毎に重みを求める
    f_weights = np.mean(f_cam_grad, axis=(1,2,3)) # (32,)
    e_weights = np.sum(e_cam_grad, axis=(1,2)) # (32,)

    # 論文通りのglobal-poolingで求めるgrad-cam
    f_cam_org = np.zeros(f_feature.shape[1:], dtype = np.float32) # (6, 7, 6)
    for i, w in enumerate(f_weights):
        f_cam_org += w * f_feature[i, :, :, :]
    
    e_cam_org = np.zeros(e_feature.shape[1:], dtype=np.float32) # (3,28)
    for i, w in enumerate(e_weights):
        # 各チャンネル毎の重みをかける
        e_cam_org += w * e_feature[i, :]
        # (3,28)
    
    # global-poolingを利用しない版のgrad-cam
    f_cam_nopool = np.sum(f_cam_grad * f_feature, axis=0) # (6, 7, 6)    
    e_cam_nopool = np.sum(e_cam_grad * e_feature, axis=0) # (3, 28)
    
    # ReLUをかける
    f_cam_org = np.maximum(f_cam_org, 0)    
    f_cam_nopool = np.maximum(f_cam_nopool, 0)
    
    e_cam_org = np.maximum(e_cam_org, 0)
    e_cam_nopool = np.maximum(e_cam_nopool, 0)
    
    return f_cam_org, f_cam_nopool, e_cam_org, e_cam_nopool, int(s_label), predicted_label, predicted_prob


def process_grad_cam_combined_sub(args,
                                  classify_type,
                                  fold,
                                  output_dir):
    
    if args.test:
        data_type = DATA_TYPE_TEST
    else:
        data_type = DATA_TYPE_VALIDATION

    test_subject_ids = get_test_subject_ids(args.test_subjects)    
    
    dataset = BrainDataset(data_type=data_type,
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
                           debug=args.debug)

    device, use_cuda = get_device(args.gpu)
    
    model_path = "{}/model_ct{}_{}.pt".format(args.model_dir, classify_type, fold)
    state = torch.load(model_path, map_location=device)
    state = fix_state_dict(state)

    fmri_ch_size = dataset.fmri_ch_size
    model = get_combined_model(args.model_type, fmri_ch_size, False, device)
    
    model.load_state_dict(state)
    
    data_size = len(dataset)
    
    f_cams = []
    f_cam_nopools = []
    e_cams = []
    e_cam_nopools = []
    labels = []
    predicted_labels = []
    predicted_probs = []
    
    bar = tqdm(total=data_size)
    
    for i in range(data_size):
        out = get_combined_cam(
            model,
            dataset,
            device,
            i)

        f_cam, f_cam_nopool, e_cam, e_cam_nopool, label, predicted_label, predicted_prob = out
        f_cams.append(f_cam)
        f_cam_nopools.append(f_cam_nopool)
        e_cams.append(e_cam)
        e_cam_nopools.append(e_cam_nopool)
        labels.append(label)
        predicted_labels.append(predicted_label)
        predicted_probs.append(predicted_prob)

        bar.update()
        
    output_file_path = "{}/grad_cam_data_combined_ct{}_{}".format(output_dir, classify_type, fold)
    np.savez_compressed(output_file_path,
                        f_cam=f_cams,
                        f_cam_nopool=f_cam_nopools,
                        e_cam=e_cams,
                        e_cam_nopool=e_cam_nopools,
                        label=labels,
                        predicted_label=predicted_labels,
                        predicted_prob=predicted_probs)


def process_grad_cam_combined():
    args = get_grad_cam_args()

    output_dir = args.output_dir + "/data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fold in range(args.fold_size):
        if args.classify_type == CLASSIFY_ALL or args.classify_type == FACE_OBJECT:
            process_grad_cam_combined_sub(args,
                                          classify_type=FACE_OBJECT,
                                          fold=fold,
                                          output_dir=output_dir)
            
        if args.classify_type == CLASSIFY_ALL or args.classify_type == MALE_FEMALE:            
            process_grad_cam_combined_sub(args,
                                          classify_type=MALE_FEMALE,
                                          fold=fold,
                                          output_dir=output_dir)

        if args.classify_type == CLASSIFY_ALL or args.classify_type == ARTIFICIAL_NATURAL:
            process_grad_cam_combined_sub(args,
                                          classify_type=ARTIFICIAL_NATURAL,
                                          fold=fold,
                                          output_dir=output_dir)


if __name__ == '__main__':
    process_grad_cam_combined()
