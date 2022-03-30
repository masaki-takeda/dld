import numpy as np
import argparse
import torch
import os
from distutils.util import strtobool
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.ndimage
import scipy.io

from dataset import BrainDataset, FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, CLASSIFY_ALL
from dataset import DATA_TYPE_TRAIN, DATA_TYPE_VALIDATION, DATA_TYPE_TEST
from model import get_fmri_model
from options import get_grad_cam_args
from utils import get_device, fix_state_dict, save_result, get_test_subject_ids
from utils import sigmoid
from guided_bp import GuidedBackprop


DEBUGGING = False


def get_fmri_grad_cam(model, data, label):
    model.eval()
    model.zero_grad()
    
    out = model.forward_grad_cam(data)
    out = torch.sum(out)
    
    logit = out.cpu().detach().numpy()
    predicted_prob = sigmoid(logit)
    
    if logit > 0:
        predicted_label = 1
    else:
        predicted_label = 0

    if label == 0:
        out = -out
    
    out.backward()
    cam_grads    = model.get_cam_gradients() # (1, 32, 6, 7, 6)
    cam_features = model.get_cam_features()  # (1, 32, 6, 7, 6)

    f_cam_grad = cam_grads.cpu().numpy()[0] # (32, 6, 7, 6)
    f_feature  = cam_features.data.cpu().numpy()[0] # (32, 6, 7, 6)

    # Take the average of the gradients of 6x7x6 voxels
    f_weights = np.mean(f_cam_grad, axis=(1,2,3)) # (32,)

    # Grad-CAM calculated with global-pooling as described in the original paper
    f_cam_org = np.zeros(f_feature.shape[1:], dtype = np.float32) # (6, 7, 6)
    for i, w in enumerate(f_weights):
        # Find the weighted sum of 32 (6,7,6) voxels
        f_cam_org += w * f_feature[i, :, :, :]
        
    # Grad-CAM calculated without global-pooling
    # Sum of 32 channels of element-wise products of (32,6,7,6) and (32,7,6,7)
    f_cam_nopool = np.sum(f_cam_grad * f_feature, axis=0) # (6, 7, 6)

    f_cam_org    = np.maximum(f_cam_org,    0)    
    f_cam_nopool = np.maximum(f_cam_nopool, 0)

    return f_cam_org, f_cam_nopool, predicted_label, predicted_prob


def get_fmri_cam(model, dataset, device, index):
    sample = dataset[index]
    s_data_f = sample['fmri_data']
    s_label  = sample['label']
    
    b_data_f = np.expand_dims(s_data_f, axis=0)
    b_label  = np.expand_dims(s_label, axis=0)
    
    b_data_f = torch.Tensor(b_data_f)
    b_label  = torch.Tensor(b_label)
    
    b_data_f, b_label = b_data_f.to(device), b_label.to(device)
    
    f_cam_org0, f_cam_nopool0, predicted_label, predicted_prob = get_fmri_grad_cam(model, b_data_f, 0)
    f_cam_org1, f_cam_nopool1, _, _                            = get_fmri_grad_cam(model, b_data_f, 1)
    # (6,7,6)

    # Resize to original voxel size
    zoom_rate = np.array((79, 95, 79)) / np.array((6,7,6))
    
    f_cam_nopool0_zoom = scipy.ndimage.zoom(f_cam_nopool0, zoom_rate)
    f_cam_nopool1_zoom = scipy.ndimage.zoom(f_cam_nopool1, zoom_rate)
    # (79, 95, 79)

    # Calculate Guided-BP
    gbp = GuidedBackprop(model)
    gbp_grad0, _ = gbp.generate_gradients(b_data_f, 0, device)
    gbp_grad1, _ = gbp.generate_gradients(b_data_f, 1, device)

    # Free the memory of GuidedBP
    gbp.clear()

    gbp_grad0 = gbp_grad0[0]
    gbp_grad1 = gbp_grad1[0]
    # (79, 95, 79)
    
    guided_cam_nopool0 = f_cam_nopool0_zoom * gbp_grad0
    guided_cam_nopool1 = f_cam_nopool1_zoom * gbp_grad1
    
    return f_cam_org0, f_cam_org1, \
        f_cam_nopool0, f_cam_nopool1, \
        gbp_grad0, gbp_grad1, \
        guided_cam_nopool0, guided_cam_nopool1, \
        int(s_label), predicted_label, predicted_prob


def process_grad_cam_fmri_sub(args,
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
                           use_eeg=False,
                           data_dir=args.data_dir,
                           fmri_frame_type=args.fmri_frame_type,
                           use_smooth=args.smooth,
                           average_trial_size=args.average_trial_size,
                           average_repeat_size=args.average_repeat_size,
                           fold=fold,
                           test_subjects=test_subject_ids,
                           subjects_per_fold=args.subjects_per_fold,
                           debug=args.debug)

    device, use_cuda = get_device(args.gpu)

    model_path = "{}/model_ct{}_{}.pt".format(args.save_dir, classify_type, fold)

    state = torch.load(model_path, map_location=device)
    state = fix_state_dict(state)

    fmri_ch_size = dataset.fmri_ch_size

    model = get_fmri_model(fmri_ch_size, False, device)
    model.load_state_dict(state)
    
    data_size = len(dataset)
    
    cams0 = []
    cams1 = []
    cam_nopools0 = []
    cam_nopools1 = []
    guided_bps0 = []
    guided_bps1 = []
    guided_cam_nopools0 = []
    guided_cam_nopools1 = []
    labels = []
    predicted_labels = []
    predicted_probs = []
    
    bar = tqdm(total=data_size)
    
    for i in range(data_size):
        out = get_fmri_cam(model,
                           dataset,
                           device,
                           i)
        cam0, cam1, cam_nopool0, cam_nopool1, guided_bp0, guided_bp1, guided_cam_nopool0, guided_cam_nopool1, \
            label, predicted_label, predicted_prob = out
        cams0.append(cam0)
        cams1.append(cam1)
        cam_nopools0.append(cam_nopool0)
        cam_nopools1.append(cam_nopool1)
        guided_bps0.append(guided_bp0)
        guided_bps1.append(guided_bp1)
        guided_cam_nopools0.append(guided_cam_nopool0)
        guided_cam_nopools1.append(guided_cam_nopool1)
        labels.append(label)
        predicted_labels.append(predicted_label)
        predicted_probs.append(predicted_prob)
        
        bar.update()

        if DEBUGGING and i == 3:
            break
        
    np_output_file_path = f"{output_dir}/cam_fmri_ct{classify_type}_{fold}"
    mat_output_file_path = f"{np_output_file_path}.mat"
    
    # Save in numpy format
    save_data = {
        'cam0' : cams0,
        'cam1' : cams1,
        'cam_nopool0' : cam_nopools0,
        'cam_nopool1' : cam_nopools1,
        'guided_bp0' : guided_bps0,
        'guided_bp1' : guided_bps1,
        'guided_cam_nopool0' : guided_cam_nopools0,
        'guided_cam_nopool1' : guided_cam_nopools1,
        'label' : labels,
        'predicted_label' : predicted_labels,
        'predicted_prob' : predicted_probs
    }
    
    np.savez_compressed(np_output_file_path, **save_data)
    scipy.io.savemat(mat_output_file_path, save_data)


def process_grad_cam_fmri():
    args = get_grad_cam_args()

    output_dir = args.save_dir + "/grad_cam/data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fold in range(args.fold_size):
        if args.classify_type == CLASSIFY_ALL or args.classify_type == FACE_OBJECT:        
            process_grad_cam_fmri_sub(args,
                                      classify_type=FACE_OBJECT,
                                      fold=fold,
                                      output_dir=output_dir)

        if args.classify_type == CLASSIFY_ALL or args.classify_type == MALE_FEMALE:
            process_grad_cam_fmri_sub(args,
                                      classify_type=MALE_FEMALE,
                                      fold=fold,
                                      output_dir=output_dir)

        if args.classify_type == CLASSIFY_ALL or args.classify_type == ARTIFICIAL_NATURAL:
            process_grad_cam_fmri_sub(args,
                                      classify_type=ARTIFICIAL_NATURAL,
                                      fold=fold,
                                      output_dir=output_dir)


if __name__ == '__main__':
    process_grad_cam_fmri()
