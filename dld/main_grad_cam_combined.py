import numpy as np
import torch
import os
from tqdm import tqdm
import scipy.ndimage
import scipy.io

from dataset import get_dataset, CLASSIFY_ALL, CLASSIFY_TYPE_MAX
from dataset import DATA_TYPE_VALIDATION, DATA_TYPE_TEST
from dataset import COMBINE_TYPE_EEG, COMBINE_TYPE_FMRI, COMBINE_TYPE_COMBINED

from model import get_combined_model
from options import get_grad_cam_args, load_args
from utils import get_device, fix_state_dict
from utils import sigmoid
from model import calc_required_level

from guided_bp import GuidedBackprop
from grad_cam import get_eeg_grad_cam_sub, get_fmri_grad_cam_sub


def get_combined_grad_cam(model, data_f, data_e, label,
                          kernel_size, level_size):
    model.eval()
    model.zero_grad()
    
    out = model.forward_grad_cam(data_f, data_e) # (1,1) logit values
    out = torch.sum(out) # (,) To scalar
    
    logit = out.cpu().detach().numpy()
    
    # Output probability with sigmoid()
    predicted_prob = sigmoid(logit)
    
    if logit > 0:
        predicted_label = 1
    else:
        predicted_label = 0
    
    if label == 0:
        # When the training label is negative
        out = -out
    
    out.backward()

    f_raw_grads, e_raw_grads = model.get_cam_gradients()
    # (1, 32, 6, 7, 6), [(1, 63, 250), ... x7]
    f_raw_features, e_raw_features = model.get_cam_features()
    # (1, 32, 6, 7, 6), [(1, 63, 250), ... x7]

    f_cam_org, f_cam_nopool = get_fmri_grad_cam_sub(f_raw_grads, f_raw_features)
    
    e_out = get_eeg_grad_cam_sub(e_raw_grads, e_raw_features,
                                 kernel_size, level_size)
    
    e_raw_grads, e_raw_features, \
        e_grad_cam_nopool, \
        e_grad_cam_org, \
        e_flat_active_grads, \
        e_flat_active_features = e_out

    model.clear_grad_cam()
    
    return f_cam_org, f_cam_nopool, \
        e_raw_grads, e_raw_features, \
        e_grad_cam_nopool, \
        e_grad_cam_org, \
        e_flat_active_grads, \
        e_flat_active_features, \
        predicted_label, predicted_prob


def get_combined_cam(model, dataset, device, index, kernel_size, level_size):
    sample = dataset[index]

    s_data_f = sample['fmri_data']
    s_data_e = sample['eeg_data'] # (5, 63, 250)
    s_label  = sample['label'] # (1,)
    
    # Add the dimension of the batch to the top of the data
    b_data_f = np.expand_dims(s_data_f, axis=0)
    b_data_e = np.expand_dims(s_data_e, axis=0)
    b_label  = np.expand_dims(s_label, axis=0)
    
    # Convert to tensor
    b_data_f = torch.Tensor(b_data_f)
    b_data_e = torch.Tensor(b_data_e)
    b_label  = torch.Tensor(b_label)
    
    # Transfer to device
    b_data_f = b_data_f.to(device)
    b_data_e = b_data_e.to(device)
    b_label = b_label.to(device)

    out0 = get_combined_grad_cam(model, b_data_f, b_data_e, 0,
                                 kernel_size, level_size)
    out1 = get_combined_grad_cam(model, b_data_f, b_data_e, 1,
                                 kernel_size, level_size)

    f_cam_org0, f_cam_nopool0, \
        e_raw_grads0, e_raw_features0, \
        e_grad_cam_nopool0, \
        e_grad_cam_org0, \
        e_flat_active_grads0, \
        e_flat_active_features0, \
        predicted_label, predicted_prob = out0

    f_cam_org1, f_cam_nopool1, \
        e_raw_grads1, e_raw_features1, \
        e_grad_cam_nopool1, \
        e_grad_cam_org1, \
        e_flat_active_grads1, \
        e_flat_active_features1, \
        _, _ = out1

    # Resize to original voxel size
    zoom_rate = np.array((79, 95, 79)) / np.array((6,7,6))
    
    f_cam_nopool0_zoom = scipy.ndimage.zoom(f_cam_nopool0, zoom_rate)
    f_cam_nopool1_zoom = scipy.ndimage.zoom(f_cam_nopool1, zoom_rate)
    # (79, 95, 79)
    
    # Calculate Guided-BP
    gbp = GuidedBackprop(model)
    
    f_gbp_grad0, e_gbp_grad0, _ = gbp.generate_gradients_for_combined(b_data_f, b_data_e, 0, device)
    f_gbp_grad1, e_gbp_grad1, _ = gbp.generate_gradients_for_combined(b_data_f, b_data_e, 1, device)
    
    # Free the memory of GuidedBP
    gbp.clear()
    
    f_gbp_grad0 = f_gbp_grad0[0]
    f_gbp_grad1 = f_gbp_grad1[0]
    # (79, 95, 79)

    e_gbp_grad0 = e_gbp_grad0[0]
    e_gbp_grad1 = e_gbp_grad1[0]
    # (79, 95, 79)
    
    f_guided_cam_nopool0 = f_cam_nopool0_zoom * f_gbp_grad0
    f_guided_cam_nopool1 = f_cam_nopool1_zoom * f_gbp_grad1

    return \
        f_cam_org0, f_cam_org1, \
        f_cam_nopool0, f_cam_nopool1, \
        f_gbp_grad0, f_gbp_grad1, \
        f_guided_cam_nopool0, f_guided_cam_nopool1, \
        e_gbp_grad0, e_gbp_grad1, \
        e_raw_grads0, e_raw_grads1, \
        e_raw_features0, \
        e_grad_cam_nopool0, e_grad_cam_nopool1, \
        e_grad_cam_org0, e_grad_cam_org1, \
        e_flat_active_grads0, e_flat_active_grads1, \
        e_flat_active_features0, \
        int(s_label), predicted_label, predicted_prob


def process_grad_cam_combined_sub(args,
                                  classify_type,
                                  fold,
                                  output_dir):

    if args.test:
        data_type = DATA_TYPE_TEST
    else:
        data_type = DATA_TYPE_VALIDATION

    combine_type = COMBINE_TYPE_COMBINED

    dataset = get_dataset(combine_type=combine_type,
                          data_type=data_type,
                          classify_type=classify_type,
                          fold=fold,
                          args=args)
    
    device, use_cuda = get_device(args.gpu)

    model_path = "{}/model_ct{}_{}.pt".format(args.save_dir, classify_type, fold)

    state = torch.load(model_path, map_location=device)
    state = fix_state_dict(state)    

    fmri_ch_size = dataset.fmri_ch_size

    level_size = args.level_size

    if level_size < 0:
        level_size = calc_required_level(args.kernel_size)

    fmri_ch_size = dataset.fmri_ch_size
    model = get_combined_model(args.model_type, fmri_ch_size,
                               args.kernel_size,
                               args.level_size,
                               args.level_hidden_size,
                               args.residual,
                               args.combined_hidden_size,
                               args.combined_layer_size,
                               False,
                               device)

    model.load_state_dict(state)
    
    data_size = len(dataset)

    f_cams0 = []
    f_cams1 = []
    f_cam_nopools0 = []
    f_cam_nopools1 = []
    f_guided_bps0 = []
    f_guided_bps1 = []
    f_guided_cam_nopools0 = []
    f_guided_cam_nopools1 = []

    e_guided_bps0 = []
    e_guided_bps1 = []
    e_raw_grads0 = []
    e_raw_grads1 = []
    e_raw_features = []
    e_grad_cam_nopools0 = []
    e_grad_cam_nopools1 = []
    e_grad_cam_orgs0 = []
    e_grad_cam_orgs1 = []
    e_flat_active_grads0 = []
    e_flat_active_grads1 = []
    e_flat_active_features = []
    
    labels = []
    predicted_labels = []
    predicted_probs = []
    
    bar = tqdm(total=data_size)
    
    for i in range(data_size):
        out = get_combined_cam(
            model,
            dataset,
            device,
            i,
            args.kernel_size,
            level_size)
        
        f_cam0, f_cam1, \
            f_cam_nopool0, f_cam_nopool1, \
            f_guided_bp0, f_guided_bp1, \
            f_guided_cam_nopool0, f_guided_cam_nopool1, \
            e_guided_bp0, e_guided_bp1, \
            e_raw_grad0, e_raw_grad1, \
            e_raw_feature, \
            e_grad_cam_nopool0, e_grad_cam_nopool1, \
            e_grad_cam_org0, e_grad_cam_org1, \
            e_flat_active_grad0, e_flat_active_grad1, \
            e_flat_active_feature, \
            label, predicted_label, predicted_prob = out

        f_cams0.append(f_cam0)
        f_cams1.append(f_cam1)
        f_cam_nopools0.append(f_cam_nopool0)
        f_cam_nopools1.append(f_cam_nopool1)
        f_guided_bps0.append(f_guided_bp0)
        f_guided_bps1.append(f_guided_bp1)
        f_guided_cam_nopools0.append(f_guided_cam_nopool0)
        f_guided_cam_nopools1.append(f_guided_cam_nopool1)

        e_guided_bps0.append(e_guided_bp0)
        e_guided_bps1.append(e_guided_bp1)
        e_raw_grads0.append(e_raw_grad0)
        e_raw_grads1.append(e_raw_grad1)
        e_raw_features.append(e_raw_feature)
        e_grad_cam_nopools0.append(e_grad_cam_nopool0)
        e_grad_cam_nopools1.append(e_grad_cam_nopool1)
        e_grad_cam_orgs0.append(e_grad_cam_org0)
        e_grad_cam_orgs1.append(e_grad_cam_org1)
        e_flat_active_grads0.append(e_flat_active_grad0)
        e_flat_active_grads1.append(e_flat_active_grad1)
        e_flat_active_features.append(e_flat_active_feature)
        
        labels.append(label)
        predicted_labels.append(predicted_label)
        predicted_probs.append(predicted_prob)
        
        bar.update()

        #if DEBUGGING and i == 3:
        #    break

    np_output_file_path = f"{output_dir}/cam_combined_ct{classify_type}_{fold}"
    mat_output_file_path = f"{np_output_file_path}.mat"

    # Save in numpy format
    save_data = {
        'f_cam0' : f_cams0,
        'f_cam1' : f_cams1,
        'f_cam_nopool0' : f_cam_nopools0,
        'f_cam_nopool1' : f_cam_nopools1,
        'f_guided_bp0' : f_guided_bps0,
        'f_guided_bp1' : f_guided_bps1,
        'f_guided_cam_nopool0' : f_guided_cam_nopools0,
        'f_guided_cam_nopool1' : f_guided_cam_nopools1,

        'e_guided_bp0' : e_guided_bps0,
        'e_guided_bp1' : e_guided_bps1,
        'e_raw_grad0' : e_raw_grads0,
        'e_raw_grad1' : e_raw_grads1,
        'e_raw_feature' : e_raw_features,
        'e_cam_nopool0' : e_grad_cam_nopools0,
        'e_cam_nopool1' : e_grad_cam_nopools1,
        'e_cam0' : e_grad_cam_orgs0,
        'e_cam1' : e_grad_cam_orgs1,
        'e_flat_active_grad0' : e_flat_active_grads0,
        'e_flat_active_grad1' : e_flat_active_grads1,
        'e_flat_active_feature' : e_flat_active_features,
        
        'label' : labels,
        'predicted_label' : predicted_labels,
        'predicted_prob' : predicted_probs
    }
    
    # Save in numpy format
    np.savez_compressed(np_output_file_path, **save_data)
    
    # Save in matlab format
    scipy.io.savemat(mat_output_file_path, save_data)


def process_grad_cam_combined():
    raw_args = get_grad_cam_args()

    output_dir = raw_args.save_dir + "/grad_cam/data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    args = load_args(raw_args.save_dir)

    # For use test data for grad cam
    args.override_params(
        {
            'test' : 1,
            'debug' : raw_args.debug,
        }
    )

    for fold in range(args.fold_size):
        for ct in range(CLASSIFY_TYPE_MAX):
            if args.classify_type == CLASSIFY_ALL or args.classify_type == ct:
                process_grad_cam_combined_sub(args,
                                              classify_type=ct,
                                              fold=fold,
                                              output_dir=output_dir)

if __name__ == '__main__':
    process_grad_cam_combined()
