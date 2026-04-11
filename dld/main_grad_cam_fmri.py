import numpy as np
import torch
import os
from tqdm import tqdm
import scipy.ndimage
import scipy.io

from dataset import get_dataset, CLASSIFY_ALL, CLASSIFY_TYPE_MAX
from dataset import DATA_TYPE_TRAIN, DATA_TYPE_VALIDATION, DATA_TYPE_TEST
from dataset import COMBINE_TYPE_EEG, COMBINE_TYPE_FMRI, COMBINE_TYPE_COMBINED

from model import get_fmri_model
from options import get_grad_cam_args, load_args
from utils import get_device, fix_state_dict
from utils import sigmoid
from guided_bp import GuidedBackprop
from grad_cam import get_fmri_grad_cam_sub


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

    f_cam_org, f_cam_nopool = get_fmri_grad_cam_sub(cam_grads, cam_features)

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

    combine_type = COMBINE_TYPE_FMRI

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
                process_grad_cam_fmri_sub(args,
                                          classify_type=ct,
                                          fold=fold,
                                          output_dir=output_dir)

if __name__ == '__main__':
    process_grad_cam_fmri()
