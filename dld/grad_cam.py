import numpy as np
from scipy import interpolate


def calc_all_active_positions(level_size, kernel_size):
    all_positions = []
    handover_hop_size = 0
    for level in range(level_size-1, -1, -1):
        dilation_size = 2 ** level

        reverse_positions = [0]
        hop_size = (kernel_size-1) * 2 + handover_hop_size
        for i in range(hop_size):
            reverse_position = dilation_size * (i+1)
            reverse_positions.append(reverse_position)

        handover_hop_size = hop_size * 2
        positions = []
        for reverse_position in reverse_positions:
            position = 250 - 1 - reverse_position
            if position >= 0:
                positions.append(position)

        positions.reverse()
        all_positions.append(positions)

    all_positions.reverse()    
    return all_positions


def calc_effective_size(kernel_size, level):
    effective_size = 0
    for i in range(level):
        dilation_size = 2 ** i
        effective_size += (dilation_size * 2)
    return effective_size


def calc_interpolate_base_positions(kernel_size, level, active_positions):
    """
    Level must be specified to a value starting from 0
    (If kernel_size=2; 0~6)
    """
    dilation_size = 2 ** level
    effective_size = calc_effective_size(kernel_size, level)
    
    half_effective_size = effective_size//2
    interpolate_base_positions = \
        [active_position - half_effective_size for active_position in active_positions]
    # This value can be negative because it is just for the key point of the interpolation source
    return interpolate_base_positions


def interpolate_values(values, kernel_size, level, active_positions):
    if len(values) == 250:
        return values
    
    x_base = calc_interpolate_base_positions(kernel_size, level, active_positions)
    
    values = list(values)
    assert type(values) == list
    assert type(x_base) == list
    
    # When inserting 0 in frame 0 and frame 249
    if x_base[0] > 0:
        x_base = [0] + x_base
        values = [0] + values
    if x_base[-1] < 249:
        x_base = x_base + [249]
        values = values + [0]
    
    # When inserting values of first frame in frame 0 and that of last frame in frame 249
    """
    if x_base[0] > 0:
        x_base = [0] + x_base
        values = [values[0]] + values
    if x_base[-1] < 249:
        x_base = x_base + [249]
        values = values + [values[-1]]
    """
    
    x = np.arange(250)
    interp = interpolate.PchipInterpolator(x_base, values)
    y = interp(x)
    return y


def get_eeg_grad_cam_sub(raw_grads,
                         raw_features,
                         kernel_size, level_size):

    # Remove batch size 1. To np.ndarray
    raw_grads    = [raw_grad[0].cpu().numpy()             for raw_grad    in raw_grads]
    raw_features = [raw_feature[0].cpu().detach().numpy() for raw_feature in raw_features]

    raw_grads    = np.array(raw_grads)
    raw_features = np.array(raw_features)
    # (7, 63, 250)
    
    all_active_positions = calc_all_active_positions(level_size, kernel_size)
    
    all_flat_active_grads    = []
    all_flat_active_features = []

    all_grad_cam_nopool_interpolated = []
    all_grad_cam_org_interpolated    = []
    
    for i,active_positions in enumerate(all_active_positions):
        active_grads    = raw_grads[i][:,active_positions]
        active_features = raw_features[i][:,active_positions]
        # e.g., (63, 250), (63, 3)...

        # gradientのglobal pooling
        active_grads_pool = np.mean(active_grads, axis=1).reshape(63,1)
        # (63, 1)

        # Compute Grad-CAM (grad x feature)
        # Sum up along channel and apply ReLU
        # (without global pooling)
        grad_cam_nopool = np.maximum(
            np.sum(active_features * active_grads,      axis=0), 0)
        # (with global pooling)
        grad_cam_org    = np.maximum(
            np.sum(active_features * active_grads_pool, axis=0), 0)

        # Expand to 250 frames
        grad_cam_nopool_interpolated = interpolate_values(
            grad_cam_nopool, kernel_size, i, active_positions)
        grad_cam_org_interpolated    = interpolate_values(
            grad_cam_org,    kernel_size, i, active_positions)

        all_grad_cam_nopool_interpolated.append(grad_cam_nopool_interpolated)
        all_grad_cam_org_interpolated.append(grad_cam_org_interpolated)
        
        all_flat_active_grads.append(active_grads.ravel())
        # Make the followings into 1-dimentional Flat: e.g., (63, 250), (63, 3)...
        # (To avoid errors when converting to ndarray at saving)
        all_flat_active_features.append(active_features.ravel())

    return raw_grads, raw_features, \
        np.array(all_grad_cam_nopool_interpolated), \
        np.array(all_grad_cam_org_interpolated), \
        all_flat_active_grads, all_flat_active_features


def get_fmri_grad_cam_sub(cam_grads, cam_features):
    cam_grad = cam_grads.cpu().numpy()[0] # (32, 6, 7, 6)
    feature  = cam_features.data.cpu().numpy()[0] # (32, 6, 7, 6)

    # Take the average of the gradients of 6x7x6 voxels
    weights = np.mean(cam_grad, axis=(1,2,3)) # (32,)

    # Grad-CAM calculated with global-pooling as described in the original paper
    cam_org = np.zeros(feature.shape[1:], dtype = np.float32) # (6, 7, 6)
    for i, w in enumerate(weights):
        # Find the weighted sum of 32 (6,7,6) voxels
        cam_org += w * feature[i, :, :, :]
        
    # Grad-CAM calculated without global-pooling
    # Sum of 32 channels of element-wise products of (32,6,7,6) and (32,7,6,7)
    cam_nopool = np.sum(cam_grad * feature, axis=0) # (6, 7, 6)

    cam_org    = np.maximum(cam_org,    0)    
    cam_nopool = np.maximum(cam_nopool, 0)

    return cam_org, cam_nopool
