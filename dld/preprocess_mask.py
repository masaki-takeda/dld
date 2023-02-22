import nibabel as nib
import numpy as np


def export_mask(input_path, output_path):
    nii = nib.load(input_path)
    
    data = np.array(nii.dataobj)    
    # (x,y,z)
    data = np.transpose(data, [2, 1, 0])
    # (z,y,x)
    
    np.savez_compressed(output_path, mask=data)


if __name__ == '__main__':
    export_mask('./data/SPMspace_GuidedGradCAM_butnot_univariate_bothFO_bin.nii',
                './experiment_data/mask_gcam')

    export_mask('./data/SPMspace_frontal.nii',
                './experiment_data/mask_frontal')

    export_mask('./data/SPMspace_occipital.nii',
                './experiment_data/mask_occipital')

    export_mask('./data/SPMspace_parietal.nii',
                './experiment_data/mask_parietal')

    export_mask('./data/SPMspace_subcortical.nii',
                './experiment_data/mask_subcortical')

    export_mask('./data/SPMspace_temporal.nii',
                './experiment_data/mask_temporal')
