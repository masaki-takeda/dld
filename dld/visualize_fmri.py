import numpy as np
import nibabel as nib
import os

from scipy import stats
from scipy.ndimage import zoom
from scipy.ndimage import label as connected_cluster
from nilearn.datasets import load_mni152_template

def process_raw_data(data):
    label = data['label'].flatten()  # (1, *)
    predicted_label = data['predicted_label'].flatten()
    guided_cam_nopool0 = data['guided_cam_nopool0']  # (782, 79, 95, 79)
    guided_cam_nopool1 = data['guided_cam_nopool1']

    success_type0 = np.where((label == 0) & (predicted_label == 0))[0]
    success_type1 = np.where((label == 1) & (predicted_label == 1))[0]

    data0 = guided_cam_nopool0[success_type0]
    data1 = guided_cam_nopool1[success_type1]

    return data0, data1

def permutation_test(data0, data1, p_threshold=0.001, n_permutations=5000, alpha=0.05):
    t_vals_original, p_vals_original = stats.ttest_ind(data0, data1, axis=0, equal_var=False)
    t_vals_original = np.nan_to_num(t_vals_original, nan=0.0, posinf=0.0, neginf=0.0)
    p_vals_original = np.nan_to_num(p_vals_original, nan=0.0, posinf=0.0, neginf=0.0)

    significant_mask = p_vals_original < p_threshold
    labeled_array_original, num_clusters_original = connected_cluster(significant_mask)
    cluster_sizes_original = np.zeros(num_clusters_original)
    for i in range(1, num_clusters_original + 1):
        cluster_sizes_original[i - 1] = np.sum(labeled_array_original == i)

    # Merge two sets of data
    all_data = np.vstack((data0, data1))
    n_total = len(all_data)
    n_type0 = len(data0)

    max_cluster_sizes_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            print(f"Perform permutations {i + 1}/{n_permutations}")

        # Randomize tags
        perm_indices = np.random.permutation(n_total)
        perm_type0_data = all_data[perm_indices[:n_type0]]
        perm_type1_data = all_data[perm_indices[n_type0:]]

        # Calculate the permuted t-statistic and p-value
        perm_t_values, perm_p_values = stats.ttest_ind(perm_type0_data, perm_type1_data, axis=0, equal_var=False)
        perm_t_values = np.nan_to_num(perm_t_values, nan=0.0, posinf=0.0, neginf=0.0)
        perm_p_values = np.nan_to_num(perm_p_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Create a saliency mask
        perm_significant_mask =  perm_p_values < p_threshold
        labeled_array, num_clusters = connected_cluster(perm_significant_mask)
        cluster_sizes_perm = np.zeros(num_clusters)
        for j in range(1, num_clusters + 1):
            cluster_sizes_perm[j - 1] = np.sum(labeled_array == j)

        if len(cluster_sizes_perm) > 0:
            max_cluster_sizes_perm[i] = np.max(cluster_sizes_perm)
        else:
            max_cluster_sizes_perm[i] = 0

    critical_cluster_size = np.percentile(max_cluster_sizes_perm, 100 * (1 - alpha))
    # Mark significant clusters (clusters in the raw data that exceed the threshold)
    significant_mask = np.zeros_like(t_vals_original, dtype=bool)
    # Create a Boolean array `significant_mask` with the same shape as `t_map_original` to store which voxels belong to significant clusters.
    for i in range(1, num_clusters_original + 1):
        this_cluster = labeled_array_original == i
        if np.sum(this_cluster) >= critical_cluster_size:
            significant_mask |= this_cluster

    return significant_mask, t_vals_original, p_vals_original

def save_nifti(t_map, p_map, significant_cluster, output_dir):
    template_img = load_mni152_template(resolution=2)
    affine = template_img.affine
    target_shape = template_img.shape  # (91, 109, 91)

    if t_map.shape != target_shape:
        zoom_factors = np.array(target_shape) / np.array(t_map.shape)
        t_map = zoom(t_map, zoom_factors, order=1) 
        significant_cluster = zoom(significant_cluster.astype(np.float32), zoom_factors, order=0) 
        significant_cluster = significant_cluster.astype(np.int16)

    # Generate a masked t-map 
    masked_t_map = t_map * significant_cluster

    t_map_nifti = nib.Nifti1Image(t_map, affine)
    nib.save(t_map_nifti, os.path.join(output_dir, f't_map.nii.gz'))

    p_map_nifti = nib.Nifti1Image(p_map, affine)
    nib.save(p_map_nifti, os.path.join(output_dir, 'p_map.nii.gz'))

    significant_clusters_nifti = nib.Nifti1Image(significant_cluster.astype(np.int16), affine)
    nib.save(significant_clusters_nifti, os.path.join(output_dir, f'significant_clusters.nii.gz'))

    # save masked t-map
    masked_t_map_nifti = nib.Nifti1Image(masked_t_map, affine)
    nib.save(masked_t_map_nifti, os.path.join(output_dir, f'masked_t_map.nii.gz'))

    print(f"Saved successfully：t_map.nii.gz、p_map.nii.gz、significant_clusters.nii.gz and masked_t_map.nii.gz")


if __name__ == "__main__":
    data = np.load("/data3/DLD2/241212_fmri/a9r9/class4/grad_cam/data/cam_fmri_ct4_3.npz")
    output_dir = "/data3/DLD2/241212_fmri/analysis_results_for_fsl"
    os.makedirs(output_dir, exist_ok=True)

    data0, data1 = process_raw_data(data)
    significant_cluster, t_map, p_map = permutation_test(data0, data1, p_threshold=0.001, n_permutations=5000, alpha=0.05)
    save_nifti(t_map, p_map,significant_cluster, output_dir)
