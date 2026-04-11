import numpy as np
import nibabel as nib
import os

from scipy import stats
from scipy.ndimage import zoom
from scipy.ndimage import label as connected_cluster
from nilearn.datasets import load_mni152_template


def process_raw_data(data):
    label = data['label'].flatten()
    predicted_label = data['predicted_label'].flatten()
    guided_cam_nopool0 = data['guided_cam_nopool0']
    guided_cam_nopool1 = data['guided_cam_nopool1']

    success_type0 = np.where((label == 0) & (predicted_label == 0))[0]
    success_type1 = np.where((label == 1) & (predicted_label == 1))[0]

    data0 = guided_cam_nopool0[success_type0]
    data1 = guided_cam_nopool1[success_type1]
    return data0, data1


def _apply_tail(p_two_sided, t_vals, tail="two-sided"):
    """
    tail:
      - "two-sided"
      - "greater": H1 mean > 0
      - "less":    H1 mean < 0
    """
    if tail == "two-sided":
        return p_two_sided

    p_one = p_two_sided / 2.0
    if tail == "greater":
        p_one[t_vals < 0] = 1.0
    elif tail == "less":
        p_one[t_vals > 0] = 1.0
    else:
        raise ValueError("tail must be 'two-sided', 'greater', or 'less'")
    return p_one


def one_sample_permutation_test(
    data,
    p_threshold=0.001,
    n_permutations=5000,
    alpha=0.05,
    tail="two-sided",
    seed=0
):
    """
    One-sample t-test vs 0 with sign-flip permutation (max cluster size) for FWER control.
    """
    # 1) original one-sample t / p
    t_vals, p_vals = stats.ttest_1samp(data, popmean=0.0, axis=0)

    valid = np.isfinite(t_vals) & np.isfinite(p_vals)
    t_vals[~valid] = 0.0
    p_vals[~valid] = 1.0

    p_vals = _apply_tail(p_vals, t_vals, tail=tail)

    # cluster-forming
    initial_mask = p_vals < p_threshold
    labeled_array_original, num_clusters_original = connected_cluster(initial_mask)

    # 2) permutation (sign-flip)
    rng = np.random.default_rng(seed)
    max_cluster_sizes_perm = np.zeros(n_permutations, dtype=np.int32)

    n_trials = data.shape[0]
    sign_shape = (n_trials,) + (1,) * (data.ndim - 1)  # (n_trials, 1, 1, 1)

    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            print(f"permutations {i + 1}/{n_permutations}")

        signs = rng.choice([-1.0, 1.0], size=sign_shape)
        perm_data = data * signs

        perm_t, perm_p = stats.ttest_1samp(perm_data, popmean=0.0, axis=0)

        perm_valid = np.isfinite(perm_t) & np.isfinite(perm_p)
        perm_t[~perm_valid] = 0.0
        perm_p[~perm_valid] = 1.0

        perm_p = _apply_tail(perm_p, perm_t, tail=tail)

        perm_mask = perm_p < p_threshold
        labeled_array, num_clusters = connected_cluster(perm_mask)

        if num_clusters > 0:
            cluster_sizes = np.array(
                [(labeled_array == j).sum() for j in range(1, num_clusters + 1)],
                dtype=np.int32
            )
            max_cluster_sizes_perm[i] = cluster_sizes.max()
        else:
            max_cluster_sizes_perm[i] = 0

    # 3) Cluster threshold and final significant clusters
    critical_cluster_size = np.percentile(max_cluster_sizes_perm, 100 * (1 - alpha))

    significant_mask = np.zeros_like(t_vals, dtype=bool)
    for j in range(1, num_clusters_original + 1):
        this_cluster = labeled_array_original == j
        if this_cluster.sum() >= critical_cluster_size:
            significant_mask |= this_cluster

    return significant_mask, t_vals, p_vals, critical_cluster_size


def save_nifti(t_map, p_map, significant_cluster, output_dir, prefix):
    # Nilearn version compatibility: 0.8.0 has no resolution kwarg
    try:
        template_img = load_mni152_template(resolution=2)
    except TypeError:
        template_img = load_mni152_template()

    affine = template_img.affine
    target_shape = template_img.shape

    if t_map.shape != target_shape:
        zoom_factors = np.array(target_shape) / np.array(t_map.shape)
        t_map = zoom(t_map, zoom_factors, order=1)
        p_map = zoom(p_map, zoom_factors, order=1)
        significant_cluster = zoom(significant_cluster.astype(np.float32), zoom_factors, order=0).astype(np.int16)

    masked_t_map = t_map * significant_cluster

    nib.save(nib.Nifti1Image(t_map, affine), os.path.join(output_dir, f"{prefix}_t_map.nii.gz"))
    nib.save(nib.Nifti1Image(p_map, affine), os.path.join(output_dir, f"{prefix}_p_map.nii.gz"))
    nib.save(nib.Nifti1Image(significant_cluster.astype(np.int16), affine),
             os.path.join(output_dir, f"{prefix}_significant_clusters.nii.gz"))
    nib.save(nib.Nifti1Image(masked_t_map, affine), os.path.join(output_dir, f"{prefix}_masked_t_map.nii.gz"))

    print(f"[{prefix}] saved: t_map / p_map / significant_clusters / masked_t_map")


def print_positive_t_range(t_map, sig_mask, label):
    """Print positive t-value range within significant clusters on the ORIGINAL grid."""
    pos_mask = sig_mask & (t_map > 0)
    pos_t = t_map[pos_mask]

    n_pos = int(pos_t.size)
    if n_pos == 0:
        print(f"[{label}] Positive significant voxels: 0 (no t>0 within significant clusters)")
        return

    print(f"[{label}] Positive significant voxels: {n_pos}")
    print(f"[{label}] Positive t-range (significant clusters): {float(pos_t.min())}  {float(pos_t.max())}")


if __name__ == "__main__":
    data = np.load("/data3/DLD2/241212_fmri/a9r9/class4/grad_cam/data/cam_fmri_ct4_3.npz")
    output_dir = "/data3/DLD2/241212_fmri/analysis_results_for_fsl/against_zero"
    os.makedirs(output_dir, exist_ok=True)

    data0, data1 = process_raw_data(data)

    # ========== type0 vs 0 ==========
    sig0, t0, p0, crit0 = one_sample_permutation_test(
        data0,
        p_threshold=0.001,
        n_permutations=5000,
        alpha=0.05,
        tail="two-sided",
        seed=0
    )
    print(f"[type0] critical_cluster_size = {crit0}")
    print_positive_t_range(t0, sig0, label="type0")
    save_nifti(t0, p0, sig0, output_dir, prefix="type0_vs0")

    # ========== type1 vs 0 ==========
    sig1, t1, p1, crit1 = one_sample_permutation_test(
        data1,
        p_threshold=0.001,
        n_permutations=5000,
        alpha=0.05,
        tail="two-sided",
        seed=1
    )
    print(f"[type1] critical_cluster_size = {crit1}")
    print_positive_t_range(t1, sig1, label="type1")
    save_nifti(t1, p1, sig1, output_dir, prefix="type1_vs0")
