import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats
from scipy.ndimage import label as connected_cluster
from scipy.ndimage import zoom
from nilearn.datasets import load_mni152_template

try:
    # Name used by the deployed visualisation workflow.
    from visualize_fmri_small_vs_large import (
        SMALL_LARGE_CLASSIFY_TYPE,
        _restore_volume,
        load_grad_cam_data,
        process_raw_data,
        resolve_fold,
        write_summary,
    )
except ModuleNotFoundError as error:
    if error.name != "visualize_fmri_small_vs_large":
        raise
    # Backward compatibility with the repository's earlier filename.
    from visualize_fmri import (
        SMALL_LARGE_CLASSIFY_TYPE,
        _restore_volume,
        load_grad_cam_data,
        process_raw_data,
        resolve_fold,
        write_summary,
    )


def one_sided_greater_p_values(p_two_sided, t_values):
    """Convert two-sided t-test p values to p values for ``mean > 0``.

    Non-positive t values are assigned p=1 because they cannot support the
    pre-specified positive alternative.
    """
    p_two_sided = np.asarray(p_two_sided, dtype=float)
    t_values = np.asarray(t_values, dtype=float)
    p_values = np.ones_like(p_two_sided, dtype=float)
    positive = t_values > 0
    p_values[positive] = p_two_sided[positive] / 2.0
    return p_values


def _largest_cluster_size(mask):
    """Return the size of the largest connected component in a binary mask."""
    labels, cluster_count = connected_cluster(mask)
    if cluster_count == 0:
        return 0
    return int(np.bincount(labels.ravel())[1:].max())


def one_sample_greater_permutation_test(data, brain_mask, p_threshold=0.001,
                                        n_permutations=5000, alpha=0.05,
                                        seed=0):
    """Run a one-sided, cluster-corrected test of mean attribution > 0.

    The first data dimension is the selected test-sample dimension.  Cluster
    forming and max-cluster permutation statistics are both restricted to
    positive t values, so the returned mask is directly interpretable as
    ``greater than zero``.
    """
    data = np.asarray(data, dtype=float)
    brain_mask = np.asarray(brain_mask, dtype=bool)
    if data.ndim != 4:
        raise ValueError(
            "Guided Grad-CAM data must have shape (samples, z, y, x).")
    if data.shape[0] < 2:
        raise ValueError("At least two samples are required for a t-test.")
    if tuple(data.shape[1:]) != tuple(brain_mask.shape):
        raise ValueError(
            "Guided Grad-CAM shape {} does not match brain-mask shape {}."
            .format(tuple(data.shape[1:]), tuple(brain_mask.shape)))
    if not np.any(brain_mask):
        raise ValueError("Brain mask contains no non-zero voxels.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    internal_shape = data.shape[1:]
    data_masked = data[:, brain_mask]
    t_masked, p_two_sided_masked = stats.ttest_1samp(
        data_masked, popmean=0.0, axis=0)
    valid = np.isfinite(t_masked) & np.isfinite(p_two_sided_masked)
    t_masked[~valid] = 0.0
    p_two_sided_masked[~valid] = 1.0
    p_masked = one_sided_greater_p_values(p_two_sided_masked, t_masked)

    t_values = _restore_volume(t_masked, internal_shape, brain_mask, 0.0)
    p_values = _restore_volume(p_masked, internal_shape, brain_mask, 1.0)
    initial_mask = (p_values < p_threshold) & (t_values > 0) & brain_mask
    labels, cluster_count = connected_cluster(initial_mask)

    rng = np.random.default_rng(seed)
    maximum_cluster_sizes = np.zeros(n_permutations, dtype=np.int32)
    sign_shape = (data_masked.shape[0], 1)
    for permutation_index in range(n_permutations):
        if (permutation_index + 1) % 100 == 0:
            print("Permutation {}/{}".format(
                permutation_index + 1, n_permutations))

        signs = rng.choice((-1.0, 1.0), size=sign_shape)
        permuted_data = data_masked * signs
        permuted_t, permuted_p_two_sided = stats.ttest_1samp(
            permuted_data, popmean=0.0, axis=0)
        permuted_valid = np.isfinite(permuted_t) & np.isfinite(
            permuted_p_two_sided)
        permuted_t[~permuted_valid] = 0.0
        permuted_p_two_sided[~permuted_valid] = 1.0
        permuted_p = one_sided_greater_p_values(
            permuted_p_two_sided, permuted_t)
        permuted_initial = (permuted_p < p_threshold) & (permuted_t > 0)
        maximum_cluster_sizes[permutation_index] = _largest_cluster_size(
            _restore_volume(permuted_initial, internal_shape, brain_mask, False))

    critical_cluster_size = int(np.ceil(np.percentile(
        maximum_cluster_sizes, 100.0 * (1.0 - alpha))))
    significant_mask = np.zeros_like(brain_mask, dtype=bool)
    for cluster_id in range(1, cluster_count + 1):
        cluster = labels == cluster_id
        if int(cluster.sum()) >= critical_cluster_size:
            significant_mask |= cluster

    return significant_mask, t_values, p_values, critical_cluster_size


def _positive_masked_t_map(t_values, significant_mask):
    """Return a display-ready map containing only positive significant t's."""
    return np.where(significant_mask & (t_values > 0), t_values, 0.0)


def _output_paths(output_dir, class_name):
    output_path = Path(output_dir)
    return {
        "t_map": output_path / "{}_vs_zero_t_map.nii.gz".format(class_name),
        "p_map": output_path / "{}_gt_zero_p_map.nii.gz".format(class_name),
        "greater_mask": output_path / "{}_gt_zero_mask.nii.gz".format(
            class_name),
        "masked_t_map": output_path / "{}_gt_zero_masked_t_map.nii.gz".format(
            class_name),
    }


def load_mni_template():
    """Load the MNI template used by the original against-zero workflow."""
    try:
        return load_mni152_template(resolution=2)
    except TypeError:
        return load_mni152_template()


def _resample_to_template(volume, template_shape, interpolation_order):
    """Apply the legacy shape-only resampling convention to one volume."""
    if tuple(volume.shape) == tuple(template_shape):
        return volume
    zoom_factors = np.asarray(template_shape, dtype=float) / np.asarray(
        volume.shape, dtype=float)
    output = zoom(volume, zoom_factors, order=interpolation_order)
    if tuple(output.shape) != tuple(template_shape):
        raise ValueError(
            "MNI resampling produced shape {}, expected {}."
            .format(tuple(output.shape), tuple(template_shape)))
    return output


def _save_mni_nifti(volume, template_image, output_path, dtype,
                    interpolation_order):
    volume_mni = _resample_to_template(
        volume, template_image.shape[:3], interpolation_order)
    output_image = nib.Nifti1Image(volume_mni.astype(dtype),
                                   template_image.affine)
    output_image.set_qform(
        template_image.get_qform(),
        code=template_image.get_qform(coded=True)[1])
    output_image.set_sform(
        template_image.get_sform(),
        code=template_image.get_sform(coded=True)[1])
    nib.save(output_image, str(output_path))


def save_mni_nifti_results(class_name, t_values, p_values, significant_mask,
                           template_image, output_dir):
    """Save one condition's maps on the automatic MNI template grid."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(output_path, class_name)
    positive_masked_t_values = _positive_masked_t_map(
        t_values, significant_mask)
    _save_mni_nifti(t_values, template_image, outputs["t_map"], np.float32,
                    interpolation_order=1)
    _save_mni_nifti(p_values, template_image, outputs["p_map"], np.float32,
                    interpolation_order=1)
    _save_mni_nifti(significant_mask, template_image,
                    outputs["greater_mask"], np.uint8,
                    interpolation_order=0)
    _save_mni_nifti(positive_masked_t_values, template_image,
                    outputs["masked_t_map"], np.float32,
                    interpolation_order=1)
    return outputs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create one-sided, cluster-corrected Small > 0 and Large > 0 "
            "Guided Grad-CAM maps for one CT4 fMRI model."))
    parser.add_argument(
        "--result_csv",
        default=None,
        help=("Path to result_ct4_raw_test.csv. Required when --fold=best; "
              "used to record the selected accuracy otherwise."))
    parser.add_argument(
        "--grad_cam_dir",
        required=True,
        help="Directory containing cam_fmri_ct4_{fold}.npz files.")
    parser.add_argument(
        "--fold",
        required=True,
        help=("Explicit fold ID, or 'best' for the highest test accuracy "
              "(requires --result_csv)."))
    parser.add_argument(
        "--classify_type",
        type=int,
        default=SMALL_LARGE_CLASSIFY_TYPE,
        help="Classification type; CT4 is Small/Large (default: 4).")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for NIfTI outputs and analysis_summary.json.")
    parser.add_argument(
        "--all_trials",
        action="store_true",
        help="Use all test trials instead of only correctly classified trials.")
    parser.add_argument("--p_threshold", type=float, default=0.001)
    parser.add_argument("--n_permutations", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.classify_type != SMALL_LARGE_CLASSIFY_TYPE:
        raise SystemExit(
            "Error: This script is specific to CT4 Small/Large; received "
            "--classify_type={}.".format(args.classify_type))
    if not 0.0 < args.p_threshold < 1.0:
        raise SystemExit("Error: --p_threshold must be between 0 and 1.")
    if args.n_permutations < 1:
        raise SystemExit("Error: --n_permutations must be at least 1.")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("Error: --alpha must be between 0 and 1.")

    try:
        fold, selected_accuracy, fold_accuracies = resolve_fold(
            args.fold, args.result_csv)
        data, grad_cam_path = load_grad_cam_data(
            args.grad_cam_dir, args.classify_type, fold)
        data_small, data_large, sample_counts = process_raw_data(
            data, only_correct=not args.all_trials)
        brain_mask = np.ones(data_small.shape[1:], dtype=bool)
        template_image = load_mni_template()

        (small_mask, small_t_values, small_p_values,
         small_critical_size) = one_sample_greater_permutation_test(
             data_small, brain_mask, p_threshold=args.p_threshold,
             n_permutations=args.n_permutations, alpha=args.alpha,
             seed=args.seed)
        (large_mask, large_t_values, large_p_values,
         large_critical_size) = one_sample_greater_permutation_test(
             data_large, brain_mask, p_threshold=args.p_threshold,
             n_permutations=args.n_permutations, alpha=args.alpha,
             seed=args.seed + 1)

        small_outputs = save_mni_nifti_results(
            "small", small_t_values, small_p_values, small_mask,
            template_image, args.output_dir)
        large_outputs = save_mni_nifti_results(
            "large", large_t_values, large_p_values, large_mask,
            template_image, args.output_dir)
        summary = {
            "analysis_type": "representative_single_test_selected_model",
            "classify_type": args.classify_type,
            "hypothesis": "one_sided_mean_guided_grad_cam_greater_than_zero",
            "tail": "greater",
            "fold": int(fold),
            "selection_rule": (
                "highest test accuracy; ties resolved by lower fold ID"
                if str(args.fold).lower() == "best" else "explicit fold ID"),
            "selected_test_accuracy": selected_accuracy,
            "candidate_test_accuracies": {
                str(key): value for key, value in sorted(fold_accuracies.items())},
            "grad_cam_file": str(grad_cam_path),
            "p_threshold": args.p_threshold,
            "n_permutations": args.n_permutations,
            "alpha": args.alpha,
            "small_seed": args.seed,
            "large_seed": args.seed + 1,
            "small_critical_cluster_size": small_critical_size,
            "large_critical_cluster_size": large_critical_size,
            "small_gt_zero_voxels": int(small_mask.sum()),
            "large_gt_zero_voxels": int(large_mask.sum()),
            "output_files": {
                "small": {key: str(value) for key, value in small_outputs.items()},
                "large": {key: str(value) for key, value in large_outputs.items()},
            },
        }
        summary.update(sample_counts)
        summary.update({
            "spatial_mode": "legacy_mni_shape_only",
            "reference_nii": None,
            "brain_mask": None,
            "uses_external_brain_mask": False,
            "spatial_warning": (
                "Maps were shape-only resampled to the Nilearn MNI152 "
                "template, matching the original against-zero workflow."),
        })
        summary_path = write_summary(args.output_dir, summary)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit("Error: {}".format(error))

    print("Selected fold: {}".format(fold))
    if selected_accuracy is not None:
        print("Selected test accuracy: {:.6f}".format(selected_accuracy))
    print("Small > 0 significant voxels: {}".format(int(small_mask.sum())))
    print("Large > 0 significant voxels: {}".format(int(large_mask.sum())))
    print("Saved analysis summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
