import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label as connected_cluster
from scipy.ndimage import zoom


SMALL_LARGE_CLASSIFY_TYPE = 4
SMALL_LABEL = 0
LARGE_LABEL = 1
LEGACY_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "guided_grad_cam" / "t_map.nii.gz")


def read_fold_accuracies(result_csv):
    """Return a {fold: test_accuracy} mapping from a raw test-result CSV."""
    result_path = Path(result_csv)
    if not result_path.is_file():
        raise FileNotFoundError("Result CSV does not exist: {}".format(result_path))

    result_frame = pd.read_csv(result_path, index_col=0)
    if "accuracy" not in result_frame.columns:
        raise ValueError(
            "Result CSV must contain an 'accuracy' column; found {}."
            .format(", ".join(str(column) for column in result_frame.columns)))

    accuracies = pd.to_numeric(result_frame["accuracy"], errors="coerce")
    if accuracies.isna().any():
        bad_rows = [str(row) for row in accuracies.index[accuracies.isna()]]
        raise ValueError(
            "The accuracy column contains a missing or non-numeric value for "
            "fold row(s): {}.".format(", ".join(bad_rows)))

    fold_accuracies = {}
    for row_index, accuracy in accuracies.items():
        try:
            fold = int(row_index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Raw test-result row index {!r} is not an integer fold ID."
                .format(row_index)) from error
        if fold < 0:
            raise ValueError("Fold IDs must be non-negative; found {}.".format(fold))
        if fold in fold_accuracies:
            raise ValueError("Duplicate fold ID {} in result CSV.".format(fold))
        fold_accuracies[fold] = float(accuracy)

    if not fold_accuracies:
        raise ValueError("Result CSV contains no fold accuracy rows.")
    return fold_accuracies


def select_best_fold(result_csv):
    """Choose the highest-accuracy fold; use the lower ID to resolve ties."""
    fold_accuracies = read_fold_accuracies(result_csv)
    highest_accuracy = max(fold_accuracies.values())
    best_fold = min(
        fold for fold, accuracy in fold_accuracies.items()
        if accuracy == highest_accuracy)
    return best_fold, highest_accuracy, fold_accuracies


def resolve_fold(fold_argument, result_csv):
    """Resolve ``best`` or an explicit fold command-line argument."""
    if str(fold_argument).lower() == "best":
        if result_csv is None:
            raise ValueError("--result_csv is required when --fold=best.")
        return select_best_fold(result_csv)

    try:
        fold = int(fold_argument)
    except ValueError as error:
        raise ValueError("--fold must be 'best' or a non-negative integer.") from error
    if fold < 0:
        raise ValueError("--fold must be non-negative.")

    fold_accuracies = read_fold_accuracies(result_csv) if result_csv else {}
    selected_accuracy = fold_accuracies.get(fold)
    return fold, selected_accuracy, fold_accuracies


def load_grad_cam_data(grad_cam_dir, classify_type, fold):
    """Load the Grad-CAM archive selected for this visualisation."""
    archive_path = Path(grad_cam_dir) / "cam_fmri_ct{}_{}.npz".format(
        classify_type, fold)
    if not archive_path.is_file():
        raise FileNotFoundError("Grad-CAM file does not exist: {}".format(archive_path))

    required_keys = {
        "label", "predicted_label", "guided_cam_nopool0", "guided_cam_nopool1"
    }
    with np.load(archive_path) as archive:
        missing_keys = sorted(required_keys - set(archive.files))
        if missing_keys:
            raise ValueError(
                "Grad-CAM archive is missing key(s): {}.".format(
                    ", ".join(missing_keys)))
        data = {key: archive[key] for key in required_keys}
    return data, archive_path


def process_raw_data(data, only_correct=True):
    """Extract Small and Large maps, using the target matching each true label."""
    labels = np.asarray(data["label"]).reshape(-1).astype(np.int32)
    predicted_labels = np.asarray(data["predicted_label"]).reshape(-1).astype(np.int32)
    small_maps = np.asarray(data["guided_cam_nopool0"])
    large_maps = np.asarray(data["guided_cam_nopool1"])

    if len(labels) != len(predicted_labels) or \
            len(labels) != len(small_maps) or len(labels) != len(large_maps):
        raise ValueError("Grad-CAM labels and attribution-map counts do not match.")
    if small_maps.ndim != 4 or large_maps.ndim != 4:
        raise ValueError(
            "Guided Grad-CAM arrays must have shape (samples, z, y, x).")
    if small_maps.shape[1:] != large_maps.shape[1:]:
        raise ValueError("Small and Large Guided Grad-CAM map shapes do not match.")

    small_selection = labels == SMALL_LABEL
    large_selection = labels == LARGE_LABEL
    if only_correct:
        small_selection &= predicted_labels == SMALL_LABEL
        large_selection &= predicted_labels == LARGE_LABEL

    data_small = small_maps[small_selection]
    data_large = large_maps[large_selection]
    if len(data_small) < 2 or len(data_large) < 2:
        raise ValueError(
            "At least two Small and two Large maps are required after sample "
            "selection; found Small={}, Large={}.".format(
                len(data_small), len(data_large)))

    return data_small, data_large, {
        "total_small_trials": int(np.sum(labels == SMALL_LABEL)),
        "total_large_trials": int(np.sum(labels == LARGE_LABEL)),
        "included_small_trials": int(len(data_small)),
        "included_large_trials": int(len(data_large)),
        "only_correct": bool(only_correct),
    }


def load_reference_and_mask(reference_nii, brain_mask, internal_shape):
    """Load a NIfTI reference and convert its mask to model (z, y, x) order."""
    reference_path = Path(reference_nii)
    if not reference_path.is_file():
        raise FileNotFoundError("Reference NIfTI does not exist: {}".format(reference_path))

    reference_image = nib.load(str(reference_path))
    reference_shape = tuple(reference_image.shape[:3])  # NIfTI order: (x, y, z)
    expected_internal_shape = tuple(reversed(reference_shape))  # model: (z, y, x)
    if tuple(internal_shape) != expected_internal_shape:
        raise ValueError(
            "Grad-CAM internal shape {} does not match the reference NIfTI "
            "shape {} after (z, y, x) -> (x, y, z) conversion."
            .format(tuple(internal_shape), reference_shape))

    mask_path = Path(brain_mask)
    if not mask_path.is_file():
        raise FileNotFoundError("Brain mask does not exist: {}".format(mask_path))

    if mask_path.suffix.lower() == ".npz":
        with np.load(mask_path) as mask_archive:
            if "mask" not in mask_archive.files:
                raise ValueError("NPZ brain mask must contain a 'mask' array.")
            mask_internal = np.asarray(mask_archive["mask"], dtype=bool)
        if mask_internal.shape != expected_internal_shape:
            raise ValueError(
                "NPZ mask has internal shape {}, but expected {}."
                .format(mask_internal.shape, expected_internal_shape))
    else:
        mask_image = nib.load(str(mask_path))
        if tuple(mask_image.shape[:3]) != reference_shape:
            raise ValueError(
                "NIfTI brain mask shape {} does not match reference shape {}."
                .format(tuple(mask_image.shape[:3]), reference_shape))
        mask_xyz = np.asarray(mask_image.get_fdata(), dtype=float) > 0
        mask_internal = np.transpose(mask_xyz, (2, 1, 0))

    if not np.any(mask_internal):
        raise ValueError("Brain mask contains no non-zero voxels.")
    return reference_image, mask_internal


def load_legacy_reference(internal_shape, template_path=LEGACY_TEMPLATE_PATH):
    """Load the bundled legacy MNI grid and use all model voxels for testing.

    This reproduces the old visualisation workflow's spatial assumptions.  It
    is intentionally separate from the reference-and-mask workflow because it
    has no subject-specific affine or externally supplied brain mask.
    """
    template_path = Path(template_path)
    if not template_path.is_file():
        raise FileNotFoundError(
            "Legacy MNI template does not exist: {}. Copy "
            "guided_grad_cam/t_map.nii.gz into the project or use "
            "--reference_nii with --brain_mask instead.".format(template_path))
    if len(internal_shape) != 3:
        raise ValueError(
            "Guided Grad-CAM maps must have three spatial dimensions; found {}."
            .format(tuple(internal_shape)))

    return nib.load(str(template_path)), np.ones(internal_shape, dtype=bool)


def _restore_volume(values, internal_shape, mask_internal, outside_value):
    """Restore masked vector data to a full model-space (z, y, x) volume."""
    volume = np.full(internal_shape, outside_value, dtype=float)
    volume[mask_internal] = values
    return volume


def _cluster_sizes(mask):
    labeled_mask, cluster_count = connected_cluster(mask)
    if cluster_count == 0:
        return labeled_mask, np.empty(0, dtype=np.int32)
    cluster_sizes = np.bincount(labeled_mask.ravel())[1:].astype(np.int32)
    return labeled_mask, cluster_sizes


def signed_cluster_masks(t_values, p_values, brain_mask, p_threshold, critical_size):
    """Return separately thresholded positive and negative cluster masks."""
    positive_initial = (p_values < p_threshold) & (t_values > 0) & brain_mask
    negative_initial = (p_values < p_threshold) & (t_values < 0) & brain_mask
    positive_labels, _ = _cluster_sizes(positive_initial)
    negative_labels, _ = _cluster_sizes(negative_initial)

    positive_mask = np.zeros_like(brain_mask, dtype=bool)
    negative_mask = np.zeros_like(brain_mask, dtype=bool)
    for labels, final_mask in ((positive_labels, positive_mask),
                               (negative_labels, negative_mask)):
        for cluster_id in range(1, int(labels.max()) + 1):
            cluster = labels == cluster_id
            if int(cluster.sum()) >= critical_size:
                final_mask |= cluster
    return positive_mask, negative_mask


def permutation_test(data_small, data_large, brain_mask, p_threshold=0.001,
                     n_permutations=5000, alpha=0.05, seed=0):
    """Two-sided, signed-cluster label-permutation test for Small minus Large.

    ``data_small`` and ``data_large`` are model-space arrays in (z, y, x)
    order.  The returned t and p maps use that same order.
    """
    if data_small.shape[1:] != data_large.shape[1:]:
        raise ValueError("Small and Large attribution-map shapes do not match.")
    if data_small.shape[1:] != brain_mask.shape:
        raise ValueError("Brain-mask shape does not match attribution-map shape.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    internal_shape = data_small.shape[1:]
    small_masked = data_small[:, brain_mask]
    large_masked = data_large[:, brain_mask]
    t_masked, p_masked = stats.ttest_ind(
        small_masked, large_masked, axis=0, equal_var=False)
    valid = np.isfinite(t_masked) & np.isfinite(p_masked)
    t_masked[~valid] = 0.0
    p_masked[~valid] = 1.0

    t_values = _restore_volume(t_masked, internal_shape, brain_mask, 0.0)
    p_values = _restore_volume(p_masked, internal_shape, brain_mask, 1.0)

    combined = np.concatenate((small_masked, large_masked), axis=0)
    small_count = len(small_masked)
    total_count = len(combined)
    rng = np.random.RandomState(seed)
    maximum_cluster_sizes = np.zeros(n_permutations, dtype=np.int32)

    for permutation_index in range(n_permutations):
        if (permutation_index + 1) % 100 == 0:
            print("Permutation {}/{}".format(
                permutation_index + 1, n_permutations))

        permutation = rng.permutation(total_count)
        perm_small = combined[permutation[:small_count]]
        perm_large = combined[permutation[small_count:]]
        perm_t, perm_p = stats.ttest_ind(
            perm_small, perm_large, axis=0, equal_var=False)
        perm_valid = np.isfinite(perm_t) & np.isfinite(perm_p)
        perm_t[~perm_valid] = 0.0
        perm_p[~perm_valid] = 1.0

        positive = (perm_p < p_threshold) & (perm_t > 0)
        negative = (perm_p < p_threshold) & (perm_t < 0)
        _, positive_sizes = _cluster_sizes(_restore_volume(
            positive, internal_shape, brain_mask, False))
        _, negative_sizes = _cluster_sizes(_restore_volume(
            negative, internal_shape, brain_mask, False))
        all_sizes = np.concatenate((positive_sizes, negative_sizes))
        if len(all_sizes) > 0:
            maximum_cluster_sizes[permutation_index] = int(all_sizes.max())

    critical_cluster_size = int(np.ceil(np.percentile(
        maximum_cluster_sizes, 100.0 * (1.0 - alpha))))
    small_greater_mask, large_greater_mask = signed_cluster_masks(
        t_values, p_values, brain_mask, p_threshold, critical_cluster_size)
    return (small_greater_mask, large_greater_mask, t_values, p_values,
            critical_cluster_size)


def _to_nifti_orientation(volume_internal):
    """Convert a model volume from (z, y, x) to NIfTI (x, y, z) order."""
    return np.transpose(volume_internal, (2, 1, 0))


def _save_nifti(volume_internal, reference_image, output_path, dtype):
    volume_xyz = _to_nifti_orientation(volume_internal).astype(dtype)
    expected_shape = tuple(reference_image.shape[:3])
    if volume_xyz.shape != expected_shape:
        raise ValueError(
            "Output volume shape {} does not match reference shape {}."
            .format(volume_xyz.shape, expected_shape))
    output_image = nib.Nifti1Image(volume_xyz, reference_image.affine)
    output_image.set_qform(reference_image.get_qform(), code=reference_image.get_qform(coded=True)[1])
    output_image.set_sform(reference_image.get_sform(), code=reference_image.get_sform(coded=True)[1])
    nib.save(output_image, str(output_path))


def _legacy_resample(volume_internal, target_shape, interpolation_order):
    """Resample a model-space volume with the legacy shape-only convention."""
    zoom_factors = np.asarray(target_shape, dtype=float) / np.asarray(
        volume_internal.shape, dtype=float)
    resampled = zoom(volume_internal, zoom_factors, order=interpolation_order)
    if tuple(resampled.shape) != tuple(target_shape):
        raise ValueError(
            "Legacy resampling produced shape {}, expected {}."
            .format(tuple(resampled.shape), tuple(target_shape)))
    return resampled


def _save_legacy_nifti(volume_internal, reference_image, output_path, dtype,
                       interpolation_order):
    """Write a legacy-resampled volume on the bundled MNI template grid."""
    volume_mni = _legacy_resample(
        volume_internal, reference_image.shape[:3], interpolation_order)
    output_image = nib.Nifti1Image(volume_mni.astype(dtype), reference_image.affine)
    output_image.set_qform(reference_image.get_qform(),
                           code=reference_image.get_qform(coded=True)[1])
    output_image.set_sform(reference_image.get_sform(),
                           code=reference_image.get_sform(coded=True)[1])
    nib.save(output_image, str(output_path))


def save_nifti_results(t_values, p_values, small_greater_mask,
                       large_greater_mask, reference_image, output_dir):
    """Write signed t maps and directional masks in reference NIfTI space."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    significant_mask = small_greater_mask | large_greater_mask
    masked_t_values = t_values * significant_mask
    outputs = {
        "t_map": output_path / "small_minus_large_t_map.nii.gz",
        "p_map": output_path / "small_minus_large_p_map.nii.gz",
        "small_greater_mask": output_path / "small_gt_large_mask.nii.gz",
        "large_greater_mask": output_path / "large_gt_small_mask.nii.gz",
        "masked_t_map": output_path / "small_minus_large_masked_t_map.nii.gz",
    }
    _save_nifti(t_values, reference_image, outputs["t_map"], np.float32)
    _save_nifti(p_values, reference_image, outputs["p_map"], np.float32)
    _save_nifti(small_greater_mask, reference_image,
                outputs["small_greater_mask"], np.uint8)
    _save_nifti(large_greater_mask, reference_image,
                outputs["large_greater_mask"], np.uint8)
    _save_nifti(masked_t_values, reference_image,
                outputs["masked_t_map"], np.float32)
    return outputs


def save_legacy_nifti_results(t_values, p_values, small_greater_mask,
                              large_greater_mask, reference_image, output_dir):
    """Write results using the legacy MNI shape-only spatial convention."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    significant_mask = small_greater_mask | large_greater_mask
    masked_t_values = t_values * significant_mask
    outputs = {
        "t_map": output_path / "small_minus_large_t_map.nii.gz",
        "p_map": output_path / "small_minus_large_p_map.nii.gz",
        "small_greater_mask": output_path / "small_gt_large_mask.nii.gz",
        "large_greater_mask": output_path / "large_gt_small_mask.nii.gz",
        "masked_t_map": output_path / "small_minus_large_masked_t_map.nii.gz",
    }
    _save_legacy_nifti(t_values, reference_image, outputs["t_map"],
                       np.float32, interpolation_order=1)
    _save_legacy_nifti(p_values, reference_image, outputs["p_map"],
                       np.float32, interpolation_order=1)
    _save_legacy_nifti(small_greater_mask, reference_image,
                       outputs["small_greater_mask"], np.uint8,
                       interpolation_order=0)
    _save_legacy_nifti(large_greater_mask, reference_image,
                       outputs["large_greater_mask"], np.uint8,
                       interpolation_order=0)
    _save_legacy_nifti(masked_t_values, reference_image,
                       outputs["masked_t_map"], np.float32,
                       interpolation_order=1)
    return outputs


def resolve_spatial_context(reference_nii, brain_mask, legacy_mni,
                            internal_shape, template_path=LEGACY_TEMPLATE_PATH):
    """Select either strict reference/mask space or explicit legacy MNI mode."""
    has_reference = reference_nii is not None
    has_mask = brain_mask is not None

    if legacy_mni:
        if has_reference or has_mask:
            raise ValueError(
                "--legacy_mni cannot be combined with --reference_nii or "
                "--brain_mask.")
        reference_image, mask_internal = load_legacy_reference(
            internal_shape, template_path=template_path)
        return reference_image, mask_internal, {
            "spatial_mode": "legacy_mni_assumed",
            "reference_nii": None,
            "brain_mask": None,
            "legacy_template_nii": str(template_path),
            "uses_external_brain_mask": False,
            "spatial_warning": (
                "Legacy MNI shape-only convention; use a real reference NIfTI "
                "and brain mask for spatially precise localisation."),
        }

    if not has_reference or not has_mask:
        raise ValueError(
            "Provide both --reference_nii and --brain_mask, or use "
            "--legacy_mni when those files are unavailable.")

    reference_image, mask_internal = load_reference_and_mask(
        reference_nii, brain_mask, internal_shape)
    return reference_image, mask_internal, {
        "spatial_mode": "reference_and_mask",
        "reference_nii": str(reference_nii),
        "brain_mask": str(brain_mask),
        "legacy_template_nii": None,
        "uses_external_brain_mask": True,
        "spatial_warning": None,
    }


def write_summary(output_dir, summary):
    summary_path = Path(output_dir) / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create a signed Small-minus-Large Guided Grad-CAM map for one "
            "CT4 fMRI model. The selected model is a representative display."))
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
        default="best",
        help="Fold ID or 'best' for the highest test accuracy (default: best).")
    parser.add_argument(
        "--classify_type",
        type=int,
        default=SMALL_LARGE_CLASSIFY_TYPE,
        help="Classification type; CT4 is Small/Large (default: 4).")
    parser.add_argument(
        "--reference_nii",
        default=None,
        help="Preprocessed fMRI NIfTI in the same spatial grid as the model input.")
    parser.add_argument(
        "--brain_mask",
        default=None,
        help=("Binary brain mask as a NIfTI file or an internal (z,y,x) NPZ "
              "mask with key 'mask'."))
    parser.add_argument(
        "--legacy_mni",
        action="store_true",
        help=("Use the bundled guided_grad_cam/t_map.nii.gz MNI grid and all "
              "model voxels. This reproduces the legacy spatial convention."))
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
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("Error: --alpha must be between 0 and 1.")

    try:
        fold, selected_accuracy, fold_accuracies = resolve_fold(
            args.fold, args.result_csv)
        data, grad_cam_path = load_grad_cam_data(
            args.grad_cam_dir, args.classify_type, fold)
        data_small, data_large, sample_counts = process_raw_data(
            data, only_correct=not args.all_trials)
        reference_image, brain_mask, spatial_metadata = resolve_spatial_context(
            args.reference_nii, args.brain_mask, args.legacy_mni,
            data_small.shape[1:])
        (small_greater_mask, large_greater_mask, t_values, p_values,
         critical_cluster_size) = permutation_test(
            data_small, data_large, brain_mask,
            p_threshold=args.p_threshold,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            seed=args.seed)
        if args.legacy_mni:
            output_paths = save_legacy_nifti_results(
                t_values, p_values, small_greater_mask, large_greater_mask,
                reference_image, args.output_dir)
        else:
            output_paths = save_nifti_results(
                t_values, p_values, small_greater_mask, large_greater_mask,
                reference_image, args.output_dir)
        summary = {
            "analysis_type": "representative_single_test_selected_model",
            "classify_type": args.classify_type,
            "contrast": "Small - Large",
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
            "seed": args.seed,
            "critical_cluster_size": critical_cluster_size,
            "small_gt_large_voxels": int(small_greater_mask.sum()),
            "large_gt_small_voxels": int(large_greater_mask.sum()),
            "output_files": {key: str(value) for key, value in output_paths.items()},
        }
        summary.update(spatial_metadata)
        summary.update(sample_counts)
        summary_path = write_summary(args.output_dir, summary)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit("Error: {}".format(error))

    print("Selected fold: {}".format(fold))
    if selected_accuracy is not None:
        print("Selected test accuracy: {:.6f}".format(selected_accuracy))
    print("Small > Large significant voxels: {}".format(
        int(small_greater_mask.sum())))
    print("Large > Small significant voxels: {}".format(
        int(large_greater_mask.sum())))
    print("Saved analysis summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
