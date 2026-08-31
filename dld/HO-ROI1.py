import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection


ALL_FOLD_COUNT = 9
REQUIRED_NPZ_KEYS = (
    'label',
    'predicted_label',
    'guided_cam_nopool0',
    'guided_cam_nopool1',
)
CLASS_INFOS = (
    {
        'name': 'small',
        'display_name': 'Small',
        'label': 0,
        'cam_key': 'guided_cam_nopool0',
    },
    {
        'name': 'large',
        'display_name': 'Large',
        'label': 1,
        'cam_key': 'guided_cam_nopool1',
    },
)


def map_region_to_category(region_name):
    if any(area in region_name for area in
           ['Temporal', 'MTG', 'STG', 'ITG', 'TP', 'Heschl', 'TPO']):
        return 'Temporal'
    elif any(area in region_name for area in
             ['Occipital', 'LOC', 'TOF', 'Visual', 'Lingual', 'Cuneal',
              'Calcarine']):
        return 'Occipital'
    elif any(area in region_name for area in
             ['Frontal', 'IFG', 'MFG', 'SFG', 'OFC', 'Precentral']):
        return 'Frontal'
    elif any(area in region_name for area in
             ['Parietal', 'SPL', 'IPL', 'SMG', 'Angular', 'Postcentral']):
        return 'Parietal'
    elif any(area in region_name for area in
             ['Cingulate', 'Paracingulate', 'Insular', 'Insula']):
        return 'Limbic'
    elif any(area in region_name for area in
             ['Putamen', 'Caudate', 'Thalamus', 'Pallidum', 'Accumbens',
              'Amygdala']):
        return 'Subcortical'
    elif 'Cerebellum' in region_name:
        return 'Cerebellum'
    elif 'Brain-Stem' in region_name:
        return 'Brain-Stem'
    elif 'Cortical' in region_name:
        return 'Cortical-Other'
    return 'Other'


def load_harvard_oxford_atlas():
    """Load separate cortical/subcortical atlases using the original layout."""
    ho_cortical = fetch_atlas_harvard_oxford(
        'cort-maxprob-thr25-2mm', symmetric_split=True)
    ho_subcortical = fetch_atlas_harvard_oxford(
        'sub-maxprob-thr25-2mm', symmetric_split=True)

    cortical_labels = ho_cortical['labels'][1:]
    subcortical_labels = ho_subcortical['labels'][1:]
    max_cortical_id = len(cortical_labels)

    formatted_labels = []
    region_types = []
    detailed_region_types = []

    for label in cortical_labels:
        formatted_name = _format_lateralized_region_name(label)
        formatted_labels.append(formatted_name)
        region_types.append('Cortical')
        detailed_region_types.append(map_region_to_category(formatted_name))

    for label in subcortical_labels:
        formatted_name = _format_lateralized_region_name(label)
        formatted_labels.append(formatted_name)
        region_types.append('Subcortical')
        detailed_region_types.append(map_region_to_category(formatted_name))

    region_labels = pd.DataFrame({
        'Region_ID': list(range(1, len(formatted_labels) + 1)),
        'Region': formatted_labels,
        'Type': region_types,
        'Detailed_Type': detailed_region_types,
    })
    print('The Harvard-Oxford Atlas has loaded {} cortical and {} '
          'subcortical regions ({} total).'.format(
              len(cortical_labels), len(subcortical_labels),
              len(region_labels)))
    return {
        'cortical_img': ho_cortical['maps'],
        'subcortical_img': ho_subcortical['maps'],
        'region_labels': region_labels,
        'max_cortical_id': max_cortical_id,
        'subcortical_count': len(subcortical_labels),
    }


def _format_lateralized_region_name(label):
    if 'Left' in label:
        return 'L. {}'.format(label.replace('Left ', ''))
    if 'Right' in label:
        return 'R. {}'.format(label.replace('Right ', ''))
    return label


def resolve_folds(fold_argument):
    """Resolve one fold ID or the fixed nine-fold CT4 experiment."""
    if fold_argument == 'all':
        return list(range(ALL_FOLD_COUNT))
    try:
        fold = int(fold_argument)
    except ValueError:
        raise ValueError('--fold must be an integer from 0 to 8, or all.')
    if fold < 0 or fold >= ALL_FOLD_COUNT:
        raise ValueError('--fold={} is outside the valid range 0-8.'.format(fold))
    return [fold]


def get_grad_cam_path(grad_cam_dir, classify_type, fold):
    return os.path.join(
        grad_cam_dir, 'cam_fmri_ct{}_{}.npz'.format(classify_type, fold))


def _read_labels_and_predictions(npz_path, fold):
    """Read and validate the lightweight metadata needed for preflight."""
    try:
        with np.load(npz_path, allow_pickle=False) as learned_data:
            missing_keys = [key for key in REQUIRED_NPZ_KEYS
                            if key not in learned_data.files]
            if missing_keys:
                raise ValueError(
                    'Fold {} file {} is missing required key(s): {}.'.format(
                        fold, npz_path, ', '.join(missing_keys)))
            labels = np.asarray(learned_data['label']).reshape(-1)
            predicted_labels = np.asarray(
                learned_data['predicted_label']).reshape(-1)
    except (IOError, OSError, ValueError) as error:
        raise ValueError('Could not read fold {} file {}: {}'.format(
            fold, npz_path, error))

    if labels.size == 0:
        raise ValueError('Fold {} file {} has no trials.'.format(fold, npz_path))
    if labels.size != predicted_labels.size:
        raise ValueError(
            'Fold {} file {} has {} labels but {} predicted labels.'.format(
                fold, npz_path, labels.size, predicted_labels.size))
    return labels, predicted_labels


def preflight_input_files(grad_cam_dir, classify_type, folds,
                          all_trials=False):
    """Check all requested files before any ROI output is written."""
    input_paths = {}
    missing_paths = []
    for fold in folds:
        input_path = get_grad_cam_path(grad_cam_dir, classify_type, fold)
        input_paths[fold] = input_path
        if not os.path.isfile(input_path):
            missing_paths.append(input_path)

    if missing_paths:
        raise FileNotFoundError(
            'Required Grad-CAM file(s) were not found:\n{}.'.format(
                '\n'.join(missing_paths)))

    success_counts = {}
    for fold in folds:
        labels, predicted_labels = _read_labels_and_predictions(
            input_paths[fold], fold)
        fold_counts = {}
        for class_info in CLASS_INFOS:
            selection = labels == class_info['label']
            if not all_trials:
                selection &= predicted_labels == class_info['label']
            selected_count = int(np.sum(selection))
            if selected_count == 0:
                qualifier = '' if all_trials else 'successful '
                raise ValueError(
                    'Fold {} has no {}{} trials; cannot calculate an ROI '
                    'table.'.format(
                        fold, qualifier, class_info['display_name']))
            fold_counts[class_info['name']] = selected_count
        success_counts[fold] = fold_counts

    return input_paths, success_counts


def load_gradcam_data(learned_data_path, fold, all_trials=False):
    """Load one fold and select true-label trials for both target maps.

    Correct predictions are retained by default.  ``all_trials=True`` keeps
    every trial of the corresponding true class, irrespective of prediction.
    """
    labels, predicted_labels = _read_labels_and_predictions(
        learned_data_path, fold)

    try:
        with np.load(learned_data_path, allow_pickle=False) as learned_data:
            selected_data = {}
            for class_info in CLASS_INFOS:
                cam_data = np.asarray(learned_data[class_info['cam_key']])
                if cam_data.ndim != 4:
                    raise ValueError(
                        '{} must have shape (samples, x, y, z); found {}.'.format(
                            class_info['cam_key'], cam_data.shape))
                if cam_data.shape[0] != labels.size:
                    raise ValueError(
                        '{} has {} samples but label has {} samples.'.format(
                            class_info['cam_key'], cam_data.shape[0], labels.size))

                selection = labels == class_info['label']
                if not all_trials:
                    selection &= predicted_labels == class_info['label']
                selected_data[class_info['name']] = cam_data[selection]
    except (IOError, OSError, ValueError) as error:
        raise ValueError('Could not load Guided Grad-CAM for fold {}: {}'.format(
            fold, error))

    return selected_data


def _validate_atlas_ids(atlas_data, expected_count, atlas_name):
    """Require atlas labels 1..N so table rows and image IDs cannot drift."""
    observed_ids = sorted(
        int(value) for value in np.unique(atlas_data) if value > 0)
    expected_ids = list(range(1, expected_count + 1))
    if observed_ids != expected_ids:
        missing_ids = sorted(set(expected_ids).difference(observed_ids))
        extra_ids = sorted(set(observed_ids).difference(expected_ids))
        parts = []
        if missing_ids:
            parts.append('missing {}'.format(
                ', '.join(str(value) for value in missing_ids)))
        if extra_ids:
            parts.append('unexpected {}'.format(
                ', '.join(str(value) for value in extra_ids)))
        raise ValueError(
            '{} atlas ROI IDs do not match its labels: {}.'.format(
                atlas_name, '; '.join(parts)))


def prepare_resampled_roi_masks(atlas_info, target_shape):
    """Resample every atlas ROI mask once into the Grad-CAM voxel grid.

    This restores the original analysis direction: atlas mask -> Grad-CAM
    grid.  Flat voxel indices are cached instead of full boolean volumes to
    reduce memory and make reuse across trials inexpensive.
    """
    target_shape = tuple(int(value) for value in target_shape)
    if len(target_shape) != 3 or any(value <= 0 for value in target_shape):
        raise ValueError(
            'Grad-CAM target shape must contain three positive dimensions; '
            'found {}.'.format(target_shape))

    cortical_img = atlas_info['cortical_img']
    subcortical_img = atlas_info['subcortical_img']
    region_labels = atlas_info['region_labels']
    cortical_count = int(atlas_info['max_cortical_id'])
    subcortical_count = int(atlas_info['subcortical_count'])

    if len(region_labels) != cortical_count + subcortical_count:
        raise ValueError(
            'Atlas metadata contains {} region labels but {} cortical + {} '
            'subcortical regions were declared.'.format(
                len(region_labels), cortical_count, subcortical_count))

    cortical_data = cortical_img.get_fdata()
    subcortical_data = subcortical_img.get_fdata()
    _validate_atlas_ids(cortical_data, cortical_count, 'Cortical')
    _validate_atlas_ids(subcortical_data, subcortical_count, 'Subcortical')

    target_img = nib.Nifti1Image(
        np.zeros(target_shape, dtype=np.uint8), cortical_img.affine)
    flat_indices = []
    voxel_counts = []
    atlas_entries = (
        ('Cortical', cortical_img, cortical_data, cortical_count, 0),
        ('Subcortical', subcortical_img, subcortical_data,
         subcortical_count, cortical_count),
    )

    for atlas_name, atlas_img, atlas_data, roi_count, row_offset in atlas_entries:
        for atlas_id in range(1, roi_count + 1):
            row_index = row_offset + atlas_id - 1
            region_name = str(region_labels.iloc[row_index]['Region'])
            region_mask_img = nib.Nifti1Image(
                (atlas_data == atlas_id).astype(np.uint8), atlas_img.affine)
            resampled_mask = resample_to_img(
                region_mask_img, target_img,
                interpolation='nearest').get_fdata() > 0.5
            indices = np.flatnonzero(resampled_mask.reshape(-1))
            if indices.size == 0:
                raise ValueError(
                    '{} ROI {} ({}) contains no voxels after resampling to '
                    'Grad-CAM shape {}.'.format(
                        atlas_name, atlas_id, region_name, target_shape))
            flat_indices.append(indices)
            voxel_counts.append(int(indices.size))

    cortical_counts = voxel_counts[:cortical_count]
    subcortical_counts = voxel_counts[cortical_count:]
    print('Cortical ROI masks: {}; voxel coverage min={} max={}.'.format(
        cortical_count, min(cortical_counts), max(cortical_counts)))
    print('Subcortical ROI masks: {}; voxel coverage min={} max={}.'.format(
        subcortical_count, min(subcortical_counts), max(subcortical_counts)))
    return {
        'target_shape': target_shape,
        'flat_indices': tuple(flat_indices),
        'voxel_counts': tuple(voxel_counts),
    }


def calculate_roi_means(gradcam_data, roi_mask_info):
    """Calculate trial-wise means using masks on the original Grad-CAM grid."""
    if gradcam_data.ndim != 4:
        raise ValueError(
            'Grad-CAM data must have shape (samples, x, y, z); found {}.'
            .format(gradcam_data.shape))
    if gradcam_data.shape[0] == 0:
        raise ValueError('ROI means cannot be calculated from zero samples.')

    target_shape = tuple(roi_mask_info['target_shape'])
    if tuple(gradcam_data.shape[1:]) != target_shape:
        raise ValueError(
            'Grad-CAM shape {} does not match prepared ROI-mask shape {}.'
            .format(tuple(gradcam_data.shape[1:]), target_shape))

    flat_data = gradcam_data.reshape(gradcam_data.shape[0], -1)
    flat_indices = roi_mask_info['flat_indices']
    roi_means = np.empty((gradcam_data.shape[0], len(flat_indices)), dtype=float)
    for region_idx, indices in enumerate(flat_indices):
        roi_means[:, region_idx] = np.mean(flat_data[:, indices], axis=1)

    if not np.isfinite(roi_means).all():
        raise ValueError('ROI means contain a non-finite value.')
    print('ROI averaging on the Grad-CAM grid is complete')
    return roi_means


def statistical_comparison_with_zero(learned_roi_means, region_labels,
                                     alpha=0.05):
    """Preserve the existing two-sided Wilcoxon and per-table FDR method."""
    if learned_roi_means.shape[0] == 0:
        raise ValueError('Statistical testing requires at least one trial.')

    num_regions = learned_roi_means.shape[1]
    p_values = np.ones(num_regions)
    effect_sizes = np.zeros(num_regions)

    for region_idx in range(num_regions):
        learned_values = learned_roi_means[:, region_idx]
        if np.any(learned_values != 0):
            _, p_values[region_idx] = wilcoxon(
                learned_values, alternative='two-sided')
        effect_sizes[region_idx] = np.mean(learned_values)

    reject, q_values = fdrcorrection(p_values, alpha=alpha, method='indep')
    results = []
    for region_idx in range(num_regions):
        region_row = region_labels.iloc[region_idx]
        results.append({
            'Region_ID': region_row['Region_ID'],
            'Region_Name': region_row['Region'],
            'Region_Type': region_row['Type'],
            'Detailed_Type': region_row['Detailed_Type'],
            'Learned_Mean': np.mean(learned_roi_means[:, region_idx]),
            'Effect_Size': effect_sizes[region_idx],
            'P_Value': p_values[region_idx],
            'FDR_Q_Value': q_values[region_idx],
            'Significant': reject[region_idx],
            'Significance': '*' if reject[region_idx] else '',
        })
    return pd.DataFrame(results)


def save_results(results_df, output_dir, classify_type, fold, class_info,
                 n_selected_trials, all_trials=False):
    """Save one self-describing Small or Large ROI table."""
    output_table = results_df[[
        'Region_Name', 'Detailed_Type', 'Learned_Mean', 'FDR_Q_Value',
        'Significance',
    ]].copy()
    output_table = output_table.rename(columns={
        'Region_Name': 'Label',
        'Detailed_Type': 'Region',
        'Learned_Mean': 'Value',
        'FDR_Q_Value': 'FDR q',
    })
    count_column = 'N_All_Trials' if all_trials else 'N_Success_Trials'
    output_table.insert(0, count_column, n_selected_trials)
    output_table.insert(0, 'Class', class_info['display_name'])
    output_table.insert(0, 'Fold', fold)

    output_path = get_per_fold_output_path(
        output_dir, classify_type, fold, class_info,
        all_trials=all_trials)
    output_table.to_csv(output_path, index=False)
    print('Saved {} fold {} ROI table: {}'.format(
        class_info['display_name'], fold, output_path))
    return output_path


def get_per_fold_output_path(output_dir, classify_type, fold, class_info,
                             all_trials=False):
    """Return the unique CSV path for one fold and one class."""
    suffix = '_all_trials' if all_trials else ''
    return os.path.join(
        output_dir,
        'gradcam_comparison_results_ct{}_fold{}_{}{}.csv'.format(
            classify_type, fold, class_info['name'], suffix))


def build_all_fold_summary_table(output_dir, classify_type, folds, class_info,
                                 all_trials=False):
    """Align all fold ``Value`` columns for one class by ROI label.

    The atlas controls the ROI count.  Every requested fold must contain the
    same non-empty, unique label set, but there is no fixed expected number of
    ROIs.
    """
    required_columns = {'Fold', 'Class', 'Label', 'Region', 'Value'}
    tables = {}
    expected_labels = None
    expected_regions = None

    for fold in folds:
        input_path = get_per_fold_output_path(
            output_dir, classify_type, fold, class_info,
            all_trials=all_trials)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(
                'Expected {} fold {} ROI table was not found: {}.'.format(
                    class_info['display_name'], fold, input_path))

        try:
            table = pd.read_csv(input_path)
        except (IOError, OSError, ValueError) as error:
            raise ValueError(
                'Could not read {} fold {} ROI table {}: {}.'.format(
                    class_info['display_name'], fold, input_path, error))

        missing_columns = required_columns.difference(table.columns)
        if missing_columns:
            raise ValueError(
                '{} fold {} ROI table is missing required column(s): {}.'.format(
                    class_info['display_name'], fold,
                    ', '.join(sorted(missing_columns))))
        if table.empty:
            raise ValueError('{} fold {} ROI table contains no ROIs.'.format(
                class_info['display_name'], fold))
        if table['Label'].isna().any() or \
                (table['Label'].astype(str).str.strip() == '').any():
            raise ValueError('{} fold {} ROI table contains an empty Label.'.format(
                class_info['display_name'], fold))
        if table['Label'].duplicated().any():
            duplicates = table.loc[table['Label'].duplicated(), 'Label']
            raise ValueError(
                '{} fold {} ROI table contains duplicate Label(s): {}.'.format(
                    class_info['display_name'], fold,
                    ', '.join(str(value) for value in duplicates.tolist())))

        try:
            values = pd.to_numeric(table['Value'], errors='raise').to_numpy(
                dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('{} fold {} has a non-numeric Value: {}.'.format(
                class_info['display_name'], fold, error))
        if not np.isfinite(values).all():
            raise ValueError('{} fold {} has a non-finite Value.'.format(
                class_info['display_name'], fold))

        reported_folds = set(pd.to_numeric(
            table['Fold'], errors='coerce').dropna().astype(int).tolist())
        if reported_folds != {fold}:
            raise ValueError(
                '{} fold {} table has inconsistent Fold metadata: {}.'.format(
                    class_info['display_name'], fold,
                    ', '.join(str(value) for value in sorted(reported_folds))))
        reported_classes = set(table['Class'].astype(str).tolist())
        if reported_classes != {class_info['display_name']}:
            raise ValueError(
                '{} fold {} table has inconsistent Class metadata: {}.'.format(
                    class_info['display_name'], fold,
                    ', '.join(sorted(reported_classes))))

        table = table[['Label', 'Region', 'Value']].copy()
        table['Value'] = values
        table = table.set_index('Label', drop=False)
        label_set = set(table.index.tolist())

        if expected_labels is None:
            expected_labels = label_set
            expected_regions = table['Region'].to_dict()
        else:
            missing_labels = expected_labels.difference(label_set)
            extra_labels = label_set.difference(expected_labels)
            if missing_labels or extra_labels:
                message_parts = []
                if missing_labels:
                    message_parts.append('missing {}'.format(
                        ', '.join(sorted(missing_labels))))
                if extra_labels:
                    message_parts.append('extra {}'.format(
                        ', '.join(sorted(extra_labels))))
                raise ValueError(
                    '{} fold {} ROI labels do not match the first fold: {}.'.format(
                        class_info['display_name'], fold,
                        '; '.join(message_parts)))

            changed_regions = [
                label for label in expected_labels
                if table.loc[label, 'Region'] != expected_regions[label]]
            if changed_regions:
                raise ValueError(
                    '{} fold {} changes Region metadata for Label(s): {}.'.format(
                        class_info['display_name'], fold,
                        ', '.join(sorted(changed_regions))))

        tables[fold] = table

    first_fold = folds[0]
    summary_table = tables[first_fold][['Label', 'Region']].reset_index(drop=True)
    for fold in folds:
        summary_table['Fold{}'.format(fold)] = summary_table['Label'].map(
            tables[fold]['Value'])

    if summary_table.isna().any().any():
        raise RuntimeError(
            '{} ROI summary contains missing values after label alignment.'.format(
                class_info['display_name']))
    return summary_table


def save_all_fold_summary_tables(output_dir, classify_type, folds,
                                 all_trials=False):
    """Write one dynamically sized all-fold Value table for each class."""
    summary_paths = []
    for class_info in CLASS_INFOS:
        summary_table = build_all_fold_summary_table(
            output_dir, classify_type, folds, class_info,
            all_trials=all_trials)
        suffix = '_all_trials' if all_trials else ''
        output_path = os.path.join(
            output_dir, 'roi_values_ct{}_{}_all_folds{}.csv'.format(
                classify_type, class_info['name'], suffix))
        summary_table.to_csv(output_path, index=False)
        print('Saved {} all-fold ROI Value summary ({} ROIs): {}'.format(
            class_info['display_name'], len(summary_table), output_path))
        summary_paths.append(output_path)
    return summary_paths


def process_fold(input_path, output_dir, classify_type, fold, atlas_info,
                 roi_mask_cache, selected_counts, alpha, all_trials=False):
    """Calculate and write the Small and Large tables for one model fold."""
    selected_data = load_gradcam_data(
        input_path, fold, all_trials=all_trials)
    region_labels = atlas_info['region_labels']
    class_shapes = {
        tuple(selected_data[class_info['name']].shape[1:])
        for class_info in CLASS_INFOS
    }
    if len(class_shapes) != 1:
        raise ValueError(
            'Fold {} Small and Large Grad-CAM shapes do not match: {}.'
            .format(fold, sorted(class_shapes)))
    target_shape = class_shapes.pop()
    if target_shape not in roi_mask_cache:
        roi_mask_cache[target_shape] = prepare_resampled_roi_masks(
            atlas_info, target_shape)
    roi_mask_info = roi_mask_cache[target_shape]
    output_paths = []

    for class_info in CLASS_INFOS:
        class_name = class_info['name']
        trial_description = 'all true-label' if all_trials else 'successful'
        print('Processing CT{} fold {} {} ({} {} trials)...'.format(
            classify_type, fold, class_info['display_name'],
            selected_counts[class_name], trial_description))
        roi_means = calculate_roi_means(
            selected_data[class_name], roi_mask_info)
        results_df = statistical_comparison_with_zero(
            roi_means, region_labels, alpha=alpha)
        output_paths.append(save_results(
            results_df, output_dir, classify_type, fold, class_info,
            selected_counts[class_name], all_trials=all_trials))

    return output_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Calculate per-fold Small and Large Harvard-Oxford ROI Guided '
            'Grad-CAM tables from fMRI Grad-CAM NPZ files.'))
    parser.add_argument(
        '--grad_cam_dir', required=True,
        help='Directory containing cam_fmri_ct{classify_type}_{fold}.npz files.')
    parser.add_argument(
        '--classify_type', type=int, default=4,
        help='Classification task encoded in the Grad-CAM file names (default: 4).')
    parser.add_argument(
        '--fold', default='all',
        help='Fold ID from 0 to 8, or all to require and process every fold (default: all).')
    parser.add_argument(
        '--output_dir', required=True,
        help='New or existing directory for the per-fold CSV tables.')
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='FDR alpha applied independently to each fold/class table (default: 0.05).')
    parser.add_argument(
        '--all_trials', action='store_true',
        help=('Use every true-label Small/Large trial instead of only '
              'correctly classified trials. Outputs receive an '
              '_all_trials suffix.'))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.alpha <= 0 or args.alpha >= 1:
            raise ValueError('--alpha must be greater than 0 and less than 1.')

        folds = resolve_folds(args.fold)
        input_paths, success_counts = preflight_input_files(
            args.grad_cam_dir, args.classify_type, folds,
            all_trials=args.all_trials)

        # Input validation succeeds before this directory is created, so a
        # missing fold or empty class never produces a partial output set.
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        atlas_info = load_harvard_oxford_atlas()
        roi_mask_cache = {}
        output_paths = []
        for fold in folds:
            output_paths.extend(process_fold(
                input_paths[fold], args.output_dir, args.classify_type, fold,
                atlas_info, roi_mask_cache, success_counts[fold], args.alpha,
                all_trials=args.all_trials))

        if args.fold == 'all':
            summary_paths = save_all_fold_summary_tables(
                args.output_dir, args.classify_type, folds,
                all_trials=args.all_trials)
        else:
            summary_paths = []
            print('A single fold was selected; all-fold ROI summaries were not created.')

        print('Completed {} fold(s) and saved {} ROI table(s).'.format(
            len(folds), len(output_paths)))
        if summary_paths:
            print('Saved {} all-fold ROI Value summary table(s).'.format(
                len(summary_paths)))
    except (IOError, OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
