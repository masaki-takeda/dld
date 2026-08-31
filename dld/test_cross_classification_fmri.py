import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import options
from dataset import (ARTIFICIAL_SMALL_LARGE, COMBINE_TYPE_FMRI,
                     DATA_TYPE_TEST, GROUP1_GROUP2, NATURAL_SMALL_LARGE,
                     SMALL_LARGE, get_dataset)
from main import eval_epoch
from model import get_fmri_model
from utils import fix_run_seed, fix_state_dict, get_device


TASK_LABELS = {
    SMALL_LARGE: OrderedDict([('0', 'Small'), ('1', 'Large')]),
    GROUP1_GROUP2: OrderedDict([('0', 'Group1'), ('1', 'Group2')]),
    ARTIFICIAL_SMALL_LARGE: OrderedDict([('0', 'Small'), ('1', 'Large')]),
    NATURAL_SMALL_LARGE: OrderedDict([('0', 'Small'), ('1', 'Large')]),
}
TASK_SHORT_NAMES = {
    SMALL_LARGE: 'ct4',
    GROUP1_GROUP2: 'ct5',
    ARTIFICIAL_SMALL_LARGE: 'ct6',
    NATURAL_SMALL_LARGE: 'ct7',
}
TASK_DISPLAY_NAMES = {
    SMALL_LARGE: 'CT4 Small/Large',
    GROUP1_GROUP2: 'CT5 Group1/Group2',
    ARTIFICIAL_SMALL_LARGE: 'CT6 Artificial Small/Large',
    NATURAL_SMALL_LARGE: 'CT7 Natural Small/Large',
}
LABEL_COLUMN_NAMES = {
    SMALL_LARGE: 'label_ct4_small_large',
    GROUP1_GROUP2: 'label_ct5_group',
    ARTIFICIAL_SMALL_LARGE: 'label_ct6_artificial_small_large',
    NATURAL_SMALL_LARGE: 'label_ct7_natural_small_large',
}
PROBABILITY_COLUMN_NAMES = {
    SMALL_LARGE: 'probability_ct4_large',
    GROUP1_GROUP2: 'probability_ct5_group2',
    ARTIFICIAL_SMALL_LARGE: 'probability_ct6_artificial_large',
    NATURAL_SMALL_LARGE: 'probability_ct7_natural_large',
}
PREDICTION_COLUMN_NAMES = {
    SMALL_LARGE: 'prediction_ct4_small_large',
    GROUP1_GROUP2: 'prediction_ct5_group1_group2',
    ARTIFICIAL_SMALL_LARGE: 'prediction_ct6_artificial_small_large',
    NATURAL_SMALL_LARGE: 'prediction_ct7_natural_small_large',
}
SUPPORTED_TASK_PAIRS = {
    (SMALL_LARGE, GROUP1_GROUP2),
    (GROUP1_GROUP2, SMALL_LARGE),
    (ARTIFICIAL_SMALL_LARGE, NATURAL_SMALL_LARGE),
    (NATURAL_SMALL_LARGE, ARTIFICIAL_SMALL_LARGE),
}


def get_task_pair_info(model_classify_type, data_classify_type):
    """Validate a supported transfer direction and return output metadata."""
    task_pair = (model_classify_type, data_classify_type)
    if task_pair not in SUPPORTED_TASK_PAIRS:
        raise ValueError(
            'Only CT4-to-CT5 (model=4, data=5), CT5-to-CT4 '
            '(model=5, data=4), CT6-to-CT7 (model=6, data=7), and '
            'CT7-to-CT6 (model=7, data=6) evaluations are supported.')

    return OrderedDict([
        ('model_task_name', TASK_DISPLAY_NAMES[model_classify_type]),
        ('data_task_name', TASK_DISPLAY_NAMES[data_classify_type]),
        ('direction', '{}_to_{}'.format(
            TASK_SHORT_NAMES[model_classify_type],
            TASK_SHORT_NAMES[data_classify_type])),
        ('model_label_mapping', TASK_LABELS[model_classify_type]),
        ('data_label_mapping', TASK_LABELS[data_classify_type]),
        ('label_column_name', LABEL_COLUMN_NAMES[data_classify_type]),
        ('probability_column_name', PROBABILITY_COLUMN_NAMES[model_classify_type]),
        ('prediction_column_name', PREDICTION_COLUMN_NAMES[model_classify_type]),
    ])


def parse_subjects(subjects_text):
    """Parse a comma-separated subject list and reject empty IDs."""
    if subjects_text is None:
        raise ValueError('--test_subjects must be specified.')

    subjects = [subject.strip() for subject in subjects_text.split(',')]
    if not subjects or any(not subject for subject in subjects):
        raise ValueError(
            '--test_subjects must be a comma-separated list without empty IDs.')
    if len(set(subjects)) != len(subjects):
        raise ValueError('--test_subjects contains the same subject more than once.')
    return subjects


def validate_test_subjects(requested_subjects, source_subjects):
    """Require target test subjects to equal the source configuration."""
    if len(requested_subjects) != len(source_subjects) or \
            set(requested_subjects) != set(source_subjects):
        raise ValueError(
            'The requested --test_subjects must match model_dir/options.json. '
            'Requested: {}. Source: {}.'.format(
                ','.join(requested_subjects), ','.join(source_subjects)))


def ensure_separate_output_dir(model_dir, output_dir):
    """Do not allow cross-task results to overwrite source model files."""
    model_dir = os.path.abspath(model_dir)
    output_dir = os.path.abspath(output_dir)

    try:
        output_is_inside_model_dir = os.path.commonpath(
            [model_dir, output_dir]) == model_dir
    except ValueError:
        # Different Windows drives cannot have a common path.  They are safe.
        output_is_inside_model_dir = False

    if output_is_inside_model_dir:
        raise ValueError(
            '--output_dir must be outside --model_dir so existing source results '
            'cannot be overwritten.')
    return output_dir


def resolve_folds(fold_argument, fold_size):
    """Return selected fold IDs and whether all source folds were requested."""
    if fold_size <= 0:
        raise ValueError('model_dir/options.json has invalid fold_size={}.'.format(fold_size))

    if fold_argument == 'all':
        return list(range(fold_size)), True

    try:
        fold = int(fold_argument)
    except ValueError:
        raise ValueError('--fold must be an integer such as 3, or the word all.')

    if fold < 0 or fold >= fold_size:
        raise ValueError(
            '--fold={} is outside the source fold range 0-{}.'.format(
                fold, fold_size - 1))
    return [fold], False


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate supported fMRI checkpoints on another task\'s test data.')
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing source options.json and model checkpoints.')
    parser.add_argument('--model_classify_type', type=int, default=SMALL_LARGE,
                        help='Source task: 4=CT4, 5=CT5, 6=artificial Small/Large, or 7=natural Small/Large (default: 4).')
    parser.add_argument('--data_classify_type', type=int, default=GROUP1_GROUP2,
                        help='Target task: 4=CT4, 5=CT5, 6=artificial Small/Large, or 7=natural Small/Large (default: 5).')
    parser.add_argument('--fold', default='3',
                        help='Source checkpoint fold, or all for every source fold (default: 3).')
    parser.add_argument('--data_dir', default=None,
                        help='Optional replacement for data_dir stored in source options.json.')
    parser.add_argument('--test_subjects', required=True,
                        help='Comma-separated held-out subjects; must match source options.json.')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU index; use -1 for the default available GPU or CPU if unavailable.')
    parser.add_argument('--output_dir', required=True,
                        help='New directory for cross-task metrics, predictions, and summary.')
    return parser


def main():
    parser = build_parser()
    raw_args = parser.parse_args()

    try:
        task_pair_info = get_task_pair_info(
            raw_args.model_classify_type, raw_args.data_classify_type)
        model_dir = os.path.abspath(raw_args.model_dir)
        output_dir = ensure_separate_output_dir(model_dir, raw_args.output_dir)
        options_path = os.path.join(model_dir, 'options.json')

        if not os.path.isfile(options_path):
            raise ValueError('Source options file was not found: {}.'.format(options_path))

        source_args = options.load_args(model_dir)
        if source_args.combine_type != 'fmri':
            raise ValueError(
                'model_dir/options.json is not an fMRI experiment '
                '(combine_type={!r}).'.format(source_args.combine_type))
        if source_args.classify_type != raw_args.model_classify_type:
            raise ValueError(
                'model_dir/options.json has classify_type={}, but '
                '--model_classify_type={}.'.format(
                    source_args.classify_type, raw_args.model_classify_type))

        folds, all_folds_requested = resolve_folds(
            raw_args.fold, source_args.fold_size)
        checkpoint_paths = OrderedDict()
        missing_checkpoints = []
        for fold in folds:
            checkpoint_path = os.path.join(
                model_dir, 'model_ct{}_{}.pt'.format(
                    raw_args.model_classify_type, fold))
            checkpoint_paths[fold] = checkpoint_path
            if not os.path.isfile(checkpoint_path):
                missing_checkpoints.append(checkpoint_path)
        if missing_checkpoints:
            raise ValueError(
                'Required source checkpoint(s) were not found:\n{}.'.format(
                    '\n'.join(missing_checkpoints)))

        requested_subjects = parse_subjects(raw_args.test_subjects)
        source_subjects = parse_subjects(source_args.test_subjects)
        validate_test_subjects(requested_subjects, source_subjects)

        effective_data_dir = (os.path.abspath(raw_args.data_dir)
                              if raw_args.data_dir is not None
                              else source_args.data_dir)

        # Keep all source data/architecture settings, but make the loader
        # select target averaged files and labels. No files are written to
        # model_dir.
        source_args.override_params({
            'classify_type': raw_args.data_classify_type,
            'data_dir': effective_data_dir,
            'test_subjects': ','.join(source_subjects),
            'test': 1,
        })

        os.makedirs(output_dir, exist_ok=True)

        device, use_cuda = get_device(raw_args.gpu)
        loader_kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        # Target test membership is defined solely by test_subjects, not by
        # the training/validation fold. Build it once so every source model
        # sees the same samples in the same order.
        dataset_fold = folds[0]
        test_dataset = get_dataset(
            combine_type=COMBINE_TYPE_FMRI,
            data_type=DATA_TYPE_TEST,
            classify_type=raw_args.data_classify_type,
            fold=dataset_fold,
            args=source_args)
        test_loader = DataLoader(
            test_dataset,
            batch_size=source_args.batch_size,
            shuffle=False,
            **loader_kwargs)

        if len(test_dataset) == 0:
            raise ValueError(
                '{} test dataset is empty for the requested subjects.'.format(
                    task_pair_info['data_task_name']))

        all_results = []
        prediction_paths = OrderedDict()
        reference_labels = None

        for fold in folds:
            if source_args.run_seed >= 0:
                fix_run_seed(source_args.run_seed + fold)

            # Build an unwrapped model because fix_state_dict removes an
            # optional DataParallel prefix from saved checkpoints.
            model = get_fmri_model(test_dataset.fmri_ch_size, False, device)
            state = torch.load(checkpoint_paths[fold], map_location=device)
            model.load_state_dict(fix_state_dict(state))

            metrics, (labels, probabilities) = eval_epoch(
                COMBINE_TYPE_FMRI,
                model,
                device,
                test_loader,
                epoch=0,
                logger=None,
                record_result=True)

            labels = np.asarray(labels, dtype=np.int64)
            probabilities = np.asarray(probabilities, dtype=np.float64)
            predictions = (probabilities > 0.5).astype(np.int64)

            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise RuntimeError(
                    'Target labels changed between folds; cross-fold results are invalid.')

            result = OrderedDict()
            result['model_classify_type'] = raw_args.model_classify_type
            result['data_classify_type'] = raw_args.data_classify_type
            result['fold'] = fold
            result['n_test_samples'] = int(labels.size)
            for name, value in metrics.items():
                result[name] = float(value)
            all_results.append(result)

            predictions_path = os.path.join(
                output_dir, 'preds_{}_fold{}_test.csv'.format(
                    task_pair_info['direction'], fold))
            pd.DataFrame({
                task_pair_info['label_column_name']: labels,
                task_pair_info['probability_column_name']: probabilities,
                task_pair_info['prediction_column_name']: predictions,
            }).to_csv(predictions_path, index=False)
            prediction_paths[fold] = predictions_path
            print('{} fold {} to {} accuracy: {:.6f}%'.format(
                task_pair_info['model_task_name'], fold,
                task_pair_info['data_task_name'], result['accuracy']))

        results_df = pd.DataFrame(all_results)
        metric_names = list(all_results[0].keys())[4:]

        if all_folds_requested:
            metrics_path = os.path.join(
                output_dir, 'result_{}_all_folds_test.csv'.format(
                    task_pair_info['direction']))
            aggregate_path = os.path.join(
                output_dir, 'result_{}_all_folds_summary.csv'.format(
                    task_pair_info['direction']))
            summary_path = os.path.join(
                output_dir, 'cross_task_summary_all_folds.json')
            aggregate_df = pd.DataFrame(
                [results_df[metric_names].mean(),
                 results_df[metric_names].std(ddof=1)],
                index=['mean', 'std'])
            aggregate_df.to_csv(aggregate_path, index=True)
            summary_metrics = OrderedDict([
                ('mean', OrderedDict(
                    (name, float(aggregate_df.loc['mean', name]))
                    for name in metric_names)),
                ('sample_sd', OrderedDict(
                    (name, float(aggregate_df.loc['std', name]))
                    for name in metric_names)),
            ])
            selection_rule = (
                'All {} folds 0-{} were evaluated on the same fixed {} test set.'.format(
                    task_pair_info['model_task_name'], source_args.fold_size - 1,
                    task_pair_info['data_task_name']))
        else:
            fold = folds[0]
            metrics_path = os.path.join(
                output_dir, 'result_{}_fold{}_test.csv'.format(
                    task_pair_info['direction'], fold))
            aggregate_path = None
            summary_path = os.path.join(output_dir, 'cross_task_summary.json')
            summary_metrics = OrderedDict(
                (name, float(all_results[0][name])) for name in metric_names)
            if raw_args.model_classify_type == SMALL_LARGE and fold == 3:
                selection_rule = (
                    'CT4 fold 3 was selected as the highest-accuracy CT4 test fold.')
            else:
                selection_rule = '{} fold {} was explicitly selected by --fold.'.format(
                    task_pair_info['model_task_name'], fold)

        results_df.to_csv(metrics_path, index=False)

        if all_folds_requested:
            legacy_checkpoint_path = None
            legacy_fold = None
            legacy_predictions_path = None
        else:
            legacy_fold = folds[0]
            legacy_checkpoint_path = checkpoint_paths[legacy_fold]
            legacy_predictions_path = prediction_paths[legacy_fold]

        summary = OrderedDict([
            ('evaluation', '{} checkpoint(s) evaluated on {} test data'.format(
                task_pair_info['model_task_name'], task_pair_info['data_task_name'])),
            ('selection_rule', selection_rule),
            # Preserve the original single-fold fields for callers that read
            # cross_task_summary.json from the first version of this script.
            ('checkpoint_path', legacy_checkpoint_path),
            ('checkpoint_paths', OrderedDict(
                (str(fold), checkpoint_paths[fold]) for fold in folds)),
            ('model_classify_type', raw_args.model_classify_type),
            ('model_label_mapping', task_pair_info['model_label_mapping']),
            ('data_classify_type', raw_args.data_classify_type),
            ('data_label_mapping', task_pair_info['data_label_mapping']),
            ('fold', legacy_fold),
            ('folds', folds),
            ('test_dataset_fold_argument', dataset_fold),
            ('test_subjects', source_subjects),
            ('source_options_test_subjects', source_subjects),
            ('n_test_samples', int(reference_labels.size)),
            ('data_config', OrderedDict([
                ('data_dir', effective_data_dir),
                ('smooth', source_args.smooth),
                ('fmri_frame_type', source_args.fmri_frame_type),
                ('fmri_offset_tr', source_args.fmri_offset_tr),
                ('average_trial_size', source_args.average_trial_size),
                ('average_repeat_size', source_args.average_repeat_size),
                ('subjects_per_fold', source_args.subjects_per_fold),
            ])),
            ('decision_rule',
             'Raw {} output 0 ({}) is compared with {} label 0 ({}); raw {} '
             'output 1 ({}) is compared with {} label 1 ({}). Labels are not '
             'inverted.'.format(
                 task_pair_info['model_task_name'],
                 task_pair_info['model_label_mapping']['0'],
                 task_pair_info['data_task_name'],
                 task_pair_info['data_label_mapping']['0'],
                 task_pair_info['model_task_name'],
                 task_pair_info['model_label_mapping']['1'],
                 task_pair_info['data_task_name'],
                 task_pair_info['data_label_mapping']['1'])),
            ('metrics', summary_metrics),
            ('output_files', OrderedDict([
                ('metrics_csv', metrics_path),
                ('aggregate_csv', aggregate_path),
                ('predictions_csv', legacy_predictions_path),
                ('prediction_csvs', OrderedDict(
                    (str(fold), prediction_paths[fold]) for fold in folds)),
                ('summary_json', summary_path),
            ])),
        ])
        with open(summary_path, 'w') as summary_file:
            json.dump(summary, summary_file, indent=2)

        print('Loaded checkpoint(s): {}'.format(', '.join(
            checkpoint_paths[fold] for fold in folds)))
        print('{} test subjects: {}'.format(
            task_pair_info['data_task_name'], ','.join(source_subjects)))
        print('Saved metrics: {}'.format(metrics_path))
        if aggregate_path is not None:
            print('Saved mean and sample SD: {}'.format(aggregate_path))
        print('Saved predictions: {}'.format(', '.join(
            prediction_paths[fold] for fold in folds)))
        print('Saved summary: {}'.format(summary_path))
    except (IOError, OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
