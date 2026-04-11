import os
import numpy as np
import pandas as pd
import options
import optuna

from main import train_full_folds, test_full_folds
from dataset import COMBINE_TYPE_EEG, COMBINE_TYPE_FMRI, COMBINE_TYPE_COMBINED


def get_summary_file_dir(save_dir_prefix):
   return '{}{}'.format(save_dir_prefix, '_summary')


def get_summary_file_path(save_dir_prefix, test=False):
    if not test:
        return '{}/{}'.format(get_summary_file_dir(save_dir_prefix), 'summary.txt')
    else:
        return '{}/{}'.format(get_summary_file_dir(save_dir_prefix), 'summary_test.txt')


def get_param_str(params):
    ret = ''
    for key,value in params.items():
        if type(value) == float:
            ret += f'{key}={value:.4f}, '
        else:
            ret += f'{key}={value}, '
    return ret


def export_summary(best_params,
                   best_trial_index,
                   trials,
                   save_dir_prefix):
    trial_size = len(trials)
    
    summary_lines = []
    
    summary_lines.append(f'[Best params]')
    
    for key,value in best_params.items():
        summary_lines.append(f'  {key} = {value}')
        
    summary_lines.append('')
    summary_lines.append('[Results]')
    
    for i in range(trial_size):
        suffix_string = f'_{i}'
        directory = '{}{}'.format(save_dir_prefix, suffix_string)
        
        trial = trials[i]
        
        if i == best_trial_index:
            best = '[best]'
        else:
            best = '[    ]'
        param_str = get_param_str(trial.params)
        summary_lines.append(f'  {best} {directory}: {trials[i].value} {param_str}')
    
    summary_file_dir = get_summary_file_dir(save_dir_prefix)
    if not os.path.exists(summary_file_dir):
        os.makedirs(summary_file_dir)
    
    summary_file_path = get_summary_file_path(save_dir_prefix)
    
    f = open(summary_file_path, 'w', encoding='utf_8')
    for line in summary_lines:
        f.write(line + '\n')
        print(line)
    f.close()

    
def export_summary_test(classify_type,
                        save_dir_prefix,
                        trial_size,
                        early_stopping_metric):

    def load_result(path):
        if not os.path.exists(path):
            return -np.inf

        df = pd.read_csv(path, index_col=0)
        # Use early stopping metric as a target metric
        return df[early_stopping_metric]['test_mean']

    results = []
    directories = []
    
    for i in range(trial_size):
        directory = "{}_{}".format(save_dir_prefix, i)
        directories.append(directory)
        
        file_path = "{}_{}/result_ct{}_test.csv".format(save_dir_prefix, i,
                                                        classify_type)
        result = load_result(file_path)
        results.append(result)


    max_index = np.argmax(results)
    summary_lines = []

    for i in range(len(directories)):
        if i == max_index:
            best = "[best]"
        else:
            best = "[    ]"
        summary_lines.append("  {} {}: {}".format(best,
                                                  directories[i],
                                                  results[i]))

    summary_file_dir = get_summary_file_dir(save_dir_prefix)
    if not os.path.exists(summary_file_dir):
        os.makedirs(summary_file_dir)
    
    summary_file_path = get_summary_file_path(save_dir_prefix, test=True)
    
    f = open(summary_file_path, 'w', encoding='utf_8')
    for line in summary_lines:
        f.write(line + '\n')
        print(line)
    f.close()


class Objective:
    def __init__(self, combine_type, classify_type, objective_func, args, save_dir_prefix):
        self.combine_type = combine_type
        self.classify_type = classify_type
        self.objective_func = objective_func
        self.args = options.Arguments(args)
        self.save_dir_prefix = save_dir_prefix
        
    def __call__(self, trial):
        # Hyper parameter setting
        params = self.objective_func(trial)
        
        params.update(
            {
                'save_dir' : self.save_dir_prefix + f'_{trial.number}'
            }
        )
        
        self.args.override_params(params)
        
        options.save_args(self.args)

        # The score is the best validtion value of args.early_stopping_metric.
        score = train_full_folds(self.combine_type, self.args, self.classify_type)
        return score


def process_optimization(raw_combine_type,
                         classify_type,
                         objective_func,
                         trial_size,
                         save_dir_prefix,
                         test):
    
    if raw_combine_type == "eeg":
        combine_type = COMBINE_TYPE_EEG
    elif raw_combine_type == "fmri":
        combine_type = COMBINE_TYPE_FMRI
    elif raw_combine_type == "combined":
        combine_type = COMBINE_TYPE_COMBINED
    else:
        assert False

    if not test:
        if combine_type == COMBINE_TYPE_EEG:
            args = options.get_eeg_args()
        elif combine_type == COMBINE_TYPE_FMRI:
            args = options.get_fmri_args()
        elif combine_type == COMBINE_TYPE_COMBINED:
            args = options.get_combined_args()
        else:
            assert False

        objective = Objective(combine_type,
                              classify_type,
                              objective_func,
                              args,
                              save_dir_prefix)
        study = optuna.create_study(direction='maximize')
    
        study.optimize(objective, n_trials=trial_size)
    
        export_summary(study.best_params,
                       study.best_trial.number,
                       study.get_trials(),
                       save_dir_prefix)
        
    else:
        # Exporting test result is only for debbbing purpose
        for i in range(trial_size):
            suffix_string = f'_{i}'
            directory = '{}{}'.format(save_dir_prefix, suffix_string)
            args = options.load_args(directory)
            
            test_full_folds(combine_type, args, classify_type)
        
        export_summary_test(classify_type,
                            save_dir_prefix,
                            trial_size,
                            args.early_stopping_metric)

