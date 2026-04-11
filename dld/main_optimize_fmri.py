from dataset import FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, FRONT_SIDE, SMALL_LARGE, CLASSIFY_ALL
from optimize import process_optimization
from options import get_optimize_args


#-----[Changes from here]-----
# Trail size for hyper-parameter optization
optimization_trial_size = 100

# Combine type
combine_type = 'fmri' # combined, fmri, eeg
classify_type = SMALL_LARGE # FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, FRONT_SIDE, SMALL_LARGE

save_dir_prefix = "./saved_fmri_optimize_a7r7/op"
# When specifying up to a subdirectory as above
# The save directory will be "./saved_eeg_op0/op_0", "./saved_eeg_op0/op_1" ... and so on


def objective_func(trial):
    params = {
        # Optimizing parameters
        'lr'          : trial.suggest_loguniform('lr', 0.0001, 0.1),
        'weight_decay': trial.suggest_uniform('weight_decay', 0.0, 0.1),
        #'kernel_size' : trial.suggest_int('kernel_size', 2, 9),
        #'residual'    : trial.suggest_categorical('residual', [0, 1]), 
        # モデルタイプといった数値でないものもcategoricalにて指定可能
        #'model_type'  : trial.suggest_categorical('model_type', ['tcn1', 'tcnx']),

        # Fixed parameters
	#'model_type' : 'fmri',
	#'eeg_frame_type' : 'normal',
        #'eeg_normalize_type' : 'pre',
        'fmri_frame_type' : 'normal',
	'smooth'	: False,
        'batch_size' : 100,
	'epochs'     : 200,
        'patience'   : 50,
	'test_subjects' : 'TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01',
        'data_dir'   : '/data3/DLD2/Data_Converted_EEG_ICAed2',
	'average_trial_size' : 7,
	'average_repeat_size' : 7,
        'fold_size'  : 1, # When only 1 Fold is targeted to speed up the processing
        'gpu'        : 1,
        #'debug'     : 1,
    }
    return params


#-----[To here]-----

    
if __name__ == '__main__':
    optimize_args = get_optimize_args()
    
    process_optimization(combine_type,
                         classify_type,
                         objective_func,
                         optimization_trial_size,
                         save_dir_prefix,
                         test=optimize_args.test)

