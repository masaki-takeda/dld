from dataset import FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, FRONT_SIDE, SMALL_LARGE, CLASSIFY_ALL
from optimize import process_optimization
from options import get_optimize_args


#-----[Changes from here]-----
# Trail size for hyper-parameter optization
optimization_trial_size = 10

# Combine type
combine_type = 'eeg' # combined, fmri, eeg
classify_type = FRONT_SIDE # FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, FRONT_SIDE, SMALL_LARGE

save_dir_prefix = "./saved_eeg_op0/op"
# When specifying up to a subdirectory as above
# The save directory will be "./saved_eeg_op0/op_0", "./saved_eeg_op0/op_1" ... and so on


def objective_func(trial):
    params = {
        # Optimizing parameters
        'lr'          : trial.suggest_loguniform('lr', 0.0001, 0.1),
        #'weight_decay': trial.suggest_uniform('weight_decay', 0.0, 0.1),
        'kernel_size' : trial.suggest_int('kernel_size', 2, 9),
        #'residual'    : trial.suggest_categorical('residual', [0, 1]), 
        # Non-numeric values, such as model types, can also be specified using the “categorical” option
        #'model_type'  : trial.suggest_categorical('model_type', ['tcn1', 'tcnx']),
        
        # Fixed parameters
        'eeg_normalize_type' : 'pre',
        'batch_size' : 100,
        'patience'   : 50,
        'data_dir'   : '/data2/DLD/Data_Converted',
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
