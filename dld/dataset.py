import numpy as np
import os
from torch.utils.data import Dataset

from eeg import EEG
from behavior import Behavior
from utils import get_test_subject_ids


# Data Type
DATA_TYPE_TRAIN      = 0
DATA_TYPE_VALIDATION = 1
DATA_TYPE_TEST       = 2

# Classification Type
FACE_OBJECT        = 0
MALE_FEMALE        = 1
ARTIFICIAL_NATURAL = 2
FRONT_SIDE         = 3 # front face vs non-front face
SMALL_LARGE        = 4 # front face/right face/left face
#FRONT_RIGHT_LEFT   = 5 # front face/right face/left face
CLASSIFY_ALL       = -1

CLASSIFY_TYPE_MAX  = 5

CATEGORY_FACE          = 0
CATEGORY_OBJECT        = 1
SUBCATEGORY_MALE       = 0
SUBCATEGORY_FEMALE     = 1
SUBCATEGORY_ARTIFICIAL = 0
SUBCATEGORY_NATURAL    = 1

FMRI_FRAME_TYPE_NORMAL  = 0
FMRI_FRAME_TYPE_AVERAGE = 1
FMRI_FRAME_TYPE_THREE   = 2

EEG_NORMALIZE_TYPE_NORMAL  = 0
EEG_NORMALIZE_TYPE_PRE     = 1
EEG_NORMALIZE_TYPE_NONE    = 2

EEG_DURATION_TYPE_NORMAL = 0
EEG_DURATION_TYPE_SHORT  = 1
EEG_DURATION_TYPE_LONG   = 2

COMBINE_TYPE_EEG      = 1
COMBINE_TYPE_FMRI     = 2
COMBINE_TYPE_COMBINED = 3


# Channel numbers for the mask
CH_Fp1  = 0
CH_Fp2  = 1
CH_F3   = 2
CH_F4   = 3
CH_C3   = 4
CH_C4   = 5
CH_P3   = 6
CH_P4   = 7
CH_O1   = 8
CH_O2   = 9
CH_F7   = 10
CH_F8   = 11
CH_T7   = 12
CH_T8   = 13
CH_P7   = 14
CH_P8   = 15
CH_Fz   = 16
CH_Cz   = 17
CH_Pz   = 18
CH_Oz   = 19
CH_FC1  = 20
CH_FC2  = 21
CH_CP1  = 22
CH_CP2  = 23
CH_FC5  = 24
CH_FC6  = 25
CH_CP5  = 26
CH_CP6  = 27
CH_TP9  = 28
CH_TP10 = 29
CH_POz  = 30
CH_F1   = 31
CH_F2   = 32
CH_C1   = 33
CH_C2   = 34
CH_P1   = 35
CH_P2   = 36
CH_AF3  = 37
CH_AF4  = 38
CH_FC3  = 39
CH_FC4  = 40
CH_CP3  = 41
CH_CP4  = 42
CH_PO3  = 43
CH_PO4  = 44
CH_F5   = 45
CH_F6   = 46
CH_C5   = 47
CH_C6   = 48
CH_P5   = 49
CH_P6   = 50
CH_AF7  = 51
CH_AF8  = 52
CH_FT7  = 53
CH_FT8  = 54
CH_TP7  = 55
CH_TP8  = 56
CH_PO7  = 57
CH_PO8  = 58
CH_FT9  = 59
CH_FT10 = 60
CH_Fpz  = 61
CH_CPz  = 62


class BrainDataset(Dataset):
    def __init__(self,
                 data_type,
                 classify_type,
                 data_seed,
                 use_fmri=False,
                 use_eeg=False,
                 data_dir='./data',
                 fmri_frame_type='normal',
                 fmri_offset_tr=2,
                 eeg_normalize_type='normal',
                 eeg_frame_type='normal',
                 eeg_duration_type='normal',
                 use_smooth=True,
                 average_trial_size=0,
                 average_repeat_size=0,
                 fold=0,
                 test_subjects=[],
                 subjects_per_fold=4,
                 unmatched=False,
                 fmri_mask_name=None,
                 pfi_seed=None,
                 debug=False):
        """
        use_fmri:
           (bool) whether to use the fMRI data
        use_eeg:
           (bool) whether to use the EEG data
        fmri_frame_type:
           (str) how the data is used: the data of 1TR, the average data of 3TR, or the all data of 3TR
        fmri_offset_tr:
           (int) offset TR of fmri data: 1,2 or 3 (Default 2)
        eeg_normalize_type:
           (str) normalize type of the EEG data
        eeg_frame_type:
           (str) frame type of the eeg data (normal, filter, ft)
        eeg_duration_type:
           (str) duration type of the eeg data (normal, short, long)
        use_smooth:
           (bool) whether to use smoothed fmri data
        average_trial_size:
           (int) average number of trials
        average_repeat_size:
           (int) number of repetitions for augmentation
        subjects_per_fold:
           (int) number of participants assined to one Fold
        unmatched:
           (bool) Whether to use unmatched avaraging trials between EEG and fMRI.
        """
        self.use_fmri = use_fmri
        self.use_eeg = use_eeg
        self.data_dir = data_dir
        self.use_smooth = use_smooth

        assert (data_type == DATA_TYPE_TRAIN or data_type == DATA_TYPE_VALIDATION or data_type == DATA_TYPE_TEST)
        assert (fmri_frame_type == 'normal' or fmri_frame_type == 'average' or fmri_frame_type == 'three')
        assert (fmri_offset_tr == 1 or fmri_offset_tr == 2 or fmri_offset_tr == 3)
        assert (eeg_normalize_type == 'normal' or eeg_normalize_type == 'pre' or eeg_normalize_type == 'none')
        # Now we are treating EEG normal frame type only.
        assert (eeg_frame_type == 'normal')
        assert (eeg_duration_type == 'normal' or eeg_duration_type == 'short' or eeg_duration_type == 'long')

        print('fmri frame type={}, eeg normalize type={}, eeg frame type={}'.format(
            fmri_frame_type, eeg_normalize_type, eeg_frame_type))

        if fmri_frame_type == 'normal':
            self.fmri_frame_type = FMRI_FRAME_TYPE_NORMAL
        elif fmri_frame_type == 'average':
            self.fmri_frame_type = FMRI_FRAME_TYPE_AVERAGE
        else:
            self.fmri_frame_type = FMRI_FRAME_TYPE_THREE

        self.fmri_offset_tr = fmri_offset_tr

        if eeg_normalize_type == 'normal':
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_NORMAL
        elif eeg_normalize_type == 'pre':
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_PRE
        else:
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_NONE

        if eeg_duration_type == 'normal':
            self.eeg_duration_type = EEG_DURATION_TYPE_NORMAL
        elif eeg_duration_type == 'short':
            self.eeg_duration_type = EEG_DURATION_TYPE_SHORT
        else:
            self.eeg_duration_type = EEG_DURATION_TYPE_LONG

        if average_trial_size > 0:
            self.use_trial_average = True
        else:
            self.use_trial_average = False

        if self.use_eeg:
            # Loading EEG data
            eeg_suffix = ''

            eeg_duration_type_suffix = ''
            if self.eeg_duration_type == EEG_DURATION_TYPE_LONG:
                eeg_duration_type_suffix = '_long'
            elif self.eeg_duration_type == EEG_DURATION_TYPE_SHORT:
                eeg_duration_type_suffix = '_short'

            if self.eeg_normalize_type == EEG_NORMALIZE_TYPE_NORMAL:
                eeg_data_path = os.path.join(self.data_dir, 'final_eeg_data{}{}'.format(
                    eeg_suffix, eeg_duration_type_suffix))
            elif self.eeg_normalize_type == EEG_NORMALIZE_TYPE_PRE:
                eeg_data_path = os.path.join(self.data_dir, 'final_eeg_data_pre{}{}'.format(
                    eeg_suffix, eeg_duration_type_suffix))
            else:
                eeg_data_path = os.path.join(self.data_dir, 'final_eeg_data_none{}{}'.format(
                    eeg_suffix, eeg_duration_type_suffix))

            if self.use_trial_average:
                # For using trial average
                eeg_data_path = eeg_data_path + '_a{}_r{}_ct{}'.format(average_trial_size,
                                                                       average_repeat_size,
                                                                       classify_type)
                if unmatched:
                    behavior_data_path = behavior_data_path + '_unmatched'
                
            eeg_data_path = eeg_data_path + '.npz'
            
            eeg_data_all = np.load(eeg_data_path)
            eeg_datas = eeg_data_all['eeg_data'] # e.g., (3940, 63, 375) or (3940, 5, 63, 375)...
            
            self.eeg_datas = eeg_datas

        if self.use_fmri:
            # Settings of fMRI data directory
            if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL:
                # For normal
                fmri_data_dir = 'final_fmri_data'
            elif self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
                # For using the average data of 3TR
                fmri_data_dir = 'final_fmri_data_av'
            else:
                # For using the all data of 3TR
                fmri_data_dir = 'final_fmri_data_th'

            if not self.use_smooth:
                # When non-smoothing data is used, '_nosmooth' is appended to the end of the directory name
                fmri_data_dir = fmri_data_dir + '_nosmooth'

            if self.fmri_offset_tr != 2:
                fmri_data_dir = fmri_data_dir + f'_tr{self.fmri_offset_tr}'

            if self.use_trial_average:
                # For using trial averaged data
                fmri_data_dir = fmri_data_dir + '_a{}_r{}_ct{}'.format(
                    average_trial_size,
                    average_repeat_size,
                    classify_type)
                if unmatched:
                    fmri_data_dir = fmri_data_dir + '_unmatched'
            
            self.fmri_data_dir = os.path.join(self.data_dir, fmri_data_dir)
            
        # The method used is to store all data as it is and change only the indexes
        
        # Loading the behavior data
        behavior_data_path = os.path.join(self.data_dir, 'final_behavior_data')

        if self.use_trial_average:
            behavior_data_path = behavior_data_path + '_a{}_r{}_ct{}'.format(
                average_trial_size, average_repeat_size, classify_type)
            if unmatched:
                behavior_data_path = behavior_data_path + '_unmatched'
        
        if debug:
            behavior_data_path = behavior_data_path + '_debug'
        
        behavior_data_all = np.load(behavior_data_path + '.npz')
        
        categories     = behavior_data_all['category']     # (3940),
        sub_categories = behavior_data_all['sub_category'] # (3940),
        subjects       = behavior_data_all['subject']      # (3940),
        identities     = behavior_data_all['identity']     # (3940),
        angles         = behavior_data_all['angle']        # (3940),
        
        unique_subjects = np.unique(behavior_data_all['subject'])
        unique_subjects = np.sort(unique_subjects)

        train_validation_subjects = np.sort(list(set(unique_subjects) - set(test_subjects)))

        # To divide subjects independent of a random seed, the seed is temporarily fixed at 0 at this point
        np.random.seed(0)
        # Shuffle subject IDs for training & validation
        np.random.shuffle(train_validation_subjects)
        
        # Fix a random seed with argument 'data_seed'
        if data_seed != 0:
            # To be compatible with the previous one, the random seed is continuously used when it is 0
            np.random.seed(data_seed)

        if debug:
            # When debugging, the number of participants per fold is forced to 1
            subjects_per_fold = 1

        assert fold < len(train_validation_subjects) // subjects_per_fold

        # Divided the participants into training/validation with 'subjects_per_fold'
        train_subjects = []
        validation_subjects = None

        for i in range(len(train_validation_subjects) // subjects_per_fold):
            fold_subjects = train_validation_subjects[i*subjects_per_fold:(i+1)*subjects_per_fold]
            if i == fold:
                validation_subjects = list(fold_subjects)
            else:
                train_subjects += list(fold_subjects)
        
        # Location of trial for training, validation, and test: [True, False, ....]
        test_subject_set       = set(test_subjects)
        train_subject_set      = set(train_subjects)
        validation_subject_set = set(validation_subjects)

        test_trials       = np.array([subject in test_subject_set       for subject in subjects])
        train_trials      = np.array([subject in train_subject_set      for subject in subjects])
        validation_trials = np.array([subject in validation_subject_set for subject in subjects])

        test_indices       = np.where(test_trials == True)[0]
        train_indices      = np.where(train_trials == True)[0]
        vaidation_indices  = np.where(validation_trials == True)[0]

        if data_type == DATA_TYPE_TRAIN:
            trial_mask = train_trials
        elif data_type == DATA_TYPE_VALIDATION:
            trial_mask = validation_trials
        elif data_type == DATA_TYPE_TEST:
            trial_mask = test_trials
        
        all_indices0, all_indices1 = self.get_indices(classify_type,
                                                      categories,
                                                      sub_categories,
                                                      identities,
                                                      angles,
                                                      trial_mask,
                                                      for_test=(data_type==DATA_TYPE_TEST))
                
        if data_type == DATA_TYPE_TRAIN or data_type == DATA_TYPE_VALIDATION:
            # For training, shuffle the index array (position in the original data) for label 0 and 1
            if classify_type == FRONT_SIDE:
                # Augment to balance label sizes
                all_indices0 = np.repeat(all_indices0, 4)
            
            np.random.shuffle(all_indices0)
            np.random.shuffle(all_indices1)
            
        indices0 = all_indices0 # Index array of class 0 (e.g., Face)
        indices1 = all_indices1 # Index array of class 1 (e.g., Object)

        if data_type == DATA_TYPE_TEST:
            if classify_type != FRONT_SIDE:
                # For test data, the number of positives and negatives should be the same
                min_size = np.min([len(indices0), len(indices1)])
                indices0 = indices0[:min_size]
                indices1 = indices1[:min_size]
            else:
                # Make index0:index 1:4
                index_size0 = len(indices0) # 1/5
                index_size1 = len(indices1) # 4/5

                if index_size1 > index_size0 * 4:
                    indices1 = indices1[:(len(indices0) * 4)]
                else:
                    # Make index1 size even                    
                    over_size1 = index_size1 % 4
                    indices1 = indices1[:(index_size1-over_size1)]
                    indices0 = indices0[:(len(indices1) // 4)]
        
        # Prepare an int array for labels, and input 0 for class 0, 1 for class 1, and -1 for all others
        labels = np.ones([len(categories)], dtype=np.int32) * -1
        labels[indices0] = 0
        labels[indices1] = 1
        
        self.labels = labels
        
        # Locations on original data
        indices = np.hstack([indices0, indices1])
        if data_type == DATA_TYPE_TRAIN:
            np.random.shuffle(indices)
        self.indices = indices
        
        # fMRI Masking
        self.fmri_org_mask = None
        self.fmri_shuffle_mask = None
        self.shuffled_indices = None
        
        if fmri_mask_name is not None:
            mask_path = os.path.join(os.path.dirname(__file__),
                                     f'experiment_data/mask_{fmri_mask_name}.npz')
            mask_data = np.load(mask_path)['mask']
            # (79, 95, 79)
            mask_data = mask_data[np.newaxis,:,:,:]
            # (1, 79, 95, 79)
            
            print(f'fmri mask loaded: {mask_path}')
            
            if pfi_seed is not None:
                # fMRI Mask for for permutation feature importance
                self.fmri_org_mask = 1.0 - mask_data
                self.fmri_shuffle_mask = mask_data
                
                rng = np.random.default_rng(pfi_seed)
                self.shuffled_indices = rng.permutation(self.indices)
            else:
                # fMRI Mask for input masking
                self.fmri_org_mask = mask_data
        
    def get_indices(self,
                    classify_type,
                    categories,
                    sub_categories,
                    identities,
                    angles,
                    trial_mask,
                    for_test):
        if classify_type == FACE_OBJECT:
            # Index array of Face
            flags0 = [w0 and w1 for w0, w1 in \
                      zip((categories == CATEGORY_FACE),
                          (trial_mask == True))]
            # Index array of Object
            flags1 = [w0 and w1 for w0, w1 in \
                      zip((categories == CATEGORY_OBJECT),
                          (trial_mask == True))]
        elif classify_type == MALE_FEMALE:
            # Index array of Male Face
            flags0 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_FACE),
                          (sub_categories == SUBCATEGORY_MALE),
                          (trial_mask == True))]
            # Index array of Female Face
            flags1 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_FACE),
                          (sub_categories == SUBCATEGORY_FEMALE),
                          (trial_mask == True))]
        elif classify_type == ARTIFICIAL_NATURAL:
            # Index array of Artificial Object
            flags0 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_OBJECT),
                          (sub_categories == SUBCATEGORY_ARTIFICIAL),
                          (trial_mask == True))]
            # Index array of Natural Object
            flags1 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_OBJECT),
                          (sub_categories == SUBCATEGORY_NATURAL),
                          (trial_mask == True))]
        elif classify_type == FRONT_SIDE:
            # Front
            flags0 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_FACE),
                          (angles == 2) | (angles == -2),
                          (trial_mask == True))]
            # Side
            flags1 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_FACE),
                          (angles == 0) | (angles == 1) | (angles == 3) | (angles == -3),
                          (trial_mask == True))]                          
        elif classify_type == SMALL_LARGE:
            # small
            flags0 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_OBJECT),
                          (identities == 0) | (identities == 2) | (identities == -2),
                          (trial_mask == True))]
            # Large
            flags1 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_OBJECT),
                          (identities == 1) | (identities == 3) | (identities == -3),
                          (trial_mask == True))]
        else:
            assert False
            
        all_indices0 = np.where(flags0)[0]
        all_indices1 = np.where(flags1)[0]
        return all_indices0, all_indices1

    def load_fmri_frame_data(self, index):
        dir_index = index // 100
        file_path = os.path.join(self.fmri_data_dir,
                                 'frames{}/frame{}.npy'.format(dir_index, index))
        data = np.load(file_path)
        if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL or \
           self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
            data = np.reshape(data, [1,79,95,79])
        # (79, 95, 79)
        return data
        
    def __getitem__(self, index):
        real_index = self.indices[index]

        label = self.labels[real_index].astype(np.float32).reshape([1])

        sample = {
            'label' : label
        }

        if self.use_eeg:
            eeg_data = self.eeg_datas[real_index]
            # (63, 375)
            # For normal
            eeg_data = eeg_data[:,125:] # Excluding from -0.5 seconds to 0 seconds # (63, 250)

            sample.update( {
                'eeg_data': eeg_data,
            })
        
        if self.use_fmri:
            fmri_data = self.load_fmri_frame_data(real_index)
            
            if self.fmri_org_mask is not None:
                # Mask fMRI input
                fmri_data = self.fmri_org_mask * fmri_data
                
            if self.fmri_shuffle_mask is not None:
                # For pemutation feature importance shuffling
                real_shuffled_index = self.shuffled_indices[index]
                shuffled_fmri_data = self.load_fmri_frame_data(real_shuffled_index)
                shuffled_masked_fmri_data = shuffled_fmri_data * self.fmri_shuffle_mask
                fmri_data = fmri_data + shuffled_masked_fmri_data
            
            sample.update( {
                'fmri_data' : fmri_data
            })
        
        return sample

    def __len__(self):
        return len(self.indices)

    # TODO: Can be deleted    
    @property
    def fmri_ch_size(self):
        if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL or \
           self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
            return 1
        else:
            return 3


class DebugDataset(Dataset):
    """
    Dummy Dataset for the test.
    """
    def __init__(self,
                 **kwargs):

        def get_arg(d, name, default):
            if name in d:
                return d[name]
            else:
                return default
        
        self.use_fmri = get_arg(kwargs, 'use_fmri', False)
        self.use_eeg = get_arg(kwargs, 'use_eeg', False)

        # TODO: Can be deleted
        fmri_frame_type = get_arg(kwargs, 'fmri_frame_type', 'normal')

        if fmri_frame_type == 'normal':
            self.fmri_frame_type = FMRI_FRAME_TYPE_NORMAL
        elif fmri_frame_type == 'average':
            self.fmri_frame_type = FMRI_FRAME_TYPE_AVERAGE
        else:
            self.fmri_frame_type = FMRI_FRAME_TYPE_THREE
            
    def __getitem__(self, index):
        if index % 2 == 0:
            label = np.array([1.0], dtype=np.float32)
        else:
            label = np.array([0.0], dtype=np.float32)
        
        sample = {
            'label' : label
        }
        
        if self.use_eeg:
            eeg_data = np.zeros((63, 250), dtype=np.float32)
            sample.update( {
                'eeg_data': eeg_data,
            })

        if self.use_fmri:
            fmri_data = np.zeros((1, 79, 95, 79), dtype=np.float32)
            sample.update( {
                'fmri_data' : fmri_data
            })
        
        return sample

    def __len__(self):
        return 256

    # TODO: Can be deleted
    @property
    def fmri_ch_size(self):
        if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL or \
           self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
            return 1
        else:
            return 3
    
def get_dataset_sub(**kwargs):
    if 'debug' in kwargs and kwargs['debug'] == True:
        return DebugDataset(**kwargs)
    else:
        return BrainDataset(**kwargs)


def get_dataset(combine_type,
                data_type,
                classify_type,
                fold,
                args,
                ignore_fmri_mask_arg=False,
                pfi_seed=None):
    
    test_subject_ids = get_test_subject_ids(args.test_subjects)

    fmri_mask_name = args.fmri_mask
    if ignore_fmri_mask_arg == True:
        fmri_mask_name = None
    
    return get_dataset_sub(data_type=data_type,
                           classify_type=classify_type,
                           data_seed=args.data_seed,
                           use_fmri=(combine_type != COMBINE_TYPE_EEG),
                           use_eeg=(combine_type != COMBINE_TYPE_FMRI),
                           data_dir=args.data_dir,
                           eeg_normalize_type=args.eeg_normalize_type,
                           eeg_frame_type=args.eeg_frame_type,
                           eeg_duration_type=args.eeg_duration_type,
                           fmri_frame_type=args.fmri_frame_type,
                           fmri_offset_tr=args.fmri_offset_tr,
                           use_smooth=args.smooth,
                           average_trial_size=args.average_trial_size,
                           average_repeat_size=args.average_repeat_size,
                           fold=fold,
                           test_subjects=test_subject_ids,
                           subjects_per_fold=args.subjects_per_fold,
                           unmatched=args.unmatched,
                           fmri_mask_name=fmri_mask_name,
                           pfi_seed=pfi_seed,
                           debug=args.debug)
