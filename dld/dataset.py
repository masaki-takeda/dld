import numpy as np
import os
from torch.utils.data import Dataset

from eeg import EEG
from behavior import Behavior


# Data Type
DATA_TYPE_TRAIN      = 0
DATA_TYPE_VALIDATION = 1
DATA_TYPE_TEST       = 2

# Classification Type
FACE_OBJECT        = 0
MALE_FEMALE        = 1
ARTIFICIAL_NATURAL = 2
CLASSIFY_ALL       = -1

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

EEG_FRAME_TYPE_NORMAL = 0
EEG_FRAME_TYPE_FILTER = 1
EEG_FRAME_TYPE_FT     = 2

EEG_DURATION_TYPE_NORMAL = 0
EEG_DURATION_TYPE_SHORT  = 1
EEG_DURATION_TYPE_LONG   = 2


class BrainDataset(Dataset):
    def __init__(self,
                 data_type,
                 classify_type,
                 data_seed,
                 use_fmri=False,
                 use_eeg=False,
                 data_dir="./data",
                 fmri_frame_type="normal",
                 fmri_offset_tr=2,
                 eeg_normalize_type="normal",
                 eeg_frame_type="normal",
                 eeg_duration_type="normal",
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
        eeg_normalize_type:
           (str) normalize type of the EEG data
        eeg_frame_type:
           (str) frame type of the eeg data (normal, filter, ft)
        use_smooth:
           (bool) whether to use smoothed fmri data
        average_trial_size:
           (int) average number of trials
        average_repeat_size:
           (int) number of repetitions for augmentation
        subjects_per_fold:
           (int) number of participants assined to one Fold
        """
        self.use_fmri = use_fmri
        self.use_eeg = use_eeg
        self.data_dir = data_dir
        self.use_smooth = use_smooth

        assert (data_type == DATA_TYPE_TRAIN or data_type == DATA_TYPE_VALIDATION or data_type == DATA_TYPE_TEST)
        assert (fmri_frame_type == "normal" or fmri_frame_type == "average" or fmri_frame_type == "three")
        assert (fmri_offset_tr == 1 or fmri_offset_tr == 2 or fmri_offset_tr == 3)
        assert (eeg_normalize_type == "normal" or eeg_normalize_type == "pre" or eeg_normalize_type == "none")
        assert (eeg_frame_type == "normal" or eeg_frame_type == "filter" or eeg_frame_type == "ft")
        assert (eeg_duration_type == "normal" or eeg_duration_type == "short" or eeg_duration_type == "long")

        print("fmri frame type={}, eeg normalize type={}, eeg frame type={}".format(
            fmri_frame_type, eeg_normalize_type, eeg_frame_type))
        
        if fmri_frame_type == "normal":
            self.fmri_frame_type = FMRI_FRAME_TYPE_NORMAL
        elif fmri_frame_type == "average":
            self.fmri_frame_type = FMRI_FRAME_TYPE_AVERAGE
        else:
            self.fmri_frame_type = FMRI_FRAME_TYPE_THREE
        
        self.fmri_offset_tr = fmri_offset_tr
        
        if eeg_normalize_type == "normal":
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_NORMAL
        elif eeg_normalize_type == "pre":
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_PRE
        else:
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_NONE

        if eeg_frame_type == "normal":
            self.eeg_frame_type = EEG_FRAME_TYPE_NORMAL
        elif eeg_frame_type == "filter":
            self.eeg_frame_type = EEG_FRAME_TYPE_FILTER
        else:
            self.eeg_frame_type = EEG_FRAME_TYPE_FT

        if eeg_duration_type == "normal":
            self.eeg_duration_type = EEG_DURATION_TYPE_NORMAL
        elif eeg_duration_type == "short":
            self.eeg_duration_type = EEG_DURATION_TYPE_SHORT
        else:
            self.eeg_duration_type = EEG_DURATION_TYPE_LONG

        if average_trial_size > 0:
            self.use_trial_average = True
        else:
            self.use_trial_average = False

        if self.use_eeg:
            # Loading EEG data
            eeg_frame_type_suffix = ""
            if self.eeg_frame_type == EEG_FRAME_TYPE_FILTER:
                # When using filters in EEG, "_filter" is added to the file name
                eeg_frame_type_suffix = "_filter"
            elif self.eeg_frame_type == EEG_FRAME_TYPE_FT:
                # When using FT_spectrogram in EEG, "_ft" is added to the file name
                eeg_frame_type_suffix = "_ft"

            eeg_duration_type_suffix = ""
            if self.eeg_duration_type == EEG_DURATION_TYPE_LONG:
                eeg_duration_type_suffix = "_long"
            elif self.eeg_duration_type == EEG_DURATION_TYPE_SHORT:
                eeg_duration_type_suffix = "_short"

            if self.eeg_normalize_type == EEG_NORMALIZE_TYPE_NORMAL:
                eeg_data_path = os.path.join(self.data_dir, "final_eeg_data{}{}".format(
                    eeg_frame_type_suffix, eeg_duration_type_suffix))
            elif self.eeg_normalize_type == EEG_NORMALIZE_TYPE_PRE:
                eeg_data_path = os.path.join(self.data_dir, "final_eeg_data_pre{}{}".format(
                    eeg_frame_type_suffix, eeg_duration_type_suffix))
            else:
                eeg_data_path = os.path.join(self.data_dir, "final_eeg_data_none{}{}".format(
                    eeg_frame_type_suffix, eeg_duration_type_suffix))

            if self.use_trial_average:
                # For using trial average
                eeg_data_path = eeg_data_path + "_a{}_r{}_ct{}".format(average_trial_size,
                                                                       average_repeat_size,
                                                                       classify_type)
                if unmatched:
                    eeg_data_path = eeg_data_path + "_unmatched"
                
            eeg_data_path = eeg_data_path + ".npz"
            
            eeg_data_all = np.load(eeg_data_path)
            eeg_datas = eeg_data_all["eeg_data"] # e.g., (3940, 63, 375) or (3940, 5, 63, 375)...

            print(f"eeg_data_path : {eeg_data_path}")
            
            self.eeg_datas = eeg_datas

        if self.use_fmri:
            # Settings of fMRI data directory
            if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL:
                # For normal
                fmri_data_dir = "final_fmri_data"
            elif self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
                # For using the average data of 3TR
                fmri_data_dir = "final_fmri_data_av"
            else:
                # For using the all data of 3TR
                fmri_data_dir = "final_fmri_data_th"

            if not self.use_smooth:
                # When non-smoothing data is used, "_nosmooth" is appended to the end of the directory name
                fmri_data_dir = fmri_data_dir + "_nosmooth"

            if self.fmri_offset_tr != 2:
                fmri_data_dir = fmri_data_dir + f"_tr{self.fmri_offset_tr}"

            if self.use_trial_average:
                # For using trial averaged data
                fmri_data_dir = fmri_data_dir + "_a{}_r{}_ct{}".format(
                    average_trial_size,
                    average_repeat_size,
                    classify_type)
                if unmatched:
                    fmri_data_dir = fmri_data_dir + "_unmatched"
            
            self.fmri_data_dir = os.path.join(self.data_dir, fmri_data_dir)

            print(f"fmri_data_dir : {self.fmri_data_dir}")
            
        # The method used is to store all data as it is and change only the indexes
        
        # Loading the behavior data
        behavior_data_path = os.path.join(self.data_dir, "final_behavior_data")

        if self.use_trial_average:
            behavior_data_path = behavior_data_path + "_a{}_r{}_ct{}".format(
                average_trial_size, average_repeat_size, classify_type)
            if unmatched:
                behavior_data_path = behavior_data_path + "_unmatched"
            
        if debug:
            behavior_data_path = behavior_data_path + "_debug"

        print(f"behavior_data_path : {behavior_data_path}.npz")
        
        behavior_data_all = np.load(behavior_data_path + ".npz")
        
        categories     = behavior_data_all["category"]     # (3940),
        sub_categories = behavior_data_all["sub_category"] # (3940),
        subjects       = behavior_data_all["subject"]      # (3940),

        unique_subjects = np.unique(behavior_data_all['subject'])
        unique_subjects = np.sort(unique_subjects)

        train_validation_subjects = np.sort(list(set(unique_subjects) - set(test_subjects)))

        # To divide subjects independent of a random seed, the seed is temporarily fixed at 0 at this point
        np.random.seed(0)
        # Shuffle subject IDs for training & validation
        np.random.shuffle(train_validation_subjects)
        
        # Fix a random seed with argument "data_seed"
        if data_seed != 0:
            # To be compatible with the previous one, the random seed is continuously used when it is 0
            np.random.seed(data_seed)

        if debug:
            # When debugging, the number of participants per fold is forced to 1
            subjects_per_fold = 1

        assert fold < len(train_validation_subjects) // subjects_per_fold

        # Divided the participants into training/validation with "subjects_per_fold"
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
                                                      trial_mask,
                                                      for_test=(data_type==DATA_TYPE_TEST))
                
        if data_type == DATA_TYPE_TRAIN:
            # For training, shuffle the index array (position in the original data) for label 0 and 1
            np.random.shuffle(all_indices0)
            np.random.shuffle(all_indices1)
            
        indices0 = all_indices0 # Index array of class 0 (e.g., Face)
        indices1 = all_indices1 # Index array of class 1 (e.g., Object)

        if data_type == DATA_TYPE_TEST:
            # For test data, the number of positives and negatives should be the same
            min_size = np.min([len(indices0), len(indices1)])
            indices0 = indices0[:min_size]
            indices1 = indices1[:min_size]
        
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

    def get_indices(self, classify_type, categories, sub_categories, trial_mask, for_test):
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

        all_indices0 = np.where(flags0)[0]
        all_indices1 = np.where(flags1)[0]
        return all_indices0, all_indices1

    def load_fmri_frame_data(self, index):
        dir_index = index // 100
        file_path = os.path.join(self.fmri_data_dir,
                                 "frames{}/frame{}.npy".format(dir_index, index))
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
            # (63, 375) or (5, 63, 375) or (17, 63, 163)
            if self.eeg_frame_type == EEG_FRAME_TYPE_FILTER:
                # For using Filter
                eeg_data = eeg_data[:,:,125:] # Excluding from -0.5 seconds to 0 seconds # (5, 63, 250)
            elif self.eeg_frame_type == EEG_FRAME_TYPE_FT:
                # For using FT
                eeg_data = eeg_data[:,:,38:] # Excluding from -0.5 seconds to 0 seconds # (5, 63, 125)
            else:
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

    @property
    def fmri_ch_size(self):
        if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL or \
           self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
            return 1
        else:
            return 3
