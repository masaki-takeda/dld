import numpy as np
import os
import argparse
from distutils.util import strtobool

from dataset import FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL
from dataset import CATEGORY_FACE, CATEGORY_OBJECT, SUBCATEGORY_MALE, SUBCATEGORY_FEMALE, SUBCATEGORY_ARTIFICIAL, SUBCATEGORY_NATURAL


class Subject:
    def __init__(self,
                 subject_id,
                 indices0,
                 indices1,
                 average_trial_size,
                 average_repeat_size):
        self.subject_id = subject_id
        
        extended_indices0 = []
        extended_indices1 = []
        repeat_indices0 = []
        repeat_indices1 = []
        
        for i in range(average_repeat_size):
            extended_indices0.extend(np.random.permutation(indices0))
            repeat_indices0.extend([i] * len(indices0))

            extended_indices1.extend(np.random.permutation(indices1))
            repeat_indices1.extend([i] * len(indices1))

        averaging_indices0 = []
        averaging_indices1 = []
        averaging_repeat_indices0 = []
        averaging_repeat_indices1 = []
        
        for i in range(len(extended_indices0) // average_trial_size):
            pos = i * average_trial_size
            averaging_indices0.append(extended_indices0[pos:pos+average_trial_size])
            # Which repat the last value to take the average is in
            averaging_repeat_index = repeat_indices0[pos+average_trial_size-1]
            averaging_repeat_indices0.append(averaging_repeat_index)

        for i in range(len(extended_indices1) // average_trial_size):
            pos = i * average_trial_size
            averaging_indices1.append(extended_indices1[pos:pos+average_trial_size])
            # Which repat the last value to take the average is in
            averaging_repeat_index = repeat_indices1[pos+average_trial_size-1]
            averaging_repeat_indices1.append(averaging_repeat_index)
            
        self.averaging_indices0 = np.array(averaging_indices0, dtype=np.int32)
        self.averaging_indices1 = np.array(averaging_indices1, dtype=np.int32)
        self.averaging_repeat_indices0 = np.array(averaging_repeat_indices0, dtype=np.int32)
        self.averaging_repeat_indices1 = np.array(averaging_repeat_indices1, dtype=np.int32)
        
        self.subject_ids0 = [self.subject_id] * len(self.averaging_indices0)
        self.subject_ids1 = [self.subject_id] * len(self.averaging_indices1)


class AveragingBehavior:
    def __init__(self,
                 classify_type,
                 indices0,
                 indices1,
                 repeat_indices0,
                 repeat_indices1,
                 subject_ids0,
                 subject_ids1):
                 
        self.classify_type  = classify_type
        self.indices0        = indices0 # (*,3)
        self.indices1        = indices1 # (*,3)
        self.repeat_indices0 = repeat_indices0 # (*,)
        self.repeat_indices1 = repeat_indices1 # (*,)
        self.subject_ids0    = subject_ids0
        self.subject_ids1    = subject_ids1
    
    @property
    def indices(self):
        return np.concatenate([self.indices0, self.indices1], axis=0)

    @property
    def repeat_indices(self):
        return np.concatenate([self.repeat_indices0, self.repeat_indices1], axis=0)

    @property
    def categories(self):
        if self.classify_type == FACE_OBJECT:
            categories = \
                ([CATEGORY_FACE] * len(self.indices0)) + \
                ([CATEGORY_OBJECT] * len(self.indices1))
        elif self.classify_type == MALE_FEMALE:
            categories = [CATEGORY_FACE] * (len(self.indices0) + len(self.indices1))
        else:
            categories = [CATEGORY_OBJECT] * (len(self.indices0) + len(self.indices1))
        return np.array(categories, dtype=np.int32)

    @property
    def sub_categories(self):
        if self.classify_type == FACE_OBJECT:
            sub_categories = [-1] * (len(self.indices0) + len(self.indices1))
        elif self.classify_type == MALE_FEMALE:
            sub_categories = \
                ([SUBCATEGORY_MALE] * len(self.indices0)) + \
                ([SUBCATEGORY_FEMALE] * len(self.indices1))
        else:
            sub_categories = \
                ([SUBCATEGORY_ARTIFICIAL] * len(self.indices0)) + \
                ([SUBCATEGORY_NATURAL] * len(self.indices1))
        return np.array(sub_categories, dtype=np.int32)

    @property
    def subject_ids(self):
        return np.concatenate([self.subject_ids0, self.subject_ids1], axis=0)
        
        
def preprocess_average_behavior(behavior_data,
                                classify_type,
                                average_trial_size,
                                average_repeat_size):
    
    categories     = behavior_data["category"]     # (3940),
    sub_categories = behavior_data["sub_category"] # (3940),
    subjects       = behavior_data["subject"]      # (3940),

    subject_ids = np.unique(subjects)

    subject_objs = []
    
    for subject_id in subject_ids:
        # Process one participant at a time
        if classify_type == FACE_OBJECT:
            # Face index array
            indices0 = np.where([w0 and w1 for w0, w1 in \
                                 zip((categories == CATEGORY_FACE),
                                     (subjects == subject_id))])[0]
            # Object index array
            indices1 = np.where([w0 and w1 for w0, w1 in \
                                 zip((categories == CATEGORY_OBJECT),
                                     (subjects == subject_id))])[0]
        elif classify_type == MALE_FEMALE:
            # Male-Face index array
            indices0 = np.where([w0 and w1 and w2 for w0, w1, w2 in \
                                 zip((categories == CATEGORY_FACE),
                                     (sub_categories == SUBCATEGORY_MALE),
                                     (subjects == subject_id))])[0]
            # Female-Face index array
            indices1 = np.where([w0 and w1 and w2 for w0, w1, w2 in \
                                 zip((categories == CATEGORY_FACE),
                                     (sub_categories == SUBCATEGORY_FEMALE),
                                     (subjects == subject_id))])[0]
        else:
            # Artificial-Object index array
            indices0 = np.where([w0 and w1 and w2 for w0, w1, w2 in \
                                 zip((categories == CATEGORY_OBJECT),
                                     (sub_categories == SUBCATEGORY_ARTIFICIAL),
                                     (subjects == subject_id))])[0]
            # Natural-Object index array
            indices1 = np.where([w0 and w1 and w2 for w0, w1, w2 in \
                                 zip((categories == CATEGORY_OBJECT),
                                     (sub_categories == SUBCATEGORY_NATURAL),
                                     (subjects == subject_id))])[0]
        subject_obj = Subject(subject_id,
                              indices0,
                              indices1,
                              average_trial_size,
                              average_repeat_size)
        subject_objs.append(subject_obj)

    averaging_indices0 = np.concatenate(
        [subject_obj.averaging_indices0 for subject_obj in subject_objs],
        axis=0) # (***, 3)
    averaging_indices1 = np.concatenate(
        [subject_obj.averaging_indices1 for subject_obj in subject_objs],
        axis=0) # (***, 3)

    averaging_repeat_indices0 = np.concatenate(
        [subject_obj.averaging_repeat_indices0 for subject_obj in subject_objs],
        axis=0) # (***,)
    averaging_repeat_indices1 = np.concatenate(
        [subject_obj.averaging_repeat_indices1 for subject_obj in subject_objs],
        axis=0) # (***,)

    subject_ids0 = np.concatenate(
        [subject_obj.subject_ids0 for subject_obj in subject_objs],
        axis=0) # (***,)
    subject_ids1 = np.concatenate(
        [subject_obj.subject_ids1 for subject_obj in subject_objs],
        axis=0) # (***,)

    averaging_behavior = AveragingBehavior(classify_type,
                                           averaging_indices0,
                                           averaging_indices1,
                                           averaging_repeat_indices0,
                                           averaging_repeat_indices1,
                                           subject_ids0,
                                           subject_ids1)
    return averaging_behavior
    

def preprocess_average_eeg(dst_base,
                           averaging_behavior,
                           average_trial_size,
                           average_repeat_size,
                           eeg_normalize_type,
                           eeg_frame_type):
    
    assert (eeg_normalize_type == "normal" or eeg_normalize_type == "pre" or eeg_normalize_type == "none")
    assert (eeg_frame_type == "normal" or eeg_frame_type == "filter" or eeg_frame_type == "ft")

    print("processing eeg: classify_type={}".format(averaging_behavior.classify_type))

    eeg_suffix = ""
    if eeg_frame_type == "filter":
        # Append "_filter" to the end of the file name when using filters in EEG
        eeg_suffix = "_filter"
    elif eeg_frame_type == "ft":
        # Append "_ft" to the end of the file name when using FT-spectrograms in EEG
        eeg_suffix = "_ft"

    if eeg_normalize_type == "normal":
        eeg_data_path_base = os.path.join(dst_base, "final_eeg_data{}".format(
            eeg_suffix))
    elif eeg_normalize_type == "pre":
        eeg_data_path_base = os.path.join(dst_base, "final_eeg_data_pre{}".format(
            eeg_suffix))
    else:
        eeg_data_path_base = os.path.join(dst_base, "final_eeg_data_none{}".format(
            eeg_suffix))
        
    eeg_data_all = np.load(eeg_data_path_base + ".npz")
    eeg_datas = eeg_data_all["eeg_data"] # (3940, 63, 375) or (3940, 5, 63, 375)等
    
    indices = averaging_behavior.indices
    avaraging_eeg_datas = eeg_datas[indices] # (1053, 3, 5, 63, 375)
    mean_eeg_datas = avaraging_eeg_datas.mean(axis=1) # (1053, 5, 63, 375)

    output_file_path = "{}_a{}_r{}_ct{}".format(eeg_data_path_base,
                                                average_trial_size,
                                                average_repeat_size,
                                                averaging_behavior.classify_type)
    np.savez(output_file_path,
             eeg_data=mean_eeg_datas)


def load_fmri_frame_data(input_fmri_data_dir, index):
    dir_index = index // 100
    file_path = os.path.join(input_fmri_data_dir,
                             "frames{}/frame{}.npy".format(dir_index, index))
    data = np.load(file_path)
    return data


def save_fmri_frame_data(frame_data,
                         output_fmri_data_dir,
                         index):

    # Separate storage directories for each 100
    dir_name = "frames{}".format(index // 100)
    dir_path = os.path.join(output_fmri_data_dir, dir_name)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    file_path = os.path.join(dir_path, "frame{}".format(index))
    np.save(file_path, frame_data)
    
    
def preprocess_average_fmri(dst_base,
                            averaging_behavior,
                            average_trial_size,
                            average_repeat_size,
                            fmri_frame_type,
                            smooth):
    assert (fmri_frame_type == "normal" or fmri_frame_type == "average" or fmri_frame_type == "three")

    print("processing fmri: classify_type={}".format(averaging_behavior.classify_type))

    if fmri_frame_type == "normal":
        # For normal
        fmri_data_dir = "final_fmri_data"
    elif fmri_frame_type == "average":
        # For using the average data of 3TR
        fmri_data_dir = "final_fmri_data_av"
    else:
        # For using the all data of 3TR
        fmri_data_dir = "final_fmri_data_th"

    if not smooth:
        # Add "_nosmooth" to the end of the directory name when using non-smoothing data
        fmri_data_dir = fmri_data_dir + "_nosmooth"

    input_fmri_data_dir = os.path.join(dst_base, fmri_data_dir)
    output_fmri_data_dir = os.path.join(dst_base, fmri_data_dir) + "_a{}_r{}_ct{}".format(
        average_trial_size,
        average_repeat_size,
        averaging_behavior.classify_type)

    if not os.path.exists(output_fmri_data_dir):
        os.mkdir(output_fmri_data_dir)
    
    averaging_indices = averaging_behavior.indices
    # e.g., (1053, 3)

    output_frame_index = 0
    
    for indices in averaging_indices:
        # (3,)
        frames = []
        for index in indices:
            frame = load_fmri_frame_data(input_fmri_data_dir, index)
            frames.append(frame)
        mean_frame = np.mean(np.array(frames), axis=0)
        save_fmri_frame_data(mean_frame,
                             output_fmri_data_dir,
                             output_frame_index)
        output_frame_index += 1


def save_averaging_behavior_data(dst_base,
                                 averaging_behavior,
                                 average_trial_size,
                                 average_repeat_size,
                                 debug):

    file_name = "final_behavior_data_a{}_r{}_ct{}".format(
        average_trial_size,
        average_repeat_size,
        averaging_behavior.classify_type)

    
    output_file_path = os.path.join(dst_base, file_name) # File name of npz
    if debug:
        output_file_path = output_file_path + "_debug"
    
    np.savez_compressed(output_file_path,
                        category=averaging_behavior.categories,
                        sub_category=averaging_behavior.sub_categories,
                        repeat_index=averaging_behavior.repeat_indices,
                        subject=averaging_behavior.subject_ids)


def preprocess_average():
    parser = argparse.ArgumentParser()
    parser.add_argument("--average_trial_size", type=int,
                        default=0)
    parser.add_argument("--average_repeat_size", type=int,
                        default=0)
    parser.add_argument("--fmri", type=strtobool,
                        default="true")
    parser.add_argument("--eeg", type=strtobool,
                        default="true")
    parser.add_argument("--smooth", type=strtobool,
                        default="true")
    parser.add_argument("--eeg_normalize_type", type=str,
                        default="normal") # normal, pre, none
    parser.add_argument("--dst_base", type=str,
                        default="./data")
    parser.add_argument("--fmri_frame_type", type=str,
                        default="normal") # normal, avarage, three
    parser.add_argument("--eeg_frame_type", type=str,
                        default="filter") # normal, filter, ft
    parser.add_argument("--debug", type=strtobool,
                        default="false")
    args = parser.parse_args()

    # Data inport/export location
    dst_base = args.dst_base

    if args.debug:
        behavior_data_path = os.path.join(dst_base, "final_behavior_data_debug.npz")
    else:
        behavior_data_path = os.path.join(dst_base, "final_behavior_data.npz")

    behavior_data = np.load(behavior_data_path)

    # Fix the random seed for using in trial average
    np.random.seed(0)

    for ct in [FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL]:
        averaging_behavior = preprocess_average_behavior(
            behavior_data,
            classify_type=ct,
            average_trial_size=args.average_trial_size,
            average_repeat_size=args.average_repeat_size)

        if args.eeg:
            preprocess_average_eeg(dst_base,
                                   averaging_behavior,
                                   average_trial_size=args.average_trial_size,
                                   average_repeat_size=args.average_repeat_size,
                                   eeg_normalize_type=args.eeg_normalize_type,
                                   eeg_frame_type=args.eeg_frame_type)

        if args.fmri:
            preprocess_average_fmri(dst_base,
                                    averaging_behavior,
                                    average_trial_size=args.average_trial_size,
                                    average_repeat_size=args.average_repeat_size,
                                    fmri_frame_type=args.fmri_frame_type,
                                    smooth=args.smooth)
        
        # Save averaging behavior data
        save_averaging_behavior_data(dst_base,
                                     averaging_behavior,
                                     average_trial_size=args.average_trial_size,
                                     average_repeat_size=args.average_repeat_size,
                                     debug=args.debug)
    
if __name__ == '__main__':
    preprocess_average()
