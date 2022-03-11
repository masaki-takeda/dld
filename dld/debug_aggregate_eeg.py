import numpy as np
import os
import argparse
from distutils.util import strtobool


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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_normalize_type", type=str,
                        default="normal") # normal, pre, none
    parser.add_argument("--eeg_frame_type", type=str,
                        default="filter") # normal, filter, ft
    parser.add_argument("--debug", type=strtobool,
                        default="false")
    parser.add_argument("--data_dir", type=str,
                        default="/data2/DLD/Data_Converted")
    
    args = parser.parse_args()

    eeg_frame_type = args.eeg_frame_type
    eeg_normalize_type = args.eeg_normalize_type
    
    suffix = ""
    
    if eeg_frame_type == "filter":
        suffix = "_filter"
    elif eeg_frame_type == "ft":
        suffix = "_ft"

    data_dir = args.data_dir
    
    if eeg_normalize_type == "normal":
        eeg_data_path = os.path.join(data_dir, "final_eeg_data{}.npz".format(suffix))
    elif eeg_normalize_type == "pre":
        eeg_data_path = os.path.join(data_dir, "final_eeg_data_pre{}.npz".format(suffix))
    else:
        eeg_data_path = os.path.join(data_dir, "final_eeg_data_none{}.npz".format(suffix))
    
    eeg_data_all = np.load(eeg_data_path)

    eeg_datas = eeg_data_all["eeg_data"] # (3950, 63, 375)等

    if args.debug:
        behavior_data_path = os.path.join(data_dir, "final_behavior_data_debug.npz")
    else:
        behavior_data_path = os.path.join(data_dir, "final_behavior_data.npz")
        
    behavior_data_all = np.load(behavior_data_path)

    categories     = behavior_data_all["category"]     # (3940),
    identities     = behavior_data_all["identity"]     # (3940),
    sub_categories = behavior_data_all["sub_category"] # (3940),
    angles         = behavior_data_all["angle"]        # (3940),

    face_indices   = np.where(categories == CATEGORY_FACE)[0]   # Face
    object_indices = np.where(categories == CATEGORY_OBJECT)[0] # Object
    face_data   = eeg_datas[face_indices]   # (406, 5, 63, 375)
    object_data = eeg_datas[object_indices] # (385, 5, 63, 375)
    face_mean   = np.mean(face_data, axis=0)   # (5, 63, 375)
    object_mean = np.mean(object_data, axis=0) # (5, 63, 375)

    male_indices = np.where([w0 and w1 for w0, w1 in \
                             zip((categories == CATEGORY_FACE),
                                 (sub_categories == SUBCATEGORY_MALE))])[0]
    female_indices = np.where([w0 and w1 for w0, w1 in \
                             zip((categories == CATEGORY_FACE),
                                 (sub_categories == SUBCATEGORY_FEMALE))])[0]
    male_data   = eeg_datas[male_indices]    # (204, 5, 63, 375)
    female_data = eeg_datas[female_indices]  # (202, 5, 63, 375)
    male_mean   = np.mean(male_data, axis=0)
    female_mean = np.mean(female_data, axis=0)

    artificial_indices = np.where([w0 and w1 for w0, w1 in \
                             zip((categories == CATEGORY_OBJECT),
                                 (sub_categories == SUBCATEGORY_ARTIFICIAL))])[0]
    natural_indices = np.where([w0 and w1 for w0, w1 in \
                             zip((categories == CATEGORY_OBJECT),
                                 (sub_categories == SUBCATEGORY_NATURAL))])[0]
    artificial_data = eeg_datas[artificial_indices]
    natural_data    = eeg_datas[natural_indices]
    artificial_mean = np.mean(artificial_data, axis=0)
    natural_mean    = np.mean(natural_data, axis=0)

    output_file_path  = "eeg_agg_data{}".format(suffix)

    np.savez(output_file_path,
             face=face_mean,
             object=object_mean,
             male=male_mean,
             female=female_mean,
             artificial=artificial_mean,
             natural=natural_mean)

    print("file saved: {}.npz".format(output_file_path))


if __name__ == '__main__':
    main()
