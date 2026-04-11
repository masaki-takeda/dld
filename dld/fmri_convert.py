import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/data2/DLD/Data_Converted_EEG_ICAed')
    parser.add_argument('--output_dir', type=str,
                        default='/data3/DLD2/Data_Converted_Matlab')
    args = parser.parse_args()
    return args


def process(args, classify_type):
    behavior_data_path = Path(args.data_dir) / f'final_behavior_data_a9_r9_ct{classify_type}.npz'
    behavior_data_all = np.load(behavior_data_path)
    
    categories     = behavior_data_all['category']     # (3940),
    sub_categories = behavior_data_all['sub_category'] # (3940),
    subjects       = behavior_data_all['subject']      # (3940),

    frame_size = len(categories)    

    index = np.arange(frame_size)
    
    behavior_df = pd.DataFrame({
        'index' : index,
        'categoy' : categories,
        'sub_category' : sub_categories,
        'subject' : subjects,
    })
    
    input_base_dir = Path(args.data_dir) /  f'final_fmri_data_nosmooth_a9_r9_ct{classify_type}'
    
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = output_dir / f'category_a9_r9_ct{classify_type}.csv'
    
    behavior_df.to_csv(csv_path, index=False)
    
    frame_output_dir = output_dir /  f'matlab_nosmooth_a9_r9_ct{classify_type}'

    os.makedirs(frame_output_dir, exist_ok=True)
    
    for frame_index in tqdm(range(frame_size)):
        dir_name = 'frames{}'.format(frame_index // 100)
        
        path = input_base_dir / dir_name / f'frame{frame_index}.npy'
        frame = np.load(path)
        # (79, 95, 79)
        # (z,y,x)
        frame = np.transpose(frame, [2, 1, 0])
        # (x,y,z)
        
        mat_output_file_path = frame_output_dir / f'frame{frame_index}.mat'
        
        save_data = {
            'data' : frame,
        }
        
        scipy.io.savemat(mat_output_file_path, save_data)
        
    
def main():
    args = get_args()
    
    for i in range(5):
        process(args, i)
    
    
if __name__ == '__main__':
    main()
