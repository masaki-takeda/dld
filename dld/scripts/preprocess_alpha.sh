#!/bin/sh

# Behavior
# EEG (normal)
# fMRI (normal, smooth)
python3 preprocess.py --fmri=true --eeg=true --behavior=true --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --fmri_frame_type=normal --eeg_normalize_type=normal --eeg_frame_type=filter --smooth=true

# EEG (ft)
python3 preprocess.py --fmri=false --eeg=true --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --eeg_normalize_type=normal --eeg_frame_type=ft

# fMRI (normal, non-smooth)
python3 preprocess.py --fmri=true --eeg=false --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --fmri_frame_type=normal --smooth=false


# EEG (pre)
#python3 preprocess.py --fmri=false --eeg=true --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --eeg_normalize_type=pre --eeg_frame_type=filter --smooth=true

# EEG (none)
#python3 preprocess.py --fmri=false --eeg=true --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --eeg_normalize_type=none --eeg_frame_type=filter --smooth=true

# fMRI (3TR average, smooth)
#python3 preprocess.py --fmri=true --eeg=false --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --fmri_frame_type=average --eeg_frame_type=filter --smooth=true

# fMRI (3TR all, smooth)
#python3 preprocess.py --fmri=true --eeg=false --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --fmri_frame_type=three --eeg_frame_type=filter --smooth=true

# fMRI (3TR average, non-smooth)
#python3 preprocess.py --fmri=true --eeg=false --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --fmri_frame_type=average --eeg_frame_type=filter --smooth=false

# fMRI (3TR all, non-smooth)
#python3 preprocess.py --fmri=true --eeg=false --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data2/DLD/Data_Converted --fmri_frame_type=three --eeg_frame_type=filter --smooth=false
