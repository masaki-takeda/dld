#!/bin/sh

# for Behavior, EEG (normal), fMRI (normal)
python3 preprocess.py --use_alpha_setting=true --fmri=true --eeg=true --behavior=true --src_base=/data1/DLD/Data_Prepared --dst_base=/data1/DLD/Data_Converted --fmri_frame_type=normal --eeg_normalize_type=normal

# for EEG (pre)
python3 preprocess.py --use_alpha_setting=true --fmri=false --eeg=true --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data1/DLD/Data_Converted --eeg_normalize_type=pre

# for EEG (none)
python3 preprocess.py --use_alpha_setting=true --fmri=false --eeg=true --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data1/DLD/Data_Converted --eeg_normalize_type=none

# for additional fMRI (3TR average)
python3 preprocess.py --use_alpha_setting=true --fmri=true --eeg=false --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data1/DLD/Data_Converted --fmri_frame_type=average

# for additional fMRI (3TR all)
python3 preprocess.py --use_alpha_setting=true --fmri=true --eeg=false --behavior=false --src_base=/data1/DLD/Data_Prepared --dst_base=/data1/DLD/Data_Converted --fmri_frame_type=three
