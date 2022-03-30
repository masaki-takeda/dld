import numpy as np
import h5py

from behavior import Behavior

# Prepare data for channel interpolation
ch_names = ['Fp1', 'Fp2', 'F3', 'F4',
            'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8',
            'T7', 'T8', 'P7', 'P8',
            'Fz', 'Cz', 'Pz', 'Oz',
            'FC1', 'FC2', 'CP1', 'CP2',
            'FC5', 'FC6', 'CP5', 'CP6',
            'TP9', 'TP10', 'POz', 'ECG',
            'F1', 'F2', 'C1', 'C2',
            'P1', 'P2', 'AF3', 'AF4',
            'FC3', 'FC4', 'CP3', 'CP4',
            'PO3', 'PO4', 'F5', 'F6',
            'C5', 'C6', 'P5', 'P6',
            'AF7', 'AF8', 'FT7', 'FT8',
            'TP7', 'TP8', 'PO7', 'PO8',
            'FT9', 'FT10', 'Fpz', 'CPz']

ch_name_index_map = {}
for i in range(64):
    ch_name = ch_names[i]
    if ch_name == 'ECG':
        # Exclude ECG(31)
        continue
    if i < 31:
        ch_index = i
    else:
        # ECG and subsequent channel numbers are set to -1
        ch_index = i-1
    ch_name_index_map[ch_name] = ch_index


def interpolate_noisy_channel(data, noisy_ch_name, neighbor_ch_names):
    noisy_ch = ch_name_index_map[noisy_ch_name]
    if data.ndim == 4:
        # For using Filter
        neighbor_data = np.empty([len(neighbor_ch_names), data.shape[1], data.shape[2],
                                  data.shape[3]],
                                 dtype=data.dtype)
    else:
        # For normal
        neighbor_data = np.empty([len(neighbor_ch_names), data.shape[1], data.shape[2]],
                             dtype=data.dtype)

    for i, neighbor_ch_name in enumerate(neighbor_ch_names):
        neighbor_ch = ch_name_index_map[neighbor_ch_name]
        neighbor_data[i] = data[neighbor_ch]

    interpolated_ch_data = neighbor_data.mean(axis=0)

    interpolated_data = np.array(data)
    interpolated_data[noisy_ch] = interpolated_ch_data
    return interpolated_data


def process_noisy_channel(behavior, data):
    """Exception handling for interpolation of noisy channels
    (Hard corded)
    """
    if behavior.date == 191009 and behavior.subject == 1 and (5 <= behavior.run <= 8):
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'AF3', ['Fp1', 'AF7', 'F1', 'F3'])
        return data
    elif behavior.date == 191015 and behavior.subject == 1 and behavior.run == 4:
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'AF4',  ['Fp2', 'AF8', 'F2', 'F4'])
        return data
    elif behavior.date == 191119 and behavior.subject == 1 and behavior.run == 2:
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'C2',  ['Cz', 'FC2', 'C4', 'CP2'])
        return data
    elif behavior.date == 191210 and behavior.subject == 1 and (8 <= behavior.run <= 10):
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'F1',  ['AF3', 'F3', 'FC1', 'Fz'])
        return data
    elif behavior.date == 191213 and behavior.subject == 1 and (9 <= behavior.run <= 12):
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'FC3',  ['F3', 'FC5', 'C3', 'FC1'])
        return data
    elif behavior.date == 200108 and behavior.subject == 1:
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'C1',  ['C3', 'FC1', 'Cz', 'CP1'])
        data = interpolate_noisy_channel(data, 'FC2', ['Fz', 'FC4', 'C2'])
        return data
    elif behavior.date == 200110 and behavior.subject == 1:
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'Fp1', ['AF7', 'AF3', 'Fpz'])
        return data
    elif behavior.date == 200117 and behavior.subject == 1 and (7 <= behavior.run <= 12):
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'AF7', ['Fp1', 'F7', 'F5', 'AF3'])
        return data
    elif behavior.date == 200130 and behavior.subject == 1:
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        if (1 <= behavior.run <= 2):
            data = interpolate_noisy_channel(data, 'FC2', ['F2', 'C2', 'FC4'])
        elif (3 <= behavior.run <= 6):
            data = interpolate_noisy_channel(data, 'FC2', ['F2', 'C2', 'FC4'])
            data = interpolate_noisy_channel(data, 'Fp2', ['Fpz', 'AF4', 'AF8'])
        elif (7 <= behavior.run <= 8):
            data = interpolate_noisy_channel(data, 'FC2', ['F2', 'C2', 'FC4'])
            data = interpolate_noisy_channel(data, 'Fp2', ['Fpz', 'AF4', 'AF8'])
            data = interpolate_noisy_channel(data, 'AF4',  ['Fp2', 'AF8', 'F2', 'F4'])
        elif (9 <= behavior.run <= 12):
            data = interpolate_noisy_channel(data, 'FC2', ['F2', 'C2', 'FC4'])
            data = interpolate_noisy_channel(data, 'Fp2', ['Fpz', 'AF4', 'AF8'])
            data = interpolate_noisy_channel(data, 'AF4',  ['Fp2', 'AF8', 'F2', 'F4'])
            data = interpolate_noisy_channel(data, 'F4',  ['F6', 'FC4', 'F2'])
        return data
    elif behavior.date == 200310 and behavior.subject == 1:
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'CPz', ['Cz', 'CP2', 'Pz', 'CP1'])
        return data
    elif behavior.date == 200629 and behavior.subject == 1 and (1 <= behavior.run <= 2):
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'FC6', ['F6', 'FT8', 'C6', 'FC4'])
        return data
    elif behavior.date == 200710 and behavior.subject == 1 and (4 <= behavior.run <= 12):
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'Fp1', ['Fpz', 'AF3', 'AF7'])
        return data
    elif behavior.date == 200727 and behavior.subject == 1 and behavior.run == 1:
        print("interpolate noisy channel: data={} subject={}".format(
            behavior.date, behavior.subject))
        data = interpolate_noisy_channel(data, 'FC2', ['F2', 'FC4', 'C2'])
        return data
    else:
        return data


class EEG(object):
    """ EEG data (for preprocess) """

    def __init__(self, src_base, behavior, normalize_type="normal",
                 frame_type="normal"):
        assert (normalize_type == "normal" or normalize_type == "pre" or normalize_type == "none")
        assert (frame_type == "normal" or frame_type == "filter" or frame_type == "ft")
        # EEG data: from -0.5 to +1.0 seconds

        path_format = "{0}/EEG/Epoched/TM_{1}_{2:0>2}/TM_{1}_{2:0>2}_{3:0>2}_Segmentation.mat"

        path = path_format.format(src_base,
                                  behavior.date,
                                  behavior.subject,
                                  behavior.run)
        
        with h5py.File(path,'r') as f:
            if frame_type == "normal":
                eeg_data = np.array(f['EEG']['data'])        # (50, 375, 64)
                eeg_data = np.transpose(eeg_data, [2, 1, 0]) # (64, 375, 50)
            elif frame_type == "filter":
                eeg_data = np.array(f['EEGfilt'])               # (5, 50, 5, 375, 64)
                eeg_data = np.transpose(eeg_data, [3, 2, 1, 0]) # (64, 375, 50, 5)
            else:
                # For FT
                eeg_data = np.array(f['FT_Specgram'])           # (50, 17, 163, 64)
                eeg_data = np.transpose(eeg_data, [3, 2, 0, 1]) # (64, 163, 50, 17)
            
        ch_indices = []
        for i in range(64):
            # Exclude ECG(31)
            if i != 31:
                ch_indices.append(i)

        # Exclude ECG
        data = eeg_data[ch_indices,:,:,...]
        # (63, 375, 50) or (63, 375, 50, 5)
        # "..." indicates it doesn't matter whether the dimension is inputted or not

        # Exception handling for interpolation of noisy channels (Hard corded)
        data = process_noisy_channel(behavior, data)

        # Normalize
        if normalize_type != "none":
            data = self.normalize_data(data, normalize_type, frame_type)

        # Extract only the valid index
        data = data[:,:,behavior.trial_indices,...].astype(np.float32)
        # (63, 375, 50), (63, 375, 50, 5), or (63, 163, 50, 17)

        if frame_type == "filter" or frame_type == "ft":
            self.data = np.transpose(data, [2, 3, 0, 1])
            # e.g., (50, 5, 63, 375) or (50, 17, 63, 163)
        else:
            self.data = np.transpose(data, [2, 0, 1])
            # e.g., (50, 63, 375)

        print(self.data.shape)

    def normalize_data(self, data, normalize_type, frame_type):
        if normalize_type == "pre":
            # Normalize using only the data between fixations (125 samples or 38 samples)
            if frame_type == "ft":
                fixattion_data = data[:,:38,:,...]
            else:
                fixattion_data = data[:,:125,:,...]
            mean = fixattion_data.mean(axis=(1,2))
            std  = fixattion_data.std(axis=(1,2))
            # (63,5)
        else:
            # For normal
            mean = data.mean(axis=(1,2))
            std  = data.std(axis=(1,2))
            # (63,)

        if data.ndim == 4:
            # For Filter or For FT
            mean = mean.reshape([-1, 1, 1, data.shape[3]])
            std  = std.reshape([-1, 1, 1, data.shape[3]])
            # (63,1,1,5)
        else:
            # For normal
            mean = mean.reshape([-1, 1, 1])
            std  = std.reshape([-1, 1, 1])
            # (63,1,1)

        # Normalize with mean and std
        norm_data = (data - mean) / std
        return norm_data
